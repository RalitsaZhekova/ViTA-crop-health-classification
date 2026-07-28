"""Build the compact, web-ready payload downlink bundle."""

from __future__ import annotations

import hashlib
import json
import math
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image, features
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds

from prithvi_payload.cloud_classifier import CLOUD_MODEL_NAME, CLOUD_MODEL_SHA256

DOWNLINK_SCHEMA_VERSION = "1.0"
DOWNLINK_PRODUCT_TYPE = "vita.crop-condition.web-bundle"
DOWNLINK_ALGORITHM_VERSION = "compact-downlink-v1"
DEFAULT_MAX_IMAGE_DIMENSION = 1600
DEFAULT_GRID_SIZE = 16
RGB_WEBP_QUALITY = 82

CONDITION_COLOR_STOPS = (
    (0.0, (215, 48, 39)),
    (35.0, (252, 141, 89)),
    (55.0, (254, 224, 139)),
    (75.0, (145, 207, 96)),
    (100.0, (26, 152, 80)),
)
OVERLAY_CLASSES = {
    "thick_cloud": {"rgba": (245, 247, 250, 235), "semantic_value": 1},
    "thin_cloud": {"rgba": (0, 184, 217, 205), "semantic_value": 2},
    "cloud_shadow": {"rgba": (126, 87, 194, 220), "semantic_value": 3},
    "invalid": {"rgba": (255, 193, 7, 225)},
    "unusable_buffer": {"rgba": (117, 117, 117, 180)},
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read JSON product: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON product must contain an object: {path}")
    return value


def _resolve_asset(value: Any, root: Path, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing {name} asset")
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name} asset: {path}")
    return path


def _validate_grid(reference: rasterio.DatasetReader, candidate: rasterio.DatasetReader) -> None:
    if (
        reference.width != candidate.width
        or reference.height != candidate.height
        or reference.crs != candidate.crs
        or not reference.transform.almost_equals(candidate.transform)
    ):
        raise ValueError("Every downlink source raster must use the source-scene grid")


def _preview_dimensions(width: int, height: int, maximum: int) -> tuple[int, int]:
    if maximum <= 0:
        raise ValueError("max_image_dimension must be positive")
    scale = min(1.0, maximum / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def _stretch_rgb(values: np.ndarray) -> np.ndarray:
    rgb = np.moveaxis(values, 0, -1).astype(np.float32, copy=False)
    finite = rgb[np.isfinite(rgb)]
    if not finite.size:
        return np.zeros(rgb.shape, dtype=np.uint8)
    low, high = np.percentile(finite, (2, 98))
    if high <= low:
        high = low + 1.0
    scaled = np.clip((rgb - low) / (high - low), 0.0, 1.0)
    return np.round(255.0 * scaled).astype(np.uint8)


def _condition_colors(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float32)
    clipped = np.where(np.isfinite(raw), np.clip(raw, 0.0, 100.0), 0.0)
    colors = np.zeros((*clipped.shape, 3), dtype=np.float32)
    for (low_value, low_color), (high_value, high_color) in zip(
        CONDITION_COLOR_STOPS[:-1], CONDITION_COLOR_STOPS[1:], strict=True
    ):
        selected = (clipped >= low_value) & (clipped <= high_value)
        fraction = np.zeros(clipped.shape, dtype=np.float32)
        fraction[selected] = (clipped[selected] - low_value) / (high_value - low_value)
        low = np.asarray(low_color, dtype=np.float32)
        high = np.asarray(high_color, dtype=np.float32)
        colors[selected] = low + fraction[selected, np.newaxis] * (high - low)
    return np.round(colors).astype(np.uint8)


def _build_overlay(
    condition: np.ndarray,
    valid_crop: np.ndarray,
    semantic: np.ndarray,
    unusable: np.ndarray,
    invalid: np.ndarray,
) -> np.ndarray:
    shape = condition.shape
    if any(array.shape != shape for array in (valid_crop, semantic, unusable, invalid)):
        raise ValueError("Preview arrays must have identical shapes")
    overlay = np.zeros((*shape, 4), dtype=np.uint8)
    measured = (valid_crop == 1) & np.isfinite(condition)
    overlay[measured, :3] = _condition_colors(condition)[measured]
    overlay[measured, 3] = 205

    safety_buffer = (unusable == 1) & np.isin(semantic, (0,)) & (invalid == 0)
    overlay[safety_buffer] = OVERLAY_CLASSES["unusable_buffer"]["rgba"]
    overlay[invalid == 1] = OVERLAY_CLASSES["invalid"]["rgba"]
    for name in ("cloud_shadow", "thin_cloud", "thick_cloud"):
        specification = OVERLAY_CLASSES[name]
        overlay[semantic == specification["semantic_value"]] = specification["rgba"]
    return overlay


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_record(path: Path, *, width: int, height: int, media_type: str) -> dict[str, Any]:
    return {
        "href": path.name,
        "media_type": media_type,
        "width": width,
        "height": height,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _round_optional(value: float | None, digits: int = 4) -> float | None:
    return None if value is None or not math.isfinite(value) else round(float(value), digits)


def _finite_summary(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {"mean": None, "median": None}
    return {
        "mean": _round_optional(float(np.mean(finite))),
        "median": _round_optional(float(np.median(finite))),
    }


def _percentage(count: int, total: int) -> float:
    return round(100.0 * count / total, 4) if total else 0.0


def _cell_bounds_wgs84(
    source: rasterio.DatasetReader,
    window: Window,
) -> list[float]:
    native = window_bounds(window, source.transform)
    west, south, east, north = transform_bounds(source.crs, "EPSG:4326", *native, densify_pts=5)
    return [round(west, 8), round(south, 8), round(east, 8), round(north, 8)]


def _build_interaction_grid(
    source: rasterio.DatasetReader,
    rasters: dict[str, rasterio.DatasetReader],
    *,
    grid_size: int,
) -> dict[str, Any]:
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    rows = min(grid_size, source.height)
    columns = min(grid_size, source.width)
    row_edges = np.rint(np.linspace(0, source.height, rows + 1)).astype(int)
    column_edges = np.rint(np.linspace(0, source.width, columns + 1)).astype(int)
    cells: list[dict[str, Any]] = []
    metric_names = ("ndvi", "gndvi", "evi", "savi")
    for row in range(rows):
        for column in range(columns):
            row_start, row_stop = row_edges[row : row + 2]
            column_start, column_stop = column_edges[column : column + 2]
            window = Window(
                column_start,
                row_start,
                column_stop - column_start,
                row_stop - row_start,
            )
            total = round(window.width) * round(window.height)
            condition = (
                rasters["condition_score"].read(1, window=window, masked=True).filled(np.nan)
            )
            valid = np.isfinite(condition) & (condition >= 0.0) & (condition <= 100.0)
            analysis_pixels = int(np.count_nonzero(valid))
            alert = rasters["alert_mask"].read(1, window=window)
            crop = rasters["crop_binary"].read(1, window=window)
            unusable = rasters["unusable_mask"].read(1, window=window)
            semantic = rasters["semantic_mask"].read(1, window=window)
            invalid = rasters["invalid_mask"].read(1, window=window)
            metrics = {
                name: _finite_summary(
                    rasters[name].read(1, window=window, masked=True).filled(np.nan)
                )["median"]
                for name in metric_names
            }
            condition_summary = _finite_summary(condition)
            cells.append(
                {
                    "id": f"r{row:02d}c{column:02d}",
                    "row": row,
                    "column": column,
                    "bounds_wgs84": _cell_bounds_wgs84(source, window),
                    "quality": {
                        "analysis_pixels": analysis_pixels,
                        "analysis_percentage": _percentage(analysis_pixels, total),
                        "crop_candidate_percentage": _percentage(
                            int(np.count_nonzero(crop == 1)), total
                        ),
                        "unusable_percentage": _percentage(
                            int(np.count_nonzero(unusable == 1)), total
                        ),
                        "thick_cloud_percentage": _percentage(
                            int(np.count_nonzero(semantic == 1)), total
                        ),
                        "thin_cloud_percentage": _percentage(
                            int(np.count_nonzero(semantic == 2)), total
                        ),
                        "cloud_shadow_percentage": _percentage(
                            int(np.count_nonzero(semantic == 3)), total
                        ),
                        "invalid_percentage": _percentage(
                            int(np.count_nonzero(invalid == 1)), total
                        ),
                    },
                    "condition_score": condition_summary["median"],
                    "alert_percentage": _percentage(
                        int(np.count_nonzero((alert == 1) & valid)), analysis_pixels
                    ),
                    "metrics": metrics,
                }
            )
    return {
        "rows": rows,
        "columns": columns,
        "cell_order": "row-major, north-to-south then west-to-east",
        "cells": cells,
    }


def _write_manifest(path: Path, manifest: dict[str, Any], asset_bytes: int) -> None:
    metadata_bytes = -1
    while True:
        manifest["package"]["metadata_bytes"] = max(metadata_bytes, 0)
        manifest["package"]["total_bytes"] = asset_bytes + max(metadata_bytes, 0)
        encoded = (
            json.dumps(
                manifest,
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        next_size = len(encoded)
        if next_size == metadata_bytes:
            path.write_bytes(encoded)
            return
        metadata_bytes = next_size


def build_downlink_bundle(
    payload_result_path: str | Path,
    *,
    output_root: str | Path,
    max_image_dimension: int = DEFAULT_MAX_IMAGE_DIMENSION,
    grid_size: int = DEFAULT_GRID_SIZE,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write exactly two web images and one authoritative metadata document."""
    if not features.check("webp"):
        raise RuntimeError("The installed Pillow build does not support WebP")
    result_path = Path(payload_result_path).resolve()
    payload = _read_json(result_path)
    if payload.get("status") != "CONDITION_COMPLETE":
        raise ValueError("Downlink packaging requires payload status CONDITION_COMPLETE")
    result_root = result_path.parent
    output = Path(output_root).resolve()
    metadata_path = output / "scene.json"
    if metadata_path.exists() and not overwrite:
        raise FileExistsError(f"Downlink bundle already exists: {metadata_path}")
    output.mkdir(parents=True, exist_ok=True)

    artifacts = payload.get("artifacts", {})
    stage_metadata = payload.get("stage_metadata", {})
    intake = stage_metadata.get("intake", {})
    condition_report_path = _resolve_asset(
        artifacts.get("condition", {}).get("report"), result_root, name="condition report"
    )
    condition_report = _read_json(condition_report_path)
    source_path = _resolve_asset(intake.get("source_path"), result_root, name="source scene")
    cloud_assets = artifacts.get("cloud", {})
    crop_assets = artifacts.get("crop", {})
    raster_paths = {
        "semantic_mask": _resolve_asset(
            cloud_assets.get("semantic_mask"), result_root, name="semantic mask"
        ),
        "unusable_mask": _resolve_asset(
            cloud_assets.get("unusable_mask"), result_root, name="unusable mask"
        ),
        "invalid_mask": _resolve_asset(
            cloud_assets.get("invalid_mask"), result_root, name="invalid mask"
        ),
        "crop_binary": _resolve_asset(
            crop_assets.get("crop_binary"), result_root, name="crop mask"
        ),
    }
    condition_assets = (
        "condition_score",
        "valid_crop_mask",
        "alert_mask",
        "ndvi",
        "gndvi",
        "evi",
        "savi",
    )
    for name in condition_assets:
        raster_paths[name] = _resolve_asset(
            condition_report.get("raster_assets", {}).get(name),
            condition_report_path.parent,
            name=name,
        )

    mapping = intake.get("logical_band_mapping", {})
    if any(role not in mapping for role in ("RED", "GREEN", "BLUE")):
        raise ValueError("Payload intake metadata is missing RGB band mapping")
    rgb_indices = [int(mapping[role]["index"]) for role in ("RED", "GREEN", "BLUE")]
    reflectance_scale = condition_report.get("radiometry", {}).get("input_scale_divisor")
    if not isinstance(reflectance_scale, (int, float)) or reflectance_scale <= 0:
        raise ValueError("Condition report is missing a positive reflectance scale")

    rgb_path = output / "scene.webp"
    overlay_path = output / "condition.png"
    with ExitStack() as stack:
        source = stack.enter_context(rasterio.open(source_path))
        if source.crs is None:
            raise ValueError("Source scene must have a CRS")
        if (
            not np.isclose(source.transform.b, 0.0)
            or not np.isclose(source.transform.d, 0.0)
            or source.transform.a <= 0
            or source.transform.e >= 0
        ):
            raise ValueError("Web image packaging requires a north-up rectilinear source grid")
        rasters = {
            name: stack.enter_context(rasterio.open(path)) for name, path in raster_paths.items()
        }
        for raster in rasters.values():
            _validate_grid(source, raster)
        preview_width, preview_height = _preview_dimensions(
            source.width, source.height, max_image_dimension
        )
        rgb = source.read(
            rgb_indices,
            out_shape=(3, preview_height, preview_width),
            resampling=Resampling.bilinear,
        ).astype(np.float32)
        rgb /= float(reflectance_scale)
        Image.fromarray(_stretch_rgb(rgb)).save(
            rgb_path,
            format="WEBP",
            quality=RGB_WEBP_QUALITY,
            method=6,
            exact=True,
        )

        condition = rasters["condition_score"].read(
            1,
            out_shape=(preview_height, preview_width),
            masked=True,
            resampling=Resampling.bilinear,
        ).filled(np.nan)
        preview_arrays = {
            name: rasters[name].read(
                1,
                out_shape=(preview_height, preview_width),
                resampling=Resampling.nearest,
            )
            for name in ("valid_crop_mask", "semantic_mask", "unusable_mask", "invalid_mask")
        }
        overlay = _build_overlay(
            condition,
            preview_arrays["valid_crop_mask"],
            preview_arrays["semantic_mask"],
            preview_arrays["unusable_mask"],
            preview_arrays["invalid_mask"],
        )
        Image.fromarray(overlay).save(
            overlay_path,
            format="PNG",
            optimize=True,
            compress_level=9,
        )
        grid = _build_interaction_grid(source, rasters, grid_size=grid_size)
        native_bounds = [float(value) for value in source.bounds]
        wgs84_bounds = [
            round(value, 8)
            for value in transform_bounds(source.crs, "EPSG:4326", *source.bounds, densify_pts=21)
        ]
        geospatial = {
            "native_crs": str(source.crs),
            "native_bounds": native_bounds,
            "native_transform": list(source.transform)[:6],
            "source_width": source.width,
            "source_height": source.height,
            "resolution": [abs(float(source.res[0])), abs(float(source.res[1]))],
            "bounds_wgs84": wgs84_bounds,
            "web_overlay_contract": "north-up bounds; images share dimensions and pixel alignment",
        }

    assets = {
        "rgb_preview": _asset_record(
            rgb_path,
            width=preview_width,
            height=preview_height,
            media_type="image/webp",
        ),
        "condition_overlay": _asset_record(
            overlay_path,
            width=preview_width,
            height=preview_height,
            media_type="image/png",
        ),
    }
    asset_bytes = sum(item["bytes"] for item in assets.values())
    crop_model = stage_metadata.get("crop", {}).get("model", {})
    manifest = {
        "schema_version": DOWNLINK_SCHEMA_VERSION,
        "product_type": DOWNLINK_PRODUCT_TYPE,
        "algorithm_version": DOWNLINK_ALGORITHM_VERSION,
        "scene_id": payload.get("scene_id"),
        "region_id": condition_report.get("region_id"),
        "sensor": payload.get("sensor"),
        "acquired_at": intake.get("acquired_at"),
        "status": condition_report.get("status"),
        "claim": "relative crop-condition screening; not an agronomic diagnosis",
        "assets": assets,
        "geospatial": geospatial,
        "quality": {
            **payload.get("summary", {}).get("cloud", {}),
            **payload.get("summary", {}).get("crop", {}),
            **condition_report.get("quality", {}),
        },
        "metrics": condition_report.get("metrics", {}),
        "condition": condition_report.get("condition", {}),
        "interaction_grid": grid,
        "legend": {
            "condition_gradient": [
                {"score": score, "rgb": list(color)} for score, color in CONDITION_COLOR_STOPS
            ],
            "classes": {
                name: {"rgba": list(specification["rgba"])}
                for name, specification in OVERLAY_CLASSES.items()
            },
            "priority": [
                "condition",
                "unusable_buffer",
                "invalid",
                "cloud_shadow",
                "thin_cloud",
                "thick_cloud",
            ],
        },
        "processing": {
            "cloud_model": {"name": CLOUD_MODEL_NAME, "sha256": CLOUD_MODEL_SHA256},
            "crop_model": crop_model,
            "index_algorithm_version": condition_report.get("index_algorithm_version"),
            "condition_algorithm_version": condition_report.get(
                "condition_algorithm_version"
            ),
            "radiometry": condition_report.get("radiometry", {}),
        },
        "limitations": condition_report.get("condition", {}).get("limitations", []),
        "warnings": condition_report.get("warnings", []),
        "package": {
            "file_count": 3,
            "metadata_file": metadata_path.name,
            "asset_bytes": asset_bytes,
            "metadata_bytes": 0,
            "total_bytes": asset_bytes,
        },
    }
    _write_manifest(metadata_path, manifest, asset_bytes)
    return _read_json(metadata_path)

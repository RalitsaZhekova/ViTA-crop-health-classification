"""Build the compact, web-ready payload downlink bundle."""

from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image, features
from prithvi_shared.files import sha256_file
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds

from prithvi_payload.balkan_crop_calibration import (
    ADAPTER_MODE,
    load_calibration,
)
from prithvi_payload.cloud_classifier import CLOUD_MODEL_NAME, CLOUD_MODEL_SHA256
from prithvi_payload.runtime_config import environment_flag

DOWNLINK_SCHEMA_VERSION = "1.0"
DOWNLINK_PRODUCT_TYPE = "vita.crop-condition.web-bundle"
DOWNLINK_ALGORITHM_VERSION = "compact-downlink-v2"
DEFAULT_MAX_IMAGE_DIMENSION = 1200
DEFAULT_GRID_SIZE = 16
MINIMUM_GRID_CELL_PIXELS = 32
RGB_WEBP_QUALITY = 82
RGB_WEBP_METHOD = int(os.environ.get("VITA_WEBP_METHOD", "3"))
PNG_COMPRESSION_LEVEL = int(os.environ.get("VITA_PNG_COMPRESSION_LEVEL", "4"))
EXPERIMENTAL_RAW_RGB_SATURATION = 0.68
EXPERIMENTAL_RAW_RGB_PERCENTILES = (1.0, 99.0)
EXPERIMENTAL_RAW_OVERLAY_ALPHA = 150
EXPERIMENTAL_RAW_OVERLAY_SATURATION = 0.72
if not 0 <= RGB_WEBP_METHOD <= 6:
    raise RuntimeError("VITA_WEBP_METHOD must be in the range 0..6")
if not 0 <= PNG_COMPRESSION_LEVEL <= 9:
    raise RuntimeError("VITA_PNG_COMPRESSION_LEVEL must be in the range 0..9")

CONDITION_COLOR_STOPS = (
    (0.0, (215, 48, 39)),
    (35.0, (252, 141, 89)),
    (55.0, (254, 224, 139)),
    (75.0, (145, 207, 96)),
    (100.0, (26, 152, 80)),
)


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


def _stretch_rgb(
    values: np.ndarray,
    *,
    channelwise: bool = False,
    channel_limits: list[list[float]] | None = None,
    percentiles: tuple[float, float] = (2.0, 98.0),
    saturation: float = 1.0,
) -> np.ndarray:
    if len(percentiles) != 2 or not 0.0 <= percentiles[0] < percentiles[1] <= 100.0:
        raise ValueError("RGB percentiles must be an increasing pair within 0..100")
    if not 0.0 <= saturation <= 1.0:
        raise ValueError("RGB saturation must be within 0..1")

    def adjust_saturation(scaled: np.ndarray) -> np.ndarray:
        if saturation == 1.0:
            return scaled
        luminance = np.sum(
            scaled * np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32),
            axis=-1,
            keepdims=True,
        )
        return np.clip(luminance + saturation * (scaled - luminance), 0.0, 1.0)

    rgb = np.moveaxis(values, 0, -1).astype(np.float32, copy=False)
    if channelwise:
        output = np.zeros(rgb.shape, dtype=np.uint8)
        valid = np.all(np.isfinite(rgb), axis=-1) & np.any(rgb != 0, axis=-1)
        for channel in range(rgb.shape[-1]):
            samples = rgb[..., channel][valid]
            if not samples.size:
                continue
            if channel_limits is not None:
                if len(channel_limits) != 3 or len(channel_limits[channel]) != 2:
                    raise ValueError("RGB channel limits must contain three low/high pairs")
                low, high = (float(value) for value in channel_limits[channel])
            else:
                low, high = np.percentile(samples, percentiles)
            if high <= low:
                high = low + 1.0
            scaled = np.clip((rgb[..., channel] - low) / (high - low), 0.0, 1.0)
            output[..., channel] = np.round(255.0 * scaled).astype(np.uint8)
        if saturation != 1.0:
            adjusted = adjust_saturation(output.astype(np.float32) / 255.0)
            output = np.round(255.0 * adjusted).astype(np.uint8)
        output[~valid] = 0
        return output
    valid = np.all(np.isfinite(rgb), axis=-1) & np.any(rgb != 0, axis=-1)
    samples = rgb[valid]
    if not samples.size:
        return np.zeros(rgb.shape, dtype=np.uint8)
    low, high = np.percentile(samples, percentiles)
    if high <= low:
        high = low + 1.0
    scaled = np.clip((rgb - low) / (high - low), 0.0, 1.0)
    output = np.zeros(rgb.shape, dtype=np.uint8)
    scaled = adjust_saturation(scaled)
    output[valid] = np.round(255.0 * scaled[valid]).astype(np.uint8)
    return output


def _calibrated_balkan_rgb(
    source: rasterio.DatasetReader,
    *,
    mapping: dict[str, Any],
    spectral_adapter: dict[str, Any],
    original_source_path: Path,
    preview_height: int,
    preview_width: int,
) -> np.ndarray:
    """Render Balkan RGB through its validated Sentinel-equivalence curves."""
    roles = ("RED", "GREEN", "BLUE")
    if any(role not in mapping for role in roles):
        raise ValueError("Balkan display calibration requires RED, GREEN, and BLUE")
    indices = [int(mapping[role]["index"]) for role in roles]
    raw = source.read(
        indices,
        out_shape=(3, preview_height, preview_width),
        resampling=Resampling.bilinear,
        out_dtype="float32",
        masked=True,
    )
    return _apply_balkan_rgb_calibration(
        raw.filled(np.nan),
        spectral_adapter=spectral_adapter,
        original_source_path=original_source_path,
    )


def _apply_balkan_rgb_calibration(
    raw: np.ndarray,
    *,
    spectral_adapter: dict[str, Any],
    original_source_path: Path,
) -> np.ndarray:
    roles = ("RED", "GREEN", "BLUE")
    calibration_path = spectral_adapter.get("calibration_path")
    if not isinstance(calibration_path, str) or not calibration_path:
        raise ValueError("Balkan display calibration path is missing")
    calibration = load_calibration(
        calibration_path,
        source_path=original_source_path,
    )
    model_units = np.asarray(raw, dtype=np.float32) * np.float32(
        calibration["source_scale_to_model_units"]
    )
    curves = {curve["band"]: curve for curve in calibration["curves"]}
    calibrated = np.empty_like(model_units)
    for index, role in enumerate(roles):
        curve = curves[role]
        calibrated[index] = np.interp(
            model_units[index],
            np.asarray(curve["source_knots"], dtype=np.float32),
            np.asarray(curve["target_values"], dtype=np.float32),
        )
    return calibrated / np.float32(10_000.0)


def prepare_rgb_preview(
    source_path: str | Path,
    *,
    mapping: dict[str, Any],
    spectral_adapter: dict[str, Any] | None,
    original_source_path: Path,
    sensor: str,
    reflectance_scale: float,
    maximum_dimension: int,
    channel_limits: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Prepare the final RGB pixels independently of condition analysis."""
    started = time.perf_counter()
    roles = ("RED", "GREEN", "BLUE")
    if any(role not in mapping for role in roles):
        raise ValueError("RGB preview requires RED, GREEN, and BLUE bands")
    rgb_indices = [int(mapping[role]["index"]) for role in roles]
    calibrated_balkan = (
        sensor == "balkan-1"
        and isinstance(spectral_adapter, dict)
        and spectral_adapter.get("mode") == ADAPTER_MODE
    )
    experimental_raw_display = calibrated_balkan and isinstance(
        spectral_adapter.get("experimental_raw_proxy"), dict
    )
    read_started = time.perf_counter()
    with rasterio.open(source_path) as source:
        width, height = _preview_dimensions(
            source.width,
            source.height,
            maximum_dimension,
        )
        if calibrated_balkan:
            rgb = _calibrated_balkan_rgb(
                source,
                mapping=mapping,
                spectral_adapter=spectral_adapter,
                original_source_path=original_source_path,
                preview_height=height,
                preview_width=width,
            )
        else:
            rgb = source.read(
                rgb_indices,
                out_shape=(3, height, width),
                resampling=Resampling.bilinear,
                out_dtype="float32",
            )
            rgb /= float(reflectance_scale)
    read_seconds = time.perf_counter() - read_started
    stretch_started = time.perf_counter()
    preview = _stretch_rgb(
        rgb,
        channelwise=sensor == "balkan-1" and not calibrated_balkan,
        channel_limits=channel_limits,
        percentiles=(EXPERIMENTAL_RAW_RGB_PERCENTILES if experimental_raw_display else (2.0, 98.0)),
        saturation=(EXPERIMENTAL_RAW_RGB_SATURATION if experimental_raw_display else 1.0),
    )
    return {
        "pixels": preview,
        "width": width,
        "height": height,
        "calibrated_balkan_display": calibrated_balkan,
        "experimental_raw_display": experimental_raw_display,
        "read_seconds": read_seconds,
        "stretch_seconds": time.perf_counter() - stretch_started,
        "seconds": time.perf_counter() - started,
    }


def _condition_colors(values: np.ndarray, *, saturation: float = 1.0) -> np.ndarray:
    if not 0.0 <= saturation <= 1.0:
        raise ValueError("Condition-overlay saturation must be within 0..1")
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
    if saturation != 1.0:
        luminance = np.sum(
            colors * np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32),
            axis=-1,
            keepdims=True,
        )
        colors = np.clip(luminance + saturation * (colors - luminance), 0.0, 255.0)
    return np.round(colors).astype(np.uint8)


def _build_overlay(
    condition: np.ndarray,
    valid_crop: np.ndarray,
    *,
    alpha: int = 205,
    saturation: float = 1.0,
) -> np.ndarray:
    shape = condition.shape
    if valid_crop.shape != shape:
        raise ValueError("Preview arrays must have identical shapes")
    overlay = np.zeros((*shape, 4), dtype=np.uint8)
    measured = (valid_crop == 1) & np.isfinite(condition)
    if isinstance(alpha, bool) or not 0 <= alpha <= 255:
        raise ValueError("Condition-overlay alpha must be within 0..255")
    overlay[measured, :3] = _condition_colors(condition[measured], saturation=saturation)
    overlay[measured, 3] = alpha
    return overlay


def _save_webp(path: Path, rgb: np.ndarray) -> None:
    Image.fromarray(rgb).save(
        path,
        format="WEBP",
        quality=RGB_WEBP_QUALITY,
        method=RGB_WEBP_METHOD,
        exact=True,
    )


def _save_png(path: Path, overlay: np.ndarray) -> None:
    Image.fromarray(overlay).save(
        path,
        format="PNG",
        optimize=False,
        compress_level=PNG_COMPRESSION_LEVEL,
    )


def _asset_record(path: Path, *, width: int, height: int, media_type: str) -> dict[str, Any]:
    return {
        "href": path.name,
        "media_type": media_type,
        "width": width,
        "height": height,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _portable_radiometry(value: Any) -> dict[str, Any]:
    """Keep calibration evidence while removing machine-local runtime paths."""
    if not isinstance(value, dict):
        return {}
    portable = dict(value)
    adapter = portable.get("spectral_adapter")
    if isinstance(adapter, dict):
        portable["spectral_adapter"] = {
            key: item for key, item in adapter.items() if key != "calibration_path"
        }
    return portable


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
    products: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    rows = min(grid_size, max(1, source.height // MINIMUM_GRID_CELL_PIXELS))
    columns = min(grid_size, max(1, source.width // MINIMUM_GRID_CELL_PIXELS))
    row_edges = np.rint(np.linspace(0, source.height, rows + 1)).astype(int)
    column_edges = np.rint(np.linspace(0, source.width, columns + 1)).astype(int)
    cells: list[dict[str, Any]] = []
    metric_names = ("ndvi", "gndvi", "evi", "savi")
    byte_names = (
        "alert_mask",
        "crop_binary",
        "unusable_mask",
        "semantic_mask",
        "invalid_mask",
    )
    full_grid_bytes = (
        source.width
        * source.height
        * (
            np.dtype(np.float32).itemsize * (1 + len(metric_names))
            + np.dtype(np.uint8).itemsize * len(byte_names)
        )
    )
    in_memory_limit = int(
        os.environ.get("VITA_DOWNLINK_GRID_IN_MEMORY_MAX_BYTES", str(512 * 1024**2))
    )
    use_in_memory_grid = products is not None or (
        environment_flag("VITA_DOWNLINK_GRID_IN_MEMORY", True)
        and full_grid_bytes <= in_memory_limit
    )
    if use_in_memory_grid:
        if products is None:
            full_condition = rasters["condition_score"].read(1, masked=True).filled(np.nan)
            full_bytes = {name: rasters[name].read(1) for name in byte_names}
            full_metrics = {
                name: rasters[name].read(1, masked=True).filled(np.nan) for name in metric_names
            }
        else:
            required = {"condition_score", *byte_names, *metric_names}
            missing = sorted(required.difference(products))
            if missing:
                raise ValueError(f"In-memory downlink products are missing: {missing}")
            expected_shape = (source.height, source.width)
            if any(np.asarray(products[name]).shape != expected_shape for name in required):
                raise ValueError("In-memory downlink products do not match the source grid")
            full_condition = np.asarray(products["condition_score"], dtype=np.float32)
            full_bytes = {name: np.asarray(products[name], dtype=np.uint8) for name in byte_names}
            full_metrics = {
                name: np.asarray(products[name], dtype=np.float32) for name in metric_names
            }

    def build_cell(
        row: int,
        column: int,
        window: Window,
        bounds_wgs84: list[float],
        condition: np.ndarray,
        byte_values: dict[str, np.ndarray],
        metric_values: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        total = round(window.width) * round(window.height)
        valid = np.isfinite(condition) & (condition >= 0.0) & (condition <= 100.0)
        analysis_pixels = int(np.count_nonzero(valid))
        metrics = {name: _finite_summary(metric_values[name])["median"] for name in metric_names}
        condition_summary = _finite_summary(condition)
        semantic = byte_values["semantic_mask"]
        return {
            "id": f"r{row:02d}c{column:02d}",
            "row": row,
            "column": column,
            "bounds_wgs84": bounds_wgs84,
            "quality": {
                "analysis_pixels": analysis_pixels,
                "analysis_percentage": _percentage(analysis_pixels, total),
                "crop_candidate_percentage": _percentage(
                    int(np.count_nonzero(byte_values["crop_binary"] == 1)), total
                ),
                "unusable_percentage": _percentage(
                    int(np.count_nonzero(byte_values["unusable_mask"] == 1)), total
                ),
                "thick_cloud_percentage": _percentage(int(np.count_nonzero(semantic == 1)), total),
                "thin_cloud_percentage": _percentage(int(np.count_nonzero(semantic == 2)), total),
                "cloud_shadow_percentage": _percentage(int(np.count_nonzero(semantic == 3)), total),
                "invalid_percentage": _percentage(
                    int(np.count_nonzero(byte_values["invalid_mask"] == 1)), total
                ),
            },
            "condition_score": condition_summary["median"],
            "alert_percentage": _percentage(
                int(np.count_nonzero((byte_values["alert_mask"] == 1) & valid)),
                analysis_pixels,
            ),
            "metrics": metrics,
        }

    jobs = []
    for row in range(rows):
        row_start, row_stop = row_edges[row : row + 2]
        for column in range(columns):
            column_start, column_stop = column_edges[column : column + 2]
            window = Window(
                column_start,
                row_start,
                column_stop - column_start,
                row_stop - row_start,
            )
            jobs.append(
                (
                    row,
                    column,
                    window,
                    _cell_bounds_wgs84(source, window),
                )
            )

    if use_in_memory_grid:

        def build_memory_cell(job: tuple[int, int, Window, list[float]]) -> dict[str, Any]:
            row, column, window, bounds_wgs84 = job
            row_slice, column_slice = window.toslices()
            return build_cell(
                row,
                column,
                window,
                bounds_wgs84,
                full_condition[row_slice, column_slice],
                {name: values[row_slice, column_slice] for name, values in full_bytes.items()},
                {name: values[row_slice, column_slice] for name, values in full_metrics.items()},
            )

        grid_threads = max(
            1,
            int(os.environ.get("VITA_DOWNLINK_GRID_THREADS", "8")),
        )
        with ThreadPoolExecutor(
            max_workers=grid_threads,
            thread_name_prefix="downlink-grid",
        ) as grid_pool:
            cells = list(grid_pool.map(build_memory_cell, jobs))
    else:
        for row in range(rows):
            row_start, row_stop = row_edges[row : row + 2]
            row_window = Window(0, row_start, source.width, row_stop - row_start)
            condition_row = (
                rasters["condition_score"].read(1, window=row_window, masked=True).filled(np.nan)
            )
            byte_rows = {name: rasters[name].read(1, window=row_window) for name in byte_names}
            metric_rows = {
                name: rasters[name].read(1, window=row_window, masked=True).filled(np.nan)
                for name in metric_names
            }
            for column in range(columns):
                column_start, column_stop = column_edges[column : column + 2]
                window = Window(
                    column_start,
                    row_start,
                    column_stop - column_start,
                    row_stop - row_start,
                )
                column_slice = slice(column_start, column_stop)
                cells.append(
                    build_cell(
                        row,
                        column,
                        window,
                        _cell_bounds_wgs84(source, window),
                        condition_row[:, column_slice],
                        {name: values[:, column_slice] for name, values in byte_rows.items()},
                        {name: values[:, column_slice] for name, values in metric_rows.items()},
                    )
                )
    return {
        "rows": rows,
        "columns": columns,
        "cell_order": "row-major, north-to-south then west-to-east",
        "cells": cells,
    }


def _resample_product(
    source: rasterio.DatasetReader,
    values: np.ndarray,
    *,
    width: int,
    height: int,
    resampling: Resampling,
) -> np.ndarray:
    """Resample an ephemeral science layer with the same GDAL kernel as disk mode."""
    source_values = np.asarray(values)
    floating = np.issubdtype(source_values.dtype, np.floating)
    nodata = -9999.0 if floating else 255
    encoded = (
        np.where(np.isfinite(source_values), source_values, nodata).astype(np.float32)
        if floating
        else source_values.astype(np.uint8, copy=False)
    )
    profile = source.profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype=str(encoded.dtype),
        nodata=nodata,
        BIGTIFF="IF_SAFER",
    )
    for option in ("compress", "predictor", "zlevel", "num_threads"):
        profile.pop(option, None)
    with MemoryFile() as memory:
        with memory.open(**profile) as destination:
            destination.write(encoded, 1)
        with memory.open() as ephemeral:
            result = ephemeral.read(
                1,
                out_shape=(height, width),
                masked=floating,
                resampling=resampling,
            )
            return result.filled(np.nan) if floating else result


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
    products: dict[str, np.ndarray] | None = None,
    prepared_rgb: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write exactly two web images and one authoritative metadata document."""
    started = time.perf_counter()
    if not features.check("webp"):
        raise RuntimeError("The installed Pillow build does not support WebP")
    result_path = Path(payload_result_path).resolve()
    payload = _read_json(result_path)
    if payload.get("status") not in {"CONDITION_COMPLETE", "DOWNLINK_READY"}:
        raise ValueError(
            "Downlink packaging requires payload status CONDITION_COMPLETE or DOWNLINK_READY"
        )
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
    original_source_path = _resolve_asset(
        intake.get("source_path"), result_root, name="source scene"
    )
    analysis = intake.get("analysis")
    use_shared_analysis = payload.get("sensor") == "balkan-1" and isinstance(analysis, dict)
    source_path = (
        _resolve_asset(analysis.get("source_path"), result_root, name="Balkan analysis scene")
        if use_shared_analysis
        else original_source_path
    )
    cloud_assets = artifacts.get("cloud", {})
    raster_paths = {}
    for name, label in (
        ("semantic_mask", "semantic mask"),
        ("unusable_mask", "unusable mask"),
        ("invalid_mask", "invalid mask"),
    ):
        if products is None or name not in products:
            raster_paths[name] = _resolve_asset(cloud_assets.get(name), result_root, name=label)
    condition_assets = (
        "condition_score",
        "valid_crop_mask",
        "alert_mask",
        "ndvi",
        "gndvi",
        "evi",
        "savi",
    )
    if products is None:
        crop_assets = artifacts.get("crop", {})
        raster_paths["crop_binary"] = _resolve_asset(
            crop_assets.get("crop_binary"), result_root, name="crop mask"
        )
        for name in condition_assets:
            raster_paths[name] = _resolve_asset(
                condition_report.get("raster_assets", {}).get(name),
                condition_report_path.parent,
                name=name,
            )

    mapping = (
        analysis.get("logical_band_mapping", {})
        if use_shared_analysis
        else intake.get("logical_band_mapping", {})
    )
    if any(role not in mapping for role in ("RED", "GREEN", "BLUE")):
        raise ValueError("Payload intake metadata is missing RGB band mapping")
    reflectance_scale = condition_report.get("radiometry", {}).get("input_scale_divisor")
    if not isinstance(reflectance_scale, (int, float)) or reflectance_scale <= 0:
        raise ValueError("Condition report is missing a positive reflectance scale")

    rgb_path = output / "scene.webp"
    overlay_path = output / "condition.png"
    image_preparation_started = time.perf_counter()
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
        interaction_products = None
        if products is not None:
            interaction_products = dict(products)
            for name in ("semantic_mask", "unusable_mask", "invalid_mask"):
                if name not in interaction_products:
                    interaction_products[name] = rasters[name].read(1)
        resource_setup_seconds = time.perf_counter() - image_preparation_started
        preview_width, preview_height = _preview_dimensions(
            source.width, source.height, max_image_dimension
        )
        spectral_adapter = condition_report.get("radiometry", {}).get("spectral_adapter")
        calibrated_balkan_display = (
            use_shared_analysis
            and isinstance(spectral_adapter, dict)
            and spectral_adapter.get("mode") == ADAPTER_MODE
        )
        experimental_raw_display = calibrated_balkan_display and isinstance(
            spectral_adapter.get("experimental_raw_proxy"), dict
        )
        overlay_alpha = EXPERIMENTAL_RAW_OVERLAY_ALPHA if experimental_raw_display else 205
        overlay_saturation = (
            EXPERIMENTAL_RAW_OVERLAY_SATURATION if experimental_raw_display else 1.0
        )
        channel_limits = None
        if use_shared_analysis:
            raw_limits = analysis.get("display", {}).get("native_source_channel_limits")
            if isinstance(raw_limits, list):
                channel_limits = [
                    [float(value) / float(reflectance_scale) for value in limits]
                    for limits in raw_limits
                ]

        def prepare_rgb() -> tuple[np.ndarray, float, float]:
            prepared = prepare_rgb_preview(
                source_path,
                mapping=mapping,
                spectral_adapter=(spectral_adapter if isinstance(spectral_adapter, dict) else None),
                original_source_path=original_source_path,
                sensor=str(payload.get("sensor")),
                reflectance_scale=float(reflectance_scale),
                maximum_dimension=max_image_dimension,
                channel_limits=channel_limits,
            )
            return (
                np.asarray(prepared["pixels"], dtype=np.uint8),
                float(prepared["read_seconds"]),
                float(prepared["stretch_seconds"]),
            )

        def prepare_compact_overlay() -> tuple[np.ndarray, float, float]:
            assert products is not None
            condition_preview_started = time.perf_counter()
            with rasterio.open(source_path) as overlay_source:
                condition = _resample_product(
                    overlay_source,
                    products["condition_score"],
                    width=preview_width,
                    height=preview_height,
                    resampling=Resampling.bilinear,
                )
                valid_crop = _resample_product(
                    overlay_source,
                    products["valid_crop_mask"],
                    width=preview_width,
                    height=preview_height,
                    resampling=Resampling.nearest,
                ).astype(np.uint8)
            condition_seconds = time.perf_counter() - condition_preview_started
            overlay_started = time.perf_counter()
            prepared = _build_overlay(
                condition,
                valid_crop,
                alpha=overlay_alpha,
                saturation=overlay_saturation,
            )
            return prepared, condition_seconds, time.perf_counter() - overlay_started

        if products is not None:
            parallel_started = time.perf_counter()
            with ThreadPoolExecutor(
                max_workers=4,
                thread_name_prefix="downlink",
            ) as pool:
                if prepared_rgb is None:
                    rgb_preparation_future = pool.submit(prepare_rgb)
                    rgb_preview = None
                else:
                    if (
                        int(prepared_rgb.get("width", 0)) != preview_width
                        or int(prepared_rgb.get("height", 0)) != preview_height
                        or bool(prepared_rgb.get("calibrated_balkan_display"))
                        != calibrated_balkan_display
                        or bool(prepared_rgb.get("experimental_raw_display"))
                        != experimental_raw_display
                    ):
                        raise ValueError("Prepared RGB preview does not match the bundle")
                    rgb_preparation_future = None
                    rgb_preview = np.asarray(prepared_rgb["pixels"], dtype=np.uint8)
                    if rgb_preview.shape != (preview_height, preview_width, 3):
                        raise ValueError("Prepared RGB preview has an invalid shape")
                    rgb_read_seconds = float(prepared_rgb["read_seconds"])
                    rgb_stretch_seconds = float(prepared_rgb["stretch_seconds"])
                overlay_preparation_future = pool.submit(prepare_compact_overlay)
                grid_started = time.perf_counter()
                grid_future = pool.submit(
                    _build_interaction_grid,
                    source,
                    rasters,
                    grid_size=grid_size,
                    products=interaction_products,
                )
                if rgb_preparation_future is not None:
                    rgb_preview, rgb_read_seconds, rgb_stretch_seconds = (
                        rgb_preparation_future.result()
                    )
                overlay, condition_preview_seconds, overlay_seconds = (
                    overlay_preparation_future.result()
                )
                image_preparation_seconds = time.perf_counter() - parallel_started
                assert rgb_preview is not None
                rgb_future = pool.submit(_save_webp, rgb_path, rgb_preview)
                overlay_future = pool.submit(_save_png, overlay_path, overlay)
                grid = grid_future.result()
                grid_seconds = time.perf_counter() - grid_started
                rgb_future.result()
                overlay_future.result()
                encoding_and_grid_seconds = time.perf_counter() - parallel_started
        else:
            rgb_preview, rgb_read_seconds, rgb_stretch_seconds = prepare_rgb()
            condition_preview_started = time.perf_counter()
            condition = (
                rasters["condition_score"]
                .read(
                    1,
                    out_shape=(preview_height, preview_width),
                    masked=True,
                    resampling=Resampling.bilinear,
                )
                .filled(np.nan)
            )
            valid_crop = rasters["valid_crop_mask"].read(
                1,
                out_shape=(preview_height, preview_width),
                resampling=Resampling.nearest,
            )
            condition_preview_seconds = time.perf_counter() - condition_preview_started
            overlay_started = time.perf_counter()
            overlay = _build_overlay(
                condition,
                valid_crop,
                alpha=overlay_alpha,
                saturation=overlay_saturation,
            )
            overlay_seconds = time.perf_counter() - overlay_started
            image_preparation_seconds = time.perf_counter() - image_preparation_started
            with ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="downlink-encode",
            ) as pool:
                encoding_started = time.perf_counter()
                rgb_future = pool.submit(_save_webp, rgb_path, rgb_preview)
                overlay_future = pool.submit(_save_png, overlay_path, overlay)
                grid_started = time.perf_counter()
                grid = _build_interaction_grid(
                    source,
                    rasters,
                    grid_size=grid_size,
                )
                grid_seconds = time.perf_counter() - grid_started
                rgb_future.result()
                overlay_future.result()
                encoding_and_grid_seconds = time.perf_counter() - encoding_started
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

    manifest_started = time.perf_counter()
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
    acquisition = stage_metadata.get("acquisition")
    if isinstance(acquisition, dict):
        source_provenance = dict(acquisition)
    else:
        source_provenance = {
            "provider": "local_file",
            "acquired_at": intake.get("acquired_at"),
        }
    cloud_summary = payload.get("summary", {}).get("cloud", {})
    source_provenance.update(
        {
            "payload_measured_thick_cloud_percentage": cloud_summary.get("thick_cloud_percentage"),
            "payload_measured_thin_cloud_percentage": cloud_summary.get("thin_cloud_percentage"),
            "payload_measured_shadow_percentage": cloud_summary.get("cloud_shadow_percentage"),
            "payload_measured_cloud_percentage": cloud_summary.get("total_cloud_percentage"),
            "payload_measured_unusable_percentage": cloud_summary.get("unusable_percentage"),
        }
    )
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
        "source": source_provenance,
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
                {
                    "score": score,
                    "rgb": _condition_colors(
                        np.asarray([score], dtype=np.float32),
                        saturation=overlay_saturation,
                    )[0].tolist(),
                }
                for score, _ in CONDITION_COLOR_STOPS
            ],
            "overlay_semantics": {
                "colored": "valid clear crop pixels with a condition score",
                "transparent": "non-crop, cloud, shadow, invalid, buffered or unmeasured pixels",
            },
        },
        "processing": {
            "cloud_model": {"name": CLOUD_MODEL_NAME, "sha256": CLOUD_MODEL_SHA256},
            "crop_model": crop_model,
            "index_algorithm_version": condition_report.get("index_algorithm_version"),
            "condition_algorithm_version": condition_report.get("condition_algorithm_version"),
            "radiometry": _portable_radiometry(condition_report.get("radiometry", {})),
            "rgb_display": {
                "input_values_modified": calibrated_balkan_display,
                "mode": (
                    "experimental raw-proxy Sentinel calibration, combined RGB "
                    "1-99% stretch, and reduced-saturation preview"
                    if experimental_raw_display
                    else "validated Balkan-1 to Sentinel calibration and combined RGB "
                    "2-98% display stretch"
                    if calibrated_balkan_display
                    else "per-channel 2-98% display stretch"
                    if payload.get("sensor") == "balkan-1"
                    else "combined RGB 2-98% display stretch"
                ),
                "scope": "web preview only",
            },
            "condition_overlay_display": {
                "alpha": overlay_alpha,
                "saturation": overlay_saturation,
                "scope": "web preview only",
            },
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
    manifest_seconds = time.perf_counter() - manifest_started
    result = _read_json(metadata_path)
    result["_runtime"] = {
        "seconds": time.perf_counter() - started,
        "image_preparation_seconds": image_preparation_seconds,
        "resource_setup_seconds": resource_setup_seconds,
        "rgb_read_seconds": rgb_read_seconds,
        "rgb_stretch_seconds": rgb_stretch_seconds,
        "condition_preview_seconds": condition_preview_seconds,
        "overlay_seconds": overlay_seconds,
        "interaction_grid_seconds": grid_seconds,
        "encoding_and_grid_seconds": encoding_and_grid_seconds,
        "manifest_seconds": manifest_seconds,
    }
    return result

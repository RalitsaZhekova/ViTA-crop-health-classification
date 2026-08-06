"""Create the single 10 m Balkan-1 raster shared by every science stage."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject, transform_bounds

ANALYSIS_BAND_ROLES = ("BLUE", "GREEN", "RED", "NIR_BROAD")
DEFAULT_ANALYSIS_RESOLUTION_METRES = 10.0
GDAL_WARP_THREADS = max(1, min(4, os.cpu_count() or 1))
DISPLAY_MAX_DIMENSION = 1600
ANALYSIS_ALGORITHM_VERSION = "balkan-shared-analysis-grid-v1"


def _cache_paths(
    intake: dict[str, Any],
    output_root: Path,
    *,
    resolution: float,
    source_indices: list[int],
) -> tuple[Path, Path, str] | None:
    source_sha256 = intake.get("source_sha256")
    if not isinstance(source_sha256, str) or len(source_sha256) != 64:
        return None
    configured = os.environ.get("VITA_BALKAN_ANALYSIS_CACHE_DIR")
    runtime_output = os.environ.get("VITA_OUTPUT_ROOT")
    if configured:
        cache_root = Path(configured).resolve()
    elif runtime_output:
        cache_root = Path(runtime_output).resolve().parent / "cache" / "balkan-analysis"
    else:
        cache_root = output_root.parent.parent / "cache" / "balkan-analysis"
    cache_key = hashlib.sha256(
        json.dumps(
            {
                "algorithm": ANALYSIS_ALGORITHM_VERSION,
                "resolution": resolution,
                "source_band_indices": source_indices,
                "source_sha256": source_sha256.lower(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    cache_root.mkdir(parents=True, exist_ok=True)
    return (
        cache_root / f"{cache_key}.tif",
        cache_root / f"{cache_key}.json",
        cache_key,
    )


def _cached_analysis(
    raster_path: Path,
    metadata_path: Path,
    cache_key: str,
    *,
    started: float,
) -> dict[str, Any] | None:
    if not raster_path.is_file() or not metadata_path.is_file():
        return None
    try:
        record = json.loads(metadata_path.read_text(encoding="utf-8"))
        with rasterio.open(raster_path) as dataset:
            valid = (
                record.get("cache", {}).get("key") == cache_key
                and dataset.tags().get("ANALYSIS_CACHE_KEY") == cache_key
                and dataset.count == len(ANALYSIS_BAND_ROLES)
                and dataset.descriptions == ANALYSIS_BAND_ROLES
            )
    except (OSError, json.JSONDecodeError, rasterio.errors.RasterioError):
        return None
    if not valid:
        return None
    record["source_path"] = str(raster_path.resolve())
    record["cache"]["hit"] = True
    record["runtime"] = {
        "seconds": time.perf_counter() - started,
        "cache_hit": True,
        "cache_key": cache_key,
    }
    return record


def _utm_crs(source_crs: CRS, bounds: rasterio.coords.BoundingBox) -> CRS:
    west, south, east, north = transform_bounds(
        source_crs,
        "EPSG:4326",
        *bounds,
        densify_pts=21,
    )
    longitude = (west + east) / 2.0
    latitude = (south + north) / 2.0
    zone = max(1, min(60, int(math.floor((longitude + 180.0) / 6.0) + 1)))
    return CRS.from_epsg((32600 if latitude >= 0 else 32700) + zone)


def materialize_balkan_analysis_grid(
    intake: dict[str, Any],
    *,
    output_root: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Reproject the validated raw reflectance bands once for all downstream work."""
    if intake.get("sensor") != "balkan-1":
        raise ValueError("Shared Balkan analysis preparation requires a Balkan-1 intake")
    source_path = Path(str(intake.get("source_path", ""))).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    mapping = intake.get("logical_band_mapping")
    if not isinstance(mapping, dict):
        raise ValueError("Balkan intake is missing its logical band mapping")
    nir_role = "NIR_BROAD" if "NIR_BROAD" in mapping else "NIR_NARROW"
    source_roles = ("BLUE", "GREEN", "RED", nir_role)
    if any(role not in mapping for role in source_roles):
        raise ValueError("Balkan intake does not provide the four analysis bands")
    source_indices = [int(mapping[role]["index"]) for role in source_roles]

    spectral_adapter = (
        intake.get("model_band_routes", {})
        .get("crop_classification", {})
        .get("spectral_adapter")
    )
    resolution = DEFAULT_ANALYSIS_RESOLUTION_METRES
    if isinstance(spectral_adapter, dict):
        resolution = float(
            spectral_adapter.get("analysis_resolution_metres", resolution)
        )
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("Balkan analysis resolution must be positive and finite")

    started = time.perf_counter()
    output_root = Path(output_root).resolve()
    output = output_root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    cache = _cache_paths(
        intake,
        output_root,
        resolution=resolution,
        source_indices=source_indices,
    )
    if cache is None:
        destination_path = output / f"{intake['scene_id']}_10m.tif"
        metadata_path = None
        cache_key = None
        if destination_path.exists() and not overwrite:
            raise FileExistsError(f"Balkan analysis grid already exists: {destination_path}")
    else:
        destination_path, metadata_path, cache_key = cache
        cached = _cached_analysis(
            destination_path,
            metadata_path,
            cache_key,
            started=started,
        )
        if cached is not None:
            return cached
    temporary_path = destination_path.with_name(
        f".{destination_path.name}.{os.getpid()}.partial"
    )
    temporary_path.unlink(missing_ok=True)

    try:
        with rasterio.open(source_path) as source:
            if source.crs is None:
                raise ValueError("Balkan source has no CRS")
            target_crs = _utm_crs(source.crs, source.bounds)
            transform, width, height = calculate_default_transform(
                source.crs,
                target_crs,
                source.width,
                source.height,
                *source.bounds,
                resolution=resolution,
            )
            nodata = source.nodata
            with rasterio.open(
                temporary_path,
                "w",
                driver="GTiff",
                width=width,
                height=height,
                count=len(ANALYSIS_BAND_ROLES),
                dtype="float32",
                crs=target_crs,
                transform=transform,
                nodata=nodata,
                tiled=True,
                blockxsize=512,
                blockysize=512,
                BIGTIFF="IF_SAFER",
            ) as destination:
                for output_index, (source_index, role) in enumerate(
                    zip(source_indices, ANALYSIS_BAND_ROLES, strict=True),
                    start=1,
                ):
                    reproject(
                        source=rasterio.band(source, source_index),
                        destination=rasterio.band(destination, output_index),
                        src_transform=source.transform,
                        src_crs=source.crs,
                        src_nodata=nodata,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        dst_nodata=nodata,
                        resampling=Resampling.average,
                        init_dest_nodata=True,
                        num_threads=GDAL_WARP_THREADS,
                    )
                    destination.set_band_description(output_index, role)
                destination.update_tags(
                    ANALYSIS_GRID="balkan-1-utm-10m-v1",
                    ANALYSIS_CACHE_KEY=cache_key or "disabled",
                    ORIGINAL_SOURCE=str(source_path),
                    RESAMPLING="average",
                    SENSOR="balkan-1",
                )
            original = {
                "crs": str(source.crs),
                "width": source.width,
                "height": source.height,
                "bounds": [float(value) for value in source.bounds],
            }
            display_scale = min(
                1.0, DISPLAY_MAX_DIMENSION / max(source.width, source.height)
            )
            display_height = max(1, round(source.height * display_scale))
            display_width = max(1, round(source.width * display_scale))
            display_rgb = source.read(
                [source_indices[2], source_indices[1], source_indices[0]],
                out_shape=(3, display_height, display_width),
                out_dtype="float32",
                resampling=Resampling.bilinear,
            )
            display_rgb = np.moveaxis(display_rgb, 0, -1)
            display_valid = np.all(np.isfinite(display_rgb), axis=-1) & np.any(
                display_rgb != 0, axis=-1
            )
            display_stretch = []
            for channel in range(3):
                samples = display_rgb[..., channel][display_valid]
                if not samples.size:
                    display_stretch.append([0.0, 1.0])
                    continue
                low, high = np.percentile(samples, (2, 98))
                if high <= low:
                    high = low + 1.0
                display_stretch.append([float(low), float(high)])
        os.replace(temporary_path, destination_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    record = {
        "schema_version": "1.0",
        "algorithm_version": ANALYSIS_ALGORITHM_VERSION,
        "source_path": str(destination_path.resolve()),
        "source_band_indices_1_based": [1, 2, 3, 4],
        "logical_band_mapping": {
            role: {"index": index, "description": role, "mapping_source": "analysis_grid"}
            for index, role in enumerate(ANALYSIS_BAND_ROLES, start=1)
        },
        "model_band_routes": {
            "cloud_detection": {
                "expected_logical_order": ["NIR_BROAD", "RED", "GREEN", "BLUE"],
                "source_band_indices": [4, 3, 2, 1],
            },
            "crop_classification": {
                "expected_logical_order": ["BLUE", "GREEN", "RED", "NIR_NARROW"],
                "source_band_indices": [1, 2, 3, 4],
            },
        },
        "raster": {
            "driver": "GTiff",
            "width": width,
            "height": height,
            "band_count": len(ANALYSIS_BAND_ROLES),
            "crs": str(target_crs),
            "transform": list(transform)[:6],
            "resolution": [resolution, resolution],
            "nodata": nodata,
            "resampling": "average",
        },
        "original_raster": original,
        "display": {
            "rgb_order": ["RED", "GREEN", "BLUE"],
            "stretch_percentiles": [2.0, 98.0],
            "native_source_channel_limits": display_stretch,
        },
        "cache": {
            "enabled": cache_key is not None,
            "hit": False,
            "key": cache_key,
        },
        "runtime": {
            "seconds": time.perf_counter() - started,
            "cache_hit": False,
            "cache_key": cache_key,
        },
    }
    if metadata_path is not None:
        temporary_metadata = metadata_path.with_name(
            f".{metadata_path.name}.{os.getpid()}.tmp"
        )
        temporary_metadata.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_metadata, metadata_path)
    return record

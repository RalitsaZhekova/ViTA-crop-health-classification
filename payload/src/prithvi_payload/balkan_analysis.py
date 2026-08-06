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
DISPLAY_MAX_DIMENSION = 1600
ANALYSIS_ALGORITHM_VERSION = "balkan-shared-analysis-grid-v3-overview"
DEFAULT_PREPARATION_MEMORY_LIMIT_BYTES = 2 * 1024 * 1024 * 1024


def _flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean value")


def _warp_threads() -> int:
    raw = os.environ.get("VITA_CPU_THREADS", "8")
    try:
        requested = int(raw)
    except ValueError as error:
        raise RuntimeError("VITA_CPU_THREADS must be an integer") from error
    return max(1, min(requested, os.cpu_count() or 1))


def _preparation_memory_limit() -> int:
    raw = os.environ.get(
        "VITA_BALKAN_PREP_MAX_BYTES",
        str(DEFAULT_PREPARATION_MEMORY_LIMIT_BYTES),
    )
    try:
        limit = int(raw)
    except ValueError as error:
        raise RuntimeError("VITA_BALKAN_PREP_MAX_BYTES must be an integer") from error
    if limit <= 0:
        raise RuntimeError("VITA_BALKAN_PREP_MAX_BYTES must be positive")
    return limit


def _shared_overview_factors(
    source: rasterio.DatasetReader,
    source_indices: list[int],
) -> list[int]:
    factors = [set(source.overviews(index)) for index in source_indices]
    return sorted(set.intersection(*factors)) if factors else []


def _preparation_strategy(
    source: rasterio.DatasetReader,
    source_indices: list[int],
    *,
    target_width: int,
    target_height: int,
) -> dict[str, Any]:
    available = _shared_overview_factors(source, source_indices)
    target_decimation = min(
        source.width / target_width,
        source.height / target_height,
    )
    eligible = [factor for factor in available if 1 < factor <= target_decimation]
    factor = max(eligible, default=None)
    enabled = _flag("VITA_BALKAN_OVERVIEW_FAST_PATH", True)
    required = _flag("VITA_BALKAN_OVERVIEW_REQUIRED", False)
    overview_width = math.ceil(source.width / factor) if factor is not None else None
    overview_height = math.ceil(source.height / factor) if factor is not None else None
    estimated_bytes = (
        len(source_indices)
        * np.dtype("float32").itemsize
        * (
            target_width * target_height
            + (
                int(overview_width) * int(overview_height)
                if overview_width is not None and overview_height is not None
                else 0
            )
        )
    )
    memory_limit = _preparation_memory_limit()
    use_overview = enabled and factor is not None and estimated_bytes <= memory_limit
    if required and not use_overview:
        reason = (
            "no common overview at or above the target resolution"
            if factor is None
            else f"estimated working set {estimated_bytes} exceeds {memory_limit} bytes"
            if estimated_bytes > memory_limit
            else "the overview fast path is disabled"
        )
        raise ValueError(f"Balkan overview fast path is required but unavailable: {reason}")
    return {
        "mode": "embedded_overview_then_average" if use_overview else "full_resolution_average",
        "overview_factor": factor if use_overview else None,
        "overview_level": available.index(factor) if use_overview else None,
        "available_overview_factors": available,
        "target_decimation": float(target_decimation),
        "overview_width": int(overview_width) if use_overview else None,
        "overview_height": int(overview_height) if use_overview else None,
        "estimated_working_set_bytes": estimated_bytes if use_overview else 0,
        "memory_limit_bytes": memory_limit,
    }


def _cache_paths(
    intake: dict[str, Any],
    output_root: Path,
    *,
    resolution: float,
    source_indices: list[int],
    strategy: dict[str, Any],
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
                "preparation_mode": strategy["mode"],
                "overview_factor": strategy["overview_factor"],
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
            raster = record.get("raster", {})
            valid = (
                record.get("algorithm_version") == ANALYSIS_ALGORITHM_VERSION
                and record.get("cache", {}).get("key") == cache_key
                and dataset.tags().get("ANALYSIS_CACHE_KEY") == cache_key
                and dataset.count == len(ANALYSIS_BAND_ROLES)
                and dataset.descriptions == ANALYSIS_BAND_ROLES
                and dataset.width == raster.get("width")
                and dataset.height == raster.get("height")
                and str(dataset.crs) == raster.get("crs")
                and list(dataset.transform)[:6] == raster.get("transform")
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


def _write_cache_metadata(path: Path, record: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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


def _display_limits(rgb: np.ndarray) -> list[list[float]]:
    stride = max(1, math.ceil(max(rgb.shape[1:]) / DISPLAY_MAX_DIMENSION))
    sampled = np.moveaxis(rgb[:, ::stride, ::stride], 0, -1)
    valid = np.all(np.isfinite(sampled), axis=-1) & np.any(sampled != 0, axis=-1)
    limits: list[list[float]] = []
    for channel in range(sampled.shape[-1]):
        values = sampled[..., channel][valid]
        if not values.size:
            limits.append([0.0, 1.0])
            continue
        low, high = np.percentile(values, (2, 98))
        if high <= low:
            high = low + 1.0
        limits.append([float(low), float(high)])
    return limits


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
    strategy_selection_started = time.perf_counter()
    output_root = Path(output_root).resolve()
    output = output_root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    overview_read_seconds = 0.0
    warp_seconds = 0.0
    write_seconds = 0.0
    display_statistics_seconds = 0.0

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
        if nodata is None or not math.isfinite(float(nodata)):
            raise ValueError("Balkan analysis requires a finite source nodata value")
        strategy = _preparation_strategy(
            source,
            source_indices,
            target_width=width,
            target_height=height,
        )
        strategy_selection_seconds = time.perf_counter() - strategy_selection_started
        original = {
            "crs": str(source.crs),
            "width": source.width,
            "height": source.height,
            "bounds": [float(value) for value in source.bounds],
            "overviews": strategy["available_overview_factors"],
        }

        cache = _cache_paths(
            intake,
            output_root,
            resolution=resolution,
            source_indices=source_indices,
            strategy=strategy,
        )
        if cache is None:
            destination_path = output / f"{intake['scene_id']}_10m.tif"
            metadata_path = None
            cache_key = None
            if destination_path.exists() and not overwrite:
                raise FileExistsError(
                    f"Balkan analysis grid already exists: {destination_path}"
                )
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
            analysis_values: np.ndarray | None = None
            if strategy["mode"] == "embedded_overview_then_average":
                overview_level = int(strategy["overview_level"])
                overview_height = int(strategy["overview_height"])
                overview_width = int(strategy["overview_width"])
                overview_started = time.perf_counter()
                with rasterio.open(
                    source_path,
                    OVERVIEW_LEVEL=f"{overview_level}only",
                ) as overview_source:
                    if (
                        overview_source.width != overview_width
                        or overview_source.height != overview_height
                        or overview_source.crs != source.crs
                    ):
                        raise ValueError("Selected Balkan overview does not match its metadata")
                    overview_values = overview_source.read(
                        source_indices,
                        out_dtype="float32",
                    )
                    overview_transform = overview_source.transform
                overview_read_seconds = time.perf_counter() - overview_started
                analysis_values = np.empty(
                    (len(ANALYSIS_BAND_ROLES), height, width),
                    dtype=np.float32,
                )
                warp_started = time.perf_counter()
                reproject(
                    source=overview_values,
                    destination=analysis_values,
                    src_transform=overview_transform,
                    src_crs=source.crs,
                    src_nodata=nodata,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    dst_nodata=nodata,
                    resampling=Resampling.average,
                    init_dest_nodata=True,
                    num_threads=_warp_threads(),
                    warp_mem_limit=1024,
                )
                warp_seconds = time.perf_counter() - warp_started

            write_started = time.perf_counter()
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
                if analysis_values is not None:
                    destination.write(analysis_values)
                else:
                    warp_started = time.perf_counter()
                    for output_index, source_index in enumerate(source_indices, start=1):
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
                            num_threads=_warp_threads(),
                            warp_mem_limit=1024,
                        )
                    warp_seconds = time.perf_counter() - warp_started
                for output_index, role in enumerate(ANALYSIS_BAND_ROLES, start=1):
                    destination.set_band_description(output_index, role)
                destination.update_tags(
                    ANALYSIS_GRID="balkan-1-utm-10m-v3",
                    ANALYSIS_CACHE_KEY=cache_key or "disabled",
                    ANALYSIS_STRATEGY=strategy["mode"],
                    ORIGINAL_SOURCE=str(source_path),
                    RESAMPLING="average",
                    SENSOR="balkan-1",
                    SOURCE_OVERVIEW_FACTOR=strategy["overview_factor"] or "none",
                )
            write_seconds = time.perf_counter() - write_started - (
                warp_seconds if analysis_values is None else 0.0
            )

            display_started = time.perf_counter()
            if analysis_values is not None:
                display_stretch = _display_limits(analysis_values[[2, 1, 0]])
            else:
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
                display_stretch = _display_limits(display_rgb)
            display_statistics_seconds = time.perf_counter() - display_started
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
        "preprocessing": {
            **strategy,
            "source_read_resampling": (
                "embedded_overview_native"
                if strategy["mode"] == "embedded_overview_then_average"
                else "full_resolution"
            ),
            "target_resampling": "average",
            "output_band_order": list(ANALYSIS_BAND_ROLES),
            "radiometry_modified": False,
        },
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
            "strategy_selection_seconds": strategy_selection_seconds,
            "overview_read_seconds": overview_read_seconds,
            "warp_seconds": warp_seconds,
            "write_seconds": write_seconds,
            "display_statistics_seconds": display_statistics_seconds,
        },
    }
    if metadata_path is not None:
        _write_cache_metadata(metadata_path, record)
    return record

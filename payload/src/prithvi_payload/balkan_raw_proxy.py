"""Experimental raw Balkan-1 adapter for local model experiments.

This module intentionally keeps the operational calibrated path unchanged.  It
turns a verified PAN-aligned raw DN raster into a compact, approximately
geolocated reflectance proxy using only scene artifacts that are already
available locally.  The result is suitable for engineering experiments, not
for publishing geospatial or radiometric measurements.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window

from prithvi_payload.balkan_alignment import ALIGNMENT_ALGORITHM, resolve_band_indices
from prithvi_payload.balkan_crop_calibration import sha256_file

RAW_PROXY_ALGORITHM = "balkan-experimental-raw-proxy-v4-single-pass-cloud-mtf"
MODEL_BANDS = ("BLUE", "GREEN", "RED", "NIR_BROAD", "PANCHROMATIC")
EXPERIMENTAL_CROP_CLASSIFICATION_THRESHOLD = 0.3
EXPERIMENTAL_HEALTH_ANALYSIS_CROP_THRESHOLD = 0.3
EXPERIMENTAL_FALLBACK_CROP_THRESHOLD = 0.5
EXPERIMENTAL_FALLBACK_HEALTH_THRESHOLD = 0.5
SPATIAL_DETAIL_SIGMA_PIXELS = 1.2
SPATIAL_DETAIL_AMOUNT = 4.0
CROP_SPATIAL_DETAIL_AMOUNT = 1.75
LOW_PAIRWISE_CORRELATION_THRESHOLD = 0.8
LOW_CORRELATION_CLOUD_DETAIL_AMOUNT = 5.0
LOW_CORRELATION_CROP_DETAIL_AMOUNT = 0.0
TELEMETRY_GROUND_CALIBRATION_VERSION = "balkan-1-attitude-ground-v1"
# Fit against all 11 supplied raw/L1ORT scene centers.  Inputs are
# [1, tan(roll), tan(pitch)]; outputs are offsets divided by spacecraft height.
LEFT_OFFSET_COEFFICIENTS = (-0.01292852, 0.99731659, -0.03316559)
ALONG_OFFSET_COEFFICIENTS = (0.00396598, -0.03294596, 1.33877149)


class RawProxyError(ValueError):
    """Raised when an experimental raw proxy cannot be built safely."""


@dataclass(frozen=True)
class TelemetryGrid:
    crs: CRS
    transform: Affine
    center_lon: float
    center_lat: float
    spacecraft_lon: float
    spacecraft_lat: float
    spacecraft_height_m: float
    roll_deg: float
    pitch_deg: float
    along_track_unit: tuple[float, float]
    left_cross_track_unit: tuple[float, float]
    detector_column_unit: tuple[float, float]


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    try:
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise RawProxyError(f"Cannot read telemetry CSV: {csv_path}") from error
    if len(rows) < 2:
        raise RawProxyError(f"Telemetry CSV has too few records: {csv_path}")
    return rows


def _finite(row: dict[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise RawProxyError(f"Telemetry field {key!r} is missing or invalid") from error
    if not math.isfinite(value):
        raise RawProxyError(f"Telemetry field {key!r} is not finite")
    return value


def _utm_for_lon_lat(lon: float, lat: float) -> CRS:
    zone = max(1, min(60, int(math.floor((lon + 180.0) / 6.0)) + 1))
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def telemetry_grid(
    position_path: str | Path,
    attitude_path: str | Path,
    *,
    width: int,
    height: int,
    pixel_size_m: float,
) -> TelemetryGrid:
    """Build an approximate rotated grid from ECEF position and roll/pitch."""
    if width <= 0 or height <= 0 or not math.isfinite(pixel_size_m) or pixel_size_m <= 0:
        raise RawProxyError("Proxy dimensions and pixel size must be positive")
    positions = _read_csv(position_path)
    attitudes = _read_csv(attitude_path)
    midpoint = positions[len(positions) // 2]
    attitude = attitudes[len(attitudes) // 2]
    ecef_to_geo = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)

    def geodetic(row: dict[str, str]) -> tuple[float, float, float]:
        lon, lat, altitude = ecef_to_geo.transform(
            _finite(row, "px"),
            _finite(row, "py"),
            _finite(row, "pz"),
        )
        return float(lon), float(lat), float(altitude)

    spacecraft_lon, spacecraft_lat, spacecraft_height = geodetic(midpoint)
    first_lon, first_lat, _ = geodetic(positions[0])
    last_lon, last_lat, _ = geodetic(positions[-1])
    crs = _utm_for_lon_lat(spacecraft_lon, spacecraft_lat)
    project = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    unproject = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    first_e, first_n = project.transform(first_lon, first_lat)
    last_e, last_n = project.transform(last_lon, last_lat)
    center_e, center_n = project.transform(spacecraft_lon, spacecraft_lat)
    delta_e = float(last_e - first_e)
    delta_n = float(last_n - first_n)
    track_norm = math.hypot(delta_e, delta_n)
    if track_norm <= 0:
        raise RawProxyError("Position telemetry does not define an along-track direction")
    along_e, along_n = delta_e / track_norm, delta_n / track_norm
    left_e, left_n = -along_n, along_e
    roll = _finite(attitude, "estRpyRoll")
    pitch = _finite(attitude, "estRpyPitch")
    attitude_features = np.asarray(
        [
            1.0,
            math.tan(math.radians(roll)),
            math.tan(math.radians(pitch)),
        ],
        dtype=np.float64,
    )
    cross_offset = spacecraft_height * float(
        np.dot(attitude_features, LEFT_OFFSET_COEFFICIENTS)
    )
    along_offset = spacecraft_height * float(
        np.dot(attitude_features, ALONG_OFFSET_COEFFICIENTS)
    )
    target_e = center_e + left_e * cross_offset + along_e * along_offset
    target_n = center_n + left_n * cross_offset + along_n * along_offset
    center_lon, center_lat = unproject.transform(target_e, target_n)

    # Sensor QA against the supplied L1ORT scene establishes that detector
    # columns run right-of-track.  Encoding that direction in the transform is
    # the geospatial equivalent of the flip-x used by the offline L1A QA,
    # without copying or reversing the large raw arrays.
    detector_column_e = -left_e
    detector_column_n = -left_n
    a = pixel_size_m * detector_column_e
    d = pixel_size_m * detector_column_n
    b = pixel_size_m * along_e
    e = pixel_size_m * along_n
    c = target_e - a * width / 2.0 - b * height / 2.0
    f = target_n - d * width / 2.0 - e * height / 2.0
    return TelemetryGrid(
        crs=crs,
        transform=Affine(a, b, c, d, e, f),
        center_lon=float(center_lon),
        center_lat=float(center_lat),
        spacecraft_lon=spacecraft_lon,
        spacecraft_lat=spacecraft_lat,
        spacecraft_height_m=spacecraft_height,
        roll_deg=roll,
        pitch_deg=pitch,
        along_track_unit=(along_e, along_n),
        left_cross_track_unit=(left_e, left_n),
        detector_column_unit=(detector_column_e, detector_column_n),
    )


def _load_json(path: str | Path, *, label: str) -> dict[str, Any]:
    json_path = Path(path)
    try:
        value = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RawProxyError(f"Cannot read {label}: {json_path}") from error
    if not isinstance(value, dict):
        raise RawProxyError(f"{label} must contain a JSON object")
    return value


def _radiometric_coefficients(path: str | Path) -> dict[str, tuple[float, float]]:
    value = _load_json(path, label="radiometric diagnostics")
    diagnostics = value.get("radiometric_diagnostics")
    if not isinstance(diagnostics, list):
        raise RawProxyError("Radiometric diagnostics contain no per-band fits")
    result: dict[str, tuple[float, float]] = {}
    aliases = {"NIR": "NIR_BROAD", "PAN": "PANCHROMATIC"}
    for item in diagnostics:
        if not isinstance(item, dict):
            continue
        band = aliases.get(str(item.get("band")), str(item.get("band")))
        if band not in MODEL_BANDS:
            continue
        try:
            slope = float(item["diagnostic_affine_slope"])
            intercept = float(item["diagnostic_affine_intercept"])
        except (KeyError, TypeError, ValueError) as error:
            raise RawProxyError(f"Radiometric fit for {band} is invalid") from error
        if not math.isfinite(slope) or slope <= 0 or not math.isfinite(intercept):
            raise RawProxyError(f"Radiometric fit for {band} is invalid")
        result[band] = (slope, intercept)
    missing = [band for band in MODEL_BANDS if band not in result]
    if missing:
        raise RawProxyError(f"Radiometric diagnostics are missing {missing}")
    return result


def _minimum_radiometric_correlation(path: str | Path) -> float | None:
    value = _load_json(path, label="radiometric diagnostics")
    diagnostics = value.get("radiometric_diagnostics")
    if not isinstance(diagnostics, list):
        return None
    correlations: list[float] = []
    for item in diagnostics:
        if not isinstance(item, dict):
            continue
        try:
            correlation = float(item["correlation"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(correlation):
            correlations.append(correlation)
    return min(correlations) if correlations else None


def _detail_restoration_amounts(
    minimum_radiometric_correlation: float | None,
) -> tuple[float, float]:
    low_pairwise_correlation = (
        minimum_radiometric_correlation is not None
        and minimum_radiometric_correlation < LOW_PAIRWISE_CORRELATION_THRESHOLD
    )
    return (
        (
            LOW_CORRELATION_CLOUD_DETAIL_AMOUNT
            if low_pairwise_correlation
            else SPATIAL_DETAIL_AMOUNT
        ),
        (
            LOW_CORRELATION_CROP_DETAIL_AMOUNT
            if low_pairwise_correlation
            else CROP_SPATIAL_DETAIL_AMOUNT
        ),
    )


def _validate_alignment(source: Path, report_path: str | Path) -> dict[str, Any]:
    report = _load_json(report_path, label="alignment report")
    if report.get("algorithm") != ALIGNMENT_ALGORITHM:
        raise RawProxyError(
            f"Raw proxy requires alignment algorithm {ALIGNMENT_ALGORITHM!r}"
        )
    output = report.get("output")
    registrations = report.get("registrations")
    if not isinstance(output, dict) or not isinstance(registrations, dict):
        raise RawProxyError("Alignment report is incomplete")
    if output.get("sha256") != sha256_file(source):
        raise RawProxyError("Alignment report does not match the aligned raw TIFF")
    if output.get("radiometry_modified") is not False:
        raise RawProxyError("Raw proxy requires a geometry-only alignment parent")
    if any(
        not isinstance(registrations.get(band), dict)
        or registrations[band].get("aligned") is not True
        for band in MODEL_BANDS
    ):
        raise RawProxyError("Every raw band must pass alignment")
    return report


def _black_reference(
    source: rasterio.io.DatasetReader,
    band_index: int,
    *,
    output_height: int,
    dark_pixels: int,
) -> tuple[np.ndarray, np.ndarray]:
    left = source.read(
        band_index,
        window=Window(0, 0, dark_pixels, source.height),
        out_shape=(output_height, 1),
        out_dtype="float32",
        resampling=Resampling.average,
    )[:, 0]
    right = source.read(
        band_index,
        window=Window(source.width - dark_pixels, 0, dark_pixels, source.height),
        out_shape=(output_height, 1),
        out_dtype="float32",
        resampling=Resampling.average,
    )[:, 0]
    return left, right


def _north_up_grid(
    sensor_grid: TelemetryGrid,
    *,
    source_width: int,
    source_height: int,
    output_pixel_size_m: float,
) -> tuple[Affine, int, int]:
    corners = [
        sensor_grid.transform * (column, row)
        for column, row in (
            (0, 0),
            (source_width, 0),
            (0, source_height),
            (source_width, source_height),
        )
    ]
    left = min(point[0] for point in corners)
    right = max(point[0] for point in corners)
    bottom = min(point[1] for point in corners)
    top = max(point[1] for point in corners)
    width = max(1, int(math.ceil((right - left) / output_pixel_size_m)))
    height = max(1, int(math.ceil((top - bottom) / output_pixel_size_m)))
    return (
        Affine(output_pixel_size_m, 0.0, left, 0.0, -output_pixel_size_m, top),
        width,
        height,
    )


def _warp_threads() -> int:
    raw = os.environ.get("VITA_CPU_THREADS", "8")
    try:
        requested = int(raw)
    except ValueError as error:
        raise RawProxyError("VITA_CPU_THREADS must be an integer") from error
    return max(1, min(requested, os.cpu_count() or 1))


def _write_proxy_calibration(
    parent_path: str | Path,
    output_path: Path,
    *,
    raw_source: Path,
    alignment_report: Path,
    proxy_report: Path,
    cloud_detail_amount: float = SPATIAL_DETAIL_AMOUNT,
    crop_detail_amount: float = CROP_SPATIAL_DETAIL_AMOUNT,
) -> Path:
    value = deepcopy(_load_json(parent_path, label="parent crop calibration"))
    uses_cross_scene_fallback = isinstance(
        value.get("experimental_cross_scene_fallback"), dict
    )
    crop_threshold = (
        EXPERIMENTAL_FALLBACK_CROP_THRESHOLD
        if uses_cross_scene_fallback
        else EXPERIMENTAL_CROP_CLASSIFICATION_THRESHOLD
    )
    health_threshold = (
        EXPERIMENTAL_FALLBACK_HEALTH_THRESHOLD
        if uses_cross_scene_fallback
        else EXPERIMENTAL_HEALTH_ANALYSIS_CROP_THRESHOLD
    )
    value["created_at"] = datetime.now(UTC).isoformat()
    value["source"] = {
        "filename": output_path.name,
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
    }
    value["experimental_raw_proxy"] = {
        "status": "UNQUALIFIED_ENGINEERING_EXPERIMENT",
        "algorithm": RAW_PROXY_ALGORITHM,
        "raw_source": raw_source.name,
        "alignment_report": alignment_report.name,
        "proxy_report": proxy_report.name,
        "parent_calibration": Path(parent_path).name,
        "crop_probability_threshold": crop_threshold,
        "health_analysis_crop_threshold": health_threshold,
        "cloud_spatial_detail_restoration": {
            "method": "nodata_aware_unsharp_mask",
            "sigma_pixels": SPATIAL_DETAIL_SIGMA_PIXELS,
            "amount": cloud_detail_amount,
            "bands": ["RED", "GREEN", "NIR_BROAD"],
        },
        "crop_spatial_detail_restoration": {
            "method": "nodata_aware_unsharp_mask",
            "sigma_pixels": SPATIAL_DETAIL_SIGMA_PIXELS,
            "amount": crop_detail_amount,
            "bands": ["BLUE", "GREEN", "RED", "NIR_BROAD"],
        },
        "threshold_basis": (
            "conservative 0.5 threshold for an unqualified cross-scene crop adapter"
            if uses_cross_scene_fallback
            else "the operational Balkan threshold is retained after model-input-only "
            "spatial-detail restoration; raw-proxy parity remains experimental"
        ),
        "warning": (
            "Parent validation applies to the supplied L1ORT product; it does not "
            "validate this telemetry-geolocated raw proxy."
        ),
    }
    destination = output_path.with_name(f"{output_path.stem}.crop_calibration.json")
    temporary = destination.with_name(f".{destination.name}.partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def build_raw_model_proxy(
    aligned_raw_path: str | Path,
    output_path: str | Path,
    *,
    alignment_report_path: str | Path,
    radiometric_diagnostics_path: str | Path,
    position_path: str | Path,
    attitude_path: str | Path,
    parent_calibration_path: str | Path,
    raw_pixel_size_m: float = 1.5,
    output_pixel_size_m: float = 10.0,
    inactive_border_pixels: int = 88,
    dark_reference_pixels: int = 68,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build a final-grid reflectance proxy from aligned native-resolution raw DN.

    Radiometric correction happens before a single native-to-10 m average warp.  The
    output is already north-up, so the shared Balkan analysis stage can use it
    directly instead of averaging the 10 m pixels a second time.
    """
    started = datetime.now(UTC)
    source_path = Path(aligned_raw_path).resolve()
    output_path = Path(output_path).resolve()
    alignment_path = Path(alignment_report_path).resolve()
    proxy_report_path = output_path.with_suffix(".raw_proxy.json")
    if not source_path.is_file():
        raise RawProxyError(f"Aligned raw TIFF does not exist: {source_path}")
    for target in (output_path, proxy_report_path):
        if target.exists() and not overwrite:
            raise RawProxyError(f"Output exists; pass overwrite=True: {target}")
    if output_pixel_size_m < raw_pixel_size_m:
        raise RawProxyError("Raw proxy output must not upsample the raw scene")
    alignment = _validate_alignment(source_path, alignment_path)
    raw_parent_path = Path(str(alignment.get("source", {}).get("path", ""))).resolve()
    if not raw_parent_path.is_file():
        raise RawProxyError(f"Alignment raw parent does not exist: {raw_parent_path}")
    if alignment["source"].get("sha256") != sha256_file(raw_parent_path):
        raise RawProxyError("Alignment report does not match its untouched raw parent")
    coefficients = _radiometric_coefficients(radiometric_diagnostics_path)
    minimum_radiometric_correlation = _minimum_radiometric_correlation(
        radiometric_diagnostics_path
    )
    cloud_detail_amount, crop_detail_amount = _detail_restoration_amounts(
        minimum_radiometric_correlation
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(f".{output_path.name}.partial")
    partial_path.unlink(missing_ok=True)

    with rasterio.open(source_path) as source, rasterio.open(raw_parent_path) as raw_parent:
        if source.count != 5:
            raise RawProxyError("Aligned raw TIFF must contain five bands")
        if (raw_parent.width, raw_parent.height, raw_parent.count) != (
            source.width,
            source.height,
            source.count,
        ):
            raise RawProxyError("Aligned TIFF geometry does not match its raw parent")
        if inactive_border_pixels * 2 >= source.width:
            raise RawProxyError("Inactive detector border removes the complete raster")
        band_indices = resolve_band_indices(source.descriptions)
        raw_band_indices = resolve_band_indices(raw_parent.descriptions)
        active_width = source.width - 2 * inactive_border_pixels
        sensor_grid = telemetry_grid(
            position_path,
            attitude_path,
            width=active_width,
            height=source.height,
            pixel_size_m=raw_pixel_size_m,
        )
        output_transform, width, height = _north_up_grid(
            sensor_grid,
            source_width=active_width,
            source_height=source.height,
            output_pixel_size_m=output_pixel_size_m,
        )
        profile = {
            "driver": "GTiff",
            "width": width,
            "height": height,
            "count": 5,
            "dtype": "float32",
            "crs": sensor_grid.crs,
            "transform": output_transform,
            "nodata": 0.0,
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
            "compress": "deflate",
            "predictor": 3,
            "BIGTIFF": "IF_SAFER",
        }
        try:
            with rasterio.open(partial_path, "w", **profile) as destination:
                destination.update_tags(
                    SENSOR="balkan-1",
                    PROCESSING_LEVEL="EXPERIMENTAL_RAW_MODEL_PROXY",
                    RAW_PROXY_ALGORITHM=RAW_PROXY_ALGORITHM,
                    RAW_PARENT_SHA256=alignment["source"]["sha256"],
                    ALIGNED_PARENT_SHA256=alignment["output"]["sha256"],
                    RADIOMETRY="APPROXIMATE_QA_AFFINE",
                    GEOLOCATION="APPROXIMATE_TELEMETRY_ROLL_PITCH",
                    DETECTOR_ORIENTATION="COLUMNS_RIGHT_OF_GROUND_TRACK",
                    DISPLAY_ORIENTATION="NORTH_UP_SINGLE_PASS_FROM_NATIVE_RAW",
                    SCIENCE_QUALIFIED="FALSE",
                    REFLECTANCE_SCALE="1.0",
                    RAW_TO_MODEL_RESAMPLING_PASSES="1",
                )
                active_window = Window(
                    inactive_border_pixels,
                    0,
                    active_width,
                    source.height,
                )
                for destination_index, band in enumerate(MODEL_BANDS, start=1):
                    source_index = band_indices[band]
                    values = source.read(
                        source_index,
                        window=active_window,
                        out_dtype="float32",
                    )
                    left, right = _black_reference(
                        raw_parent,
                        raw_band_indices[band],
                        output_height=source.height,
                        dark_pixels=dark_reference_pixels,
                    )
                    values[values <= 0] = np.nan
                    difference = right - left
                    correction_chunk_width = 512
                    denominator = max(active_width - 1, 1)
                    for column_start in range(0, active_width, correction_chunk_width):
                        column_stop = min(
                            active_width,
                            column_start + correction_chunk_width,
                        )
                        fraction = (
                            np.arange(column_start, column_stop, dtype=np.float32)
                            / np.float32(denominator)
                        )
                        values[:, column_start:column_stop] -= (
                            left[:, None] + difference[:, None] * fraction[None, :]
                        )
                    slope, intercept = coefficients[band]
                    values *= np.float32(slope)
                    values += np.float32(intercept)
                    np.clip(values, 0.0, 1.5, out=values)
                    reflectance = np.zeros((height, width), dtype=np.float32)
                    reproject(
                        source=values,
                        destination=reflectance,
                        src_transform=sensor_grid.transform,
                        src_crs=sensor_grid.crs,
                        src_nodata=np.nan,
                        dst_transform=output_transform,
                        dst_crs=sensor_grid.crs,
                        dst_nodata=0.0,
                        resampling=Resampling.average,
                        init_dest_nodata=True,
                        num_threads=_warp_threads(),
                        warp_mem_limit=1024,
                    )
                    reflectance[~np.isfinite(reflectance)] = 0.0
                    destination.write(reflectance, destination_index)
                    destination.set_band_description(destination_index, band)
                factors = [factor for factor in (2, 4, 8) if min(width, height) // factor >= 128]
                if factors:
                    destination.build_overviews(factors, Resampling.average)
                    destination.update_tags(ns="rio_overview", resampling="average")
            os.replace(partial_path, output_path)
        except Exception:
            partial_path.unlink(missing_ok=True)
            raise

    proxy_report: dict[str, Any] = {
        "schema_version": 1,
        "algorithm": RAW_PROXY_ALGORITHM,
        "status": "UNQUALIFIED_ENGINEERING_EXPERIMENT",
        "created_at": started.isoformat(),
        "source": {
            "path": str(source_path),
            "sha256": alignment["output"]["sha256"],
            "raw_parent_sha256": alignment["source"]["sha256"],
            "alignment_report": alignment_path.name,
        },
        "output": {
            "path": str(output_path),
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
            "width": width,
            "height": height,
            "crs": sensor_grid.crs.to_string(),
            "transform": list(output_transform)[:6],
            "pixel_size_m": output_pixel_size_m,
            "band_order": list(MODEL_BANDS),
            "units": "approximate top-of-atmosphere reflectance proxy",
            "grid_orientation": "north_up",
            "raw_to_model_resampling_passes": 1,
            "cloud_input_spatial_detail_restoration": {
                "method": "nodata-aware unsharp mask",
                "sigma_pixels": SPATIAL_DETAIL_SIGMA_PIXELS,
                "amount": cloud_detail_amount,
                "bands": ["RED", "GREEN", "NIR_BROAD"],
                "basis": (
                    "same-grid L1ORT semantic agreement experiments on raw scenes "
                    "3086 and 3408"
                ),
                "application": "cloud model input only; source/display reflectance is unchanged",
            },
            "radiometric_pairwise_minimum_correlation": minimum_radiometric_correlation,
        },
        "radiometry": {
            "dark_reference": (
                "per-native-source-line mean of untouched raw-parent left/right "
                "dark pixels with cross-track interpolation"
            ),
            "dark_reference_source": raw_parent_path.name,
            "inactive_border_pixels_removed_each_side": inactive_border_pixels,
            "dark_reference_pixels_per_side": dark_reference_pixels,
            "coefficients": {
                band: {"slope": slope, "intercept": intercept}
                for band, (slope, intercept) in coefficients.items()
            },
            "diagnostics": Path(radiometric_diagnostics_path).name,
            "column_fixed_pattern_correction": "not applied",
        },
        "geolocation": {
            "method": "ECEF track plus midpoint roll/pitch flat-Earth intersection",
            "spacecraft_lon_lat_height": [
                sensor_grid.spacecraft_lon,
                sensor_grid.spacecraft_lat,
                sensor_grid.spacecraft_height_m,
            ],
            "estimated_scene_center_lon_lat": [
                sensor_grid.center_lon,
                sensor_grid.center_lat,
            ],
            "roll_deg": sensor_grid.roll_deg,
            "pitch_deg": sensor_grid.pitch_deg,
            "detector_column_direction": "right_of_ground_track",
            "orientation_correction": "north-up warp from the corrected native raw grid",
            "detector_column_unit": list(sensor_grid.detector_column_unit),
            "terrain_model": "not applied",
            "camera_boresight_model": "not available",
            "attitude_ground_calibration": {
                "version": TELEMETRY_GROUND_CALIBRATION_VERSION,
                "fit_scene_count": 11,
                "leave_one_out_position_rmse_m": 4198.95,
                "left_offset_coefficients": list(LEFT_OFFSET_COEFFICIENTS),
                "along_offset_coefficients": list(ALONG_OFFSET_COEFFICIENTS),
            },
        },
        "warnings": [
            "Geolocation is approximate and must not be used for measurement or navigation.",
            "Radiometry uses scene QA affine fits, not the missing absolute sensor calibration.",
            "The parent crop calibration validation does not qualify this raw proxy.",
        ],
    }
    temporary_report = proxy_report_path.with_name(f".{proxy_report_path.name}.partial")
    temporary_report.write_text(
        json.dumps(proxy_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_report, proxy_report_path)
    calibration_path = _write_proxy_calibration(
        parent_calibration_path,
        output_path,
        raw_source=Path(alignment["source"]["path"]),
        alignment_report=alignment_path,
        proxy_report=proxy_report_path,
        cloud_detail_amount=cloud_detail_amount,
        crop_detail_amount=crop_detail_amount,
    )
    proxy_report["output"]["crop_calibration"] = calibration_path.name
    proxy_report_path.write_text(
        json.dumps(proxy_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return proxy_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vita-balkan-raw-proxy",
        description="Build an explicitly experimental model proxy from PAN-aligned raw DN",
    )
    parser.add_argument("aligned_raw", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--alignment-report", type=Path, required=True)
    parser.add_argument("--radiometric-diagnostics", type=Path, required=True)
    parser.add_argument("--position", type=Path, required=True)
    parser.add_argument("--attitude", type=Path, required=True)
    parser.add_argument("--parent-calibration", type=Path, required=True)
    parser.add_argument("--raw-pixel-size", type=float, default=1.5)
    parser.add_argument("--output-pixel-size", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        result = build_raw_model_proxy(
            args.aligned_raw,
            args.output,
            alignment_report_path=args.alignment_report,
            radiometric_diagnostics_path=args.radiometric_diagnostics,
            position_path=args.position,
            attitude_path=args.attitude,
            parent_calibration_path=args.parent_calibration,
            raw_pixel_size_m=args.raw_pixel_size,
            output_pixel_size_m=args.output_pixel_size,
            overwrite=args.overwrite,
        )
    except (RawProxyError, OSError, rasterio.errors.RasterioError) as error:
        print(f"Balkan raw proxy failed: {error}")
        raise SystemExit(2) from None
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

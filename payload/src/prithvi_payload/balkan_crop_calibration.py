"""Validated Balkan-1 to Sentinel crop-model radiometric calibration."""

from __future__ import annotations

import hashlib
import json
import string
from concurrent.futures import Executor
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from prithvi_shared.calibration import (
    CROP_CLASSIFICATION_THRESHOLD,
    HEALTH_ANALYSIS_CROP_THRESHOLD,
)

CALIBRATION_SCHEMA_VERSION = "1.0"
ADAPTER_MODE = "BALKAN_1_SENTINEL_MONOTONIC_V1"
MODEL_BAND_ORDER = ("BLUE", "GREEN", "RED", "NIR_NARROW")
SOURCE_BAND_ORDER = ("BLUE", "GREEN", "RED", "NIR_BROAD")
DEFAULT_ANALYSIS_RESOLUTION_METRES = 10.0
MIN_VALIDATION_PIXELS = 10_000
MIN_BAND_CORRELATION = 0.75
MIN_MEAN_BAND_CORRELATION = 0.80
# Balkan reflectance is calibrated into the selected model's Sentinel-equivalent
# input domain, so it uses the same model-validation probability thresholds.
BALKAN_CROP_CLASSIFICATION_THRESHOLD = CROP_CLASSIFICATION_THRESHOLD
BALKAN_HEALTH_ANALYSIS_CROP_THRESHOLD = HEALTH_ANALYSIS_CROP_THRESHOLD


class CalibrationError(ValueError):
    """Raised when a calibration sidecar cannot be trusted for inference."""


def default_calibration_path(source_path: str | Path) -> Path:
    """Return the non-image sidecar path associated with a Balkan product."""
    source = Path(source_path)
    return source.with_name(f"{source.stem}.crop_calibration.json")


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=16)
def _sha256_for_unchanged_file(
    resolved_path: str,
    size: int,
    modified_ns: int,
    changed_ns: int,
) -> str:
    """Hash once per process while an immutable payload file's identity is unchanged."""
    del size, modified_ns, changed_ns
    return sha256_file(resolved_path)


def verified_file_sha256(path: str | Path) -> str:
    """Return a cached digest invalidated by path, size, mtime or metadata change."""
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return _sha256_for_unchanged_file(
        str(resolved),
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _finite_numbers(values: Any, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"{name} must contain numeric values") from exc
    if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
        raise CalibrationError(f"{name} must contain at least two finite values")
    return array


def load_calibration(
    path: str | Path,
    *,
    source_path: str | Path,
    verify_sha256: bool = True,
) -> dict[str, Any]:
    """Load and fail-closed validate a crop calibration sidecar."""
    calibration_path = Path(path)
    source = Path(source_path)
    try:
        value = json.loads(calibration_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"Cannot read crop calibration: {calibration_path}") from exc
    if not isinstance(value, dict):
        raise CalibrationError("Crop calibration root must be an object")
    if value.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise CalibrationError(
            f"Unsupported crop calibration schema: {value.get('schema_version')!r}"
        )
    if value.get("adapter_mode") != ADAPTER_MODE:
        raise CalibrationError(
            f"Unsupported crop calibration adapter: {value.get('adapter_mode')!r}"
        )
    if value.get("sensor") != "balkan-1":
        raise CalibrationError("Crop calibration sensor must be balkan-1")
    if value.get("source_band_order") != list(SOURCE_BAND_ORDER):
        raise CalibrationError("Crop calibration source band order is invalid")
    if value.get("model_band_order") != list(MODEL_BAND_ORDER):
        raise CalibrationError("Crop calibration model band order is invalid")
    source_indices = value.get("source_band_indices_1_based")
    if (
        not isinstance(source_indices, list)
        or len(source_indices) != 4
        or any(type(index) is not int or index <= 0 for index in source_indices)
        or len(set(source_indices)) != 4
    ):
        raise CalibrationError("Crop calibration source band indices are invalid")

    provenance = value.get("source")
    if not isinstance(provenance, dict):
        raise CalibrationError("Crop calibration source provenance is missing")
    if provenance.get("bytes") != source.stat().st_size:
        raise CalibrationError("Crop calibration does not match the source file size")
    expected_sha = provenance.get("sha256")
    if (
        not isinstance(expected_sha, str)
        or len(expected_sha) != 64
        or any(character not in string.hexdigits for character in expected_sha)
    ):
        raise CalibrationError("Crop calibration source SHA-256 is invalid")
    if verify_sha256 and verified_file_sha256(source) != expected_sha.lower():
        raise CalibrationError("Crop calibration does not match the source SHA-256")

    resolution = value.get("analysis_resolution_metres")
    if (
        not isinstance(resolution, (int, float))
        or float(resolution) != DEFAULT_ANALYSIS_RESOLUTION_METRES
    ):
        raise CalibrationError("Crop calibration analysis resolution must be 10 metres")
    source_scale = value.get("source_scale_to_model_units")
    if (
        not isinstance(source_scale, (int, float))
        or isinstance(source_scale, bool)
        or not np.isfinite(source_scale)
        or float(source_scale) <= 0
    ):
        raise CalibrationError("Crop calibration source scale must be positive")

    acquired_at = value.get("acquired_at")
    if not isinstance(acquired_at, str):
        raise CalibrationError("Crop calibration acquisition timestamp is missing")
    try:
        parsed_acquired_at = datetime.fromisoformat(acquired_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CalibrationError("Crop calibration acquisition timestamp is invalid") from exc
    if parsed_acquired_at.tzinfo is None:
        raise CalibrationError("Crop calibration acquisition timestamp must be timezone-aware")

    curves = value.get("curves")
    if not isinstance(curves, list) or len(curves) != 4:
        raise CalibrationError("Crop calibration must contain exactly four band curves")
    maximum_target = 0.0
    for index, curve in enumerate(curves):
        if not isinstance(curve, dict) or curve.get("band") != SOURCE_BAND_ORDER[index]:
            raise CalibrationError(f"Crop calibration curve {index + 1} has the wrong band")
        source_knots = _finite_numbers(
            curve.get("source_knots"), name=f"curve {index + 1} source_knots"
        )
        target_values = _finite_numbers(
            curve.get("target_values"), name=f"curve {index + 1} target_values"
        )
        if source_knots.size != target_values.size:
            raise CalibrationError(f"Crop calibration curve {index + 1} lengths differ")
        if np.any(np.diff(source_knots) <= 0):
            raise CalibrationError(f"Crop calibration curve {index + 1} knots are not increasing")
        if np.any(np.diff(target_values) < 0):
            raise CalibrationError(f"Crop calibration curve {index + 1} is not monotonic")
        if source_knots.min() < 0 or source_knots.max() > 20_000:
            raise CalibrationError(f"Crop calibration curve {index + 1} source scale is invalid")
        if target_values.min() < 0 or target_values.max() > 20_000:
            raise CalibrationError(f"Crop calibration curve {index + 1} target scale is invalid")
        maximum_target = max(maximum_target, float(target_values.max()))
    if maximum_target <= 100:
        raise CalibrationError("Crop calibration target is not in Sentinel model scale units")

    reference = value.get("reference")
    if not isinstance(reference, dict) or reference.get("sensor") != "sentinel-2":
        raise CalibrationError("Crop calibration Sentinel reference provenance is missing")
    reference_sha = reference.get("sha256")
    if (
        not isinstance(reference_sha, str)
        or len(reference_sha) != 64
        or any(character not in string.hexdigits for character in reference_sha)
    ):
        raise CalibrationError("Crop calibration Sentinel reference SHA-256 is invalid")

    validation = value.get("validation")
    if not isinstance(validation, dict) or validation.get("status") != "PASS":
        raise CalibrationError("Crop calibration validation did not pass")
    valid_pixels = validation.get("held_out_valid_pixels")
    correlations = validation.get("held_out_band_correlations")
    if not isinstance(valid_pixels, int) or valid_pixels < MIN_VALIDATION_PIXELS:
        raise CalibrationError("Crop calibration has too few held-out validation pixels")
    correlation_values = _finite_numbers(correlations, name="held-out band correlations")
    if correlation_values.size != 4:
        raise CalibrationError("Crop calibration requires four held-out correlations")
    if np.any((correlation_values < -1.0) | (correlation_values > 1.0)):
        raise CalibrationError("Held-out band correlations must be within [-1, 1]")
    if correlation_values.min() < MIN_BAND_CORRELATION:
        raise CalibrationError("A crop calibration band correlation is below the quality gate")
    if correlation_values.mean() < MIN_MEAN_BAND_CORRELATION:
        raise CalibrationError("Mean crop calibration correlation is below the quality gate")
    return value


def apply_calibration(
    values: np.ndarray,
    calibration: dict[str, Any],
    *,
    executor: Executor | None = None,
) -> np.ndarray:
    """Apply four monotonic lookup curves to values in model scale units."""
    image = np.asarray(values, dtype=np.float32)
    if image.ndim < 2 or image.shape[0] != 4:
        raise ValueError("Calibrated crop input must have four bands on axis zero")
    result = np.empty_like(image)
    def interpolate(index: int) -> np.ndarray:
        curve = calibration["curves"][index]
        return np.interp(
            image[index],
            np.asarray(curve["source_knots"], dtype=np.float32),
            np.asarray(curve["target_values"], dtype=np.float32),
        )
    if executor is None:
        for index in range(4):
            result[index] = interpolate(index)
    else:
        futures = [executor.submit(interpolate, index) for index in range(4)]
        for index, future in enumerate(futures):
            result[index] = future.result()
    return result

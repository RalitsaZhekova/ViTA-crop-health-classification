"""Memory-bounded Balkan-1 band registration and GeoTIFF preparation.

The registration method is adapted from the Test-2 branch of
``stelspirou-git/balkan1-band-alignment`` at commit
``5688c564f733c754601240e6263b48f2792aafeb``.  That project established
the important method: measure local cross-correlation shifts against PAN,
fit a smooth shift field, and retain a global-shift fallback.

This integration adds the production boundaries needed by ViTA:

* windowed raster reads and writes instead of full-scene dense tensors;
* raw Balkan band-name and BandStartRow metadata handling;
* fail-closed registration quality checks;
* a deterministic five-band GeoTIFF contract for pipeline intake; and
* explicit separation between alignment and georeferencing/radiometry.

Alignment does not create a CRS, orthorectify an L0 image, repair detector
defects, or calibrate digital numbers to reflectance.  A no-CRS result may be
created for inspection, but it is deliberately marked as not pipeline-ready.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.windows import Window
from scipy.ndimage import map_coordinates, sobel

from prithvi_payload.balkan_crop_calibration import sha256_file

ALIGNMENT_ALGORITHM = "balkan-pan-local-shift-field-v1"
ALIGNMENT_SOURCE_URL = "https://github.com/stelspirou-git/balkan1-band-alignment/tree/Test-2"
ALIGNMENT_SOURCE_COMMIT = "5688c564f733c754601240e6263b48f2792aafeb"

OUTPUT_BAND_ORDER = ("BLUE", "GREEN", "RED", "NIR_BROAD", "PANCHROMATIC")
REFERENCE_BAND = "PANCHROMATIC"

_BAND_ALIASES = {
    "BLUE": {"BLUE", "B02", "B2", "BAND1"},
    "GREEN": {"GREEN", "B03", "B3", "BAND2"},
    "RED": {"RED", "B04", "B4", "BAND3"},
    "NIR_BROAD": {"NIR", "NIRBROAD", "B08", "B8", "BAND7"},
    "PANCHROMATIC": {"PAN", "PANCHROMATIC", "BAND0"},
}
_RAW_SENSOR_BAND_BY_ROLE = {
    "BLUE": 1,
    "GREEN": 2,
    "RED": 3,
    "NIR_BROAD": 7,
    "PANCHROMATIC": 0,
}


class BalkanAlignmentError(ValueError):
    """Raised when registration cannot produce a trustworthy result."""


@dataclass(frozen=True)
class AlignmentConfig:
    """Tunable registration and bounded-write settings."""

    grid_rows: int = 9
    grid_cols: int = 5
    measurement_tile_size: int = 256
    local_search_radius_px: int = 48
    global_search_radius_px: int = 64
    # The upstream PyTorch prototype used 0.25. NumPy's FFT/PSR implementation
    # produces slightly lower values on the real 3408 products; 0.20 is paired
    # with a separate spatial-consensus and residual gate below.
    minimum_confidence: float = 0.20
    minimum_local_measurements: int = 4
    maximum_fit_mean_residual_px: float = 3.0
    maximum_fit_residual_px: float = 10.0
    write_block_size: int = 512

    def validate(self) -> None:
        if self.grid_rows < 2 or self.grid_cols < 2:
            raise BalkanAlignmentError("Alignment grids require at least two rows and columns")
        if self.measurement_tile_size < 32:
            raise BalkanAlignmentError("Alignment measurement tiles must be at least 32 pixels")
        if self.local_search_radius_px < 1 or self.global_search_radius_px < 1:
            raise BalkanAlignmentError("Alignment search radii must be positive")
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise BalkanAlignmentError("Alignment minimum confidence must be within [0, 1]")
        if self.minimum_local_measurements < 4:
            raise BalkanAlignmentError("At least four local measurements are required")
        if (
            self.maximum_fit_mean_residual_px <= 0
            or self.maximum_fit_residual_px <= 0
            or self.write_block_size < 16
        ):
            raise BalkanAlignmentError("Residual limits and write block size must be positive")


@dataclass(frozen=True)
class LocalMeasurement:
    row: float
    column: float
    dy: float
    dx: float
    confidence: float
    method: str


@dataclass(frozen=True)
class ShiftField:
    """Bilinear shift field in normalized image coordinates."""

    coefficients_dy: tuple[float, float, float, float]
    coefficients_dx: tuple[float, float, float, float]
    height: int
    width: int

    def evaluate(self, rows: np.ndarray, columns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        normalized_rows = _normalized_axis(rows, self.height)
        normalized_columns = _normalized_axis(columns, self.width)
        design = np.stack(
            (
                np.ones_like(normalized_rows),
                normalized_rows,
                normalized_columns,
                normalized_rows * normalized_columns,
            ),
            axis=-1,
        )
        dy = design @ np.asarray(self.coefficients_dy, dtype=np.float64)
        dx = design @ np.asarray(self.coefficients_dx, dtype=np.float64)
        return dy.astype(np.float32), dx.astype(np.float32)


@dataclass(frozen=True)
class BandRegistrationResult:
    band: str
    method: str
    aligned: bool
    confidence: float
    initial_dy: float
    initial_dx: float = 0.0
    measurement_count: int = 0
    fit_mean_residual_px: float | None = None
    fit_max_residual_px: float | None = None
    shift_dy: float | None = None
    shift_dx: float | None = None
    shift_field: ShiftField | None = None

    def shifts(self, rows: np.ndarray, columns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.aligned:
            raise BalkanAlignmentError(f"Band {self.band} has no accepted registration")
        if self.method == "reference":
            return np.zeros_like(rows, dtype=np.float32), np.zeros_like(columns, dtype=np.float32)
        if self.shift_field is not None:
            return self.shift_field.evaluate(rows, columns)
        if self.shift_dy is None or self.shift_dx is None:
            raise BalkanAlignmentError(f"Band {self.band} has incomplete shift metadata")
        return (
            np.full_like(rows, self.shift_dy, dtype=np.float32),
            np.full_like(columns, self.shift_dx, dtype=np.float32),
        )


def _normalise_band_name(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def _logical_band(value: str | None) -> str | None:
    normalised = _normalise_band_name(value)
    if normalised is None:
        return None
    for role, aliases in _BAND_ALIASES.items():
        if normalised in aliases:
            return role
    return None


def resolve_band_indices(
    descriptions: Sequence[str | None],
    *,
    explicit_band_order: Sequence[str] | None = None,
) -> dict[str, int]:
    """Resolve one-based source indices for the five Balkan band roles."""
    effective = list(explicit_band_order) if explicit_band_order is not None else list(descriptions)
    if len(effective) != len(descriptions):
        raise BalkanAlignmentError(
            "Explicit alignment band order must contain exactly one role per source band"
        )
    mapping: dict[str, int] = {}
    for index, description in enumerate(effective, start=1):
        role = _logical_band(description)
        if role is None:
            raise BalkanAlignmentError(
                f"Cannot resolve Balkan alignment role for source band {index}: {description!r}"
            )
        if role in mapping:
            raise BalkanAlignmentError(f"Multiple source bands resolve to {role}")
        mapping[role] = index
    missing = [role for role in OUTPUT_BAND_ORDER if role not in mapping]
    if missing or len(mapping) != len(OUTPUT_BAND_ORDER):
        raise BalkanAlignmentError(
            f"Balkan alignment requires exactly {OUTPUT_BAND_ORDER}; missing {missing}"
        )
    return mapping


def load_band_start_rows(path: str | Path) -> dict[str, float]:
    """Read Balkan ``BandStartRow`` values from an L0 manifest or metadata JSON."""
    metadata_path = Path(path)
    try:
        value = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BalkanAlignmentError(
            f"Cannot read alignment metadata JSON: {metadata_path}"
        ) from error
    configuration = value.get("imager_configuration") if isinstance(value, dict) else None
    rows = configuration.get("BandStartRow") if isinstance(configuration, dict) else None
    if not isinstance(rows, list) or len(rows) < 8:
        raise BalkanAlignmentError("Alignment metadata has no eight-entry BandStartRow array")
    result: dict[str, float] = {}
    for role, raw_index in _RAW_SENSOR_BAND_BY_ROLE.items():
        raw_value = rows[raw_index]
        if not isinstance(raw_value, (int, float)) or not math.isfinite(float(raw_value)):
            raise BalkanAlignmentError(f"BandStartRow for {role} is not finite")
        result[role] = float(raw_value)
    return result


def initial_row_shifts(
    band_start_rows: dict[str, float] | None,
    *,
    scale: float = 1.0,
) -> dict[str, float]:
    """Convert detector start rows to source-sampling shifts relative to PAN.

    This preserves the upstream repository's seed convention:
    ``band_start - PAN_start``.  Delivered Balkan TIFFs may store the detector
    row direction along raster columns; ``align_balkan_geotiff`` exposes that
    orientation explicitly through ``band_start_axis``.
    """
    if not math.isfinite(scale) or scale == 0:
        raise BalkanAlignmentError("BandStartRow scale must be finite and non-zero")
    if band_start_rows is None:
        return {role: 0.0 for role in OUTPUT_BAND_ORDER}
    missing = [role for role in OUTPUT_BAND_ORDER if role not in band_start_rows]
    if missing:
        raise BalkanAlignmentError(f"BandStartRow metadata is missing {missing}")
    reference = float(band_start_rows[REFERENCE_BAND])
    return {
        role: (float(band_start_rows[role]) - reference) * float(scale)
        for role in OUTPUT_BAND_ORDER
    }


def _prepare_correlation_pair(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    if source.shape != target.shape or source.ndim != 2:
        raise BalkanAlignmentError("Correlation arrays must be same-sized two-dimensional images")
    valid = np.isfinite(source) & np.isfinite(target)
    if np.count_nonzero(valid) < max(64, int(valid.size * 0.25)):
        return None
    source_values = np.where(valid, source, np.nanmedian(source[valid])).astype(np.float32)
    target_values = np.where(valid, target, np.nanmedian(target[valid])).astype(np.float32)
    source_values -= float(source_values.mean())
    target_values -= float(target_values.mean())
    source_std = float(source_values.std())
    target_std = float(target_values.std())
    if source_std <= 1e-6 or target_std <= 1e-6:
        return None
    window = np.outer(np.hanning(source.shape[0]), np.hanning(source.shape[1])).astype(np.float32)
    return source_values / source_std * window, target_values / target_std * window


def cross_correlate_shift(
    source: np.ndarray,
    target: np.ndarray,
    *,
    search_radius_px: int,
    sidelobe_exclusion_px: int = 5,
) -> tuple[float, float, float]:
    """Return residual ``(dy, dx, confidence)`` using FFT phase correlation."""
    pair = _prepare_correlation_pair(source, target)
    if pair is None:
        return 0.0, 0.0, 0.0
    source_values, target_values = pair
    cross_power = np.fft.rfft2(source_values) * np.conj(np.fft.rfft2(target_values))
    cross_power /= np.maximum(np.abs(cross_power), 1e-12)
    correlation = np.fft.fftshift(
        np.fft.irfft2(cross_power, s=source_values.shape).real
    )
    height, width = correlation.shape
    center_y, center_x = height // 2, width // 2
    y0 = max(0, center_y - search_radius_px)
    y1 = min(height, center_y + search_radius_px + 1)
    x0 = max(0, center_x - search_radius_px)
    x1 = min(width, center_x + search_radius_px + 1)
    search = correlation[y0:y1, x0:x1]
    peak_local = np.unravel_index(int(np.argmax(search)), search.shape)
    peak_y = peak_local[0] + y0
    peak_x = peak_local[1] + x0
    peak_value = float(correlation[peak_y, peak_x])

    sidelobe_mask = np.ones(search.shape, dtype=bool)
    local_y, local_x = peak_y - y0, peak_x - x0
    exclusion_y0 = max(0, local_y - sidelobe_exclusion_px)
    exclusion_y1 = min(search.shape[0], local_y + sidelobe_exclusion_px + 1)
    exclusion_x0 = max(0, local_x - sidelobe_exclusion_px)
    exclusion_x1 = min(search.shape[1], local_x + sidelobe_exclusion_px + 1)
    sidelobe_mask[exclusion_y0:exclusion_y1, exclusion_x0:exclusion_x1] = False
    sidelobe = search[sidelobe_mask]
    if sidelobe.size < 10 or float(sidelobe.std()) <= 1e-12:
        confidence = 0.0
    else:
        psr = (peak_value - float(sidelobe.mean())) / float(sidelobe.std())
        confidence = float(np.clip(psr / 20.0, 0.0, 1.0))
    return float(peak_y - center_y), float(peak_x - center_x), confidence


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    finite = np.isfinite(image)
    if not np.any(finite):
        return np.full_like(image, np.nan, dtype=np.float32)
    filled = np.where(finite, image, np.nanmedian(image[finite])).astype(np.float32)
    standard_deviation = float(filled.std())
    if standard_deviation > 1e-6:
        filled = (filled - float(filled.mean())) / standard_deviation
    gradient_x = sobel(filled, axis=1, mode="nearest")
    gradient_y = sobel(filled, axis=0, mode="nearest")
    result = np.hypot(gradient_x, gradient_y).astype(np.float32)
    result[~finite] = np.nan
    return result


def _read_float_window(
    source: rasterio.io.DatasetReader,
    band_index: int,
    window: Window,
) -> np.ndarray:
    values = source.read(
        band_index,
        window=window,
        boundless=True,
        masked=True,
        out_dtype="float32",
    )
    return np.asarray(values.filled(np.nan), dtype=np.float32)


def _correlate_with_gradient_fallback(
    source: np.ndarray,
    target: np.ndarray,
    *,
    search_radius_px: int,
    minimum_confidence: float,
) -> tuple[float, float, float, str]:
    dy, dx, confidence = cross_correlate_shift(
        source,
        target,
        search_radius_px=search_radius_px,
    )
    if confidence >= minimum_confidence:
        return dy, dx, confidence, "direct"
    gradient_dy, gradient_dx, gradient_confidence = cross_correlate_shift(
        _gradient_magnitude(source),
        _gradient_magnitude(target),
        search_radius_px=search_radius_px,
    )
    if gradient_confidence > confidence:
        return gradient_dy, gradient_dx, gradient_confidence, "gradient"
    return dy, dx, confidence, "direct"


def _common_axis_centers(
    length: int,
    *,
    half_tile: int,
    initial_shift: float,
    count: int,
) -> np.ndarray:
    rounded_shift = int(round(initial_shift))
    minimum = max(half_tile, half_tile - rounded_shift)
    maximum = min(length - half_tile, length - half_tile - rounded_shift)
    if maximum < minimum:
        return np.empty(0, dtype=np.float64)
    if count == 1:
        return np.asarray([(minimum + maximum) / 2.0], dtype=np.float64)
    return np.linspace(minimum, maximum, count, dtype=np.float64)


def measure_local_shifts(
    source: rasterio.io.DatasetReader,
    *,
    source_band_index: int,
    reference_band_index: int,
    initial_dy: float,
    initial_dx: float,
    config: AlignmentConfig,
) -> list[LocalMeasurement]:
    """Measure total source-sampling shifts on a grid of bounded tiles."""
    tile_size = config.measurement_tile_size
    half = tile_size // 2
    row_centers = _common_axis_centers(
        source.height,
        half_tile=half,
        initial_shift=initial_dy,
        count=config.grid_rows,
    )
    column_centers = _common_axis_centers(
        source.width,
        half_tile=half,
        initial_shift=initial_dx,
        count=config.grid_cols,
    )
    seed_dy = int(round(initial_dy))
    seed_dx = int(round(initial_dx))
    measurements: list[LocalMeasurement] = []
    for row_center in row_centers:
        for column_center in column_centers:
            row = int(round(row_center))
            column = int(round(column_center))
            reference_window = Window(column - half, row - half, tile_size, tile_size)
            source_window = Window(
                column - half + seed_dx,
                row - half + seed_dy,
                tile_size,
                tile_size,
            )
            reference_values = _read_float_window(source, reference_band_index, reference_window)
            source_values = _read_float_window(source, source_band_index, source_window)
            residual_dy, residual_dx, confidence, method = _correlate_with_gradient_fallback(
                source_values,
                reference_values,
                search_radius_px=config.local_search_radius_px,
                minimum_confidence=config.minimum_confidence,
            )
            if confidence >= config.minimum_confidence:
                measurements.append(
                    LocalMeasurement(
                        row=float(row),
                        column=float(column),
                        dy=float(seed_dy + residual_dy),
                        dx=float(seed_dx + residual_dx),
                        confidence=confidence,
                        method=method,
                    )
                )
    return measurements


def _normalized_axis(values: np.ndarray, length: int) -> np.ndarray:
    if length <= 1:
        return np.zeros_like(values, dtype=np.float64)
    return np.asarray(values, dtype=np.float64) * (2.0 / float(length - 1)) - 1.0


def _fit_shift_field_once(
    measurements: Sequence[LocalMeasurement],
    *,
    height: int,
    width: int,
) -> ShiftField:
    rows = _normalized_axis(np.asarray([item.row for item in measurements]), height)
    columns = _normalized_axis(np.asarray([item.column for item in measurements]), width)
    design = np.stack((np.ones_like(rows), rows, columns, rows * columns), axis=1)
    weights = np.sqrt(np.asarray([item.confidence for item in measurements], dtype=np.float64))
    weighted_design = design * weights[:, None]
    dy = np.asarray([item.dy for item in measurements], dtype=np.float64) * weights
    dx = np.asarray([item.dx for item in measurements], dtype=np.float64) * weights
    coefficients_dy = np.linalg.lstsq(weighted_design, dy, rcond=None)[0]
    coefficients_dx = np.linalg.lstsq(weighted_design, dx, rcond=None)[0]
    return ShiftField(
        coefficients_dy=tuple(float(value) for value in coefficients_dy),
        coefficients_dx=tuple(float(value) for value in coefficients_dx),
        height=height,
        width=width,
    )


def fit_shift_field(
    measurements: Sequence[LocalMeasurement],
    *,
    height: int,
    width: int,
    minimum_measurements: int,
) -> tuple[ShiftField, list[LocalMeasurement], float, float]:
    """Fit a weighted field after deterministic spatial-consensus filtering."""
    retained = list(measurements)
    if len(retained) < minimum_measurements:
        raise BalkanAlignmentError("Too few local measurements to fit a shift field")

    all_rows = np.asarray([item.row for item in retained], dtype=np.float32)
    all_columns = np.asarray([item.column for item in retained], dtype=np.float32)
    all_dy = np.asarray([item.dy for item in retained], dtype=np.float32)
    all_dx = np.asarray([item.dx for item in retained], dtype=np.float32)
    required_consensus = max(minimum_measurements, math.ceil(len(retained) * 0.35))
    random = np.random.default_rng(0)
    candidate_indices: list[np.ndarray] = []
    if len(retained) == minimum_measurements:
        candidate_indices.append(np.arange(len(retained)))
    else:
        attempts = min(768, max(128, len(retained) * 32))
        for _ in range(attempts):
            candidate_indices.append(
                np.sort(random.choice(len(retained), minimum_measurements, replace=False))
            )

    best_inliers: np.ndarray | None = None
    best_score: tuple[int, float] | None = None
    for indices in candidate_indices:
        candidate_measurements = [retained[int(index)] for index in indices]
        candidate_rows = np.asarray(
            [item.row for item in candidate_measurements], dtype=np.float64
        )
        candidate_columns = np.asarray(
            [item.column for item in candidate_measurements], dtype=np.float64
        )
        design = np.stack(
            (
                np.ones_like(candidate_rows),
                _normalized_axis(candidate_rows, height),
                _normalized_axis(candidate_columns, width),
                _normalized_axis(candidate_rows, height)
                * _normalized_axis(candidate_columns, width),
            ),
            axis=1,
        )
        if np.linalg.matrix_rank(design) < 4:
            continue
        candidate = _fit_shift_field_once(
            candidate_measurements,
            height=height,
            width=width,
        )
        candidate_dy, candidate_dx = candidate.evaluate(all_rows, all_columns)
        candidate_residuals = np.hypot(candidate_dy - all_dy, candidate_dx - all_dx)
        inliers = candidate_residuals <= 4.0
        count = int(np.count_nonzero(inliers))
        if count < required_consensus:
            continue
        mean_residual = float(np.mean(candidate_residuals[inliers]))
        score = (count, -mean_residual)
        if best_score is None or score > best_score:
            best_score = score
            best_inliers = inliers

    if best_inliers is not None:
        retained = [
            item
            for item, accepted in zip(retained, best_inliers, strict=True)
            if accepted
        ]

    field = _fit_shift_field_once(retained, height=height, width=width)
    rows = np.asarray([item.row for item in retained], dtype=np.float32)
    columns = np.asarray([item.column for item in retained], dtype=np.float32)
    measured_dy = np.asarray([item.dy for item in retained], dtype=np.float32)
    measured_dx = np.asarray([item.dx for item in retained], dtype=np.float32)
    predicted_dy, predicted_dx = field.evaluate(rows, columns)
    residuals = np.hypot(predicted_dy - measured_dy, predicted_dx - measured_dx)
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    threshold = max(3.0, median + 3.0 * max(mad, 0.25))
    keep = residuals <= threshold
    if not np.all(keep) and int(np.count_nonzero(keep)) >= minimum_measurements:
        retained = [item for item, accepted in zip(retained, keep, strict=True) if accepted]
        field = _fit_shift_field_once(retained, height=height, width=width)
        rows = np.asarray([item.row for item in retained], dtype=np.float32)
        columns = np.asarray([item.column for item in retained], dtype=np.float32)
        measured_dy = np.asarray([item.dy for item in retained], dtype=np.float32)
        measured_dx = np.asarray([item.dx for item in retained], dtype=np.float32)
        predicted_dy, predicted_dx = field.evaluate(rows, columns)
        residuals = np.hypot(predicted_dy - measured_dy, predicted_dx - measured_dx)
    return field, retained, float(np.mean(residuals)), float(np.max(residuals))


def _global_registration(
    source: rasterio.io.DatasetReader,
    *,
    source_band_index: int,
    reference_band_index: int,
    initial_dy: float,
    initial_dx: float,
    config: AlignmentConfig,
    band: str,
) -> BandRegistrationResult:
    tile_size = min(
        max(config.measurement_tile_size * 2, 512),
        source.height,
        source.width,
    )
    half = tile_size // 2
    row_centers = _common_axis_centers(
        source.height,
        half_tile=half,
        initial_shift=initial_dy,
        count=1,
    )
    column_centers = _common_axis_centers(
        source.width,
        half_tile=half,
        initial_shift=initial_dx,
        count=1,
    )
    if not row_centers.size or not column_centers.size:
        return BandRegistrationResult(
            band=band,
            method="FAILED_NO_COMMON_FOOTPRINT",
            aligned=False,
            confidence=0.0,
            initial_dy=initial_dy,
            initial_dx=initial_dx,
        )
    seed_dy = int(round(initial_dy))
    seed_dx = int(round(initial_dx))
    row = int(round(row_centers[0]))
    column = int(round(column_centers[0]))
    reference_window = Window(column - half, row - half, tile_size, tile_size)
    source_window = Window(
        column - half + seed_dx,
        row - half + seed_dy,
        tile_size,
        tile_size,
    )
    reference_values = _read_float_window(source, reference_band_index, reference_window)
    source_values = _read_float_window(source, source_band_index, source_window)
    residual_dy, residual_dx, confidence, method = _correlate_with_gradient_fallback(
        source_values,
        reference_values,
        search_radius_px=config.global_search_radius_px,
        minimum_confidence=config.minimum_confidence,
    )
    aligned = confidence >= config.minimum_confidence
    return BandRegistrationResult(
        band=band,
        method=f"global_{method}" if aligned else "FAILED_LOW_CONFIDENCE",
        aligned=aligned,
        confidence=confidence,
        initial_dy=initial_dy,
        initial_dx=initial_dx,
        shift_dy=float(seed_dy + residual_dy) if aligned else None,
        shift_dx=float(seed_dx + residual_dx) if aligned else None,
    )


def register_band(
    source: rasterio.io.DatasetReader,
    *,
    source_band_index: int,
    reference_band_index: int,
    initial_dy: float,
    initial_dx: float,
    config: AlignmentConfig,
    band: str,
) -> BandRegistrationResult:
    measurements = measure_local_shifts(
        source,
        source_band_index=source_band_index,
        reference_band_index=reference_band_index,
        initial_dy=initial_dy,
        initial_dx=initial_dx,
        config=config,
    )
    if len(measurements) >= config.minimum_local_measurements:
        field, retained, mean_residual, max_residual = fit_shift_field(
            measurements,
            height=source.height,
            width=source.width,
            minimum_measurements=config.minimum_local_measurements,
        )
        if (
            mean_residual <= config.maximum_fit_mean_residual_px
            and max_residual <= config.maximum_fit_residual_px
        ):
            return BandRegistrationResult(
                band=band,
                method="local_field",
                aligned=True,
                confidence=float(np.mean([item.confidence for item in retained])),
                initial_dy=initial_dy,
                initial_dx=initial_dx,
                measurement_count=len(retained),
                fit_mean_residual_px=mean_residual,
                fit_max_residual_px=max_residual,
                shift_field=field,
            )
    return _global_registration(
        source,
        source_band_index=source_band_index,
        reference_band_index=reference_band_index,
        initial_dy=initial_dy,
        initial_dx=initial_dx,
        config=config,
        band=band,
    )


def register_all_bands(
    source: rasterio.io.DatasetReader,
    *,
    band_indices: dict[str, int],
    initial_shifts: dict[str, float],
    initial_column_shifts: dict[str, float] | None = None,
    config: AlignmentConfig,
) -> dict[str, BandRegistrationResult]:
    config.validate()
    reference_index = band_indices[REFERENCE_BAND]
    column_shifts = initial_column_shifts or {}
    results = {
        REFERENCE_BAND: BandRegistrationResult(
            band=REFERENCE_BAND,
            method="reference",
            aligned=True,
            confidence=1.0,
            initial_dy=0.0,
            shift_dy=0.0,
            shift_dx=0.0,
        )
    }
    for band in OUTPUT_BAND_ORDER:
        if band == REFERENCE_BAND:
            continue
        results[band] = register_band(
            source,
            source_band_index=band_indices[band],
            reference_band_index=reference_index,
            initial_dy=float(initial_shifts.get(band, 0.0)),
            initial_dx=float(column_shifts.get(band, 0.0)),
            config=config,
            band=band,
        )
    failed = [band for band, result in results.items() if not result.aligned]
    if failed:
        details = ", ".join(
            f"{band} ({results[band].method}, confidence={results[band].confidence:.3f})"
            for band in failed
        )
        raise BalkanAlignmentError(f"Band alignment failed closed: {details}")
    return results


def _sample_registered_window(
    source: rasterio.io.DatasetReader,
    *,
    band_index: int,
    output_window: Window,
    registration: BandRegistrationResult,
    nodata: float,
) -> np.ndarray:
    row_start = int(output_window.row_off)
    column_start = int(output_window.col_off)
    height = int(output_window.height)
    width = int(output_window.width)
    rows, columns = np.meshgrid(
        np.arange(row_start, row_start + height, dtype=np.float32),
        np.arange(column_start, column_start + width, dtype=np.float32),
        indexing="ij",
    )
    dy, dx = registration.shifts(rows, columns)
    source_rows = rows + dy
    source_columns = columns + dx
    minimum_row = math.floor(float(np.min(source_rows))) - 1
    maximum_row = math.ceil(float(np.max(source_rows))) + 2
    minimum_column = math.floor(float(np.min(source_columns))) - 1
    maximum_column = math.ceil(float(np.max(source_columns))) + 2
    read_window = Window(
        minimum_column,
        minimum_row,
        maximum_column - minimum_column,
        maximum_row - minimum_row,
    )
    source_values = source.read(
        band_index,
        window=read_window,
        boundless=True,
        masked=True,
        out_dtype="float32",
    )
    source_mask = np.ma.getmaskarray(source_values)
    source_data = np.asarray(source_values.filled(0.0), dtype=np.float32)
    coordinates = np.stack(
        (source_rows - minimum_row, source_columns - minimum_column),
        axis=0,
    )
    sampled = map_coordinates(
        source_data,
        coordinates,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    valid_weight = map_coordinates(
        (~source_mask).astype(np.float32),
        coordinates,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    inside = (
        (source_rows >= 0.0)
        & (source_rows <= source.height - 1)
        & (source_columns >= 0.0)
        & (source_columns <= source.width - 1)
    )
    valid = inside & (valid_weight >= 0.999)
    result = np.full((height, width), np.float32(nodata), dtype=np.float32)
    result[valid] = sampled[valid]
    return result


def _output_profile(
    source: rasterio.io.DatasetReader,
    *,
    nodata: float,
    block_size: int,
) -> dict[str, Any]:
    profile = source.profile.copy()
    profile.update(
        driver="GTiff",
        count=len(OUTPUT_BAND_ORDER),
        dtype="float32",
        nodata=nodata,
        compress="deflate",
        predictor=3,
        interleave="band",
        BIGTIFF="IF_SAFER",
    )
    if source.width >= 16 and source.height >= 16:
        maximum_block = min(block_size, source.width, source.height)
        tiled_block = max(16, maximum_block - maximum_block % 16)
        profile.update(tiled=True, blockxsize=tiled_block, blockysize=tiled_block)
    else:
        profile.update(tiled=False)
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
    return profile


def _has_georeferencing(source: rasterio.io.DatasetReader) -> bool:
    return source.crs is not None and source.transform != Affine.identity()


def _serialise_results(
    results: dict[str, BandRegistrationResult],
) -> dict[str, dict[str, Any]]:
    return {band: asdict(result) for band, result in results.items()}


def align_balkan_geotiff(
    source_path: str | Path,
    output_path: str | Path,
    *,
    explicit_band_order: Sequence[str] | None = None,
    metadata_path: str | Path | None = None,
    band_start_rows: dict[str, float] | None = None,
    band_start_row_scale: float = 1.0,
    band_start_axis: str = "row",
    config: AlignmentConfig | None = None,
    require_georeferencing: bool = True,
    overwrite: bool = False,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Align five Balkan bands and write a bounded-memory GeoTIFF product."""
    started = time.perf_counter()
    configuration = config or AlignmentConfig()
    configuration.validate()
    source_file = Path(source_path).resolve()
    output_file = Path(output_path).resolve()
    report_file = (
        Path(report_path).resolve()
        if report_path is not None
        else output_file.with_suffix(".alignment.json")
    )
    if not source_file.is_file():
        raise BalkanAlignmentError(f"Balkan alignment source does not exist: {source_file}")
    if output_file == source_file or report_file in {source_file, output_file}:
        raise BalkanAlignmentError("Alignment source, output, and report paths must be distinct")
    if output_file.exists() and not overwrite:
        raise BalkanAlignmentError(f"Alignment output exists; pass overwrite=True: {output_file}")
    if report_file.exists() and not overwrite:
        raise BalkanAlignmentError(f"Alignment report exists; pass overwrite=True: {report_file}")
    if metadata_path is not None and band_start_rows is not None:
        raise BalkanAlignmentError("Supply metadata_path or band_start_rows, not both")
    resolved_start_rows = (
        load_band_start_rows(metadata_path) if metadata_path is not None else band_start_rows
    )
    if band_start_axis not in {"row", "column"}:
        raise BalkanAlignmentError("band_start_axis must be row or column")
    detector_shifts = initial_row_shifts(
        resolved_start_rows,
        scale=band_start_row_scale,
    )
    row_shifts = (
        detector_shifts
        if band_start_axis == "row"
        else {role: 0.0 for role in OUTPUT_BAND_ORDER}
    )
    column_shifts = (
        detector_shifts
        if band_start_axis == "column"
        else {role: 0.0 for role in OUTPUT_BAND_ORDER}
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    partial_file = output_file.with_name(f".{output_file.name}.partial")
    if partial_file.exists():
        partial_file.unlink()

    with rasterio.open(source_file) as source:
        if source.driver != "GTiff" or source.count != len(OUTPUT_BAND_ORDER):
            raise BalkanAlignmentError("Balkan alignment input must be a five-band GeoTIFF")
        georeferenced = _has_georeferencing(source)
        if require_georeferencing and not georeferenced:
            raise BalkanAlignmentError(
                "Balkan source has no CRS/affine georeferencing. Alignment can remove band "
                "fringing, but model intake requires orthorectification. Use "
                "require_georeferencing=False only to create an alignment-only intermediate."
            )
        band_indices = resolve_band_indices(
            source.descriptions,
            explicit_band_order=explicit_band_order,
        )
        registrations = register_all_bands(
            source,
            band_indices=band_indices,
            initial_shifts=row_shifts,
            initial_column_shifts=column_shifts,
            config=configuration,
        )
        source_sha256 = sha256_file(source_file)
        nodata = float(source.nodata) if source.nodata is not None else 0.0
        profile = _output_profile(
            source,
            nodata=nodata,
            block_size=configuration.write_block_size,
        )
        try:
            with rasterio.open(partial_file, "w", **profile) as destination:
                source_tags = source.tags()
                if source_tags:
                    destination.update_tags(**source_tags)
                destination.update_tags(
                    SENSOR="balkan-1",
                    PROCESSING_LEVEL=(
                        "L1_REGISTERED_GEOREFERENCED" if georeferenced else "L0_REGISTERED"
                    ),
                    ALIGNMENT_ALGORITHM=ALIGNMENT_ALGORITHM,
                    ALIGNMENT_REFERENCE=REFERENCE_BAND,
                    ALIGNMENT_SOURCE_COMMIT=ALIGNMENT_SOURCE_COMMIT,
                    ALIGNMENT_PARENT_SHA256=source_sha256,
                    RADIOMETRY_MODIFIED="FALSE",
                    PIPELINE_READY="TRUE" if georeferenced else "FALSE",
                )
                for output_index, band in enumerate(OUTPUT_BAND_ORDER, start=1):
                    destination.set_band_description(output_index, band)
                    registration = registrations[band]
                    for _, window in destination.block_windows(output_index):
                        values = _sample_registered_window(
                            source,
                            band_index=band_indices[band],
                            output_window=window,
                            registration=registration,
                            nodata=nodata,
                        )
                        destination.write(values, output_index, window=window)
                overview_factors = [
                    factor
                    for factor in (2, 4, 8, 16)
                    if min(source.height, source.width) // factor >= 128
                ]
                if overview_factors:
                    destination.build_overviews(overview_factors, Resampling.average)
                    destination.update_tags(ns="rio_overview", resampling="average")
            os.replace(partial_file, output_file)
        except Exception:
            if partial_file.exists():
                partial_file.unlink()
            raise

        output_sha256 = sha256_file(output_file)
        report: dict[str, Any] = {
            "schema_version": "1.0",
            "algorithm": ALIGNMENT_ALGORITHM,
            "source_repository": ALIGNMENT_SOURCE_URL,
            "source_commit": ALIGNMENT_SOURCE_COMMIT,
            "created_at": datetime.now(UTC).isoformat(),
            "source": {
                "path": str(source_file),
                "bytes": source_file.stat().st_size,
                "sha256": source_sha256,
                "width": source.width,
                "height": source.height,
                "band_indices_1_based": band_indices,
                "georeferenced": georeferenced,
            },
            "output": {
                "path": str(output_file),
                "bytes": output_file.stat().st_size,
                "sha256": output_sha256,
                "band_order": list(OUTPUT_BAND_ORDER),
                "pipeline_ready": georeferenced,
                "radiometry_modified": False,
            },
            "band_start_rows": resolved_start_rows,
            "band_start_axis": band_start_axis,
            "band_start_row_scale": band_start_row_scale,
            "initial_row_shifts": row_shifts,
            "initial_column_shifts": column_shifts,
            "configuration": asdict(configuration),
            "registrations": _serialise_results(registrations),
            "runtime_seconds": time.perf_counter() - started,
            "warnings": (
                []
                if georeferenced
                else [
                    "Alignment-only output has no georeferencing and cannot enter model intake"
                ]
            ),
        }
    temporary_report = report_file.with_name(f".{report_file.name}.partial")
    temporary_report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary_report, report_file)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vita-balkan-align",
        description=(
            "Align Balkan-1 BLUE/GREEN/RED/NIR bands to PAN and write a bounded-memory "
            "five-band GeoTIFF for ViTA intake."
        ),
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--band-order",
        nargs=5,
        default=None,
        metavar=("B1", "B2", "B3", "B4", "B5"),
        help="Explicit source roles when GeoTIFF descriptions are absent",
    )
    parser.add_argument("--metadata", type=Path, help="L0 manifest containing BandStartRow")
    parser.add_argument("--band-start-row-scale", type=float, default=1.0)
    parser.add_argument(
        "--band-start-axis",
        choices=("row", "column"),
        default="row",
        help="Raster axis corresponding to detector BandStartRow offsets",
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--allow-ungeoreferenced", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--grid-rows", type=int, default=AlignmentConfig.grid_rows)
    parser.add_argument("--grid-cols", type=int, default=AlignmentConfig.grid_cols)
    parser.add_argument(
        "--measurement-tile-size",
        type=int,
        default=AlignmentConfig.measurement_tile_size,
    )
    parser.add_argument(
        "--local-search-radius",
        type=int,
        default=AlignmentConfig.local_search_radius_px,
    )
    parser.add_argument(
        "--global-search-radius",
        type=int,
        default=AlignmentConfig.global_search_radius_px,
    )
    parser.add_argument(
        "--minimum-confidence",
        type=float,
        default=AlignmentConfig.minimum_confidence,
    )
    parser.add_argument("--write-block-size", type=int, default=AlignmentConfig.write_block_size)
    return parser


def main() -> None:
    args = _parser().parse_args()
    configuration = AlignmentConfig(
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        measurement_tile_size=args.measurement_tile_size,
        local_search_radius_px=args.local_search_radius,
        global_search_radius_px=args.global_search_radius,
        minimum_confidence=args.minimum_confidence,
        write_block_size=args.write_block_size,
    )
    try:
        report = align_balkan_geotiff(
            args.input,
            args.output,
            explicit_band_order=args.band_order,
            metadata_path=args.metadata,
            band_start_row_scale=args.band_start_row_scale,
            band_start_axis=args.band_start_axis,
            config=configuration,
            require_georeferencing=not args.allow_ungeoreferenced,
            overwrite=args.overwrite,
            report_path=args.report,
        )
    except (BalkanAlignmentError, OSError, rasterio.errors.RasterioError) as error:
        print(f"Balkan alignment failed: {error}")
        raise SystemExit(2) from None
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

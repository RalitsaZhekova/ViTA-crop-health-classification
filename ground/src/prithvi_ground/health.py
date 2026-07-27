"""Sensor-neutral crop-condition measurements for one calibrated image window."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray
from prithvi_shared.calibration import HEALTH_ANALYSIS_CROP_THRESHOLD

SUPPORTED_SENSORS = {"balkan-1", "sentinel-2"}
SCHEMA_VERSION = "1.0"
ALGORITHM_VERSION = "health-indices-v2"
CROP_BINARY_NODATA = 255

FloatArray = NDArray[np.floating[Any]]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class HealthLayers:
    """Per-pixel measurements and the final mask used to compute them."""

    values: dict[str, FloatArray]
    analysis_mask: BoolArray


@dataclass(frozen=True)
class MetricSummary:
    valid_pixels: int
    mean: float | None
    median: float | None
    standard_deviation: float | None
    percentile_10: float | None
    percentile_90: float | None


@dataclass(frozen=True)
class HealthObservation:
    """Compact record intended for JSON storage and a future API."""

    scene_id: str
    region_id: str
    sensor: str
    acquired_at: str
    status: str
    total_pixels: int
    analysis_pixels: int
    analysis_percentage: float
    metrics: dict[str, MetricSummary]
    raster_assets: dict[str, str]
    schema_version: str = SCHEMA_VERSION
    algorithm_version: str = ALGORITHM_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "algorithm_version": self.algorithm_version,
            "scene_id": self.scene_id,
            "region_id": self.region_id,
            "sensor": self.sensor,
            "acquired_at": self.acquired_at,
            "status": self.status,
            "quality": {
                "total_pixels": self.total_pixels,
                "analysis_pixels": self.analysis_pixels,
                "analysis_percentage": self.analysis_percentage,
            },
            "metrics": {name: asdict(summary) for name, summary in sorted(self.metrics.items())},
            "raster_assets": dict(sorted(self.raster_assets.items())),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            indent=indent,
            sort_keys=True,
        )


def build_analysis_mask(
    crop_binary: NDArray[np.integer[Any] | np.bool_],
    cloud_unusable: NDArray[Any],
    *,
    crop_probability: FloatArray,
    minimum_crop_probability: float = HEALTH_ANALYSIS_CROP_THRESHOLD,
    nodata: NDArray[Any] | None = None,
) -> BoolArray:
    """Select clear, confident crop pixels from the deployed binary products.

    ``crop_binary`` must use the payload contract ``0 = non-crop``, ``1 = crop``
    and optional ``255 = unusable/nodata``. ``cloud_unusable`` must contain only
    Boolean or ``0/1`` values. Non-finite crop probabilities are always excluded.
    """
    crop = np.asarray(crop_binary)
    unusable_raw = np.asarray(cloud_unusable)
    probability = np.asarray(crop_probability)

    if crop.ndim != 2:
        raise ValueError("Crop binary mask must be a 2D array")
    if unusable_raw.shape != crop.shape:
        raise ValueError("Unusable mask shape does not match crop binary mask")
    if probability.shape != crop.shape:
        raise ValueError("Crop probability shape does not match crop binary mask")
    if not (np.issubdtype(crop.dtype, np.integer) or np.issubdtype(crop.dtype, np.bool_)):
        raise TypeError("Crop binary mask must use an integer or Boolean dtype")
    if not (
        np.issubdtype(unusable_raw.dtype, np.integer) or np.issubdtype(unusable_raw.dtype, np.bool_)
    ):
        raise TypeError("Unusable mask must use an integer or Boolean dtype")

    if not np.all(np.isin(np.unique(crop), (0, 1, CROP_BINARY_NODATA))):
        raise ValueError("Crop binary mask contains values outside 0, 1 and 255")
    if not np.all(np.isin(np.unique(unusable_raw), (0, 1))):
        raise ValueError("Unusable mask contains values outside 0 and 1")
    if not 0 <= minimum_crop_probability <= 1:
        raise ValueError("minimum_crop_probability must be between 0 and 1")

    unusable = unusable_raw.astype(bool, copy=False)
    mask = (crop == 1) & ~unusable
    if nodata is not None:
        nodata_raw = np.asarray(nodata)
        if nodata_raw.shape != crop.shape:
            raise ValueError("No-data mask shape does not match crop binary mask")
        if not (
            np.issubdtype(nodata_raw.dtype, np.integer) or np.issubdtype(nodata_raw.dtype, np.bool_)
        ):
            raise TypeError("No-data mask must use an integer or Boolean dtype")
        if not np.all(np.isin(np.unique(nodata_raw), (0, 1))):
            raise ValueError("No-data mask contains values outside 0 and 1")
        mask &= ~nodata_raw.astype(bool, copy=False)

    mask &= np.isfinite(probability) & (probability >= minimum_crop_probability)
    return mask


def _safe_ratio(
    numerator: FloatArray,
    denominator: FloatArray,
    mask: BoolArray,
    epsilon: float,
) -> FloatArray:
    stable = mask & np.isfinite(numerator) & (np.abs(denominator) > epsilon)
    output = np.full(numerator.shape, np.nan, dtype=np.float32)
    np.divide(numerator, denominator, out=output, where=stable)
    return output


def calculate_health_layers(
    blue: FloatArray,
    green: FloatArray,
    red: FloatArray,
    nir: FloatArray,
    analysis_mask: NDArray[Any],
    *,
    epsilon: float = 1e-6,
    reflectance_range: tuple[float, float] = (-0.2, 2.0),
    savi_soil_factor: float = 0.5,
) -> HealthLayers:
    """Calculate vegetation indices and RGB features for one image window.

    Inputs must be floating-point calibrated reflectance, not raw digital
    numbers or display-stretched RGB values.
    """
    raw_bands = {
        "blue": np.asarray(blue),
        "green": np.asarray(green),
        "red": np.asarray(red),
        "nir": np.asarray(nir),
    }
    shapes = {array.shape for array in raw_bands.values()}
    if len(shapes) != 1 or len(next(iter(shapes))) != 2:
        raise ValueError("All reflectance bands must be matching 2D arrays")
    if any(not np.issubdtype(array.dtype, np.floating) for array in raw_bands.values()):
        raise TypeError("Reflectance bands must be floating point, not raw integer DN")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not 0 <= savi_soil_factor <= 1:
        raise ValueError("savi_soil_factor must be between 0 and 1")
    lower, upper = reflectance_range
    if lower >= upper:
        raise ValueError("Invalid reflectance range")

    requested = np.asarray(analysis_mask, dtype=bool)
    shape = next(iter(shapes))
    if requested.shape != shape:
        raise ValueError("Analysis mask shape does not match reflectance bands")

    bands = {name: array.astype(np.float32, copy=False) for name, array in raw_bands.items()}
    finite_and_calibrated = np.ones(shape, dtype=bool)
    for array in bands.values():
        finite_and_calibrated &= np.isfinite(array) & (array >= lower) & (array <= upper)
    valid = requested & finite_and_calibrated

    blue_array = bands["blue"]
    green_array = bands["green"]
    red_array = bands["red"]
    nir_array = bands["nir"]

    ndvi = _safe_ratio(nir_array - red_array, nir_array + red_array, valid, epsilon)
    gndvi = _safe_ratio(
        nir_array - green_array,
        nir_array + green_array,
        valid,
        epsilon,
    )
    evi = _safe_ratio(
        2.5 * (nir_array - red_array),
        nir_array + 6.0 * red_array - 7.5 * blue_array + 1.0,
        valid,
        epsilon,
    )
    savi = _safe_ratio(
        (1.0 + savi_soil_factor) * (nir_array - red_array),
        nir_array + red_array + savi_soil_factor,
        valid,
        epsilon,
    )
    cvi = _safe_ratio(
        nir_array * red_array,
        green_array * green_array,
        valid,
        epsilon,
    )
    vari = _safe_ratio(
        green_array - red_array,
        green_array + red_array - blue_array,
        valid,
        epsilon,
    )

    brightness = np.full(shape, np.nan, dtype=np.float32)
    excess_green = np.full(shape, np.nan, dtype=np.float32)
    brightness[valid] = (blue_array[valid] + green_array[valid] + red_array[valid]) / 3.0
    excess_green[valid] = 2.0 * green_array[valid] - red_array[valid] - blue_array[valid]

    return HealthLayers(
        values={
            "cvi": cvi,
            "evi": evi,
            "excess_green": excess_green,
            "gndvi": gndvi,
            "ndvi": ndvi,
            "rgb_brightness": brightness,
            "savi": savi,
            "vari": vari,
        },
        analysis_mask=valid,
    )


def summarize_metric(values: FloatArray) -> MetricSummary:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return MetricSummary(0, None, None, None, None, None)
    return MetricSummary(
        valid_pixels=int(finite.size),
        mean=float(np.mean(finite)),
        median=float(np.median(finite)),
        standard_deviation=float(np.std(finite)),
        percentile_10=float(np.percentile(finite, 10)),
        percentile_90=float(np.percentile(finite, 90)),
    )


def build_health_observation(
    *,
    scene_id: str,
    region_id: str,
    sensor: str,
    acquired_at: datetime,
    layers: HealthLayers,
    minimum_analysis_pixels: int = 1,
    raster_assets: dict[str, str] | None = None,
) -> HealthObservation:
    """Build a JSON-safe measurement record without making a health diagnosis."""
    if not scene_id.strip() or not region_id.strip():
        raise ValueError("scene_id and region_id must be non-empty")
    sensor_name = sensor.lower()
    if sensor_name not in SUPPORTED_SENSORS:
        raise ValueError(f"Unsupported sensor: {sensor}")
    if acquired_at.tzinfo is None or acquired_at.utcoffset() is None:
        raise ValueError("acquired_at must include a timezone")
    if minimum_analysis_pixels <= 0:
        raise ValueError("minimum_analysis_pixels must be positive")

    total = int(layers.analysis_mask.size)
    analysis = int(layers.analysis_mask.sum())
    status = "MEASURED" if analysis >= minimum_analysis_pixels else "INSUFFICIENT_DATA"
    return HealthObservation(
        scene_id=scene_id,
        region_id=region_id,
        sensor=sensor_name,
        acquired_at=acquired_at.isoformat(),
        status=status,
        total_pixels=total,
        analysis_pixels=analysis,
        analysis_percentage=100.0 * analysis / total if total else 0.0,
        metrics={name: summarize_metric(values) for name, values in layers.values.items()},
        raster_assets=raster_assets or {},
    )

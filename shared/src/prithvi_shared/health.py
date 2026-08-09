"""Sensor-neutral crop-condition measurements shared with payload processing."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from prithvi_shared.calibration import HEALTH_ANALYSIS_CROP_THRESHOLD

ALGORITHM_VERSION = "health-indices-v2"
CROP_BINARY_NODATA = 255

FloatArray = NDArray[np.floating[Any]]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class HealthLayers:
    """Per-pixel measurements and the final mask used to compute them."""

    values: dict[str, FloatArray]
    analysis_mask: BoolArray


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
    executor: Executor | None = None,
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

    def brightness() -> FloatArray:
        output = np.full(shape, np.nan, dtype=np.float32)
        output[valid] = (
            blue_array[valid] + green_array[valid] + red_array[valid]
        ) / 3.0
        return output

    def excess_green() -> FloatArray:
        output = np.full(shape, np.nan, dtype=np.float32)
        output[valid] = (
            2.0 * green_array[valid] - red_array[valid] - blue_array[valid]
        )
        return output

    calculations: dict[str, Callable[[], FloatArray]] = {
        "cvi": lambda: _safe_ratio(
            nir_array * red_array,
            green_array * green_array,
            valid,
            epsilon,
        ),
        "evi": lambda: _safe_ratio(
            2.5 * (nir_array - red_array),
            nir_array + 6.0 * red_array - 7.5 * blue_array + 1.0,
            valid,
            epsilon,
        ),
        "excess_green": excess_green,
        "gndvi": lambda: _safe_ratio(
            nir_array - green_array,
            nir_array + green_array,
            valid,
            epsilon,
        ),
        "ndvi": lambda: _safe_ratio(
            nir_array - red_array,
            nir_array + red_array,
            valid,
            epsilon,
        ),
        "rgb_brightness": brightness,
        "savi": lambda: _safe_ratio(
            (1.0 + savi_soil_factor) * (nir_array - red_array),
            nir_array + red_array + savi_soil_factor,
            valid,
            epsilon,
        ),
        "vari": lambda: _safe_ratio(
            green_array - red_array,
            green_array + red_array - blue_array,
            valid,
            epsilon,
        ),
    }
    if executor is None:
        values = {name: calculation() for name, calculation in calculations.items()}
    else:
        futures = {
            name: executor.submit(calculation)
            for name, calculation in calculations.items()
        }
        values = {name: future.result() for name, future in futures.items()}

    return HealthLayers(
        values=values,
        analysis_mask=valid,
    )

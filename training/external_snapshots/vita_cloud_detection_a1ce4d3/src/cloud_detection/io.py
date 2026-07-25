from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio


class RasterValidationError(ValueError):
    """Raised when a GeoTIFF does not satisfy the input contract."""


def read_geotiff(
    path: str | Path,
    expected_bands: list[str],
    require_descriptions: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    raster_path = Path(path)
    if not raster_path.exists():
        raise RasterValidationError(f"Input does not exist: {raster_path}")

    with rasterio.open(raster_path) as dataset:
        if dataset.count != len(expected_bands):
            raise RasterValidationError(
                f"Expected {len(expected_bands)} bands, found {dataset.count}."
            )
        array = dataset.read()
        profile = dataset.profile.copy()
        descriptions = list(dataset.descriptions)

    if require_descriptions and descriptions != expected_bands:
        raise RasterValidationError(
            f"Band descriptions {descriptions} do not match {expected_bands}."
        )
    if any(descriptions) and all(descriptions) and descriptions != expected_bands:
        raise RasterValidationError(
            f"GeoTIFF declares band order {descriptions}; expected {expected_bands}."
        )
    if not np.issubdtype(array.dtype, np.number):
        raise RasterValidationError(f"Expected numeric bands, found {array.dtype}.")
    return array, profile


def write_mask(
    path: str | Path,
    array: np.ndarray,
    source_profile: dict[str, Any],
    dtype: str,
    nodata: int | float | None,
    description: str,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if array.ndim != 2:
        raise ValueError(f"Expected two-dimensional output, found {array.shape}.")

    profile = source_profile.copy()
    profile.update(count=1, dtype=dtype, nodata=nodata, compress="deflate")
    with rasterio.open(output_path, "w", **profile) as dataset:
        dataset.write(array.astype(dtype), 1)
        dataset.set_band_description(1, description)

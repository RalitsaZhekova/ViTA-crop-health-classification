"""Small raster operations shared by payload execution stages."""

from __future__ import annotations

import math

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from rasterio.windows import Window


def utm_crs_for_bounds(
    source_crs: CRS,
    bounds: rasterio.coords.BoundingBox,
) -> CRS:
    """Select the UTM zone containing the geographic centre of raster bounds."""
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


def read_padded_tile(
    dataset: rasterio.DatasetReader,
    indices: list[int],
    *,
    y: int,
    x: int,
    tile_size: int,
    halo: int,
    out_dtype: str | None = None,
) -> np.ndarray:
    """Read a haloed tile and reflect-pad portions outside the raster extent."""
    requested_y = y - halo
    requested_x = x - halo
    read_y_start = max(0, requested_y)
    read_x_start = max(0, requested_x)
    read_y_end = min(dataset.height, requested_y + tile_size)
    read_x_end = min(dataset.width, requested_x + tile_size)
    window = Window(
        read_x_start,
        read_y_start,
        read_x_end - read_x_start,
        read_y_end - read_y_start,
    )
    read_options = {"out_dtype": out_dtype} if out_dtype is not None else {}
    values = dataset.read(indices, window=window, **read_options)
    top = read_y_start - requested_y
    left = read_x_start - requested_x
    bottom = requested_y + tile_size - read_y_end
    right = requested_x + tile_size - read_x_end
    mode = "reflect" if values.shape[-2] > 1 and values.shape[-1] > 1 else "edge"
    return np.pad(values, ((0, 0), (top, bottom), (left, right)), mode=mode)

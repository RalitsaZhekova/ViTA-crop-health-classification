"""Deterministic ten-metre UTM grid for bounded live Sentinel acquisition."""

from __future__ import annotations

import math
from dataclasses import dataclass

from affine import Affine
from pyproj import Transformer

from prithvi_payload.acquisition.errors import AcquisitionError

OUTPUT_RESOLUTION_METRES = 10.0
MINIMUM_OUTPUT_DIMENSION = 64
MAXIMUM_OUTPUT_DIMENSION = 1000
OUTPUT_BAND_COUNT = 5
OUTPUT_BYTES_PER_SAMPLE = 2


@dataclass(frozen=True)
class TargetGrid:
    crs: str
    transform: Affine
    width: int
    height: int
    bounds: tuple[float, float, float, float]
    estimated_uncompressed_bytes: int


def _utm_crs(bbox_wgs84: tuple[float, float, float, float]) -> str:
    west, south, east, north = bbox_wgs84
    longitude = (west + east) / 2.0
    latitude = (south + north) / 2.0
    zone = min(60, max(1, int(math.floor((longitude + 180.0) / 6.0)) + 1))
    epsg = (32600 if latitude >= 0 else 32700) + zone
    return f"EPSG:{epsg}"


def calculate_target_grid(
    bbox_wgs84: tuple[float, float, float, float],
) -> TargetGrid:
    west, south, east, north = bbox_wgs84
    if not all(math.isfinite(value) for value in bbox_wgs84):
        raise AcquisitionError("INVALID_REGION", "Area coordinates must be finite")
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise AcquisitionError(
            "INVALID_REGION",
            "Area coordinates must form one valid west-south-east-north rectangle",
        )
    crs = _utm_crs(bbox_wgs84)
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    corners = (
        transformer.transform(west, south),
        transformer.transform(west, north),
        transformer.transform(east, south),
        transformer.transform(east, north),
    )
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    left = math.floor(min(xs) / OUTPUT_RESOLUTION_METRES) * OUTPUT_RESOLUTION_METRES
    right = math.ceil(max(xs) / OUTPUT_RESOLUTION_METRES) * OUTPUT_RESOLUTION_METRES
    bottom = math.floor(min(ys) / OUTPUT_RESOLUTION_METRES) * OUTPUT_RESOLUTION_METRES
    top = math.ceil(max(ys) / OUTPUT_RESOLUTION_METRES) * OUTPUT_RESOLUTION_METRES
    width = int(round((right - left) / OUTPUT_RESOLUTION_METRES))
    height = int(round((top - bottom) / OUTPUT_RESOLUTION_METRES))
    if width < MINIMUM_OUTPUT_DIMENSION or height < MINIMUM_OUTPUT_DIMENSION:
        raise AcquisitionError(
            "REGION_TOO_SMALL",
            "Select an area at least 640 metres wide and high",
            details={"width": width, "height": height},
        )
    if width > MAXIMUM_OUTPUT_DIMENSION or height > MAXIMUM_OUTPUT_DIMENSION:
        raise AcquisitionError(
            "REGION_TOO_LARGE",
            "Select an area no more than 10 kilometres wide and high",
            details={
                "width": width,
                "height": height,
                "maximum_dimension": MAXIMUM_OUTPUT_DIMENSION,
            },
        )
    transform = Affine(OUTPUT_RESOLUTION_METRES, 0.0, left, 0.0, -OUTPUT_RESOLUTION_METRES, top)
    return TargetGrid(
        crs=crs,
        transform=transform,
        width=width,
        height=height,
        bounds=(left, bottom, right, top),
        estimated_uncompressed_bytes=(
            width * height * OUTPUT_BAND_COUNT * OUTPUT_BYTES_PER_SAMPLE
        ),
    )

"""Deterministic ten-metre UTM target-grid calculation."""

from __future__ import annotations

import math
from dataclasses import dataclass

from affine import Affine
from pyproj import Transformer

from prithvi_payload.acquisition.errors import AcquisitionError

OUTPUT_RESOLUTION_METRES = 10.0
MAX_OUTPUT_DIMENSION = 1024
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


def utm_crs_for_bbox(bbox_wgs84: tuple[float, float, float, float]) -> str:
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
    crs = utm_crs_for_bbox(bbox_wgs84)
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
    if (
        width <= 0
        or height <= 0
        or width > MAX_OUTPUT_DIMENSION
        or height > MAX_OUTPUT_DIMENSION
    ):
        raise AcquisitionError(
            "REGION_TOO_LARGE",
            "Requested region cannot be represented on the fixed 10-metre grid",
            details={
                "width": width,
                "height": height,
                "maximum_dimension": MAX_OUTPUT_DIMENSION,
            },
        )
    transform = Affine(
        OUTPUT_RESOLUTION_METRES,
        0.0,
        left,
        0.0,
        -OUTPUT_RESOLUTION_METRES,
        top,
    )
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

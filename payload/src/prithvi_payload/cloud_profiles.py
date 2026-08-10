"""Shared operational cloud-model input profiles.

The fixed-scene TensorRT builder and startup warmup must exercise the exact
arrays submitted by the timed cloud executor.  Sentinel scenes are processed
as 1000 px haloed tiles even when the source GeoTIFF itself is only 700 px;
Balkan scenes use their complete prepared analysis grids.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import rasterio

from prithvi_payload.raster_ops import read_padded_tile

CLOUD_MODEL_TILE_SIZE = 1000
CLOUD_MODEL_HALO = 150
CLOUD_MODEL_CORE_SIZE = CLOUD_MODEL_TILE_SIZE - 2 * CLOUD_MODEL_HALO


def read_fixed_scene_cloud_input(profile: dict[str, Any]) -> np.ndarray:
    """Read the exact cloud-model array used by one fixed qualification scene."""

    sensor = profile.get("sensor")
    if sensor not in {"sentinel-2", "balkan-1"}:
        raise ValueError("Cloud scene profile has no supported sensor")
    with rasterio.open(profile["source_path"]) as source:
        if sensor == "sentinel-2":
            return read_padded_tile(
                source,
                profile["source_band_indices"],
                y=0,
                x=0,
                tile_size=CLOUD_MODEL_TILE_SIZE,
                halo=CLOUD_MODEL_HALO,
            )
        return source.read(profile["source_band_indices"])


def select_fixed_scene_cloud_output(
    profile: dict[str, Any],
    values: np.ndarray,
) -> np.ndarray:
    """Select the pixels delivered by the operational executor for parity."""

    output = np.asarray(values)
    if profile.get("sensor") != "sentinel-2":
        return output
    if output.shape[-2:] != (CLOUD_MODEL_TILE_SIZE, CLOUD_MODEL_TILE_SIZE):
        raise ValueError(
            "Sentinel cloud parity requires the operational 1000 px model output"
        )
    core_end = CLOUD_MODEL_HALO + CLOUD_MODEL_CORE_SIZE
    return output[
        ...,
        CLOUD_MODEL_HALO:core_end,
        CLOUD_MODEL_HALO:core_end,
    ]

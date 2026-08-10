from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from prithvi_payload.cloud_profiles import (
    CLOUD_MODEL_CORE_SIZE,
    CLOUD_MODEL_HALO,
    CLOUD_MODEL_TILE_SIZE,
    read_fixed_scene_cloud_input,
)
from rasterio.transform import from_origin


def test_fixed_sentinel_profile_reads_the_operational_haloed_tile(tmp_path: Path) -> None:
    source_path = tmp_path / "sentinel.tif"
    values = np.stack(
        [np.full((700, 700), band, dtype=np.uint16) for band in range(1, 5)]
    )
    with rasterio.open(
        source_path,
        "w",
        driver="GTiff",
        width=700,
        height=700,
        count=4,
        dtype="uint16",
        crs="EPSG:32635",
        transform=from_origin(500_000.0, 4_500_000.0, 10.0, 10.0),
    ) as destination:
        destination.write(values)

    sentinel = read_fixed_scene_cloud_input(
        {
            "sensor": "sentinel-2",
            "source_path": source_path,
            "source_band_indices": [1, 2, 3, 4],
        }
    )
    balkan = read_fixed_scene_cloud_input(
        {
            "sensor": "balkan-1",
            "source_path": source_path,
            "source_band_indices": [1, 2, 3, 4],
        }
    )

    assert CLOUD_MODEL_CORE_SIZE == 700
    assert sentinel.shape == (4, CLOUD_MODEL_TILE_SIZE, CLOUD_MODEL_TILE_SIZE)
    np.testing.assert_array_equal(
        sentinel[
            :,
            CLOUD_MODEL_HALO : CLOUD_MODEL_HALO + CLOUD_MODEL_CORE_SIZE,
            CLOUD_MODEL_HALO : CLOUD_MODEL_HALO + CLOUD_MODEL_CORE_SIZE,
        ],
        values,
    )
    np.testing.assert_array_equal(balkan, values)

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import torch
from prithvi_payload.crop_parity import build_crop_parity_inputs
from prithvi_shared import CROP_CLASSIFICATION_THRESHOLD, HEALTH_ANALYSIS_CROP_THRESHOLD
from rasterio.transform import from_origin


def _write_scene(path: Path, value: float) -> None:
    bands = np.full((4, 256, 256), value, dtype=np.float32)
    bands[:, 0, 0] = -9999.0
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=256,
        height=256,
        count=4,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(24.0, 43.0, 0.0001, 0.0001),
        nodata=-9999.0,
    ) as destination:
        destination.write(bands)


def test_crop_parity_batch_uses_balanced_real_scene_tiles(tmp_path: Path) -> None:
    first = tmp_path / "first.tif"
    second = tmp_path / "second.tif"
    _write_scene(first, 1000.0)
    _write_scene(second, 2000.0)
    profiles = [
        {
            "sensor": "sentinel-2",
            "source_path": str(path),
            "source_band_indices": [1, 2, 3, 4],
            "training_scale_multiplier": 1.0,
            "acquired_at": acquired_at,
        }
        for path, acquired_at in (
            (first, "2026-01-15T14:06:08+00:00"),
            (second, "2026-06-10T09:13:31+00:00"),
        )
    ]

    image, temporal, location, thresholds, valid = build_crop_parity_inputs(
        profiles,
        batch_size=4,
    )

    assert image.shape == (4, 4, 1, 224, 224)
    assert temporal.shape == (4, 1, 2)
    assert location.shape == (4, 2)
    assert valid.shape == (4, 224, 224)
    assert bool(torch.isfinite(image).all())
    torch.testing.assert_close(
        thresholds,
        torch.tensor(
            [[CROP_CLASSIFICATION_THRESHOLD, HEALTH_ANALYSIS_CROP_THRESHOLD]] * 4
        ),
    )
    assert temporal[:, 0, 0].tolist() == [2026.0] * 4
    assert temporal[:, 0, 1].tolist() == [15.0, 15.0, 161.0, 161.0]
    assert bool(valid.all())

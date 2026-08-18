from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import torch
from prithvi_payload.crop_executor import (
    _compact_invalid_input_mask,
    prepare_compact_crop_inputs,
)
from prithvi_payload.crop_parity import build_crop_parity_inputs
from prithvi_payload.raster_ops import read_padded_array
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


def test_compact_invalid_mask_matches_the_original_per_tile_rule() -> None:
    source = np.arange(4 * 8 * 9, dtype=np.float32).reshape(4, 8, 9)
    source[0, 2, 3] = -9999.0
    source[:, 4, 5] = -9999.0
    model = source.copy()
    model[2, 6, 7] = np.inf

    for calibrated in (False, True):
        compact = _compact_invalid_input_mask(
            source,
            model,
            nodata=-9999.0,
            calibrated=calibrated,
        )
        for y, x in ((0, 0), (2, 3), (7, 8)):
            raw_tile = read_padded_array(
                source,
                y=y,
                x=x,
                tile_size=5,
                halo=1,
            )
            model_tile = read_padded_array(
                model,
                y=y,
                x=x,
                tile_size=5,
                halo=1,
            )
            expected = ~np.isfinite(model_tile).all(axis=0)
            expected |= (
                np.any(raw_tile == -9999.0, axis=0)
                if calibrated
                else np.all(raw_tile == -9999.0, axis=0)
            )
            actual = read_padded_array(
                compact[np.newaxis, ...],
                y=y,
                x=x,
                tile_size=5,
                halo=1,
            )[0]
            np.testing.assert_array_equal(actual, expected)


def test_overlapped_crop_preparation_matches_sequential_source_routing() -> None:
    cloud_source = np.arange(4 * 13 * 17, dtype=np.float32).reshape(4, 13, 17)
    cloud_source[:, 2, 3] = -9999.0
    source_valid = np.ones((13, 17), dtype=bool)
    source_valid[2, 3] = False

    prepared = prepare_compact_crop_inputs(
        cloud_source,
        source_valid,
        source_band_indices=(4, 3, 2, 1),
        crop_band_indices=(1, 2, 3, 4),
        multiplier=2.5,
        nodata=-9999.0,
    )
    expected_source = cloud_source[[3, 2, 1, 0]]
    expected_model = expected_source * np.float32(2.5)
    expected_invalid = _compact_invalid_input_mask(
        expected_source,
        expected_model,
        nodata=-9999.0,
        calibrated=False,
    )

    np.testing.assert_array_equal(prepared["source_bands"], expected_source)
    np.testing.assert_array_equal(prepared["source_valid_mask"], source_valid)
    np.testing.assert_array_equal(prepared["model_bands"], expected_model)
    np.testing.assert_array_equal(prepared["invalid_input_mask"], expected_invalid)


def test_raw_crop_detail_restoration_is_model_input_only_and_nodata_aware() -> None:
    source = np.ones((4, 17, 17), dtype=np.float32)
    source[:, 8, 8] = 4.0
    source[:, 0, :] = 0.0
    valid = np.ones((17, 17), dtype=bool)
    valid[0, :] = False
    original = source.copy()

    prepared = prepare_compact_crop_inputs(
        source,
        valid,
        source_band_indices=(1, 2, 3, 4),
        crop_band_indices=(1, 2, 3, 4),
        multiplier=1.0,
        nodata=0.0,
        spatial_detail_restoration={
            "method": "nodata_aware_unsharp_mask",
            "sigma_pixels": 1.2,
            "amount": 1.75,
            "bands": ["BLUE", "GREEN", "RED", "NIR_BROAD"],
        },
    )

    np.testing.assert_array_equal(source, original)
    np.testing.assert_array_equal(prepared["source_bands"], original)
    assert np.all(prepared["model_bands"][:, 8, 8] > source[:, 8, 8])
    np.testing.assert_array_equal(prepared["model_bands"][:, 0, :], 0.0)
    assert bool(prepared["invalid_input_mask"][0, 0])


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

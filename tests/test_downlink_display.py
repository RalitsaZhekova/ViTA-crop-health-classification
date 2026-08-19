from __future__ import annotations

import numpy as np
import rasterio
from prithvi_payload.downlink import (
    _build_interaction_grid,
    _build_overlay,
    _calibrated_balkan_rgb,
    _stretch_rgb,
    prepare_rgb_preview,
)
from rasterio.transform import from_origin


def test_balkan_web_rgb_uses_validated_sentinel_curves(tmp_path, monkeypatch) -> None:
    source = tmp_path / "analysis.tif"
    profile = {
        "driver": "GTiff",
        "width": 2,
        "height": 2,
        "count": 4,
        "dtype": "float32",
        "crs": "EPSG:32635",
        "transform": from_origin(500_000, 4_700_000, 10, 10),
        "nodata": 0.0,
    }
    with rasterio.open(source, "w", **profile) as dataset:
        dataset.write(np.full((4, 2, 2), 0.5, dtype=np.float32))

    calibration = {
        "source_scale_to_model_units": 10_000.0,
        "curves": [
            {
                "band": band,
                "source_knots": [0, 10_000],
                "target_values": [0, target],
            }
            for band, target in zip(
                ("BLUE", "GREEN", "RED", "NIR_BROAD"),
                (1_000, 2_000, 3_000, 4_000),
                strict=True,
            )
        ],
    }
    adapter = {"mode": "BALKAN_1_SENTINEL_MONOTONIC_V1", "calibration_path": "cal.json"}
    mapping = {
        role: {"index": index}
        for index, role in enumerate(("BLUE", "GREEN", "RED", "NIR_BROAD"), start=1)
    }

    with rasterio.open(source) as dataset:
        # The file/provenance validation is covered by calibration tests; isolate
        # the display routing here so the expected RGB order is explicit.
        import prithvi_payload.downlink as downlink

        monkeypatch.setattr(downlink, "load_calibration", lambda *args, **kwargs: calibration)
        rgb = _calibrated_balkan_rgb(
            dataset,
            mapping=mapping,
            spectral_adapter=adapter,
            original_source_path=source,
            preview_height=2,
            preview_width=2,
        )

    np.testing.assert_allclose(rgb[:, 0, 0], (0.15, 0.10, 0.05))


def test_combined_rgb_stretch_keeps_invalid_pixels_black() -> None:
    values = np.asarray(
        [
            [[0.1, np.nan], [0.0, 0.8]],
            [[0.2, np.nan], [0.0, 0.7]],
            [[0.3, np.nan], [0.0, 0.6]],
        ],
        dtype=np.float32,
    )

    result = _stretch_rgb(values)

    np.testing.assert_array_equal(result[0, 1], 0)
    np.testing.assert_array_equal(result[1, 0], 0)


def test_reduced_saturation_display_preserves_brightness_but_mutes_chroma() -> None:
    values = np.asarray(
        [
            [[0.9, 0.7], [0.5, 0.3]],
            [[0.4, 0.3], [0.2, 0.1]],
            [[0.1, 0.08], [0.05, 0.02]],
        ],
        dtype=np.float32,
    )
    saturated = _stretch_rgb(values, percentiles=(0.0, 100.0))
    muted = _stretch_rgb(values, percentiles=(0.0, 100.0), saturation=0.68)

    assert np.ptp(muted[0, 0]) < np.ptp(saturated[0, 0])
    assert np.any(muted != saturated)


def test_strict_positive_rgb_hides_incomplete_raw_color_fringe() -> None:
    values = np.ones((3, 8, 8), dtype=np.float32)
    values[:, 2:6, 2:6] = np.array([0.7, 0.5, 0.3], dtype=np.float32)[:, None, None]
    values[2, :, 0] = 0.0

    result = _stretch_rgb(
        values,
        channelwise=True,
        percentiles=(0.0, 100.0),
        require_all_positive=True,
    )

    assert np.all(result[:, 0] == 0)
    assert np.any(result[:, 1:] > 0)


def test_experimental_raw_preview_uses_neutral_per_channel_stretch(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "raw-proxy.tif"
    relative_brightness = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=4,
        dtype="float32",
        crs="EPSG:32635",
        transform=from_origin(500_000, 4_700_000, 10, 10),
    ) as dataset:
        dataset.write(
            np.stack(
                (
                    10.0 + 10.0 * relative_brightness,
                    1.0 + relative_brightness,
                    0.1 + 0.1 * relative_brightness,
                    np.ones_like(relative_brightness),
                )
            )
        )
    import prithvi_payload.downlink as downlink

    monkeypatch.setattr(
        downlink,
        "_calibrated_balkan_rgb",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Experimental raw display must not reapply crop calibration")
        ),
    )
    preview = prepare_rgb_preview(
        source,
        mapping={
            role: {"index": index}
            for index, role in enumerate(("BLUE", "GREEN", "RED", "NIR_BROAD"), start=1)
        },
        spectral_adapter={
            "mode": "BALKAN_1_SENTINEL_MONOTONIC_V1",
            "experimental_raw_proxy": {"status": "UNQUALIFIED_ENGINEERING_EXPERIMENT"},
        },
        original_source_path=source,
        sensor="balkan-1",
        reflectance_scale=1.0,
        maximum_dimension=2,
    )

    assert preview["experimental_raw_display"] is True
    assert preview["calibrated_balkan_display"] is False
    pixels = preview["pixels"].astype(np.int16)
    assert np.max(np.ptp(pixels, axis=-1)) <= 1


def test_experimental_condition_overlay_can_be_muted() -> None:
    condition = np.asarray([[0.0, 100.0]], dtype=np.float32)
    valid = np.ones_like(condition, dtype=np.uint8)
    saturated = _build_overlay(condition, valid)
    muted = _build_overlay(condition, valid, alpha=150, saturation=0.72)

    np.testing.assert_array_equal(muted[..., 3], 150)
    assert np.ptp(muted[0, 0, :3]) < np.ptp(saturated[0, 0, :3])


def test_parallel_interaction_grid_matches_serial(tmp_path, monkeypatch) -> None:
    height, width = 96, 128
    source_path = tmp_path / "grid-source.tif"
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:32635",
        "transform": from_origin(500_000, 4_700_000, 10, 10),
    }
    with rasterio.open(source_path, "w", **profile) as dataset:
        dataset.write(np.ones((1, height, width), dtype=np.float32))

    row, column = np.indices((height, width))
    condition = ((row * 7 + column * 3) % 101).astype(np.float32)
    condition[0, 0] = np.nan
    products = {
        "condition_score": condition,
        "ndvi": ((row - column) / 128.0).astype(np.float32),
        "gndvi": ((row + column) / 224.0).astype(np.float32),
        "evi": ((row * 2 - column) / 192.0).astype(np.float32),
        "savi": ((column * 2 - row) / 256.0).astype(np.float32),
        "alert_mask": ((row + column) % 5 == 0).astype(np.uint8),
        "crop_binary": ((row + column) % 3 == 0).astype(np.uint8),
        "unusable_mask": ((row + column) % 7 == 0).astype(np.uint8),
        "semantic_mask": ((row + column) % 4).astype(np.uint8),
        "invalid_mask": ((row + column) % 11 == 0).astype(np.uint8),
    }

    with rasterio.open(source_path) as source:
        monkeypatch.setenv("VITA_DOWNLINK_GRID_THREADS", "1")
        serial = _build_interaction_grid(
            source,
            {},
            grid_size=3,
            products=products,
        )
        monkeypatch.setenv("VITA_DOWNLINK_GRID_THREADS", "8")
        parallel = _build_interaction_grid(
            source,
            {},
            grid_size=3,
            products=products,
        )

    assert parallel == serial

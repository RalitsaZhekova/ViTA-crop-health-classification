from __future__ import annotations

import numpy as np
from cloud_detection.postprocessing import postprocess
from prithvi_payload.balkan_crop_calibration import apply_calibration
from prithvi_shared.condition import ConditionConfig, calculate_condition_score_layers
from prithvi_shared.health import build_analysis_mask, calculate_health_layers


def test_analysis_mask_requires_clear_confident_crop() -> None:
    crop = np.array([[1, 1, 0, 255]], dtype=np.uint8)
    unusable = np.array([[0, 1, 0, 0]], dtype=np.uint8)
    probability = np.array([[0.9, 0.9, 0.9, 0.9]], dtype=np.float32)
    mask = build_analysis_mask(
        crop,
        unusable,
        crop_probability=probability,
        minimum_crop_probability=0.645,
    )
    np.testing.assert_array_equal(mask, [[True, False, False, False]])


def test_health_and_condition_formulas_remain_transparent() -> None:
    blue = np.full((2, 2), 0.10, dtype=np.float32)
    green = np.full((2, 2), 0.30, dtype=np.float32)
    red = np.full((2, 2), 0.20, dtype=np.float32)
    nir = np.full((2, 2), 0.60, dtype=np.float32)
    health = calculate_health_layers(blue, green, red, nir, np.ones((2, 2), dtype=bool))
    np.testing.assert_allclose(health.values["ndvi"], 0.5, atol=1e-6)
    scores = calculate_condition_score_layers(
        health,
        config=ConditionConfig(minimum_analysis_pixels=1),
    )
    assert np.isfinite(scores.condition_score).all()
    assert ((scores.condition_score >= 0) & (scores.condition_score <= 100)).all()


def test_cloud_postprocessing_combines_classes_invalidity_and_dilation() -> None:
    semantic = np.zeros((7, 7), dtype=np.uint8)
    semantic[3, 3] = 1
    invalid = np.zeros_like(semantic, dtype=bool)
    invalid[0, 0] = True
    result = postprocess(
        semantic,
        {"clear": 0, "thick_cloud": 1, "thin_cloud": 2, "cloud_shadow": 3},
        {"include_shadow_as_unusable": True, "minimum_region_pixels": 1, "dilation_pixels": 1},
        invalid,
    )
    assert result[0, 0] == 1
    assert result[3, 3] == 1
    assert int(result.sum()) > 2


def test_balkan_calibration_applies_one_monotonic_curve_per_band() -> None:
    image = np.full((4, 2, 2), 0.5, dtype=np.float32)
    calibration = {
        "curves": [
            {"source_knots": [0.0, 1.0], "target_values": [0.0, float(index + 1)]}
            for index in range(4)
        ]
    }
    calibrated = apply_calibration(image, calibration)
    np.testing.assert_allclose(calibrated[:, 0, 0], [0.5, 1.0, 1.5, 2.0])

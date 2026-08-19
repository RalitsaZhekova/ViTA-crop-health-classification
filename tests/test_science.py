from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from cloud_detection.postprocessing import postprocess
from prithvi_payload.balkan_crop_calibration import (
    BALKAN_CROP_CLASSIFICATION_THRESHOLD,
    BALKAN_HEALTH_ANALYSIS_CROP_THRESHOLD,
    apply_calibration,
)
from prithvi_shared import CROP_CLASSIFICATION_THRESHOLD, HEALTH_ANALYSIS_CROP_THRESHOLD
from prithvi_shared.condition import (
    ConditionConfig,
    calculate_condition_score_layers,
    calculate_spatial_alert_masks,
    calculate_spatial_condition_layers,
)
from prithvi_shared.health import build_analysis_mask, calculate_health_layers


def test_runtime_thresholds_are_sensor_specific() -> None:
    assert CROP_CLASSIFICATION_THRESHOLD == 0.40
    assert HEALTH_ANALYSIS_CROP_THRESHOLD == 0.40
    assert BALKAN_CROP_CLASSIFICATION_THRESHOLD == 0.49
    assert BALKAN_HEALTH_ANALYSIS_CROP_THRESHOLD == 0.645


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


def test_parallel_condition_math_is_bitwise_equivalent_to_sequential_math() -> None:
    rng = np.random.default_rng(19)
    bands = rng.uniform(0.01, 0.9, (4, 47, 53)).astype(np.float32)
    mask = rng.random((47, 53)) > 0.2
    sequential_health = calculate_health_layers(*bands, mask)
    with ThreadPoolExecutor(max_workers=8) as executor:
        parallel_health = calculate_health_layers(*bands, mask, executor=executor)
        parallel_scores = calculate_condition_score_layers(
            parallel_health,
            executor=executor,
        )
    sequential_scores = calculate_condition_score_layers(sequential_health)

    assert parallel_health.values.keys() == sequential_health.values.keys()
    for name in sequential_health.values:
        np.testing.assert_array_equal(
            parallel_health.values[name],
            sequential_health.values[name],
        )
    for name in sequential_scores.component_scores:
        np.testing.assert_array_equal(
            parallel_scores.component_scores[name],
            sequential_scores.component_scores[name],
        )
    np.testing.assert_array_equal(
        parallel_scores.condition_score,
        sequential_scores.condition_score,
    )


def test_compact_spatial_masks_match_full_spatial_layers() -> None:
    rng = np.random.default_rng(23)
    score = rng.uniform(0, 100, (83, 79)).astype(np.float32)
    valid = rng.random(score.shape) > 0.17
    score[4, 5] = np.nan
    config = ConditionConfig()
    full = calculate_spatial_condition_layers(
        score,
        valid,
        median_score=57.123456,
        median_absolute_deviation=8.765432,
        config=config,
    )
    relative, low, alert = calculate_spatial_alert_masks(
        score,
        valid,
        median_score=57.123456,
        median_absolute_deviation=8.765432,
        config=config,
    )

    np.testing.assert_array_equal(relative, full.relative_anomaly_mask)
    np.testing.assert_array_equal(low, full.low_vigor_mask)
    np.testing.assert_array_equal(alert, full.alert_mask)


def test_condition_combination_matches_the_original_float64_accumulation() -> None:
    rng = np.random.default_rng(23)
    bands = rng.uniform(0.01, 0.9, (4, 37, 41)).astype(np.float32)
    health = calculate_health_layers(*bands, np.ones((37, 41), dtype=bool))
    config = ConditionConfig()
    scores = calculate_condition_score_layers(health, config=config)
    numerator = np.zeros((37, 41), dtype=np.float64)
    denominator = np.zeros((37, 41), dtype=np.float64)
    for name in ("ndvi", "gndvi", "evi", "savi"):
        component = scores.component_scores[name]
        available = health.analysis_mask & np.isfinite(component)
        numerator[available] += config.weights[name] * component[available]
        denominator[available] += config.weights[name]
    expected = np.full((37, 41), np.nan, dtype=np.float32)
    usable = health.analysis_mask & (denominator > 0)
    expected[usable] = (numerator[usable] / denominator[usable]).astype(np.float32)

    np.testing.assert_array_equal(scores.condition_score, expected)


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

    with ThreadPoolExecutor(max_workers=4) as executor:
        parallel = apply_calibration(image, calibration, executor=executor)
    np.testing.assert_array_equal(parallel, calibrated)

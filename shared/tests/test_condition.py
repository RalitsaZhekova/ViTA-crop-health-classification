from __future__ import annotations

import json

import numpy as np
import pytest
from prithvi_shared.condition import ConditionConfig, assess_crop_condition
from prithvi_shared.health import HealthLayers, calculate_health_layers


def _constant_layers(
    *,
    blue: float,
    green: float,
    red: float,
    nir: float,
    shape: tuple[int, int] = (20, 20),
) -> HealthLayers:
    bands = [np.full(shape, value, dtype=np.float32) for value in (blue, green, red, nir)]
    return calculate_health_layers(*bands, np.ones(shape, dtype=bool))


def test_high_vigor_region_is_nominal_and_bounded() -> None:
    layers = _constant_layers(blue=0.05, green=0.10, red=0.05, nir=0.80)

    result = assess_crop_condition(layers, mean_crop_probability=0.9)

    assessment = result.assessment
    assert assessment.status == "MEASURED"
    assert assessment.label == "Nominal"
    assert assessment.condition_score == pytest.approx(100.0)
    assert assessment.relative_anomaly_percentage == 0.0
    assert assessment.low_vigor_percentage == 0.0
    assert assessment.evidence_quality_label == "HIGH"
    assert assessment.configuration["ndvi_weight"] == pytest.approx(0.40)
    assert np.nanmin(result.layers.condition_score) >= 0
    assert np.nanmax(result.layers.condition_score) <= 100
    assert not np.any(result.layers.alert_mask)


def test_uniform_low_vigor_region_is_high_anomaly_without_false_spatial_anomaly() -> None:
    layers = _constant_layers(blue=0.20, green=0.25, red=0.30, nir=0.35)

    result = assess_crop_condition(layers)

    assert result.assessment.label == "High anomaly"
    assert result.assessment.condition_score is not None
    assert result.assessment.condition_score < 35
    assert result.assessment.relative_anomaly_percentage == 0.0
    assert result.assessment.low_vigor_percentage == 100.0
    assert not np.any(result.layers.relative_anomaly_mask)
    assert np.all(result.layers.low_vigor_mask)


def test_local_low_vigor_patch_creates_robust_watch_alert() -> None:
    shape = (20, 20)
    blue = np.full(shape, 0.05, dtype=np.float32)
    green = np.full(shape, 0.10, dtype=np.float32)
    red = np.full(shape, 0.05, dtype=np.float32)
    nir = np.full(shape, 0.80, dtype=np.float32)
    blue[:5, :5] = 0.20
    green[:5, :5] = 0.25
    red[:5, :5] = 0.30
    nir[:5, :5] = 0.35
    layers = calculate_health_layers(blue, green, red, nir, np.ones(shape, dtype=bool))

    result = assess_crop_condition(layers)

    assert result.assessment.label == "Watch"
    assert result.assessment.relative_anomaly_percentage == pytest.approx(6.25)
    assert result.assessment.low_vigor_percentage == pytest.approx(6.25)
    assert np.count_nonzero(result.layers.relative_anomaly_mask) == 25
    assert np.count_nonzero(result.layers.alert_mask) == 25
    assert any("crop-region median" in item for item in result.assessment.explanations)


def test_tiny_valid_region_returns_insufficient_data() -> None:
    layers = _constant_layers(
        blue=0.05,
        green=0.10,
        red=0.05,
        nir=0.80,
        shape=(4, 4),
    )

    result = assess_crop_condition(layers)

    assert result.assessment.status == "INSUFFICIENT_DATA"
    assert result.assessment.label == "Insufficient data"
    assert result.assessment.condition_score is None
    assert not np.any(result.layers.alert_mask)


def test_tiny_low_vigor_region_does_not_emit_anomaly_claims() -> None:
    layers = _constant_layers(
        blue=0.20,
        green=0.25,
        red=0.30,
        nir=0.35,
        shape=(4, 4),
    )

    result = assess_crop_condition(layers)

    assert result.assessment.status == "INSUFFICIENT_DATA"
    assert not np.any(result.layers.relative_anomaly_mask)
    assert not np.any(result.layers.low_vigor_mask)
    assert not np.any(result.layers.alert_mask)


def test_component_maps_are_nan_outside_analysis_mask() -> None:
    layers = _constant_layers(blue=0.05, green=0.10, red=0.05, nir=0.80)
    layers.analysis_mask[0, 0] = False

    result = assess_crop_condition(layers)

    assert np.isnan(result.layers.condition_score[0, 0])
    assert all(np.isnan(score[0, 0]) for score in result.layers.component_scores.values())


def test_missing_scored_index_is_rejected() -> None:
    layers = _constant_layers(blue=0.05, green=0.10, red=0.05, nir=0.80)
    values = dict(layers.values)
    del values["savi"]

    with pytest.raises(ValueError, match="savi"):
        assess_crop_condition(HealthLayers(values, layers.analysis_mask))


def test_invalid_mean_crop_probability_is_rejected() -> None:
    layers = _constant_layers(blue=0.05, green=0.10, red=0.05, nir=0.80)
    with pytest.raises(ValueError, match="between 0 and 1"):
        assess_crop_condition(layers, mean_crop_probability=1.1)


def test_condition_config_rejects_invalid_ranges_and_thresholds() -> None:
    with pytest.raises(ValueError, match="low < high"):
        ConditionConfig(ndvi_reference=(0.8, 0.2))
    with pytest.raises(ValueError, match="strictly descending"):
        ConditionConfig(nominal_minimum=50, watch_minimum=60)
    with pytest.raises(ValueError, match="sum to 1"):
        ConditionConfig(median_weight=0.5, lower_quartile_weight=0.4)
    with pytest.raises(ValueError, match="non-negative"):
        ConditionConfig(ndvi_weight=-0.1)


def test_random_reflectance_produces_json_safe_bounded_results() -> None:
    generator = np.random.default_rng(42)
    shape = (48, 64)
    blue = generator.uniform(0.02, 0.25, shape).astype(np.float32)
    green = generator.uniform(0.03, 0.35, shape).astype(np.float32)
    red = generator.uniform(0.02, 0.40, shape).astype(np.float32)
    nir = generator.uniform(0.10, 0.90, shape).astype(np.float32)
    requested = generator.random(shape) > 0.15
    layers = calculate_health_layers(blue, green, red, nir, requested)

    result = assess_crop_condition(layers, mean_crop_probability=0.8)

    assert result.assessment.condition_score is not None
    assert 0 <= result.assessment.condition_score <= 100
    assert 0 <= result.assessment.evidence_quality_score <= 100
    valid_scores = result.layers.condition_score[np.isfinite(result.layers.condition_score)]
    assert np.all((valid_scores >= 0) & (valid_scores <= 100))
    json.dumps(result.assessment.to_dict(), allow_nan=False)


def test_condition_score_improves_with_stronger_nir_response() -> None:
    weak = _constant_layers(blue=0.10, green=0.20, red=0.25, nir=0.30)
    strong = _constant_layers(blue=0.10, green=0.20, red=0.25, nir=0.75)

    weak_score = assess_crop_condition(weak).assessment.condition_score
    strong_score = assess_crop_condition(strong).assessment.condition_score

    assert weak_score is not None
    assert strong_score is not None
    assert strong_score > weak_score

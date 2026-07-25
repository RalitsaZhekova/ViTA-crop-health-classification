import json
from datetime import UTC, datetime

import numpy as np
import pytest

from prithvi_crop.health import (
    build_analysis_mask,
    build_health_observation,
    calculate_health_layers,
)


def test_analysis_mask_excludes_cloud_fallow_other_and_low_confidence() -> None:
    classification = np.asarray([[2, 9, 12], [7, 3, 0]])
    unusable = np.asarray([[False, False, False], [True, False, False]])
    confidence = np.asarray(
        [[0.9, 0.9, 0.9], [0.9, 0.4, 0.9]],
        dtype=np.float32,
    )

    mask = build_analysis_mask(
        classification,
        unusable,
        classification_confidence=confidence,
        minimum_confidence=0.5,
    )

    assert mask.tolist() == [[True, False, False], [False, False, False]]


def test_requested_indices_and_rgb_features_match_definitions() -> None:
    blue = np.full((1, 1), 0.1, dtype=np.float32)
    green = np.full((1, 1), 0.2, dtype=np.float32)
    red = np.full((1, 1), 0.3, dtype=np.float32)
    nir = np.full((1, 1), 0.7, dtype=np.float32)

    result = calculate_health_layers(
        blue,
        green,
        red,
        nir,
        np.ones((1, 1), dtype=bool),
    )

    assert result.values["ndvi"][0, 0] == pytest.approx(0.4)
    assert result.values["evi"][0, 0] == pytest.approx(1.0 / 2.75)
    assert result.values["gndvi"][0, 0] == pytest.approx(0.5 / 0.9)
    assert result.values["cvi"][0, 0] == pytest.approx(5.25)
    assert result.values["rgb_brightness"][0, 0] == pytest.approx(0.2)
    assert result.values["excess_green"][0, 0] == pytest.approx(0.0, abs=1e-7)
    assert result.values["vari"][0, 0] == pytest.approx(-0.25)


def test_invalid_reflectance_and_unstable_denominators_become_nan() -> None:
    blue = np.asarray([[0.1, 0.1]], dtype=np.float32)
    green = np.asarray([[0.0, 4.0]], dtype=np.float32)
    red = np.asarray([[0.0, 0.2]], dtype=np.float32)
    nir = np.asarray([[0.0, 0.8]], dtype=np.float32)

    result = calculate_health_layers(
        blue,
        green,
        red,
        nir,
        np.ones((1, 2), dtype=bool),
    )

    assert np.isnan(result.values["ndvi"][0, 0])
    assert np.isnan(result.values["cvi"][0, 0])
    assert not result.analysis_mask[0, 1]
    assert all(np.isnan(layer[0, 1]) for layer in result.values.values())


def test_raw_integer_digital_numbers_are_rejected() -> None:
    band = np.ones((2, 2), dtype=np.uint16)
    with pytest.raises(TypeError, match="raw integer"):
        calculate_health_layers(band, band, band, band, np.ones((2, 2)))


def test_observation_is_strict_json_and_does_not_diagnose_health() -> None:
    band = np.full((2, 2), 0.2, dtype=np.float32)
    layers = calculate_health_layers(
        band,
        band,
        band,
        band,
        np.zeros((2, 2), dtype=bool),
    )
    observation = build_health_observation(
        scene_id="S2_DEMO_001",
        region_id="field-42",
        sensor="Sentinel-2",
        acquired_at=datetime(2026, 7, 25, 9, 30, tzinfo=UTC),
        layers=layers,
        minimum_analysis_pixels=2,
    )

    record = json.loads(observation.to_json())
    assert record["status"] == "INSUFFICIENT_DATA"
    assert record["quality"]["analysis_pixels"] == 0
    assert record["metrics"]["ndvi"]["median"] is None
    assert "healthy" not in observation.to_json().lower()

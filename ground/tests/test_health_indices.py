from __future__ import annotations

import numpy as np
import pytest
from prithvi_ground.health import calculate_health_layers


def test_calculate_health_layers_matches_index_formulas() -> None:
    blue = np.array([[0.10]], dtype=np.float32)
    green = np.array([[0.20]], dtype=np.float32)
    red = np.array([[0.30]], dtype=np.float32)
    nir = np.array([[0.70]], dtype=np.float32)

    layers = calculate_health_layers(
        blue,
        green,
        red,
        nir,
        np.array([[True]]),
    )

    expected = {
        "ndvi": 0.4,
        "gndvi": 5.0 / 9.0,
        "evi": 1.0 / 2.75,
        "savi": 0.4,
        "cvi": 5.25,
        "vari": -0.25,
        "rgb_brightness": 0.2,
        "excess_green": 0.0,
    }
    assert set(layers.values) == set(expected)
    for name, value in expected.items():
        np.testing.assert_allclose(
            layers.values[name][0, 0],
            value,
            rtol=1e-6,
            atol=1e-7,
        )


def test_calculate_health_layers_masks_invalid_reflectance_and_denominators() -> None:
    blue = np.array([[0.1, 0.1, 0.1]], dtype=np.float32)
    green = np.array([[0.2, 0.2, 0.0]], dtype=np.float32)
    red = np.array([[0.3, 3.0, 0.0]], dtype=np.float32)
    nir = np.array([[0.7, 0.7, 0.0]], dtype=np.float32)
    requested = np.array([[True, True, True]])

    layers = calculate_health_layers(blue, green, red, nir, requested)

    np.testing.assert_array_equal(layers.analysis_mask, [[True, False, True]])
    assert np.isnan(layers.values["ndvi"][0, 1])
    assert np.isnan(layers.values["ndvi"][0, 2])
    assert np.isnan(layers.values["cvi"][0, 2])
    assert np.isfinite(layers.values["rgb_brightness"][0, 2])


def test_calculate_health_layers_rejects_raw_integer_bands() -> None:
    band = np.ones((2, 2), dtype=np.uint16)
    with pytest.raises(TypeError, match="floating point"):
        calculate_health_layers(band, band, band, band, np.ones((2, 2), dtype=bool))


@pytest.mark.parametrize("soil_factor", [-0.01, 1.01])
def test_calculate_health_layers_rejects_invalid_savi_factor(soil_factor: float) -> None:
    band = np.ones((1, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="savi_soil_factor"):
        calculate_health_layers(
            band,
            band,
            band,
            band,
            np.ones((1, 1), dtype=bool),
            savi_soil_factor=soil_factor,
        )

from __future__ import annotations

import warnings

import numpy as np
import pytest
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import MemoryFile

from scripts.balkan1.process_l1a import (
    _dark_surface,
    _estimate_dark_reference,
    _remap_strip,
)
from scripts.balkan1.validate_l1a import _radiometric_diagnostic


def _raw_detector_fixture() -> np.ndarray:
    height = 5
    width = 40
    calibration_pixels = 8
    left_x = (calibration_pixels - 1) / 2
    right_x = width - (calibration_pixels + 1) / 2
    rows = np.arange(height, dtype=np.float32)
    left = 100 + rows
    right = 120 + 2 * rows
    detector_x = np.arange(width, dtype=np.float32)
    alpha = (detector_x - left_x) / (right_x - left_x)
    dark = left[:, None] + (right - left)[:, None] * alpha[None, :]
    raw = np.rint(dark + 50).astype(np.uint16)
    raw[:, :calibration_pixels] = np.rint(left[:, None]).astype(np.uint16)
    raw[:, -calibration_pixels:] = np.rint(right[:, None]).astype(np.uint16)
    return raw[None, ...]


def test_dark_reference_uses_real_calibration_columns_and_removes_plane() -> None:
    raw = _raw_detector_fixture()
    profile = {
        "driver": "GTiff",
        "height": raw.shape[1],
        "width": raw.shape[2],
        "count": 1,
        "dtype": "uint16",
    }
    with MemoryFile() as memory, warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with memory.open(**profile) as dataset:
            dataset.write(raw)
            model = _estimate_dark_reference(dataset, 8, None)
            corrected = _remap_strip(
                dataset,
                band_number=1,
                inverse_matrix=np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64),
                border=10,
                output_width=20,
                output_height=raw.shape[1],
                y_start=0,
                strip_height=raw.shape[1],
                dark_reference=model,
                column_correction=np.zeros(20, dtype=np.float32),
            )

    assert model.source.startswith("per-line median of 8")
    np.testing.assert_allclose(model.left_dn[0], [100, 101, 102, 103, 104])
    np.testing.assert_allclose(model.right_dn[0], [120, 122, 124, 126, 128])
    np.testing.assert_allclose(corrected, 50, atol=1)


def test_dark_surface_interpolates_missing_rows_and_constant_override() -> None:
    raw = _raw_detector_fixture()
    raw[:, 2, :] = 0
    profile = {
        "driver": "GTiff",
        "height": raw.shape[1],
        "width": raw.shape[2],
        "count": 1,
        "dtype": "uint16",
    }
    with MemoryFile() as memory, warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with memory.open(**profile) as dataset:
            dataset.write(raw)
            measured = _estimate_dark_reference(dataset, 8, None)
            overridden = _estimate_dark_reference(dataset, 8, 17.5)

    assert measured.left_dn[0, 2] == 102
    assert measured.right_dn[0, 2] == 124
    surface = _dark_surface(
        overridden,
        band_index=0,
        detector_x=np.array([0, 20, 39], dtype=np.float32),
        row_y=np.array([0, 2, 4], dtype=np.float32),
    )
    np.testing.assert_allclose(surface, 17.5)
    assert overridden.source == "explicit --black-level-dn override"


def test_reference_radiometric_fit_is_diagnostic_and_deterministic() -> None:
    moving = np.arange(1, 40_001, dtype=np.float32).reshape(200, 200)
    reference = moving * 0.25 + 0.125
    diagnostic = _radiometric_diagnostic(
        moving,
        reference,
        np.ones(moving.shape, dtype=bool),
    )

    assert diagnostic["samples"] > 10_000
    assert diagnostic["diagnostic_affine_slope"] == pytest.approx(0.25)
    assert diagnostic["diagnostic_affine_intercept"] == pytest.approx(0.125)
    assert diagnostic["pearson_correlation"] == pytest.approx(1)
    assert diagnostic["rmse_after_diagnostic_affine"] < 1e-9

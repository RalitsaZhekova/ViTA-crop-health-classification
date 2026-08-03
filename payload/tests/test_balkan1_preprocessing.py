from __future__ import annotations

import warnings

import cv2
import numpy as np
import pytest
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import MemoryFile

from scripts.balkan1.process_l1a import (
    _dark_surface,
    _estimate_dark_reference,
    _metadata_from_extraction_log,
    _phase_shift,
    _remap_strip,
    _remap_strip_cuda,
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


def test_cuda_remap_matches_cpu_dark_correction() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    raw = _raw_detector_fixture()
    profile = {
        "driver": "GTiff",
        "height": raw.shape[1],
        "width": raw.shape[2],
        "count": 1,
        "dtype": "uint16",
    }
    arguments = {
        "band_number": 1,
        "inverse_matrix": np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64),
        "border": 10,
        "output_width": 20,
        "output_height": raw.shape[1],
        "y_start": 0,
        "strip_height": raw.shape[1],
        "column_correction": np.zeros(20, dtype=np.float32),
    }
    with MemoryFile() as memory, warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with memory.open(**profile) as dataset:
            dataset.write(raw)
            model = _estimate_dark_reference(dataset, 8, None)
            cpu = _remap_strip(dataset, dark_reference=model, **arguments)
            cuda, cuda_seconds = _remap_strip_cuda(
                dataset, dark_reference=model, **arguments
            )

    np.testing.assert_allclose(cuda, cpu, atol=1e-3)
    assert cuda_seconds > 0


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


def test_phase_correlation_recovers_real_subpixel_translation() -> None:
    moving = np.random.default_rng(7).normal(size=(256, 256)).astype(np.float32)
    expected = np.array([4.25, -2.5])
    target = cv2.warpAffine(
        moving,
        np.array([[1, 0, expected[0]], [0, 1, expected[1]]], dtype=np.float64),
        (moving.shape[1], moving.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )

    shift, response = _phase_shift(moving, target)

    np.testing.assert_allclose(shift, expected, atol=0.4)
    assert response > 0.5


def test_extraction_log_recovers_line_timing_and_utc_anchors(tmp_path) -> None:
    log = tmp_path / "log_extract.txt"
    log.write_text(
        """
PlatformID = 0
InstrumentID = 0
PacketVersion = [1, 0]
Closed = True
LinePeriod = 880
SpectralBands = 8
ExposureTime = 848
BandSetup = [4, 7, 10, 16, 0, 0, 0, 11]
BandStartRow = [3396, 4616, 4212, 3764, 2804, 2312, 1924, 3044]
BandCWL = [625, 490, 560, 665, 0, 0, 0, 842]
PGAGain = 125
ADCGain = 38
DarkOffset = -700
SceneWidth = 9520
SceneHeight = 1
ExposureStart Timestamp = 101
LineData SpectralBand = 1 LineNumber = 0
ExposureStart Timestamp = 102LineData SpectralBand = 2 LineNumber = 0
ExposureStart Timestamp = 103
LineData SpectralBand = 3 LineNumber = 0
ExposureStart Timestamp = 104LineData SpectralBand = 7 LineNumber = 0
ExposureStart Timestamp = 105
LineData SpectralBand = 0 LineNumber = 0
{'ImagerTime': 1000000, 'PPS': True}
{'ExposureTimestamp': 101, 'Data': b'1770953894.4859014'}
""",
        encoding="utf-8",
    )

    metadata = _metadata_from_extraction_log(log)

    assert metadata["Scenes"][0]["1"][0] == [0, 101, 101]
    assert metadata["Scenes"][0]["2"][0] == [0, 102, 102]
    assert metadata["Timesync"] == [{"ImagerTime": 1_000_000, "PPS": True}]
    assert metadata["UserData"]["5"] == [
        {"LastExposureTimestamp": 101, "Data": "1770953894.4859014"}
    ]

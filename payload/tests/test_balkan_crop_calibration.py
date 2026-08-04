from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from prithvi_payload.balkan_crop_calibration import (
    ADAPTER_MODE,
    CALIBRATION_SCHEMA_VERSION,
    CalibrationError,
    apply_calibration,
    load_calibration,
    sha256_file,
)


def _calibration(source: Path) -> dict:
    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "adapter_mode": ADAPTER_MODE,
        "sensor": "balkan-1",
        "acquired_at": "2026-06-16T12:00:00+00:00",
        "source_band_order": ["BLUE", "GREEN", "RED", "NIR_BROAD"],
        "model_band_order": ["BLUE", "GREEN", "RED", "NIR_NARROW"],
        "source_band_indices_1_based": [1, 2, 3, 4],
        "source_scale_to_model_units": 10_000.0,
        "analysis_resolution_metres": 10.0,
        "source": {
            "filename": source.name,
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        },
        "reference": {"sensor": "sentinel-2", "sha256": "a" * 64},
        "curves": [
            {"band": band, "source_knots": [0.0, 10_000.0], "target_values": [100.0, 9_000.0]}
            for band in ("BLUE", "GREEN", "RED", "NIR_BROAD")
        ],
        "validation": {
            "status": "PASS",
            "held_out_valid_pixels": 20_000,
            "held_out_band_correlations": [0.9, 0.85, 0.8, 0.75],
        },
    }


def test_load_and_apply_validated_monotonic_calibration(tmp_path: Path) -> None:
    source = tmp_path / "scene.tif"
    source.write_bytes(b"real-source-provenance")
    sidecar = tmp_path / "scene.crop_calibration.json"
    sidecar.write_text(json.dumps(_calibration(source)), encoding="utf-8")

    calibration = load_calibration(sidecar, source_path=source)
    values = np.full((4, 2, 2), 5_000.0, dtype=np.float32)

    assert np.allclose(apply_calibration(values, calibration), 4_550.0)


def test_calibration_fails_closed_on_source_or_quality_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "scene.tif"
    source.write_bytes(b"original")
    value = _calibration(source)
    sidecar = tmp_path / "scene.crop_calibration.json"
    sidecar.write_text(json.dumps(value), encoding="utf-8")
    source.write_bytes(b"changed!")

    with pytest.raises(CalibrationError, match="SHA-256"):
        load_calibration(sidecar, source_path=source)

    source.write_bytes(b"original")
    value["validation"]["held_out_band_correlations"][3] = 0.2
    sidecar.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(CalibrationError, match="quality gate"):
        load_calibration(sidecar, source_path=source)

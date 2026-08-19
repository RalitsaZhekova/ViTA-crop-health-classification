from __future__ import annotations

from prithvi_payload.balkan_crop_calibration import ADAPTER_MODE
from prithvi_payload.crop_stage import build_crop_stage_plan


def _cloud_inputs() -> tuple[dict, dict]:
    return (
        {"input": {"reflectance_scale": 1.0}},
        {"cloud_percentage": 0.0, "output_files": {}},
    )


def test_sentinel_uses_lower_crop_and_health_thresholds() -> None:
    cloud_plan, cloud_metadata = _cloud_inputs()
    intake = {
        "sensor": "sentinel-2",
        "acquired_at": "2026-06-01T12:00:00+00:00",
        "readiness": {"crop": "READY"},
        "model_band_routes": {
            "crop_classification": {
                "source_band_indices": [1, 2, 3, 4],
                "expected_logical_order": ["BLUE", "GREEN", "RED", "NIR_NARROW"],
            }
        },
    }

    plan = build_crop_stage_plan(
        intake,
        cloud_plan,
        cloud_metadata,
        unusable_mask_available=True,
    )

    assert plan["model"]["crop_probability_threshold"] == 0.40
    assert plan["model"]["health_analysis_crop_threshold"] == 0.40
    assert plan["model"]["threshold_source"] == "selected_model_internal_validation"


def test_processed_balkan_uses_strict_sensor_thresholds(tmp_path) -> None:
    calibration = tmp_path / "balkan.crop_calibration.json"
    calibration.write_text("{}", encoding="utf-8")
    cloud_plan, cloud_metadata = _cloud_inputs()
    intake = {
        "sensor": "balkan-1",
        "acquired_at": "2026-06-01T12:00:00+00:00",
        "readiness": {"crop": "READY"},
        "analysis": {
            "source_path": "analysis.tif",
            "model_band_routes": {
                "crop_classification": {
                    "source_band_indices": [1, 2, 3, 4],
                    "expected_logical_order": ["BLUE", "GREEN", "RED", "NIR_NARROW"],
                }
            },
        },
        "model_band_routes": {
            "crop_classification": {
                "source_band_indices": [1, 2, 3, 4],
                "expected_logical_order": ["BLUE", "GREEN", "RED", "NIR_NARROW"],
                "spectral_adapter": {
                    "mode": ADAPTER_MODE,
                    "validation_status": "VALIDATED_SENTINEL_EQUIVALENCE",
                    "source_scale_to_model_units": 10_000.0,
                    "calibration_path": str(calibration),
                },
            }
        },
    }

    plan = build_crop_stage_plan(
        intake,
        cloud_plan,
        cloud_metadata,
        unusable_mask_available=True,
    )

    assert plan["model"]["crop_probability_threshold"] == 0.49
    assert plan["model"]["health_analysis_crop_threshold"] == 0.645
    assert plan["model"]["threshold_source"] == "balkan_1_strict_sensor_policy"


def test_raw_balkan_uses_strict_sensor_thresholds(tmp_path) -> None:
    calibration = tmp_path / "raw.crop_calibration.json"
    calibration.write_text("{}", encoding="utf-8")
    cloud_plan, cloud_metadata = _cloud_inputs()
    intake = {
        "sensor": "balkan-1",
        "acquired_at": "2026-06-01T12:00:00+00:00",
        "readiness": {"crop": "READY"},
        "analysis": {
            "source_path": "analysis.tif",
            "model_band_routes": {
                "crop_classification": {
                    "source_band_indices": [1, 2, 3, 4],
                    "expected_logical_order": ["BLUE", "GREEN", "RED", "NIR_NARROW"],
                }
            },
        },
        "model_band_routes": {
            "crop_classification": {
                "source_band_indices": [1, 2, 3, 4],
                "expected_logical_order": ["BLUE", "GREEN", "RED", "NIR_NARROW"],
                "spectral_adapter": {
                    "mode": ADAPTER_MODE,
                    "validation_status": "EXPERIMENTAL_RAW_PROXY",
                    "source_scale_to_model_units": 10_000.0,
                    "calibration_path": str(calibration),
                    "experimental_raw_proxy": {
                        "status": "UNQUALIFIED_ENGINEERING_EXPERIMENT",
                        "crop_probability_threshold": 0.49,
                        "health_analysis_crop_threshold": 0.645,
                    },
                },
            }
        },
    }

    plan = build_crop_stage_plan(
        intake,
        cloud_plan,
        cloud_metadata,
        unusable_mask_available=True,
        allow_experimental_raw_proxy=True,
    )

    assert plan["model"]["crop_probability_threshold"] == 0.49
    assert plan["model"]["health_analysis_crop_threshold"] == 0.645
    assert plan["model"]["threshold_source"] == "balkan_1_raw_strict_sensor_policy"

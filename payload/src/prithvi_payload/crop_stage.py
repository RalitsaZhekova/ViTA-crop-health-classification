"""Plan cloud-gated crop segmentation for a validated scene."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from prithvi_shared import (
    CROP_CLASSIFICATION_THRESHOLD,
    INPUT_HEIGHT,
    MODEL_BANDS,
    SELECTED_CHECKPOINT_NAME,
    SELECTED_CHECKPOINT_SHA256,
)

CROP_STAGE_SCHEMA_VERSION = "0.1-draft"
DEFAULT_MAX_CLOUD_PERCENTAGE = 60.0
CALIBRATED_BALKAN_ADAPTER = "BALKAN_1_SENTINEL_MONOTONIC_V1"


def _temporal_coordinate(acquired_at: str | None) -> list[float] | None:
    if acquired_at is None:
        return None
    parsed = datetime.fromisoformat(acquired_at.replace("Z", "+00:00"))
    return [float(parsed.year), float(parsed.timetuple().tm_yday)]


def build_crop_stage_plan(
    intake: dict[str, Any],
    cloud_plan: dict[str, Any],
    cloud_metadata: dict[str, Any],
    *,
    max_cloud_percentage: float = DEFAULT_MAX_CLOUD_PERCENTAGE,
) -> dict[str, Any]:
    """Build a deterministic crop-stage plan without loading the crop model."""
    if not 0 < max_cloud_percentage <= 100:
        raise ValueError("max_cloud_percentage must be within (0, 100]")

    cloud_percentage = float(cloud_metadata["cloud_percentage"])
    gate_passed = cloud_percentage < max_cloud_percentage
    route = intake.get("model_band_routes", {}).get("crop_classification", {})
    source_indices = route.get("source_band_indices")
    expected_order = route.get("expected_logical_order")
    errors: list[str] = []
    warnings: list[str] = []
    crop_input_readiness = intake.get("readiness", {}).get("crop")
    spectral_adapter = route.get("spectral_adapter")
    calibrated_balkan = (
        intake.get("sensor") == "balkan-1"
        and isinstance(spectral_adapter, dict)
        and spectral_adapter.get("mode") == CALIBRATED_BALKAN_ADAPTER
    )

    if gate_passed and crop_input_readiness not in {"READY", "READY_PROVISIONAL"}:
        errors.append(
            "Crop input is not spectrally ready; the selected model requires "
            "BLUE, GREEN, RED and NIR_NARROW"
        )
    if gate_passed and crop_input_readiness == "READY_PROVISIONAL":
        if (
            not isinstance(spectral_adapter, dict)
            or spectral_adapter.get("validation_status") != "EXECUTION_ONLY_UNVALIDATED"
        ):
            errors.append("Provisional crop readiness is missing its spectral adapter record")
        warnings.extend(
            [
                "Balkan-1 crop inference uses an unvalidated NIR transfer",
                "Results demonstrate software execution only, not crop accuracy",
            ]
        )
    if gate_passed and calibrated_balkan:
        if spectral_adapter.get("validation_status") != "VALIDATED_SENTINEL_EQUIVALENCE":
            errors.append("Balkan-1 crop calibration is not validated")
        calibration_path = spectral_adapter.get("calibration_path")
        if not isinstance(calibration_path, str) or not Path(calibration_path).is_file():
            errors.append("Balkan-1 crop calibration sidecar is unavailable")
    if gate_passed and (
        not isinstance(source_indices, list)
        or len(source_indices) != 4
        or any(not isinstance(index, int) or index <= 0 for index in source_indices)
    ):
        errors.append("A complete four-band crop route is unavailable")
    if gate_passed and expected_order != list(MODEL_BANDS):
        errors.append("Crop route has an unexpected logical band order")

    temporal_coordinate = _temporal_coordinate(intake.get("acquired_at"))
    if gate_passed and temporal_coordinate is None:
        errors.append("Crop classification requires the image acquisition timestamp")

    cloud_reflectance_scale = cloud_plan.get("input", {}).get("reflectance_scale")
    if calibrated_balkan:
        training_scale_multiplier = spectral_adapter.get("source_scale_to_model_units")
        if not isinstance(training_scale_multiplier, (int, float)):
            training_scale_multiplier = None
    elif not isinstance(cloud_reflectance_scale, (int, float)) or cloud_reflectance_scale <= 0:
        training_scale_multiplier = None
    else:
        training_scale_multiplier = 10000.0 / float(cloud_reflectance_scale)
    if gate_passed and training_scale_multiplier is None:
        errors.append("A verified reflectance scale is required for crop classification")

    unusable_mask = cloud_metadata.get("output_files", {}).get("unusable_mask")
    if gate_passed and (not isinstance(unusable_mask, str) or not Path(unusable_mask).is_file()):
        errors.append("The cloud stage did not provide an unusable-pixel mask")

    if not gate_passed:
        readiness = "SKIPPED_CLOUD_GATE"
    elif errors:
        readiness = "BLOCKED"
    else:
        readiness = "READY"

    return {
        "schema_version": CROP_STAGE_SCHEMA_VERSION,
        "stage": "crop_classification",
        "scene_id": intake.get("scene_id"),
        "source_path": intake.get("source_path"),
        "sensor": intake.get("sensor"),
        "acquired_at": intake.get("acquired_at"),
        "readiness": readiness,
        "compatibility": (
            "CALIBRATED_BALKAN_1_SENTINEL_EQUIVALENCE"
            if calibrated_balkan
            else (
                "PROVISIONAL_EXECUTION_ONLY"
                if crop_input_readiness == "READY_PROVISIONAL"
                else "VALIDATED_MODEL_INPUT_CONTRACT"
            )
        ),
        "gate": {
            "metric": "cloud_percentage",
            "comparison": "strictly_less_than",
            "observed_percentage": cloud_percentage,
            "maximum_percentage": float(max_cloud_percentage),
            "passed": gate_passed,
        },
        "input": {
            "logical_band_order": expected_order,
            "source_logical_band_order": route.get("source_logical_order"),
            "source_band_indices_1_based": source_indices,
            "spectral_adapter": spectral_adapter,
            "unusable_mask": unusable_mask,
            "training_scale_multiplier": training_scale_multiplier,
            "temporal_coordinate_year_doy": temporal_coordinate,
        },
        "model": {
            "artifact": SELECTED_CHECKPOINT_NAME,
            "sha256": SELECTED_CHECKPOINT_SHA256,
            "crop_probability_threshold": CROP_CLASSIFICATION_THRESHOLD,
            "output_classes": ["non_crop", "crop"],
        },
        "execution": {
            "mode": "WINDOWED_GEOTIFF",
            "full_scene_materialization_allowed": False,
            "tile_size": INPUT_HEIGHT,
            "halo": 16,
            "batch_size": 4,
        },
        "output_contract": {
            "masked_value_uint8": 255,
            "masked_value_float": -9999.0,
            "required_rasters": [
                "crop_probability",
                "crop_binary",
                "crop_confidence",
            ],
            "fraction_denominator": "usable_pixels",
        },
        "errors": errors,
        "warnings": warnings,
    }

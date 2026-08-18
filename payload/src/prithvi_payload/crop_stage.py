"""Plan cloud-gated crop segmentation for a validated scene."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from prithvi_shared import (
    CROP_CLASSIFICATION_THRESHOLD,
    HEALTH_ANALYSIS_CROP_THRESHOLD,
    INPUT_HEIGHT,
    MODEL_BANDS,
    SELECTED_CHECKPOINT_NAME,
    SELECTED_CHECKPOINT_SHA256,
)

from prithvi_payload.balkan_crop_calibration import (
    ADAPTER_MODE,
    BALKAN_CROP_CLASSIFICATION_THRESHOLD,
    BALKAN_HEALTH_ANALYSIS_CROP_THRESHOLD,
)

CROP_STAGE_SCHEMA_VERSION = "0.1-draft"
DEFAULT_MAX_CLOUD_PERCENTAGE = 60.0


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
    unusable_mask_available: bool = False,
    allow_experimental_raw_proxy: bool = False,
) -> dict[str, Any]:
    """Build a deterministic crop-stage plan without loading the crop model."""
    if not 0 < max_cloud_percentage <= 100:
        raise ValueError("max_cloud_percentage must be within (0, 100]")

    cloud_percentage = float(cloud_metadata["cloud_percentage"])
    gate_passed = cloud_percentage < max_cloud_percentage
    original_route = intake.get("model_band_routes", {}).get("crop_classification", {})
    analysis = intake.get("analysis")
    analysis_ready = intake.get("sensor") == "balkan-1" and isinstance(analysis, dict)
    analysis_route = (
        analysis.get("model_band_routes", {}).get("crop_classification", {})
        if analysis_ready
        else {}
    )
    route = original_route
    source_indices = (
        analysis_route.get("source_band_indices")
        if analysis_ready
        else route.get("source_band_indices")
    )
    expected_order = (
        analysis_route.get("expected_logical_order")
        if analysis_ready
        else route.get("expected_logical_order")
    )
    errors: list[str] = []
    warnings: list[str] = []
    experimental_thresholds: tuple[float, float] | None = None
    crop_spatial_detail_restoration: dict[str, Any] | None = None
    crop_input_readiness = intake.get("readiness", {}).get("crop")
    spectral_adapter = route.get("spectral_adapter")
    calibrated_balkan = (
        intake.get("sensor") == "balkan-1"
        and isinstance(spectral_adapter, dict)
        and spectral_adapter.get("mode") == ADAPTER_MODE
    )

    if gate_passed and crop_input_readiness != "READY":
        errors.append(
            "Crop input is not spectrally ready; the selected model requires "
            "BLUE, GREEN, RED and NIR_NARROW"
        )
    if gate_passed and calibrated_balkan:
        validation_status = spectral_adapter.get("validation_status")
        experimental_proxy = spectral_adapter.get("experimental_raw_proxy")
        experimental_allowed = (
            allow_experimental_raw_proxy
            and validation_status == "EXPERIMENTAL_RAW_PROXY"
            and isinstance(experimental_proxy, dict)
            and experimental_proxy.get("status") == "UNQUALIFIED_ENGINEERING_EXPERIMENT"
        )
        if validation_status != "VALIDATED_SENTINEL_EQUIVALENCE" and not experimental_allowed:
            errors.append("Balkan-1 crop calibration is not validated")
        elif experimental_allowed:
            warnings.append(
                "Crop inference is running on an explicitly enabled, unqualified raw proxy"
            )
            raw_crop_threshold = experimental_proxy.get("crop_probability_threshold")
            raw_health_threshold = experimental_proxy.get("health_analysis_crop_threshold")
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.0 <= float(value) <= 1.0
                for value in (raw_crop_threshold, raw_health_threshold)
            ):
                errors.append("Experimental raw-proxy probability thresholds are invalid")
            else:
                experimental_thresholds = (
                    float(raw_crop_threshold),
                    float(raw_health_threshold),
                )
            raw_detail = experimental_proxy.get("crop_spatial_detail_restoration")
            if raw_detail is not None:
                expected_bands = ["BLUE", "GREEN", "RED", "NIR_BROAD"]
                valid_detail = (
                    isinstance(raw_detail, dict)
                    and raw_detail.get("method") == "nodata_aware_unsharp_mask"
                    and raw_detail.get("bands") == expected_bands
                    and isinstance(raw_detail.get("sigma_pixels"), (int, float))
                    and not isinstance(raw_detail.get("sigma_pixels"), bool)
                    and 0.1 <= float(raw_detail["sigma_pixels"]) <= 5.0
                    and isinstance(raw_detail.get("amount"), (int, float))
                    and not isinstance(raw_detail.get("amount"), bool)
                    and 0.0 <= float(raw_detail["amount"]) <= 8.0
                )
                if not valid_detail:
                    errors.append(
                        "Experimental raw-proxy crop spatial-detail restoration is invalid"
                    )
                else:
                    crop_spatial_detail_restoration = {
                        "method": raw_detail["method"],
                        "sigma_pixels": float(raw_detail["sigma_pixels"]),
                        "amount": float(raw_detail["amount"]),
                        "bands": list(raw_detail["bands"]),
                    }
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
    if (
        gate_passed
        and not unusable_mask_available
        and (not isinstance(unusable_mask, str) or not Path(unusable_mask).is_file())
    ):
        errors.append("The cloud stage did not provide an unusable-pixel mask")

    if not gate_passed:
        readiness = "SKIPPED_CLOUD_GATE"
    elif errors:
        readiness = "BLOCKED"
    else:
        readiness = "READY"

    if calibrated_balkan:
        if experimental_thresholds is not None:
            crop_probability_threshold, health_analysis_crop_threshold = experimental_thresholds
            threshold_source = "balkan_1_experimental_raw_proxy_scene_parity"
        else:
            crop_probability_threshold = BALKAN_CROP_CLASSIFICATION_THRESHOLD
            health_analysis_crop_threshold = BALKAN_HEALTH_ANALYSIS_CROP_THRESHOLD
            threshold_source = (
                "balkan_1_experimental_raw_proxy_invalid_fallback"
                if spectral_adapter.get("validation_status") == "EXPERIMENTAL_RAW_PROXY"
                else "balkan_1_operational_calibration"
            )
    else:
        crop_probability_threshold = CROP_CLASSIFICATION_THRESHOLD
        health_analysis_crop_threshold = HEALTH_ANALYSIS_CROP_THRESHOLD
        threshold_source = "selected_model_internal_validation"

    batch_size = int(os.environ.get("VITA_CROP_BATCH_SIZE", "4"))
    if not 1 <= batch_size <= 16:
        raise ValueError("VITA_CROP_BATCH_SIZE must be within 1..16")
    return {
        "schema_version": CROP_STAGE_SCHEMA_VERSION,
        "stage": "crop_classification",
        "scene_id": intake.get("scene_id"),
        "source_path": (
            analysis.get("source_path") if analysis_ready else intake.get("source_path")
        ),
        "sensor": intake.get("sensor"),
        "acquired_at": intake.get("acquired_at"),
        "readiness": readiness,
        "compatibility": (
            "CALIBRATED_BALKAN_1_SENTINEL_EQUIVALENCE"
            if calibrated_balkan
            else "VALIDATED_MODEL_INPUT_CONTRACT"
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
            "calibration_source_path": (intake.get("source_path") if calibrated_balkan else None),
            "analysis_grid_ready": analysis_ready,
            "unusable_mask": unusable_mask,
            "training_scale_multiplier": training_scale_multiplier,
            "temporal_coordinate_year_doy": temporal_coordinate,
            "spatial_detail_restoration": crop_spatial_detail_restoration,
        },
        "model": {
            "artifact": SELECTED_CHECKPOINT_NAME,
            "sha256": SELECTED_CHECKPOINT_SHA256,
            "crop_probability_threshold": crop_probability_threshold,
            "health_analysis_crop_threshold": health_analysis_crop_threshold,
            "threshold_source": threshold_source,
            "output_classes": ["non_crop", "crop"],
        },
        "execution": {
            "mode": "WINDOWED_GEOTIFF",
            "full_scene_materialization_allowed": False,
            "tile_size": INPUT_HEIGHT,
            "halo": 16,
            "batch_size": batch_size,
            # The dashboard builds its own final visualization from these rasters.
            "save_preview": False,
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

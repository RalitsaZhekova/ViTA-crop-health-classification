"""Plan the cloud-detection stage from a validated scene-intake report."""

from __future__ import annotations

import math
from typing import Any

from prithvi_payload.cloud_profiles import CLOUD_MODEL_HALO, CLOUD_MODEL_TILE_SIZE

CLOUD_STAGE_SCHEMA_VERSION = "0.1-draft"


class CloudStagePlanningError(ValueError):
    """Raised when rejected intake metadata is passed to the cloud stage."""


def _positive_scale(value: float | None) -> float | None:
    if value is None:
        return None
    scale = float(value)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("reflectance_scale must be a positive finite number")
    return scale


def build_cloud_stage_plan(
    intake: dict[str, Any],
    *,
    reflectance_scale: float | None = None,
) -> dict[str, Any]:
    """Build a deterministic, non-executing cloud-stage plan."""
    if intake.get("readiness", {}).get("intake") != "READY":
        raise CloudStagePlanningError(
            "Cloud planning requires an intake report with readiness.intake=READY"
        )

    sensor = intake.get("sensor")
    if sensor not in {"sentinel-2", "balkan-1"}:
        raise CloudStagePlanningError(f"Unsupported sensor in intake report: {sensor}")

    analysis = intake.get("analysis")
    analysis_ready = sensor == "balkan-1" and isinstance(analysis, dict)
    routes = (
        analysis.get("model_band_routes", {})
        if analysis_ready
        else intake.get("model_band_routes", {})
    )
    route = routes.get("cloud_detection", {})
    source_indices = route.get("source_band_indices")
    expected_order = route.get("expected_logical_order")
    errors: list[str] = []
    warnings: list[str] = []
    if (
        not isinstance(source_indices, list)
        or len(source_indices) != 4
        or any(not isinstance(index, int) or index <= 0 for index in source_indices)
    ):
        errors.append("A complete four-band cloud route is unavailable")
    if (
        not isinstance(expected_order, list)
        or len(expected_order) != 4
        or expected_order[0] not in {"NIR_BROAD", "NIR_NARROW"}
        or expected_order[1:] != ["RED", "GREEN", "BLUE"]
    ):
        errors.append("Cloud route has an unexpected logical band order")

    supplied_scale = _positive_scale(reflectance_scale)
    inferred_scale = _positive_scale(intake.get("radiometry", {}).get("cloud_reflectance_divisor"))
    selected_scale = supplied_scale if supplied_scale is not None else inferred_scale
    if supplied_scale is not None:
        scale_source = "explicit_override"
    elif inferred_scale is not None:
        scale_source = "intake"
    else:
        scale_source = "unresolved"
    if selected_scale is None:
        errors.append("Reflectance calibration is unresolved; supply a verified reflectance scale")

    crop_route = intake.get("model_band_routes", {}).get("crop_classification", {})
    spectral_adapter = crop_route.get("spectral_adapter")
    experimental_proxy = (
        spectral_adapter.get("experimental_raw_proxy")
        if isinstance(spectral_adapter, dict)
        else None
    )
    spatial_detail_restoration = None
    if isinstance(experimental_proxy, dict):
        candidate = experimental_proxy.get("cloud_spatial_detail_restoration")
        if isinstance(candidate, dict):
            try:
                method = str(candidate["method"])
                sigma_pixels = float(candidate["sigma_pixels"])
                amount = float(candidate["amount"])
                bands = list(candidate["bands"])
            except (KeyError, TypeError, ValueError):
                errors.append("Experimental raw cloud detail restoration is invalid")
            else:
                if (
                    method != "nodata_aware_unsharp_mask"
                    or not math.isfinite(sigma_pixels)
                    or not 0.1 <= sigma_pixels <= 5.0
                    or not math.isfinite(amount)
                    or not 0.0 <= amount <= 8.0
                    or bands != ["RED", "GREEN", "NIR_BROAD"]
                ):
                    errors.append("Experimental raw cloud detail restoration is invalid")
                else:
                    spatial_detail_restoration = {
                        "method": method,
                        "sigma_pixels": sigma_pixels,
                        "amount": amount,
                        "bands": bands,
                    }

    if sensor == "sentinel-2":
        compatibility = "OMNICLOUDMASK_SENTINEL_2"
        validation_status = "UPSTREAM_VALIDATED_SENTINEL_2"
    else:
        compatibility = "OMNICLOUDMASK_BALKAN_1"
        validation_status = "SUPPLIED_ZERO_SHOT_BALKAN_1_BENCHMARK"
        warnings.extend(
            [
                "Balkan-1 validation applies to L1ORT imagery resampled to 10 m",
                "Benchmark annotation caveats remain scene-specific",
                "Panchromatic is retained in the source but is not used by this model",
            ]
        )
        if supplied_scale is not None:
            warnings.append("The explicit scale must come from Balkan-1 calibration metadata")

    return {
        "schema_version": CLOUD_STAGE_SCHEMA_VERSION,
        "stage": "cloud_detection",
        "scene_id": intake.get("scene_id"),
        "source_path": (
            analysis.get("source_path") if analysis_ready else intake.get("source_path")
        ),
        "sensor": sensor,
        "acquired_at": intake.get("acquired_at"),
        "readiness": "READY" if not errors else "BLOCKED",
        "compatibility": compatibility,
        "validation_status": validation_status,
        "input": {
            "logical_band_order": expected_order,
            "source_band_indices_1_based": source_indices,
            "reflectance_scale": selected_scale,
            "reflectance_scale_source": scale_source,
            "nodata_value": (
                analysis.get("raster", {}).get("nodata")
                if analysis_ready
                else intake.get("raster", {}).get("nodata")
            ),
            "analysis_grid_ready": analysis_ready,
            "pan_used": False,
            "spatial_detail_restoration": spatial_detail_restoration,
        },
        "execution": {
            "mode": "WINDOWED_GEOTIFF",
            "source_full_scene_materialization_allowed": False,
            "balkan_resampled_analysis_grid_materialization_allowed": True,
            "tile_size": CLOUD_MODEL_TILE_SIZE,
            # Adjacent model tiles overlap by 300 px: 150 px of context is
            # discarded on each side before the core is written.
            "overlap": CLOUD_MODEL_HALO,
            "model_patch_overlap": 300,
        },
        "output_contract": {
            "semantic_classes": [
                "clear",
                "thick_cloud",
                "thin_cloud",
                "cloud_shadow",
            ],
            "required_rasters": [
                "semantic_mask",
                "unusable_mask",
                "invalid_mask",
            ],
            "required_metadata": [
                "class_fractions",
                "usable_percentage",
                "decision",
                "runtime_seconds",
            ],
        },
        "errors": errors,
        "warnings": warnings,
    }

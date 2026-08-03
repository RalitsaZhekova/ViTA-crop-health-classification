"""Plan the cloud-detection stage from a validated scene-intake report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

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

    route = intake.get("model_band_routes", {}).get("cloud_detection", {})
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
    inferred_scale = _positive_scale(
        intake.get("radiometry", {}).get("cloud_reflectance_divisor")
    )
    selected_scale = supplied_scale if supplied_scale is not None else inferred_scale
    if supplied_scale is not None:
        scale_source = "explicit_override"
    elif inferred_scale is not None:
        scale_source = "intake"
    else:
        scale_source = "unresolved"
    if selected_scale is None:
        errors.append(
            "Reflectance calibration is unresolved; supply a verified reflectance scale"
        )

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
            warnings.append(
                "The explicit scale must come from Balkan-1 calibration metadata"
            )

    return {
        "schema_version": CLOUD_STAGE_SCHEMA_VERSION,
        "stage": "cloud_detection",
        "scene_id": intake.get("scene_id"),
        "source_path": intake.get("source_path"),
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
            "nodata_value": intake.get("raster", {}).get("nodata"),
            "pan_used": False,
        },
        "execution": {
            "mode": "WINDOWED_GEOTIFF",
            "source_full_scene_materialization_allowed": False,
            "balkan_resampled_analysis_grid_materialization_allowed": True,
            "tile_size": 1000,
            # Adjacent model tiles overlap by 300 px: 150 px of context is
            # discarded on each side before the core is written.
            "overlap": 150,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan cloud detection from a scene-inspect JSON report."
    )
    parser.add_argument("intake_report", type=Path)
    parser.add_argument("--reflectance-scale", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    intake = json.loads(args.intake_report.read_text(encoding="utf-8"))
    plan = build_cloud_stage_plan(
        intake,
        reflectance_scale=args.reflectance_scale,
    )
    rendered = json.dumps(plan, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if plan["readiness"] == "READY" else 2)


if __name__ == "__main__":
    main()

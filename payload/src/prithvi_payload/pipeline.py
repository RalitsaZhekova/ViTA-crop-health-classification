"""Manual, stage-gated payload pipeline entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cloud_detection.backend import CloudBackend
from cloud_detection.cli import DEFAULT_CONFIG
from cloud_detection.config import load_config
from cloud_detection.pipeline import CloudDetectionPipeline

from prithvi_payload.cloud_executor import execute_cloud_stage
from prithvi_payload.cloud_stage import build_cloud_stage_plan
from prithvi_payload.crop_stage import (
    DEFAULT_MAX_CLOUD_PERCENTAGE,
    build_crop_stage_plan,
)
from prithvi_payload.scene_intake import SUPPORTED_SENSORS, inspect_scene


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _finish(output_root: Path, result: dict[str, Any]) -> dict[str, Any]:
    summary_path = output_root / "result.json"
    result["artifacts"]["run_summary"] = str(summary_path.resolve())
    _write_json(summary_path, result)
    return result


def run_scene(
    input_path: str | Path,
    *,
    sensor: str,
    output_root: str | Path,
    acquired_at: str | None = None,
    scene_id: str | None = None,
    reflectance_scale: float | None = None,
    stop_after: str = "cloud",
    max_crop_cloud_percentage: float = DEFAULT_MAX_CLOUD_PERCENTAGE,
    cloud_config_path: str | Path = DEFAULT_CONFIG,
    cloud_backend: CloudBackend | None = None,
    cloud_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run only the explicitly selected stages for one preprocessed scene."""
    if stop_after not in {"intake", "cloud", "crop"}:
        raise ValueError("stop_after must be intake, cloud or crop")
    output_root = Path(output_root)
    intake = inspect_scene(
        input_path,
        sensor=sensor,
        acquired_at=acquired_at,
        scene_id=scene_id,
    )
    resolved_scene_id = intake["scene_id"]
    intake_path = output_root / "metadata" / f"{resolved_scene_id}_intake.json"
    _write_json(intake_path, intake)
    result: dict[str, Any] = {
        "schema_version": "0.1-draft",
        "scene_id": resolved_scene_id,
        "sensor": sensor,
        "requested_stop_after": stop_after,
        "completed_stages": ["intake"],
        "status": "INTAKE_READY",
        "artifacts": {"intake": str(intake_path.resolve())},
        "summary": {
            "intake": {
                "readiness": intake["readiness"]["intake"],
                "band_count": intake["raster"]["band_count"],
                "width": intake["raster"]["width"],
                "height": intake["raster"]["height"],
            }
        },
        "stage_metadata": {"intake": intake},
        "warnings": list(intake["warnings"]),
        "errors": list(intake["errors"]),
    }
    if intake["readiness"]["intake"] != "READY":
        result["status"] = "REJECTED_AT_INTAKE"
        return _finish(output_root, result)
    if stop_after == "intake":
        return _finish(output_root, result)

    plan = build_cloud_stage_plan(
        intake,
        reflectance_scale=reflectance_scale,
    )
    plan_path = output_root / "metadata" / f"{resolved_scene_id}_cloud_plan.json"
    _write_json(plan_path, plan)
    result["artifacts"]["cloud_plan"] = str(plan_path.resolve())
    result["stage_metadata"]["cloud_plan"] = plan
    result["warnings"].extend(plan["warnings"])
    result["errors"].extend(plan["errors"])
    if plan["readiness"] != "READY":
        result["status"] = "BLOCKED_AT_CLOUD_PLAN"
        return _finish(output_root, result)

    if cloud_backend is None:
        runtime = CloudDetectionPipeline.from_yaml(cloud_config_path)
        cloud_backend = runtime.backend
        cloud_config = runtime.cfg
    elif cloud_config is None:
        cloud_config = load_config(cloud_config_path)
    cloud_metadata = execute_cloud_stage(
        plan,
        output_root=output_root,
        backend=cloud_backend,
        config=cloud_config,
    )
    result["completed_stages"].append("cloud")
    result["status"] = "CLOUD_COMPLETE"
    result["cloud_decision"] = cloud_metadata["decision"]
    result["artifacts"]["cloud"] = cloud_metadata["output_files"]
    percentages = cloud_metadata["class_percentages"]
    result["summary"]["cloud"] = {
        "decision": cloud_metadata["decision"],
        "clear_percentage": percentages["clear"],
        "thick_cloud_percentage": percentages["thick_cloud"],
        "thin_cloud_percentage": percentages["thin_cloud"],
        "cloud_shadow_percentage": percentages["cloud_shadow"],
        "total_cloud_percentage": cloud_metadata["cloud_percentage"],
        "usable_percentage": cloud_metadata["usable_percentage"],
        "unusable_percentage": cloud_metadata["unusable_percentage"],
        "invalid_percentage": cloud_metadata["invalid_percentage"],
        "runtime_seconds": cloud_metadata["runtime"]["seconds"],
    }
    result["stage_metadata"]["cloud"] = cloud_metadata
    if stop_after == "cloud":
        return _finish(output_root, result)

    crop_plan = build_crop_stage_plan(
        intake,
        plan,
        cloud_metadata,
        max_cloud_percentage=max_crop_cloud_percentage,
    )
    crop_plan_path = output_root / "metadata" / f"{resolved_scene_id}_crop_plan.json"
    _write_json(crop_plan_path, crop_plan)
    result["artifacts"]["crop_plan"] = str(crop_plan_path.resolve())
    result["stage_metadata"]["crop_plan"] = crop_plan
    result["warnings"].extend(crop_plan["warnings"])
    result["errors"].extend(crop_plan["errors"])
    if crop_plan["readiness"] == "SKIPPED_CLOUD_GATE":
        result["status"] = "CROP_SKIPPED_CLOUD_GATE"
        result["crop_decision"] = "SKIP_CLOUD_COVERAGE"
        result["summary"]["crop"] = {
            "decision": "SKIP_CLOUD_COVERAGE",
            "cloud_percentage": crop_plan["gate"]["observed_percentage"],
            "maximum_cloud_percentage": crop_plan["gate"]["maximum_percentage"],
        }
        return _finish(output_root, result)
    if crop_plan["readiness"] != "READY":
        result["status"] = "BLOCKED_AT_CROP_PLAN"
        return _finish(output_root, result)

    # Keep the 100M crop model and TerraTorch out of intake/cloud-only runs.
    from prithvi_payload.crop_executor import execute_crop_stage

    crop_metadata = execute_crop_stage(crop_plan, output_root=output_root)
    result["completed_stages"].append("crop")
    result["status"] = "CROP_COMPLETE"
    result["crop_decision"] = "CLASSIFIED"
    result["artifacts"]["crop"] = crop_metadata["output_files"]
    result["summary"]["crop"] = {
        "decision": "CLASSIFIED",
        "crop_percentage_usable": crop_metadata["crop_percentage_usable"],
        "usable_percentage": crop_metadata["usable_percentage"],
        "mean_crop_probability_usable": crop_metadata[
            "mean_crop_probability_usable"
        ],
        "mean_confidence_usable": crop_metadata["mean_confidence_usable"],
        "runtime_seconds": crop_metadata["runtime"]["seconds"],
        "device": crop_metadata["runtime"]["device"],
    }
    result["stage_metadata"]["crop"] = crop_metadata
    return _finish(output_root, result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the stage-gated payload pipeline on one preprocessed GeoTIFF."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--sensor", required=True, choices=SUPPORTED_SENSORS)
    parser.add_argument("--output", type=Path, default=Path("testing/runs"))
    parser.add_argument("--acquired-at")
    parser.add_argument("--scene-id")
    parser.add_argument("--reflectance-scale", type=float)
    parser.add_argument(
        "--stop-after",
        choices=("intake", "cloud", "crop"),
        default="cloud",
    )
    parser.add_argument(
        "--max-crop-cloud-percentage",
        type=float,
        default=DEFAULT_MAX_CLOUD_PERCENTAGE,
    )
    parser.add_argument("--cloud-config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    result = run_scene(
        args.input,
        sensor=args.sensor,
        output_root=args.output,
        acquired_at=args.acquired_at,
        scene_id=args.scene_id,
        reflectance_scale=args.reflectance_scale,
        stop_after=args.stop_after,
        max_crop_cloud_percentage=args.max_crop_cloud_percentage,
        cloud_config_path=args.cloud_config,
    )
    console_result = {
        key: result[key]
        for key in (
            "schema_version",
            "scene_id",
            "sensor",
            "status",
            "completed_stages",
            "summary",
            "artifacts",
            "warnings",
            "errors",
        )
    }
    print(json.dumps(console_result, indent=2, sort_keys=True))
    successful_statuses = {
        "INTAKE_READY",
        "CLOUD_COMPLETE",
        "CROP_COMPLETE",
        "CROP_SKIPPED_CLOUD_GATE",
    }
    raise SystemExit(0 if result["status"] in successful_statuses else 2)


if __name__ == "__main__":
    main()

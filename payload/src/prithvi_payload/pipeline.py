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
from prithvi_payload.scene_intake import SUPPORTED_SENSORS, inspect_scene


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _finish(output_root: Path, result: dict[str, Any]) -> dict[str, Any]:
    summary_path = output_root / "metadata" / f"{result['scene_id']}_run.json"
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
    cloud_config_path: str | Path = DEFAULT_CONFIG,
    cloud_backend: CloudBackend | None = None,
    cloud_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run only the explicitly selected stages for one preprocessed scene."""
    if stop_after not in {"intake", "cloud"}:
        raise ValueError("stop_after must be intake or cloud")
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
    return _finish(output_root, result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the stage-gated payload pipeline on one preprocessed GeoTIFF."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--sensor", required=True, choices=SUPPORTED_SENSORS)
    parser.add_argument("--output", type=Path, default=Path("outputs/pipeline"))
    parser.add_argument("--acquired-at")
    parser.add_argument("--scene-id")
    parser.add_argument("--reflectance-scale", type=float)
    parser.add_argument("--stop-after", choices=("intake", "cloud"), default="cloud")
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
        cloud_config_path=args.cloud_config,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] in {"INTAKE_READY", "CLOUD_COMPLETE"} else 2)


if __name__ == "__main__":
    main()

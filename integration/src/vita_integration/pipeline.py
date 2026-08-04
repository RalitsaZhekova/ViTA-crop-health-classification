"""One-command Sentinel payload-to-downlink demonstration pipeline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from cloud_detection.backend import CloudBackend
from cloud_detection.cli import DEFAULT_CONFIG
from prithvi_payload.pipeline import run_scene

INTEGRATION_SCHEMA_VERSION = "1.0"
INTEGRATION_ALGORITHM_VERSION = "sentinel-end-to-end-v2"


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def run_sentinel_end_to_end(
    input_path: str | Path,
    *,
    output_root: str | Path,
    acquired_at: str,
    scene_id: str | None = None,
    region_id: str | None = None,
    reflectance_scale: float | None = None,
    max_crop_cloud_percentage: float = 60.0,
    condition_tile_size: int = 512,
    downlink_max_image_dimension: int = 1600,
    downlink_grid_size: int = 16,
    cloud_config_path: str | Path = DEFAULT_CONFIG,
    cloud_backend: CloudBackend | None = None,
    cloud_config: dict[str, Any] | None = None,
    crop_model: Any | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run Sentinel intake, cloud, crop and payload condition stages."""
    started = time.perf_counter()
    source = Path(input_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "end_to_end_result.json"
    if summary_path.exists() and not overwrite:
        raise FileExistsError(f"End-to-end result already exists: {summary_path}")

    payload_root = output / "payload"
    payload_result = run_scene(
        source,
        sensor="sentinel-2",
        output_root=payload_root,
        acquired_at=acquired_at,
        scene_id=scene_id,
        reflectance_scale=reflectance_scale,
        stop_after="downlink",
        max_crop_cloud_percentage=max_crop_cloud_percentage,
        region_id=region_id,
        condition_tile_size=condition_tile_size,
        downlink_max_image_dimension=downlink_max_image_dimension,
        downlink_grid_size=downlink_grid_size,
        overwrite=overwrite,
        cloud_config_path=cloud_config_path,
        cloud_backend=cloud_backend,
        cloud_config=cloud_config,
        crop_model=crop_model,
    )
    payload_result_path = payload_root / "result.json"
    if payload_result.get("status") != "DOWNLINK_READY":
        summary = {
            "schema_version": INTEGRATION_SCHEMA_VERSION,
            "algorithm_version": INTEGRATION_ALGORITHM_VERSION,
            "scene_id": payload_result.get("scene_id"),
            "sensor": "sentinel-2",
            "status": "PAYLOAD_STOPPED",
            "completed_stages": ["payload"],
            "payload_status": payload_result.get("status"),
            "payload_result": _relative(payload_result_path, output),
            "condition_report": None,
            "downlink_manifest": None,
            "summary": {"payload": payload_result.get("summary", {})},
            "runtime_seconds": time.perf_counter() - started,
        }
        _write_json_atomic(summary_path, summary)
        return summary

    condition_report = payload_result["stage_metadata"]["condition"]
    condition_report_path = Path(payload_result["artifacts"]["condition"]["report"])
    downlink_manifest_path = Path(payload_result["artifacts"]["downlink"]["metadata"])
    summary = {
        "schema_version": INTEGRATION_SCHEMA_VERSION,
        "algorithm_version": INTEGRATION_ALGORITHM_VERSION,
        "scene_id": payload_result["scene_id"],
        "sensor": "sentinel-2",
        "status": "COMPLETE",
        "completed_stages": ["payload", "downlink"],
        "payload_status": payload_result["status"],
        "condition_status": condition_report["status"],
        "payload_result": _relative(payload_result_path, output),
        "condition_report": _relative(condition_report_path, output),
        "downlink_manifest": _relative(downlink_manifest_path, output),
        "summary": {
            "cloud": payload_result["summary"]["cloud"],
            "crop": payload_result["summary"]["crop"],
            "condition": {
                "label": condition_report["condition"]["label"],
                "score": condition_report["condition"]["condition_score"],
                "evidence_quality_label": condition_report["condition"]["evidence_quality_label"],
                "analysis_percentage": condition_report["quality"]["analysis_percentage"],
            },
            "downlink": payload_result["summary"]["downlink"],
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json_atomic(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete Sentinel cloud, crop and condition demonstration."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--acquired-at", required=True)
    parser.add_argument("--scene-id")
    parser.add_argument("--region-id")
    parser.add_argument("--reflectance-scale", type=float)
    parser.add_argument("--max-crop-cloud-percentage", type=float, default=60.0)
    parser.add_argument("--condition-tile-size", type=int, default=512)
    parser.add_argument("--downlink-max-image-dimension", type=int, default=1600)
    parser.add_argument("--downlink-grid-size", type=int, default=16)
    parser.add_argument("--cloud-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = run_sentinel_end_to_end(
        args.input,
        output_root=args.output,
        acquired_at=args.acquired_at,
        scene_id=args.scene_id,
        region_id=args.region_id,
        reflectance_scale=args.reflectance_scale,
        max_crop_cloud_percentage=args.max_crop_cloud_percentage,
        condition_tile_size=args.condition_tile_size,
        downlink_max_image_dimension=args.downlink_max_image_dimension,
        downlink_grid_size=args.downlink_grid_size,
        cloud_config_path=args.cloud_config,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    raise SystemExit(0 if result["status"] == "COMPLETE" else 2)


if __name__ == "__main__":
    main()

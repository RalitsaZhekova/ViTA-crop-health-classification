"""The two supported local MVP commands: Sentinel-2 and preprocessed Balkan-1."""

from __future__ import annotations

import argparse
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GROUND_STORE = REPOSITORY_ROOT / "runtime" / "ground"
DEFAULT_RUN_ROOT = REPOSITORY_ROOT / "runtime" / "runs"
BALKAN_BAND_ORDER = ("BLUE", "GREEN", "RED", "NIR", "PAN")

TIMING_ROWS = (
    ("command_overhead_seconds", "Command setup/import overhead"),
    ("startup_total_seconds", "Startup total"),
    ("earth_engine_initialization_seconds", "  Earth Engine initialization"),
    ("cloud_model_load_seconds", "  Cloud model loading"),
    ("crop_model_load_seconds", "  Crop model loading"),
    ("model_warmup_seconds", "  Model warm-up"),
    ("pipeline_total_seconds", "Pipeline total"),
    ("candidate_search_seconds", "  Candidate search"),
    ("earth_engine_acquisition_seconds", "  Acquisition total"),
    ("earth_engine_download_seconds", "    GeoTIFF download"),
    ("geotiff_validation_seconds", "    GeoTIFF validation"),
    ("intake_seconds", "  Scene intake"),
    ("cloud_plan_seconds", "  Cloud planning"),
    ("cloud_stage_seconds", "  Cloud stage total"),
    ("cloud_analysis_grid_seconds", "    Cloud analysis-grid preparation"),
    ("cloud_inference_seconds", "    Cloud inference"),
    ("cloud_mask_processing_seconds", "    Cloud mask processing"),
    ("cloud_mask_reprojection_seconds", "    Cloud mask reprojection"),
    ("crop_plan_seconds", "  Crop planning"),
    ("crop_stage_seconds", "  Crop stage total"),
    ("crop_inference_seconds", "    Crop inference"),
    ("condition_stage_seconds", "  Condition stage total"),
    ("condition_first_pass_seconds", "    Indices and absolute scores"),
    ("condition_robust_statistics_seconds", "    Robust statistics"),
    ("condition_spatial_pass_seconds", "    Spatial anomaly pass"),
    ("condition_preview_seconds", "    Condition preview"),
    ("downlink_packaging_seconds", "  Downlink packaging"),
    ("ground_ingest_seconds", "Ground catalog ingest"),
    ("end_to_end_seconds", "END TO END"),
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _nested_seconds(value: Any, *keys: str) -> float | None:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        return None
    return max(0.0, float(current))


def _render_timing_report(timing: dict[str, float]) -> str:
    lines = [
        "Timing (seconds)",
        "Top-level rows sum to END TO END; indented rows are included substeps.",
    ]
    for key, label in TIMING_ROWS:
        value = timing.get(key)
        if value is not None:
            lines.append(f"{label:<42} {value:>10.4f}")
    return "\n".join(lines)


def _balkan_pipeline_timings(result: dict[str, Any]) -> dict[str, float]:
    stages = result.get("stage_metadata", {})
    pipeline = result.get("timing", {})
    cloud = stages.get("cloud", {}).get("runtime", {})
    crop = stages.get("crop", {}).get("runtime", {})
    condition = stages.get("condition", {}).get("runtime", {})
    downlink = stages.get("downlink", {}).get("runtime", {})
    values = {
        "intake_seconds": _nested_seconds(pipeline, "intake_seconds"),
        "cloud_plan_seconds": _nested_seconds(pipeline, "cloud_plan_seconds"),
        "cloud_stage_seconds": _nested_seconds(cloud, "seconds"),
        "cloud_analysis_grid_seconds": _nested_seconds(
            cloud, "analysis_grid_preparation_seconds"
        ),
        "cloud_inference_seconds": _nested_seconds(cloud, "inference_seconds"),
        "cloud_mask_processing_seconds": _nested_seconds(
            cloud, "mask_processing_seconds"
        ),
        "cloud_mask_reprojection_seconds": _nested_seconds(
            cloud, "mask_reprojection_seconds"
        ),
        "crop_plan_seconds": _nested_seconds(pipeline, "crop_plan_seconds"),
        "crop_stage_seconds": _nested_seconds(crop, "seconds"),
        "crop_inference_seconds": _nested_seconds(crop, "inference_seconds"),
        "condition_stage_seconds": _nested_seconds(condition, "seconds"),
        "condition_first_pass_seconds": _nested_seconds(condition, "first_pass_seconds"),
        "condition_robust_statistics_seconds": _nested_seconds(
            condition, "robust_statistics_seconds"
        ),
        "condition_spatial_pass_seconds": _nested_seconds(
            condition, "spatial_pass_seconds"
        ),
        "condition_preview_seconds": _nested_seconds(condition, "preview_seconds"),
        "downlink_packaging_seconds": _nested_seconds(downlink, "seconds"),
    }
    return {key: value for key, value in values.items() if value is not None}


def _bbox(value: str) -> tuple[float, float, float, float]:
    try:
        coordinates = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "bbox must be west,south,east,north"
        ) from error
    if len(coordinates) != 4:
        raise argparse.ArgumentTypeError("bbox must be west,south,east,north")
    return coordinates


def _safe_id(value: str) -> str:
    identifier = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    if not identifier:
        raise ValueError("Could not derive a safe scene identifier")
    return identifier[:80]


def _new_job_id(region_id: str, start: str) -> str:
    return _safe_id(f"{region_id[:40]}_{start.replace('-', '')}_{uuid.uuid4().hex[:10]}")


def _run_directory(requested: Path | None, name: str) -> Path:
    directory = (requested or DEFAULT_RUN_ROOT / name).resolve()
    if directory.is_file():
        raise ValueError(f"Output path is a file: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _ingest(bundle: Path, ground_store: Path) -> tuple[dict[str, Any], bool]:
    from prithvi_ground.catalog import SceneCatalog

    return SceneCatalog(ground_store).ingest(bundle)


def _final_record(
    *,
    mode: str,
    output: Path,
    bundle: Path,
    pipeline: dict[str, Any],
    scene: dict[str, Any],
    ground_created: bool,
    timing: dict[str, float],
) -> dict[str, Any]:
    record = {
        "schema_version": "1.0",
        "mode": mode,
        "status": "MVP_READY",
        "output": str(output),
        "bundle": str(bundle),
        "pipeline": pipeline,
        "timing_seconds": timing,
        "ground": {
            "created": ground_created,
            "scene": scene,
            "dashboard": "http://127.0.0.1:8000/",
        },
    }
    _write_json(output / "mvp_result.json", record)
    return record


def run_sentinel(args: argparse.Namespace) -> int:
    command_started = time.perf_counter()
    from prithvi_payload.runtime import PayloadRuntime
    from prithvi_shared import PayloadAcquisitionCommand

    job_id = _new_job_id(args.region_id, args.start)
    output = _run_directory(args.output, job_id)
    command = PayloadAcquisitionCommand.model_validate(
        {
            "schema_version": "1.0",
            "job_id": job_id,
            "region_id": args.region_id,
            "source": {
                "provider": "earth_engine",
                "bbox_wgs84": args.bbox,
                "start_date": args.start,
                "end_date": args.end,
                "selection_policy": args.selection_policy,
                "target_cloud_min_percent": args.target_cloud_min,
                "target_cloud_max_percent": args.target_cloud_max,
                "target_cloud_ideal_percent": args.target_cloud_ideal,
            },
        }
    )

    def progress(state: str, fields: dict[str, Any]) -> None:
        visible = {
            "state": state,
            "candidate": fields.get("current_candidate_number"),
            "scene": fields.get("safe_candidate_scene_id"),
            "metadata_cloud_percent": fields.get("safe_metadata_cloud_percentage"),
            "payload_cloud_percent": fields.get("safe_payload_measured_cloud_percentage"),
        }
        print("progress:", json.dumps(visible, sort_keys=True), flush=True)

    runtime = PayloadRuntime()
    runtime.initialize()
    pipeline_started = time.perf_counter()
    result = runtime.process(command, output, progress)
    pipeline_seconds = time.perf_counter() - pipeline_started
    bundle = output / result["artifact_directory"]
    ingest_started = time.perf_counter()
    scene, created = _ingest(bundle, args.ground_store.resolve())
    ground_ingest_seconds = time.perf_counter() - ingest_started
    raw_timing = result.get("timing", {})
    timing = {
        "startup_total_seconds": _nested_seconds(
            raw_timing, "runtime_initialization_seconds"
        )
        or 0.0,
        "pipeline_total_seconds": pipeline_seconds,
        "ground_ingest_seconds": ground_ingest_seconds,
    }
    sentinel_timing_keys = (
        "earth_engine_initialization_seconds",
        "cloud_model_load_seconds",
        "crop_model_load_seconds",
        "model_warmup_seconds",
        "candidate_search_seconds",
        "earth_engine_acquisition_seconds",
        "earth_engine_download_seconds",
        "geotiff_validation_seconds",
        "intake_seconds",
        "cloud_plan_seconds",
        "cloud_stage_seconds",
        "cloud_analysis_grid_seconds",
        "cloud_inference_seconds",
        "mask_processing_seconds",
        "cloud_mask_reprojection_seconds",
        "crop_plan_seconds",
        "crop_stage_seconds",
        "crop_inference_seconds",
        "condition_calculation_seconds",
        "condition_first_pass_seconds",
        "condition_robust_statistics_seconds",
        "condition_spatial_pass_seconds",
        "condition_preview_seconds",
        "downlink_packaging_seconds",
    )
    for key in sentinel_timing_keys:
        value = _nested_seconds(raw_timing, key)
        if value is not None:
            timing[key] = value
    timing["cloud_mask_processing_seconds"] = timing.pop("mask_processing_seconds", 0.0)
    timing["condition_stage_seconds"] = timing.pop("condition_calculation_seconds", 0.0)
    end_to_end_seconds = time.perf_counter() - command_started
    timing["command_overhead_seconds"] = max(
        0.0,
        end_to_end_seconds
        - timing["startup_total_seconds"]
        - pipeline_seconds
        - ground_ingest_seconds,
    )
    timing["end_to_end_seconds"] = end_to_end_seconds
    record = _final_record(
        mode="sentinel-2",
        output=output,
        bundle=bundle,
        pipeline=result,
        scene=scene,
        ground_created=created,
        timing=timing,
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    print()
    print(_render_timing_report(timing))
    return 0


def run_balkan(args: argparse.Namespace) -> int:
    command_started = time.perf_counter()
    source = args.input.resolve()
    if not source.is_file():
        raise ValueError(f"Input GeoTIFF does not exist: {source}")
    calibration = (
        args.crop_calibration.resolve()
        if args.crop_calibration
        else source.with_name(f"{source.stem}.crop_calibration.json")
    )
    if not calibration.is_file():
        raise ValueError(
            "A source-bound *.crop_calibration.json sidecar is required beside the "
            "preprocessed Balkan GeoTIFF (or pass --crop-calibration)."
        )
    acquired_at = args.acquired_at
    if not acquired_at:
        calibration_record = json.loads(calibration.read_text(encoding="utf-8"))
        acquired_at = calibration_record.get("acquired_at")
    if not isinstance(acquired_at, str) or not acquired_at:
        raise ValueError(
            "The calibration has no acquisition time; pass --acquired-at explicitly."
        )
    scene_id = _safe_id(args.scene_id or re.sub(r"(?i)_L1ORT$", "", source.stem))
    output = _run_directory(args.output, f"balkan1_{scene_id}")
    if any(output.iterdir()) and not args.overwrite:
        raise ValueError(f"Run directory is not empty; pass --overwrite: {output}")

    import torch
    from prithvi_payload.cloud_classifier import load_cloud_model
    from prithvi_payload.inference import PayloadCropModel
    from prithvi_payload.pipeline import run_scene

    startup_started = time.perf_counter()
    cloud_model_started = time.perf_counter()
    cloud = load_cloud_model()
    cloud_model_load_seconds = time.perf_counter() - cloud_model_started
    crop_model_started = time.perf_counter()
    crop_model = PayloadCropModel.load(device="cuda" if torch.cuda.is_available() else "cpu")
    crop_model_load_seconds = time.perf_counter() - crop_model_started
    startup_seconds = time.perf_counter() - startup_started

    def progress(state: str) -> None:
        print(f"progress: {state}", flush=True)

    pipeline_started = time.perf_counter()
    result = run_scene(
        source,
        sensor="balkan-1",
        output_root=output,
        acquired_at=acquired_at,
        scene_id=scene_id,
        band_order=BALKAN_BAND_ORDER,
        crop_calibration_path=calibration,
        reflectance_scale=args.reflectance_scale,
        stop_after="downlink",
        region_id=args.region_id,
        condition_tile_size=args.condition_tile_size,
        overwrite=args.overwrite,
        cloud_backend=cloud.backend,
        cloud_config=cloud.config,
        crop_model=crop_model,
        progress_callback=progress,
    )
    pipeline_seconds = time.perf_counter() - pipeline_started
    if result.get("status") != "DOWNLINK_READY":
        raise RuntimeError(f"Balkan pipeline stopped with status {result.get('status')}")
    bundle = output / "downlink"
    ingest_started = time.perf_counter()
    scene, created = _ingest(bundle, args.ground_store.resolve())
    ground_ingest_seconds = time.perf_counter() - ingest_started
    end_to_end_seconds = time.perf_counter() - command_started
    timing = {
        "command_overhead_seconds": max(
            0.0,
            end_to_end_seconds - startup_seconds - pipeline_seconds - ground_ingest_seconds,
        ),
        "startup_total_seconds": startup_seconds,
        "cloud_model_load_seconds": cloud_model_load_seconds,
        "crop_model_load_seconds": crop_model_load_seconds,
        "pipeline_total_seconds": pipeline_seconds,
        **_balkan_pipeline_timings(result),
        "ground_ingest_seconds": ground_ingest_seconds,
        "end_to_end_seconds": end_to_end_seconds,
    }
    record = _final_record(
        mode="balkan-1",
        output=output,
        bundle=bundle,
        pipeline={
            "status": result["status"],
            "scene_id": result["scene_id"],
            "summary": result["summary"],
            "result": str(output / "result.json"),
        },
        scene=scene,
        ground_created=created,
        timing=timing,
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    print()
    print(_render_timing_report(timing))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="vita-mvp",
        description="Run one of the two supported ViTA MVP pipelines locally.",
    )
    commands = root.add_subparsers(dest="mvp", required=True)

    sentinel = commands.add_parser("sentinel", help="Acquire Sentinel-2 with Earth Engine")
    sentinel.add_argument("--bbox", type=_bbox, required=True)
    sentinel.add_argument("--start", required=True, help="YYYY-MM-DD")
    sentinel.add_argument("--end", required=True, help="YYYY-MM-DD")
    sentinel.add_argument("--region-id", required=True)
    sentinel.add_argument(
        "--selection-policy",
        choices=("target_cloud_range", "least_cloudy"),
        default="target_cloud_range",
    )
    sentinel.add_argument("--target-cloud-min", type=float, default=15.0)
    sentinel.add_argument("--target-cloud-max", type=float, default=35.0)
    sentinel.add_argument("--target-cloud-ideal", type=float, default=25.0)
    sentinel.add_argument("--output", type=Path)
    sentinel.add_argument("--ground-store", type=Path, default=DEFAULT_GROUND_STORE)
    sentinel.set_defaults(handler=run_sentinel)

    balkan = commands.add_parser("balkan", help="Run one preprocessed Balkan-1 GeoTIFF")
    balkan.add_argument("input", type=Path)
    balkan.add_argument(
        "--acquired-at",
        help="ISO-8601 time; defaults to the calibration sidecar value",
    )
    balkan.add_argument("--region-id", required=True)
    balkan.add_argument("--scene-id")
    balkan.add_argument("--crop-calibration", type=Path)
    balkan.add_argument("--reflectance-scale", type=float)
    balkan.add_argument("--condition-tile-size", type=int, default=512)
    balkan.add_argument("--output", type=Path)
    balkan.add_argument("--ground-store", type=Path, default=DEFAULT_GROUND_STORE)
    balkan.add_argument("--overwrite", action="store_true")
    balkan.set_defaults(handler=run_balkan)
    return root


def main() -> None:
    args = parser().parse_args()
    try:
        raise SystemExit(args.handler(args))
    except (OSError, RuntimeError, ValueError) as error:
        print(f"MVP failed: {error}")
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()

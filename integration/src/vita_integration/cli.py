"""The two supported local MVP commands: Sentinel-2 and preprocessed Balkan-1."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import time
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GROUND_STORE = REPOSITORY_ROOT / "runtime" / "ground"
DEFAULT_RUN_ROOT = REPOSITORY_ROOT / "runtime" / "runs"
BALKAN_BAND_ORDER = ("BLUE", "GREEN", "RED", "NIR", "PAN")
SENTINEL_BAND_ORDER = ("B02", "B03", "B04", "B08", "B8A")

_IGNORED_RASTERIO_MESSAGES = (
    "Sum of Photometric type-related color channels and ExtraSamples doesn't match",
    "Creating TIFF with legacy Deflate codec identifier",
)

TIMING_ROWS = (
    ("command_overhead_seconds", "Command setup/import overhead"),
    ("startup_total_seconds", "Startup total"),
    ("cloud_model_load_seconds", "  Cloud model loading"),
    ("crop_model_load_seconds", "  Crop model loading"),
    ("pipeline_total_seconds", "Pipeline total"),
    ("intake_seconds", "  Scene intake"),
    ("shared_analysis_grid_seconds", "  Balkan shared 10 m preparation"),
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


class _ExpectedProviderTiffFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(fragment in message for fragment in _IGNORED_RASTERIO_MESSAGES)


def _configure_runtime_logging() -> None:
    """Hide only known harmless provider/optional-tool warnings."""
    rasterio_logger = logging.getLogger("rasterio._env")
    if not any(isinstance(item, _ExpectedProviderTiffFilter) for item in rasterio_logger.filters):
        rasterio_logger.addFilter(_ExpectedProviderTiffFilter())
    logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
    warnings.filterwarnings(
        "ignore",
        message=r"Significant no-data areas detected\..*",
        category=UserWarning,
        module=r"omnicloudmask\.cloud_mask",
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


def _local_pipeline_timings(result: dict[str, Any]) -> dict[str, float]:
    stages = result.get("stage_metadata", {})
    pipeline = result.get("timing", {})
    cloud = stages.get("cloud", {}).get("runtime", {})
    crop = stages.get("crop", {}).get("runtime", {})
    condition = stages.get("condition", {}).get("runtime", {})
    downlink = stages.get("downlink", {}).get("runtime", {})
    shared_analysis_seconds = (
        _nested_seconds(pipeline, "shared_analysis_grid_seconds")
        if result.get("sensor") == "balkan-1"
        else None
    )
    values = {
        "intake_seconds": _nested_seconds(pipeline, "intake_seconds"),
        "shared_analysis_grid_seconds": shared_analysis_seconds,
        "cloud_plan_seconds": _nested_seconds(pipeline, "cloud_plan_seconds"),
        "cloud_stage_seconds": _nested_seconds(cloud, "seconds"),
        "cloud_analysis_grid_seconds": _nested_seconds(cloud, "analysis_grid_preparation_seconds"),
        "cloud_inference_seconds": _nested_seconds(cloud, "inference_seconds"),
        "cloud_mask_processing_seconds": _nested_seconds(cloud, "mask_processing_seconds"),
        "cloud_mask_reprojection_seconds": _nested_seconds(cloud, "mask_reprojection_seconds"),
        "crop_plan_seconds": _nested_seconds(pipeline, "crop_plan_seconds"),
        "crop_stage_seconds": _nested_seconds(crop, "seconds"),
        "crop_inference_seconds": _nested_seconds(crop, "inference_seconds"),
        "condition_stage_seconds": _nested_seconds(condition, "seconds"),
        "condition_first_pass_seconds": _nested_seconds(condition, "first_pass_seconds"),
        "condition_robust_statistics_seconds": _nested_seconds(
            condition, "robust_statistics_seconds"
        ),
        "condition_spatial_pass_seconds": _nested_seconds(condition, "spatial_pass_seconds"),
        "condition_preview_seconds": _nested_seconds(condition, "preview_seconds"),
        "downlink_packaging_seconds": _nested_seconds(downlink, "seconds"),
    }
    return {key: value for key, value in values.items() if value is not None}


def _safe_id(value: str) -> str:
    identifier = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    if not identifier:
        raise ValueError("Could not derive a safe scene identifier")
    return identifier[:80]


def _execution_scene_id(source: Path, requested: str | None) -> str:
    if requested:
        return _safe_id(requested)
    execution_time = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    suffix = f"{execution_time}_{uuid.uuid4().hex[:8]}"
    source_id = _safe_id(source.stem)
    return f"{source_id[: 79 - len(suffix)]}_{suffix}"


def _resolve_sentinel_source(input_path: Path, image: str | None) -> Path:
    source = input_path.resolve()
    if source.is_file():
        if image:
            raise ValueError("--image can only be used when the Sentinel input is a folder")
        return source
    if not source.is_dir():
        raise ValueError(f"Sentinel input does not exist: {source}")
    if image:
        selected = (source / image).resolve()
        if selected.parent != source:
            raise ValueError("--image must name a file directly inside the Sentinel folder")
        if not selected.is_file() or selected.suffix.casefold() not in {".tif", ".tiff"}:
            raise ValueError(f"Sentinel GeoTIFF does not exist: {selected}")
        return selected
    candidates = sorted(
        path
        for path in source.iterdir()
        if path.is_file() and path.suffix.casefold() in {".tif", ".tiff"}
    )
    if len(candidates) == 1:
        return candidates[0].resolve()
    names = ", ".join(path.name for path in candidates) or "none"
    raise ValueError(
        f"Sentinel folder contains {len(candidates)} GeoTIFFs; pass --image with one of: {names}"
    )


def _normalise_acquired_at(value: str | None, source: Path) -> str:
    if not value:
        match = re.search(r"(20\d{6}T\d{6})", source.stem)
        if match:
            parsed = datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
            return parsed.isoformat()
        raise ValueError(
            "Sentinel acquisition time is missing; embed ACQUIRED_AT in the TIFF or pass "
            "--acquired-at"
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("Sentinel acquisition time must be valid ISO-8601") from error
    if parsed.tzinfo is None:
        raise ValueError("Sentinel acquisition time must include a UTC offset")
    return parsed.isoformat()


def _sentinel_contract(
    source: Path,
    *,
    acquired_at: str | None,
    reflectance_scale: float | None,
) -> tuple[str, float]:
    import rasterio

    with rasterio.open(source) as dataset:
        if dataset.count != len(SENTINEL_BAND_ORDER):
            raise ValueError("Sentinel GeoTIFF must contain exactly five bands")
        if tuple(dataset.descriptions) != SENTINEL_BAND_ORDER:
            raise ValueError(f"Sentinel band descriptions must be {', '.join(SENTINEL_BAND_ORDER)}")
        tags = dataset.tags()
    if tags.get("SENSOR") not in {None, "sentinel-2"}:
        raise ValueError("Sentinel GeoTIFF has an incompatible SENSOR tag")
    resolved_time = _normalise_acquired_at(acquired_at or tags.get("ACQUIRED_AT"), source)
    raw_scale: Any = reflectance_scale
    if raw_scale is None:
        raw_scale = tags.get("REFLECTANCE_SCALE")
    try:
        resolved_scale = float(raw_scale)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Sentinel reflectance scale is missing; embed REFLECTANCE_SCALE or pass "
            "--reflectance-scale"
        ) from error
    if not math.isfinite(resolved_scale) or resolved_scale <= 0:
        raise ValueError("Sentinel reflectance scale must be positive and finite")
    return resolved_time, resolved_scale


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


def _execute_local_mvp(
    args: argparse.Namespace,
    *,
    command_started: float,
    source: Path,
    sensor: str,
    mode: str,
    run_prefix: str,
    acquired_at: str,
    scene_id: str,
    reflectance_scale: float | None,
    band_order: tuple[str, ...] | None = None,
    crop_calibration_path: Path | None = None,
) -> int:
    output = _run_directory(args.output, f"{run_prefix}_{scene_id}")
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
        sensor=sensor,
        output_root=output,
        acquired_at=acquired_at,
        scene_id=scene_id,
        band_order=band_order,
        crop_calibration_path=crop_calibration_path,
        reflectance_scale=reflectance_scale,
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
        raise RuntimeError(f"{mode} pipeline stopped with status {result.get('status')}")
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
        **_local_pipeline_timings(result),
        "ground_ingest_seconds": ground_ingest_seconds,
        "end_to_end_seconds": end_to_end_seconds,
    }
    record = _final_record(
        mode=mode,
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


def run_sentinel(args: argparse.Namespace) -> int:
    command_started = time.perf_counter()
    source = _resolve_sentinel_source(args.input, args.image)
    acquired_at, reflectance_scale = _sentinel_contract(
        source,
        acquired_at=args.acquired_at,
        reflectance_scale=args.reflectance_scale,
    )
    scene_id = _execution_scene_id(source, args.scene_id)
    return _execute_local_mvp(
        args,
        command_started=command_started,
        source=source,
        sensor="sentinel-2",
        mode="sentinel-2",
        run_prefix="sentinel2",
        acquired_at=acquired_at,
        scene_id=scene_id,
        reflectance_scale=reflectance_scale,
    )


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
        raise ValueError("The calibration has no acquisition time; pass --acquired-at explicitly.")
    source_id = re.sub(r"(?i)_L1ORT$", "", source.stem)
    scene_id = _execution_scene_id(source.with_stem(source_id), args.scene_id)
    return _execute_local_mvp(
        args,
        command_started=command_started,
        source=source,
        sensor="balkan-1",
        mode="balkan-1",
        run_prefix="balkan1",
        acquired_at=acquired_at,
        scene_id=scene_id,
        reflectance_scale=args.reflectance_scale,
        band_order=BALKAN_BAND_ORDER,
        crop_calibration_path=calibration,
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="vita-mvp",
        description="Run one of the two supported ViTA MVP pipelines locally.",
    )
    commands = root.add_subparsers(dest="mvp", required=True)

    sentinel = commands.add_parser("sentinel", help="Run one local Sentinel-2 GeoTIFF")
    sentinel.add_argument("input", type=Path, help="Sentinel GeoTIFF or folder")
    sentinel.add_argument(
        "--image",
        help="GeoTIFF filename when input is a folder containing multiple scenes",
    )
    sentinel.add_argument(
        "--acquired-at",
        help="ISO-8601 time; defaults to the TIFF ACQUIRED_AT tag or filename timestamp",
    )
    sentinel.add_argument("--region-id", required=True)
    sentinel.add_argument("--scene-id")
    sentinel.add_argument("--reflectance-scale", type=float)
    sentinel.add_argument("--condition-tile-size", type=int, default=512)
    sentinel.add_argument("--output", type=Path)
    sentinel.add_argument("--ground-store", type=Path, default=DEFAULT_GROUND_STORE)
    sentinel.add_argument("--overwrite", action="store_true")
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
    _configure_runtime_logging()
    args = parser().parse_args()
    try:
        raise SystemExit(args.handler(args))
    except (OSError, RuntimeError, ValueError) as error:
        print(f"MVP failed: {error}")
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()

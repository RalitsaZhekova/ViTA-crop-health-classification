"""Isolated raw Balkan-1 preprocessing and accelerated payload workload.

This entry point deliberately does not extend the operational HTTP service.  It
creates the normal downlink bundle in a separate runtime after CUDA alignment,
raw-to-model reconstruction, and fail-closed TensorRT cloud/crop inference.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import torch
from prithvi_shared.files import sha256_file

from prithvi_payload.balkan_alignment import AlignmentConfig, align_balkan_geotiff
from prithvi_payload.balkan_raw_proxy import build_raw_model_proxy
from prithvi_payload.cloud_classifier import load_cloud_model
from prithvi_payload.inference import PayloadCropModel
from prithvi_payload.pipeline import run_scene
from prithvi_payload.tensorrt_runtime import load_accepted_manifest

SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
BALKAN_RAW_BAND_ORDER = ("BLUE", "GREEN", "RED", "NIR", "PAN")
DOWNLINK_FILES = ("scene.json", "scene.webp", "condition.png")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _require_file(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} does not exist: {resolved}")
    return resolved


def _require_safe_id(value: str, *, label: str) -> str:
    if not SAFE_ID.fullmatch(value):
        raise ValueError(
            f"{label} must contain only letters, digits, dot, underscore or dash"
        )
    return value


def _synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _load_warm_accelerated_models() -> tuple[Any, PayloadCropModel, dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("The isolated raw payload requires a CUDA device")
    if os.environ.get("VITA_CLOUD_BACKEND", "").strip().casefold() != "tensorrt":
        raise RuntimeError("VITA_CLOUD_BACKEND=tensorrt is required for the raw payload")
    if os.environ.get("VITA_CROP_BACKEND", "").strip().casefold() != "tensorrt":
        raise RuntimeError("VITA_CROP_BACKEND=tensorrt is required for the raw payload")

    manifest = load_accepted_manifest(required_models=("cloud", "crop"))
    cloud = load_cloud_model()
    crop = PayloadCropModel.load(device="cuda")
    cloud_backend = getattr(cloud.backend, "execution_backend", None)
    if cloud_backend != "tensorrt" or cloud.backend.tensorrt_engine_count < 1:
        raise RuntimeError("Raw payload cloud inference did not load accepted TensorRT plans")
    if crop.backend != "tensorrt" or crop.tensorrt_engine_count < 1:
        raise RuntimeError("Raw payload crop inference did not load an accepted TensorRT plan")
    fixed_cloud_patch_size = getattr(cloud.backend, "raw_fixed_patch_size", None)
    if fixed_cloud_patch_size != cloud.backend.patch_size:
        raise RuntimeError(
            "The raw payload requires the accepted fixed-size cloud TensorRT profile"
        )
    fixed_profiles = [
        profile
        for profile in cloud.backend.tensorrt_profiles
        if int(profile["patch_size"]) == fixed_cloud_patch_size
    ]
    if (
        len(fixed_profiles) != 1
        or int(fixed_profiles[0]["minimum_batch_size"]) != 1
        or cloud.backend.batch_size != 1
    ):
        raise RuntimeError(
            "The accepted fixed-size raw cloud profile requires logical batch size one"
        )

    accepted_patch_sizes = tuple(
        sorted(
            {
                int(profile["patch_size"])
                for profile in cloud.backend.tensorrt_profiles
            }
        )
    )
    if not accepted_patch_sizes:
        raise RuntimeError("Accepted cloud TensorRT plans have no executable profiles")
    cloud_warmups = cloud.backend.warmup(accepted_patch_sizes)
    crop.predict(
        torch.zeros((1, 4, 1, 224, 224), dtype=torch.float32),
        temporal_coords=torch.tensor([[[2026.0, 1.0]]]),
        location_coords=torch.zeros((1, 2)),
    )
    _synchronize_cuda()
    freeze_profiles = getattr(cloud.backend, "freeze_tensorrt_profiles", None)
    if callable(freeze_profiles):
        freeze_profiles()
    gc.collect()
    torch.cuda.empty_cache()
    return cloud, crop, {
        "accepted_manifest_sha256": manifest["manifest_sha256"],
        "cloud_backend": cloud_backend,
        "cloud_engine_count": cloud.backend.tensorrt_engine_count,
        "cloud_profiles": cloud.backend.tensorrt_profiles,
        "cloud_warmups": cloud_warmups,
        "cloud_raw_fixed_patch_size": fixed_cloud_patch_size,
        "crop_backend": crop.backend,
        "crop_device": crop.device.type,
        "crop_engine_count": crop.tensorrt_engine_count,
        "crop_precision": crop.tensorrt_precision,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }


def _assert_cuda_alignment(report: dict[str, Any]) -> None:
    execution = report.get("execution", {})
    if (
        execution.get("resolved_device") != "cuda"
        or execution.get("cuda_batched_phase_correlation") is not True
        or execution.get("cuda_fused_four_band_warp") is not True
    ):
        raise RuntimeError("Raw alignment did not satisfy the required CUDA execution contract")


def run_raw_payload_job(
    *,
    raw_path: str | Path,
    metadata_path: str | Path,
    radiometric_diagnostics_path: str | Path,
    position_path: str | Path,
    attitude_path: str | Path,
    parent_calibration_path: str | Path,
    output_root: str | Path,
    job_id: str,
    region_id: str,
    band_start_row_scale: float = 0.19,
    band_start_axis: str = "column",
    warp_tile_size: int = 2048,
    compression: str = "zstd",
    condition_tile_size: int = 4096,
    preloaded_cloud: Any | None = None,
    preloaded_crop: PayloadCropModel | None = None,
    preloaded_acceleration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one isolated raw-derived downlink bundle without touching service state."""

    _require_safe_id(job_id, label="job_id")
    _require_safe_id(region_id, label="region_id")
    if band_start_axis not in {"row", "column"}:
        raise ValueError("band_start_axis must be row or column")
    if condition_tile_size < 1:
        raise ValueError("condition_tile_size must be positive")
    preloaded_values = (preloaded_cloud, preloaded_crop, preloaded_acceleration)
    models_preloaded = all(value is not None for value in preloaded_values)
    if any(value is not None for value in preloaded_values) and not models_preloaded:
        raise ValueError(
            "preloaded_cloud, preloaded_crop and preloaded_acceleration must be supplied together"
        )
    inputs = {
        "raw": _require_file(raw_path, label="Raw TIFF"),
        "metadata": _require_file(metadata_path, label="L0 metadata"),
        "radiometric_diagnostics": _require_file(
            radiometric_diagnostics_path,
            label="Radiometric diagnostics",
        ),
        "position": _require_file(position_path, label="Position telemetry"),
        "attitude": _require_file(attitude_path, label="Attitude telemetry"),
        "parent_calibration": _require_file(
            parent_calibration_path,
            label="Parent crop calibration",
        ),
    }
    output_root_path = Path(output_root).resolve()
    output_root_path.mkdir(parents=True, exist_ok=True)
    job_root = output_root_path / job_id
    try:
        job_root.mkdir()
    except FileExistsError as error:
        raise FileExistsError(
            f"Raw payload job already exists; use a new job ID: {job_root}"
        ) from error
    preprocess_root = job_root / "preprocess"
    preprocess_root.mkdir()
    status_path = job_root / "raw-job.json"
    aligned_path = preprocess_root / f"{inputs['raw'].stem}_aligned.tif"
    proxy_path = preprocess_root / f"{inputs['raw'].stem}_model_proxy.tif"
    started = time.perf_counter()
    timings: dict[str, float] = {}

    try:
        _write_json(
            status_path,
            {
                "schema_version": "1.0",
                "status": "ALIGNING_RAW_BANDS",
                "job_id": job_id,
                "region_id": region_id,
            },
        )
        _synchronize_cuda()
        stage_started = time.perf_counter()
        alignment = align_balkan_geotiff(
            inputs["raw"],
            aligned_path,
            explicit_band_order=BALKAN_RAW_BAND_ORDER,
            metadata_path=inputs["metadata"],
            band_start_row_scale=band_start_row_scale,
            band_start_axis=band_start_axis,
            config=AlignmentConfig(
                measurement_tile_size=512,
                global_search_radius_px=96,
                minimum_confidence=0.25,
                device="cuda",
                warp_tile_size=warp_tile_size,
                compression=compression,
                build_overviews=False,
            ),
            require_georeferencing=False,
        )
        _synchronize_cuda()
        timings["cuda_alignment_seconds"] = time.perf_counter() - stage_started
        _assert_cuda_alignment(alignment)

        _write_json(
            status_path,
            {
                "schema_version": "1.0",
                "status": "RECONSTRUCTING_MODEL_GRID",
                "job_id": job_id,
                "region_id": region_id,
                "timing_seconds": dict(timings),
            },
        )
        stage_started = time.perf_counter()
        proxy = build_raw_model_proxy(
            aligned_path,
            proxy_path,
            alignment_report_path=aligned_path.with_suffix(".alignment.json"),
            radiometric_diagnostics_path=inputs["radiometric_diagnostics"],
            position_path=inputs["position"],
            attitude_path=inputs["attitude"],
            parent_calibration_path=inputs["parent_calibration"],
        )
        timings["raw_model_reconstruction_seconds"] = time.perf_counter() - stage_started
        proxy_calibration = proxy_path.with_name(f"{proxy_path.stem}.crop_calibration.json")
        calibration = json.loads(proxy_calibration.read_text(encoding="utf-8"))
        acquired_at = calibration.get("acquired_at")
        if not isinstance(acquired_at, str) or not acquired_at:
            raise RuntimeError("Generated raw proxy calibration has no acquisition time")

        _write_json(
            status_path,
            {
                "schema_version": "1.0",
                "status": "LOADING_ACCELERATED_MODELS",
                "job_id": job_id,
                "region_id": region_id,
                "timing_seconds": dict(timings),
            },
        )
        if models_preloaded:
            cloud = preloaded_cloud
            crop = preloaded_crop
            acceleration = dict(preloaded_acceleration or {})
            timings["model_load_and_warmup_seconds"] = 0.0
        else:
            stage_started = time.perf_counter()
            cloud, crop, acceleration = _load_warm_accelerated_models()
            timings["model_load_and_warmup_seconds"] = time.perf_counter() - stage_started
        acceleration["models_preloaded"] = models_preloaded

        _write_json(
            status_path,
            {
                "schema_version": "1.0",
                "status": "RUNNING_ACCELERATED_PIPELINE",
                "job_id": job_id,
                "region_id": region_id,
                "acceleration": acceleration,
                "timing_seconds": dict(timings),
            },
        )
        stage_started = time.perf_counter()
        pipeline = run_scene(
            proxy_path,
            sensor="balkan-1",
            output_root=job_root,
            acquired_at=acquired_at,
            scene_id=job_id,
            band_order=BALKAN_RAW_BAND_ORDER,
            crop_calibration_path=proxy_calibration,
            reflectance_scale=1.0,
            stop_after="downlink",
            region_id=region_id,
            condition_tile_size=condition_tile_size,
            cloud_backend=cloud.backend,
            cloud_config=cloud.config,
            crop_model=crop,
            allow_experimental_raw_proxy=True,
            acquisition_metadata={
                "raw_payload_job": job_id,
                "raw_source_sha256": alignment["source"]["sha256"],
                "alignment_report": str(aligned_path.with_suffix(".alignment.json")),
                "raw_proxy_report": str(proxy_path.with_suffix(".raw_proxy.json")),
            },
        )
        _synchronize_cuda()
        timings["accelerated_pipeline_seconds"] = time.perf_counter() - stage_started
        if pipeline.get("status") != "DOWNLINK_READY":
            raise RuntimeError(f"Raw-derived pipeline stopped with status {pipeline.get('status')}")

        bundle = job_root / "downlink"
        files: dict[str, dict[str, Any]] = {}
        for name in DOWNLINK_FILES:
            path = bundle / name
            files[name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        timings["end_to_end_seconds"] = time.perf_counter() - started
        response = {
            "schema_version": "1.0",
            "status": "DOWNLINK_READY",
            "qualification": "UNQUALIFIED_ENGINEERING_EXPERIMENT",
            "job_id": job_id,
            "scene_id": pipeline["scene_id"],
            "sensor": "balkan-1-raw",
            "region_id": region_id,
            "bundle_relative": f"runs/{job_id}/downlink",
            "files": files,
            "timing_seconds": timings,
            "acceleration": {
                **acceleration,
                "alignment_device": alignment["execution"]["resolved_device"],
                "alignment_cuda_batched_phase_correlation": alignment["execution"][
                    "cuda_batched_phase_correlation"
                ],
                "alignment_cuda_fused_four_band_warp": alignment["execution"][
                    "cuda_fused_four_band_warp"
                ],
                "raw_model_reconstruction": proxy["execution"][
                    "reconstruction_backend"
                ],
                "raw_model_reconstruction_band_workers": proxy["execution"][
                    "band_workers"
                ],
                "raw_model_reconstruction_cpu_thread_budget": proxy["execution"][
                    "cpu_thread_budget"
                ],
            },
            "artifacts": {
                "alignment": str(aligned_path),
                "alignment_report": str(aligned_path.with_suffix(".alignment.json")),
                "raw_proxy": str(proxy_path),
                "raw_proxy_report": str(proxy_path.with_suffix(".raw_proxy.json")),
                "pipeline_result": str(job_root / "result.json"),
            },
            "summary": pipeline.get("summary", {}),
        }
        _write_json(status_path, response)
        return response
    except Exception as error:
        timings["end_to_end_seconds"] = time.perf_counter() - started
        _write_json(
            status_path,
            {
                "schema_version": "1.0",
                "status": "FAILED",
                "job_id": job_id,
                "region_id": region_id,
                "error_type": type(error).__name__,
                "detail": str(error),
                "timing_seconds": timings,
            },
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vita-raw-payload",
        description="Run isolated CUDA raw alignment and TensorRT payload downlink",
    )
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--radiometric-diagnostics", type=Path, required=True)
    parser.add_argument("--position", type=Path, required=True)
    parser.add_argument("--attitude", type=Path, required=True)
    parser.add_argument("--parent-calibration", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/runtime/runs"))
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--region-id", required=True)
    parser.add_argument("--band-start-row-scale", type=float, default=0.19)
    parser.add_argument("--band-start-axis", choices=("row", "column"), default="column")
    parser.add_argument("--warp-tile-size", type=int, default=2048)
    parser.add_argument("--compression", choices=("zstd", "deflate", "none"), default="zstd")
    parser.add_argument("--condition-tile-size", type=int, default=4096)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        result = run_raw_payload_job(
            raw_path=args.raw,
            metadata_path=args.metadata,
            radiometric_diagnostics_path=args.radiometric_diagnostics,
            position_path=args.position,
            attitude_path=args.attitude,
            parent_calibration_path=args.parent_calibration,
            output_root=args.output_root,
            job_id=args.job_id,
            region_id=args.region_id,
            band_start_row_scale=args.band_start_row_scale,
            band_start_axis=args.band_start_axis,
            warp_tile_size=args.warp_tile_size,
            compression=args.compression,
            condition_tile_size=args.condition_tile_size,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Raw payload failed: {error}", flush=True)
        raise SystemExit(2) from None
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

"""Loopback-only, warm-model payload service for SSH-tunnel orchestration."""

from __future__ import annotations

import argparse
import gc
import importlib
import importlib.metadata
import json
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import anyio
import torch
from fastapi import FastAPI, HTTPException
from prithvi_shared.files import sha256_file
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from prithvi_payload.cloud_classifier import load_cloud_model
from prithvi_payload.inference import PayloadCropModel
from prithvi_payload.nvtx import range as nvtx_range
from prithvi_payload.pipeline import run_scene
from prithvi_payload.runtime_config import environment_flag

SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
BALKAN_BAND_ORDER = ("BLUE", "GREEN", "RED", "NIR", "PAN")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _pipeline_timings(
    result: dict[str, Any],
    *,
    payload_seconds: float | None = None,
) -> dict[str, float]:
    """Flatten the persisted stage timings for the ground response."""
    stages = result.get("stage_metadata", {})
    pipeline = result.get("timing", {})
    mappings = {
        "intake_seconds": (pipeline, "intake_seconds"),
        "shared_analysis_grid_seconds": (pipeline, "shared_analysis_grid_seconds"),
        "cloud_plan_seconds": (pipeline, "cloud_plan_seconds"),
        "cloud_stage_seconds": (stages.get("cloud", {}).get("runtime", {}), "seconds"),
        "cloud_inference_seconds": (
            stages.get("cloud", {}).get("runtime", {}),
            "inference_seconds",
        ),
        "cloud_input_preparation_seconds": (
            stages.get("cloud", {}).get("runtime", {}),
            "input_preparation_seconds",
        ),
        "cloud_mask_processing_seconds": (
            stages.get("cloud", {}).get("runtime", {}),
            "mask_processing_seconds",
        ),
        "crop_plan_seconds": (pipeline, "crop_plan_seconds"),
        "crop_stage_seconds": (stages.get("crop", {}).get("runtime", {}), "seconds"),
        "crop_inference_seconds": (
            stages.get("crop", {}).get("runtime", {}),
            "inference_seconds",
        ),
        "crop_product_preparation_seconds": (
            stages.get("crop", {}).get("runtime", {}),
            "product_preparation_seconds",
        ),
        "crop_overlapped_product_preparation_seconds": (
            stages.get("crop", {}).get("runtime", {}),
            "overlapped_product_preparation_seconds",
        ),
        "crop_tile_preparation_seconds": (
            stages.get("crop", {}).get("runtime", {}),
            "tile_preparation_seconds",
        ),
        "crop_batch_assembly_seconds": (
            stages.get("crop", {}).get("runtime", {}),
            "batch_assembly_seconds",
        ),
        "crop_stitching_seconds": (
            stages.get("crop", {}).get("runtime", {}),
            "stitching_seconds",
        ),
        "crop_finalization_seconds": (
            stages.get("crop", {}).get("runtime", {}),
            "finalization_seconds",
        ),
        "condition_stage_seconds": (
            stages.get("condition", {}).get("runtime", {}),
            "seconds",
        ),
        "condition_input_read_seconds": (
            stages.get("condition", {}).get("runtime", {}),
            "input_read_seconds",
        ),
        "condition_health_calculation_seconds": (
            stages.get("condition", {}).get("runtime", {}),
            "health_calculation_seconds",
        ),
        "condition_score_calculation_seconds": (
            stages.get("condition", {}).get("runtime", {}),
            "score_calculation_seconds",
        ),
        "condition_metric_aggregation_seconds": (
            stages.get("condition", {}).get("runtime", {}),
            "metric_aggregation_seconds",
        ),
        "condition_metric_summary_seconds": (
            stages.get("condition", {}).get("runtime", {}),
            "metric_summary_seconds",
        ),
        "condition_product_materialization_seconds": (
            stages.get("condition", {}).get("runtime", {}),
            "product_materialization_seconds",
        ),
        "condition_robust_statistics_seconds": (
            stages.get("condition", {}).get("runtime", {}),
            "robust_statistics_seconds",
        ),
        "condition_spatial_pass_seconds": (
            stages.get("condition", {}).get("runtime", {}),
            "spatial_pass_seconds",
        ),
        "downlink_packaging_seconds": (
            stages.get("downlink", {}).get("runtime", {}),
            "seconds",
        ),
        "downlink_image_preparation_seconds": (
            stages.get("downlink", {}).get("runtime", {}),
            "image_preparation_seconds",
        ),
        "downlink_interaction_grid_seconds": (
            stages.get("downlink", {}).get("runtime", {}),
            "interaction_grid_seconds",
        ),
        "downlink_manifest_seconds": (
            stages.get("downlink", {}).get("runtime", {}),
            "manifest_seconds",
        ),
    }
    timings: dict[str, float] = {}
    for name, (record, key) in mappings.items():
        value = record.get(key) if isinstance(record, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            timings[name] = max(0.0, float(value))
    top_level = (
        "intake_seconds",
        "shared_analysis_grid_seconds",
        "cloud_plan_seconds",
        "cloud_stage_seconds",
        "crop_plan_seconds",
        "crop_stage_seconds",
        "condition_stage_seconds",
        "downlink_packaging_seconds",
    )
    reported_total = sum(timings.get(name, 0.0) for name in top_level)
    timings["reported_stage_total_seconds"] = reported_total
    if payload_seconds is not None:
        timings["orchestration_seconds"] = max(0.0, payload_seconds - reported_total)
    return timings


def _preload_pipeline_modules() -> float:
    """Move lazy full-pipeline imports out of the first measured request."""
    started = time.perf_counter()
    for name in (
        "prithvi_payload.balkan_analysis",
        "prithvi_payload.crop_executor",
        "prithvi_payload.condition_stage",
        "prithvi_payload.downlink",
    ):
        importlib.import_module(name)
    return time.perf_counter() - started


def _safe_relative(root: Path, value: str, *, name: str) -> Path:
    relative = PurePosixPath(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"{name} must be a relative path below the payload data root")
    candidate = root.joinpath(*relative.parts).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} resolves outside the payload data root") from error
    return candidate


def _balkan_prepare_inputs() -> tuple[str, ...]:
    """Return fixed Balkan inputs, accepting the old single-input variable."""
    raw = os.environ.get("VITA_BALKAN_PREPARE_INPUTS")
    if raw is None:
        raw = os.environ.get("VITA_BALKAN_PREPARE_INPUT", "")
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if len(values) != len(set(values)):
        raise RuntimeError("Configured Balkan prepare inputs must be distinct")
    return values


def _sentinel_demo_inputs() -> tuple[str, ...]:
    raw = os.environ.get("VITA_DEMO_SENTINEL_IMAGES", "")
    values = tuple(value.strip() for value in raw.split(",") if value.strip())
    if len(values) != len(set(values)):
        raise RuntimeError("Configured Sentinel demo inputs must be distinct")
    return values


def _configure_tensorrt_cudagraphs() -> bool:
    """Reject the retired Torch-TensorRT-specific CUDA graph switch."""
    if environment_flag("VITA_TRT_CUDAGRAPHS", False):
        raise RuntimeError(
            "VITA_TRT_CUDAGRAPHS is not supported by the direct TensorRT runtime"
        )
    return False


class JobRequest(BaseModel):
    """Only metadata crosses the uplink; image paths are payload-local."""

    model_config = ConfigDict(extra="forbid")

    sensor: Literal["sentinel-2", "balkan-1"]
    input: str = Field(min_length=1, max_length=512)
    image: str | None = Field(default=None, min_length=1, max_length=255)
    region_id: str = Field(min_length=1, max_length=80)
    job_id: str = Field(min_length=1, max_length=80)
    acquired_at: str | None = Field(default=None, max_length=64)
    reflectance_scale: float | None = Field(default=None, gt=0)
    crop_calibration: str | None = Field(default=None, min_length=1, max_length=512)


class PayloadRuntime:
    def __init__(self) -> None:
        if environment_flag("CUDA_REQUIRED", True) and not torch.cuda.is_available():
            raise RuntimeError("CUDA_REQUIRED=1 but torch.cuda.is_available() is false")
        self.input_root = Path(os.environ.get("VITA_INPUT_ROOT", "/data")).resolve()
        self.output_root = Path(os.environ.get("VITA_OUTPUT_ROOT", "/runtime/runs")).resolve()
        if not self.input_root.is_dir():
            raise RuntimeError(f"Payload data root does not exist: {self.input_root}")
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.job_lock = threading.Lock()

        started = time.perf_counter()
        pipeline_import_seconds = _preload_pipeline_modules()
        self._scene_cloud_warmups: list[dict[str, Any]] = []
        self._crop_parity_profiles: list[dict[str, Any]] = []
        balkan_cache_started = time.perf_counter()
        self.balkan_analysis_caches = self._prepare_balkan_analysis_caches()
        self._prepare_sentinel_cloud_warmups()
        self.balkan_analysis_cache = (
            self.balkan_analysis_caches[0] if self.balkan_analysis_caches else None
        )
        balkan_cache_seconds = time.perf_counter() - balkan_cache_started
        cloud_started = time.perf_counter()
        self.cloud = load_cloud_model()
        cloud_seconds = time.perf_counter() - cloud_started
        crop_started = time.perf_counter()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.crop = PayloadCropModel.load(device=device)
        crop_seconds = time.perf_counter() - crop_started
        self.tensorrt_cudagraphs = _configure_tensorrt_cudagraphs()
        self.cloud_warmup_profiles: list[dict[str, int]] = []
        self.cloud_scene_warmup_profiles: list[dict[str, Any]] = []
        self.cloud_scene_warmup_profile: dict[str, Any] | None = None
        warmup_seconds = self._warmup() if environment_flag("VITA_WARMUP", True) else 0.0
        self.startup_timing = {
            "cloud_model_load_seconds": cloud_seconds,
            "crop_model_load_seconds": crop_seconds,
            "pipeline_import_seconds": pipeline_import_seconds,
            "balkan_analysis_cache_seconds": balkan_cache_seconds,
            "warmup_seconds": warmup_seconds,
            "total_seconds": time.perf_counter() - started,
        }
        self.stack = self._stack_record()

    def _prepare_balkan_analysis_caches(self) -> list[dict[str, Any]]:
        return [
            self._prepare_balkan_analysis_cache(relative_input, index=index)
            for index, relative_input in enumerate(_balkan_prepare_inputs(), start=1)
        ]

    def _prepare_balkan_analysis_cache(
        self,
        relative_input: str,
        *,
        index: int,
    ) -> dict[str, Any]:
        source = _safe_relative(self.input_root, relative_input, name="Balkan prepare input")
        calibration = source.with_name(f"{source.stem}.crop_calibration.json")
        if not source.is_file() or not calibration.is_file():
            raise RuntimeError(
                "VITA_BALKAN_PREPARE_INPUT requires a source and adjacent calibration"
            )
        calibration_record = json.loads(calibration.read_text(encoding="utf-8"))
        acquired_at = calibration_record.get("acquired_at")
        if not isinstance(acquired_at, str) or not acquired_at:
            raise RuntimeError("Balkan prepare calibration has no acquisition time")
        from prithvi_payload.balkan_analysis import materialize_balkan_analysis_grid
        from prithvi_payload.scene_intake import inspect_scene

        intake = inspect_scene(
            source,
            sensor="balkan-1",
            acquired_at=acquired_at,
            scene_id=f"startup-balkan-cache-{index}",
            band_order=BALKAN_BAND_ORDER,
            crop_calibration_path=calibration,
        )
        if intake.get("readiness", {}).get("intake") != "READY":
            raise RuntimeError("Configured Balkan prepare input failed intake validation")
        analysis = materialize_balkan_analysis_grid(
            intake,
            output_root=self.output_root.parent / ".startup-balkan-cache" / source.stem,
        )
        self._scene_cloud_warmups.append(
            {
                "sensor": "balkan-1",
                "source_path": analysis["source_path"],
                "source_band_indices": analysis["model_band_routes"]["cloud_detection"][
                    "source_band_indices"
                ],
                "reflectance_scale": float(
                    intake["radiometry"]["cloud_reflectance_divisor"]
                ),
                "nodata_value": analysis["raster"].get("nodata"),
                "input": relative_input.replace("\\", "/"),
            }
        )
        crop_adapter = intake["model_band_routes"]["crop_classification"][
            "spectral_adapter"
        ]
        self._crop_parity_profiles.append(
            {
                "sensor": "balkan-1",
                "input": relative_input.replace("\\", "/"),
                "source_path": analysis["source_path"],
                "source_band_indices": analysis["model_band_routes"]["crop_classification"][
                    "source_band_indices"
                ],
                "training_scale_multiplier": crop_adapter["source_scale_to_model_units"],
                "calibration_path": crop_adapter["calibration_path"],
                "calibration_source_path": intake["source_path"],
                "acquired_at": intake["acquired_at"],
            }
        )
        return {
            "input": relative_input.replace("\\", "/"),
            "cache_hit": bool(analysis["runtime"].get("cache_hit")),
            "cache_key": analysis["runtime"].get("cache_key"),
            "seconds": float(analysis["runtime"]["seconds"]),
            "width": int(analysis["raster"]["width"]),
            "height": int(analysis["raster"]["height"]),
            "preprocessing_mode": analysis.get("preprocessing", {}).get("mode"),
            "overview_factor": analysis.get("preprocessing", {}).get("overview_factor"),
        }

    def _prepare_sentinel_cloud_warmups(self) -> None:
        import rasterio

        from prithvi_payload.scene_intake import inspect_scene

        for index, relative_input in enumerate(_sentinel_demo_inputs(), start=1):
            source = _safe_relative(
                self.input_root,
                relative_input,
                name="Sentinel demo input",
            )
            if not source.is_file():
                raise RuntimeError(f"Configured Sentinel demo input is missing: {source}")
            with rasterio.open(source) as dataset:
                acquired_at = dataset.tags().get("ACQUIRED_AT")
            if not acquired_at:
                raise RuntimeError(
                    f"Configured Sentinel demo input has no ACQUIRED_AT tag: {source}"
                )
            intake = inspect_scene(
                source,
                sensor="sentinel-2",
                acquired_at=acquired_at,
                scene_id=f"startup-sentinel-warmup-{index}",
            )
            if intake.get("readiness", {}).get("intake") != "READY":
                raise RuntimeError("Configured Sentinel demo input failed intake validation")
            route = intake["model_band_routes"]["cloud_detection"]
            self._scene_cloud_warmups.append(
                {
                    "sensor": "sentinel-2",
                    "source_path": intake["source_path"],
                    "source_band_indices": route["source_band_indices"],
                    "reflectance_scale": float(
                        intake["radiometry"]["cloud_reflectance_divisor"]
                    ),
                    "nodata_value": intake["raster"].get("nodata"),
                    "input": relative_input.replace("\\", "/"),
                }
            )
            crop_route = intake["model_band_routes"]["crop_classification"]
            self._crop_parity_profiles.append(
                {
                    "sensor": "sentinel-2",
                    "input": relative_input.replace("\\", "/"),
                    "source_path": intake["source_path"],
                    "source_band_indices": crop_route["source_band_indices"],
                    "training_scale_multiplier": 10000.0
                    / float(intake["radiometry"]["cloud_reflectance_divisor"]),
                    "acquired_at": intake["acquired_at"],
                }
            )

    def _warm_scene_cloud_profile(self, profile: dict[str, Any]) -> dict[str, Any]:
        """Exercise one fixed scene path without retaining its prediction."""

        import numpy as np
        from cloud_detection.preprocessing import (
            normalize_reflectance,
            strict_valid_mask,
        )

        from prithvi_payload.cloud_profiles import read_fixed_scene_cloud_input

        raw_image = read_fixed_scene_cloud_input(profile)
        image, invalid = normalize_reflectance(
            raw_image,
            scale=profile["reflectance_scale"],
            clip_min=self.cloud.config["input"].get("clip_min"),
            clip_max=self.cloud.config["input"].get("clip_max"),
            nodata_value=profile["nodata_value"],
        )
        if bool(self.cloud.config["input"].get("strict_positive_rgn", True)):
            invalid |= ~strict_valid_mask(image[[1, 2, 0]])
            image[:, invalid] = 0.0
        self.cloud.backend.predict_semantic(image.astype(np.float32, copy=False))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        result = {
            "kind": "fixed_input_profile",
            "sensor": profile["sensor"],
            "input": profile["input"],
            "height": int(image.shape[1]),
            "width": int(image.shape[2]),
            "prediction_retained": False,
        }
        del raw_image, image, invalid
        return result

    def _warmup(self) -> float:
        started = time.perf_counter()
        if getattr(self.cloud.backend, "execution_backend", "pytorch") == "tensorrt":
            accepted_sizes = {
                int(profile["patch_size"])
                for profile in self.cloud.backend.tensorrt_profiles
            }
            if not accepted_sizes:
                raise RuntimeError("Accepted cloud TensorRT plans have no warmup shapes")
            patch_sizes = tuple(sorted(accepted_sizes))
        else:
            raw_patch_sizes = os.environ.get(
                "VITA_CLOUD_WARMUP_PATCH_SIZES",
                "869,891,1000",
            )
            try:
                patch_sizes = tuple(
                    int(value.strip())
                    for value in raw_patch_sizes.split(",")
                    if value.strip()
                )
            except ValueError as error:
                raise RuntimeError(
                    "VITA_CLOUD_WARMUP_PATCH_SIZES must be comma-separated integers"
                ) from error
        warmup = getattr(self.cloud.backend, "warmup", None)
        if not callable(warmup):
            raise RuntimeError("The configured cloud backend does not support warmup")
        self.cloud_warmup_profiles = warmup(patch_sizes)
        crop_input = torch.zeros((1, 4, 1, 224, 224), dtype=torch.float32)
        self.crop.predict(
            crop_input,
            temporal_coords=torch.tensor([[[2026.0, 1.0]]]),
            location_coords=torch.zeros((1, 2)),
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        del crop_input
        gc.collect()
        # The large crop model can displace CUDA convolution/workspace state used
        # by the cloud ensemble. Warm the stage that runs first in a job last.
        self.cloud_scene_warmup_profiles = [
            self._warm_scene_cloud_profile(profile)
            for profile in self._scene_cloud_warmups
        ]
        self.cloud_scene_warmup_profile = (
            self.cloud_scene_warmup_profiles[-1]
            if self.cloud_scene_warmup_profiles
            else None
        )
        freeze_profiles = getattr(self.cloud.backend, "freeze_tensorrt_profiles", None)
        if callable(freeze_profiles):
            freeze_profiles()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return time.perf_counter() - started

    def _stack_record(self) -> dict[str, Any]:
        try:
            import tensorrt

            tensorrt_version = tensorrt.__version__
        except ImportError:
            tensorrt_version = None
        return {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "tensorrt": tensorrt_version,
            "modelopt": _package_version("nvidia-modelopt"),
            "tensorrt_runtime": "native-python",
            "crop_backend": self.crop.backend,
            "crop_device": self.crop.device.type,
            "crop_inference_dtype": (
                "fp32" if self.crop.backend == "pytorch" else self.crop.tensorrt_precision
            ),
            "crop_tf32": bool(
                torch.backends.cuda.matmul.allow_tf32
                or torch.backends.cudnn.allow_tf32
            ),
            "crop_tensorrt_engine_count": self.crop.tensorrt_engine_count,
            "crop_tensorrt_parity": self.crop.tensorrt_parity,
            "crop_parity_scene_inputs": [
                profile["input"] for profile in self._crop_parity_profiles
            ],
            "crop_tensorrt_precision": self.crop.tensorrt_precision,
            "crop_tensorrt_tf32": self.crop.tensorrt_tf32,
            "crop_batch_size": self.crop.fixed_batch_size,
            "tensorrt_cudagraphs": self.tensorrt_cudagraphs,
            "cloud_backend": (
                "omnicloudmask_tensorrt_" + str(self.cloud.backend.inference_dtype)
                if getattr(self.cloud.backend, "execution_backend", "pytorch") == "tensorrt"
                else "omnicloudmask_cuda_" + str(self.cloud.backend.inference_dtype)
            ),
            "cloud_batch_size": self.cloud.backend.batch_size,
            "cloud_tensorrt_engine_count": int(
                getattr(self.cloud.backend, "tensorrt_engine_count", 0)
            ),
            "cloud_tensorrt_profiles": list(
                getattr(self.cloud.backend, "tensorrt_profiles", [])
            ),
            "cloud_tensorrt_parity": dict(
                getattr(self.cloud.backend, "tensorrt_parity", {})
            ),
            "tensorrt_manifest_sha256": getattr(
                self.cloud.backend,
                "tensorrt_manifest_sha256",
                None,
            ),
            "cloud_warmup_profiles": self.cloud_warmup_profiles,
            "cloud_scene_warmup_profile": self.cloud_scene_warmup_profile,
            "cloud_scene_warmup_profiles": self.cloud_scene_warmup_profiles,
            "balkan_analysis_cache": self.balkan_analysis_cache,
            "balkan_analysis_caches": self.balkan_analysis_caches,
        }

    def health(self) -> dict[str, Any]:
        cuda_probe: dict[str, Any] = {"status": "not_available"}
        status = "ready"
        if self.stack["cuda_available"]:
            try:
                # torch.cuda.is_available() only describes initialization-time
                # capability. A laptop sleep, driver reset, or GPU switch can
                # invalidate a long-lived CUDA context while it still returns
                # True, so readiness must execute and synchronize real work.
                probe = torch.ones(1, device="cuda")
                torch.cuda.synchronize()
                cuda_probe = {
                    "status": "ready",
                    "device": probe.device.type,
                }
            except RuntimeError as error:
                status = "error"
                cuda_probe = {
                    "status": "error",
                    "detail": str(error),
                }
        return {
            "status": status,
            "cuda_probe": cuda_probe,
            "stack": self.stack,
            "startup_timing_seconds": self.startup_timing,
        }

    def run(self, request: JobRequest) -> dict[str, Any]:
        if not SAFE_ID.fullmatch(request.job_id):
            raise ValueError("job_id must contain only letters, digits, dot, underscore or dash")
        if not SAFE_ID.fullmatch(request.region_id):
            raise ValueError("region_id must contain only letters, digits, dot, underscore or dash")
        source = _safe_relative(self.input_root, request.input, name="input")
        if request.sensor == "sentinel-2":
            from vita_integration.cli import (
                SENTINEL_BAND_ORDER,
                _resolve_sentinel_source,
                _sentinel_contract,
            )

            source = _resolve_sentinel_source(source, request.image)
            acquired_at, reflectance_scale = _sentinel_contract(
                source,
                acquired_at=request.acquired_at,
                reflectance_scale=request.reflectance_scale,
            )
            band_order = SENTINEL_BAND_ORDER
            calibration = None
        else:
            if request.image is not None:
                raise ValueError("image is only valid for a Sentinel folder input")
            if not source.is_file():
                raise ValueError(f"Balkan input GeoTIFF does not exist: {source}")
            calibration = (
                _safe_relative(
                    self.input_root,
                    request.crop_calibration,
                    name="crop_calibration",
                )
                if request.crop_calibration
                else source.with_name(f"{source.stem}.crop_calibration.json")
            )
            if not calibration.is_file():
                raise ValueError(f"Balkan crop calibration does not exist: {calibration}")
            calibration_record = json.loads(calibration.read_text(encoding="utf-8"))
            acquired_at = request.acquired_at or calibration_record.get("acquired_at")
            if not isinstance(acquired_at, str) or not acquired_at:
                raise ValueError("Balkan acquisition time is missing")
            reflectance_scale = request.reflectance_scale
            band_order = BALKAN_BAND_ORDER

        output = self.output_root / request.job_id
        if output.exists():
            raise FileExistsError(f"Payload job already exists: {request.job_id}")
        progress: list[str] = []
        started = time.perf_counter()
        with nvtx_range(f"vita.job.{request.sensor}.{request.job_id}"):
            result = run_scene(
                source,
                sensor=request.sensor,
                output_root=output,
                acquired_at=acquired_at,
                scene_id=request.job_id,
                band_order=band_order,
                crop_calibration_path=calibration,
                reflectance_scale=reflectance_scale,
                stop_after="downlink",
                region_id=request.region_id,
                cloud_backend=self.cloud.backend,
                cloud_config=self.cloud.config,
                crop_model=self.crop,
                condition_tile_size=int(
                    os.environ.get("VITA_CONDITION_TILE_SIZE", "4096")
                ),
                progress_callback=progress.append,
            )
        payload_seconds = time.perf_counter() - started
        if result.get("status") != "DOWNLINK_READY":
            raise RuntimeError(f"Pipeline stopped with status {result.get('status')}")
        bundle = output / "downlink"
        files = {}
        for name in ("scene.json", "scene.webp", "condition.png"):
            path = bundle / name
            files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        return {
            "schema_version": "1.0",
            "status": "DOWNLINK_READY",
            "job_id": request.job_id,
            "scene_id": result["scene_id"],
            "sensor": request.sensor,
            "bundle_relative": f"runs/{request.job_id}/downlink",
            "files": files,
            "payload_seconds": payload_seconds,
            "under_two_seconds": payload_seconds < 2.0,
            "under_five_seconds": payload_seconds < 5.0,
            "pipeline_timing_seconds": _pipeline_timings(
                result,
                payload_seconds=payload_seconds,
            ),
            "summary": result.get("summary", {}),
            "progress": progress,
            "stack": self.stack,
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    # CUDA/cuDNN setup includes thread-local state. Construct, warm and execute
    # models on one persistent worker so the first accepted job stays warm.
    limiter = anyio.to_thread.current_default_thread_limiter()
    previous_tokens = limiter.total_tokens
    limiter.total_tokens = 1
    app.state.payload = await run_in_threadpool(PayloadRuntime)
    try:
        yield
    finally:
        limiter.total_tokens = previous_tokens


app = FastAPI(
    title="ViTA payload service",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)


@app.get("/healthz")
async def health() -> dict[str, Any]:
    # Use the same single worker thread that constructs and executes the
    # models so the probe validates the CUDA context used by real jobs.
    return await run_in_threadpool(app.state.payload.health)


@app.post("/v1/jobs")
async def run_job(request: JobRequest) -> dict[str, Any]:
    runtime: PayloadRuntime = app.state.payload
    if not runtime.job_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="The single payload worker is busy")
    try:
        return await run_in_threadpool(runtime.run, request)
    except FileExistsError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except (OSError, RuntimeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        runtime.job_lock.release()


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the warm ViTA Jetson payload service.")
    parser.add_argument("--host", default=os.environ.get("VITA_PAYLOAD_HOST", "0.0.0.0"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("VITA_PAYLOAD_PORT", "8090")),
    )
    args = parser.parse_args()
    uvicorn.run(
        "prithvi_payload.service:app",
        host=args.host,
        port=args.port,
        workers=1,
        access_log=environment_flag("VITA_ACCESS_LOG", False),
    )


if __name__ == "__main__":
    main()

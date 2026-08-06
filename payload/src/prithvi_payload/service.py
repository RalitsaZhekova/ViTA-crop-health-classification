"""Loopback-only, warm-model payload service for SSH-tunnel orchestration."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from prithvi_payload.cloud_classifier import load_cloud_model
from prithvi_payload.inference import PayloadCropModel
from prithvi_payload.pipeline import run_scene

SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
BALKAN_BAND_ORDER = ("BLUE", "GREEN", "RED", "NIR", "PAN")


def _flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean value")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pipeline_timings(result: dict[str, Any]) -> dict[str, float]:
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
        "condition_stage_seconds": (
            stages.get("condition", {}).get("runtime", {}),
            "seconds",
        ),
        "downlink_packaging_seconds": (
            stages.get("downlink", {}).get("runtime", {}),
            "seconds",
        ),
    }
    timings: dict[str, float] = {}
    for name, (record, key) in mappings.items():
        value = record.get(key) if isinstance(record, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            timings[name] = max(0.0, float(value))
    return timings


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
        if _flag("CUDA_REQUIRED", True) and not torch.cuda.is_available():
            raise RuntimeError("CUDA_REQUIRED=1 but torch.cuda.is_available() is false")
        self.input_root = Path(os.environ.get("VITA_INPUT_ROOT", "/data")).resolve()
        self.output_root = Path(os.environ.get("VITA_OUTPUT_ROOT", "/runtime/runs")).resolve()
        if not self.input_root.is_dir():
            raise RuntimeError(f"Payload data root does not exist: {self.input_root}")
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.job_lock = threading.Lock()

        started = time.perf_counter()
        cloud_started = time.perf_counter()
        self.cloud = load_cloud_model()
        cloud_seconds = time.perf_counter() - cloud_started
        crop_started = time.perf_counter()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.crop = PayloadCropModel.load(device=device)
        crop_seconds = time.perf_counter() - crop_started
        self.cloud_warmup_profiles: list[dict[str, int]] = []
        warmup_seconds = self._warmup() if _flag("VITA_WARMUP", True) else 0.0
        self.startup_timing = {
            "cloud_model_load_seconds": cloud_seconds,
            "crop_model_load_seconds": crop_seconds,
            "warmup_seconds": warmup_seconds,
            "total_seconds": time.perf_counter() - started,
        }
        self.stack = self._stack_record()

    def _warmup(self) -> float:
        started = time.perf_counter()
        raw_patch_sizes = os.environ.get("VITA_CLOUD_WARMUP_PATCH_SIZES", "869")
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
            "torch_tensorrt": _package_version("torch-tensorrt"),
            "modelopt": _package_version("nvidia-modelopt"),
            "crop_backend": self.crop.backend,
            "crop_tensorrt_engine_count": self.crop.tensorrt_engine_count,
            "cloud_backend": "omnicloudmask_cuda_"
            + str(self.cloud.backend.inference_dtype),
            "cloud_batch_size": self.cloud.backend.batch_size,
            "cloud_warmup_profiles": self.cloud_warmup_profiles,
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": "ready",
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
            progress_callback=progress.append,
        )
        payload_seconds = time.perf_counter() - started
        if result.get("status") != "DOWNLINK_READY":
            raise RuntimeError(f"Pipeline stopped with status {result.get('status')}")
        bundle = output / "downlink"
        files = {}
        for name in ("scene.json", "scene.webp", "condition.png"):
            path = bundle / name
            files[name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        return {
            "schema_version": "1.0",
            "status": "DOWNLINK_READY",
            "job_id": request.job_id,
            "scene_id": result["scene_id"],
            "sensor": request.sensor,
            "bundle_relative": f"runs/{request.job_id}/downlink",
            "files": files,
            "payload_seconds": payload_seconds,
            "under_five_seconds": payload_seconds < 5.0,
            "pipeline_timing_seconds": _pipeline_timings(result),
            "summary": result.get("summary", {}),
            "progress": progress,
            "stack": self.stack,
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.payload = PayloadRuntime()
    yield


app = FastAPI(
    title="ViTA payload service",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)


@app.get("/healthz")
def health() -> dict[str, Any]:
    return app.state.payload.health()


@app.post("/v1/jobs")
def run_job(request: JobRequest) -> dict[str, Any]:
    runtime: PayloadRuntime = app.state.payload
    if not runtime.job_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="The single payload worker is busy")
    try:
        return runtime.run(request)
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
        access_log=_flag("VITA_ACCESS_LOG", False),
    )


if __name__ == "__main__":
    main()

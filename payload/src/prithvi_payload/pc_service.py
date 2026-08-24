"""PC-only warm CUDA service spanning every local ViTA workflow.

This module deliberately composes the existing operational and raw runtimes
without changing either Jetson entry point.  A single process owns the CUDA
models so laptop GPUs do not waste VRAM on duplicate warm services.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any, Literal

import anyio
import torch
from fastapi import FastAPI, HTTPException
from prithvi_shared.files import sha256_file
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.concurrency import run_in_threadpool

from prithvi_payload.acquisition import (
    AcquisitionError,
    EarthEngineAcquisitionProvider,
    earth_engine_configuration,
)
from prithvi_payload.raw_payload_workload import run_raw_payload_job
from prithvi_payload.raw_service import (
    RawJobRequest,
    RawPayloadRuntime,
    _warm_alignment_cuda,
)
from prithvi_payload.runtime_config import environment_flag
from prithvi_payload.service import SAFE_ID, JobRequest, PayloadRuntime, _pipeline_timings

PC_RAW_CACHE_SCHEMA = "vita-pc-raw-preprocess-v2"
PC_RAW_CACHE_HASH_INPUT_MAX_BYTES = 16 * 1024 * 1024
PC_RAW_CACHE_ASSETS = {
    "aligned": "aligned.tif",
    "alignment_report": "alignment.json",
    "proxy": "proxy.tif",
    "proxy_report": "proxy.json",
    "proxy_calibration": "proxy.crop_calibration.json",
}


def _raw_cache_identity(paths: dict[str, Path]) -> tuple[str, dict[str, Any]]:
    implementation_paths = (
        Path(__file__).with_name("balkan_alignment.py"),
        Path(__file__).with_name("balkan_raw_proxy.py"),
    )
    input_records = {}
    for name, path in sorted(paths.items()):
        stat = path.stat()
        input_record = {
            "path": str(path),
            "bytes": stat.st_size,
            "modified_ns": stat.st_mtime_ns,
        }
        if stat.st_size <= PC_RAW_CACHE_HASH_INPUT_MAX_BYTES:
            input_record["sha256"] = sha256_file(path)
        input_records[name] = input_record
    record = {
        "schema": PC_RAW_CACHE_SCHEMA,
        "inputs": input_records,
        "implementation": {
            path.name: sha256_file(path) for path in implementation_paths
        },
        "parameters": {
            "band_start_row_scale": 0.19,
            "band_start_axis": "column",
            "warp_tile_size": 2048,
            "alignment_compression": "zstd",
            "output_pixel_size_m": 10.0,
        },
    }
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), record


def _validated_raw_cache(
    cache_directory: Path,
    *,
    cache_key: str,
) -> dict[str, Path] | None:
    manifest_path = cache_directory / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if manifest.get("schema") != PC_RAW_CACHE_SCHEMA or manifest.get("cache_key") != cache_key:
        return None
    assets = manifest.get("assets")
    if not isinstance(assets, dict) or set(assets) != set(PC_RAW_CACHE_ASSETS):
        return None
    resolved: dict[str, Path] = {}
    for name, filename in PC_RAW_CACHE_ASSETS.items():
        path = cache_directory / filename
        asset = assets.get(name)
        if (
            not path.is_file()
            or not isinstance(asset, dict)
            or path.stat().st_size != asset.get("bytes")
            or path.stat().st_mtime_ns != asset.get("modified_ns")
        ):
            return None
        resolved[name] = path
    return resolved


def _publish_raw_cache(
    cache_directory: Path,
    *,
    cache_key: str,
    identity: dict[str, Any],
    response: dict[str, Any],
) -> None:
    if cache_directory.exists():
        return
    cache_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_directory.with_name(
        f".{cache_directory.name}.{os.getpid()}.{time.time_ns()}.partial"
    )
    temporary.mkdir()
    try:
        artifacts = response["artifacts"]
        proxy_path = Path(artifacts["raw_proxy"]).resolve()
        sources = {
            "aligned": Path(artifacts["alignment"]).resolve(),
            "alignment_report": Path(artifacts["alignment_report"]).resolve(),
            "proxy": proxy_path,
            "proxy_report": Path(artifacts["raw_proxy_report"]).resolve(),
            "proxy_calibration": proxy_path.with_name(
                f"{proxy_path.stem}.crop_calibration.json"
            ),
        }
        asset_records = {}
        for name, filename in PC_RAW_CACHE_ASSETS.items():
            source = sources[name]
            destination = temporary / filename
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
            asset_records[name] = {
                "bytes": destination.stat().st_size,
                "modified_ns": destination.stat().st_mtime_ns,
                "sha256": sha256_file(destination),
            }
        (temporary / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": PC_RAW_CACHE_SCHEMA,
                    "cache_key": cache_key,
                    "identity": identity,
                    "assets": asset_records,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, cache_directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


class PCJobRequest(BaseModel):
    """One local request using stored, raw, or bounded live imagery."""

    model_config = ConfigDict(extra="forbid")

    sensor: Literal["sentinel-2", "sentinel-2-live", "balkan-1", "balkan-1-raw"]
    input: str = Field(min_length=1, max_length=512)
    image: str | None = Field(default=None, min_length=1, max_length=255)
    region_id: str = Field(min_length=1, max_length=80)
    job_id: str = Field(min_length=1, max_length=80)
    acquired_at: str | None = Field(default=None, max_length=64)
    reflectance_scale: float | None = Field(default=None, gt=0)
    crop_calibration: str | None = Field(default=None, min_length=1, max_length=512)
    bbox_wgs84: tuple[float, float, float, float] | None = None
    start_date: date | None = None
    end_date: date | None = None

    @model_validator(mode="after")
    def validate_sensor_fields(self) -> PCJobRequest:
        live_values = (self.bbox_wgs84, self.start_date, self.end_date)
        if self.sensor == "sentinel-2-live":
            if any(value is None for value in live_values):
                raise ValueError("Live Sentinel analysis requires an area and date range")
            if self.input != "earth-engine":
                raise ValueError("Live Sentinel input must be earth-engine")
            if self.image is not None:
                raise ValueError("Live Sentinel analysis does not accept an image name")
        elif any(value is not None for value in live_values):
            raise ValueError("Area and date range are only valid for live Sentinel analysis")
        if self.sensor != "sentinel-2" and self.image is not None:
            raise ValueError("image is only valid for a stored Sentinel folder input")
        return self


class PCEarthEngineAcquisitionProvider(EarthEngineAcquisitionProvider):
    """Initialize local service-account credentials with Earth Engine scopes."""

    def initialize(self) -> None:
        configuration = earth_engine_configuration()
        if not configuration["configured"]:
            raise AcquisitionError(
                "EARTH_ENGINE_NOT_CONFIGURED",
                "Live Sentinel acquisition is not configured on the payload computer",
            )
        try:
            import ee
            import google.auth

            credentials, _ = google.auth.load_credentials_from_file(
                os.environ["VITA_EE_CREDENTIALS"]
            )
            if getattr(credentials, "requires_scopes", False):
                credentials = credentials.with_scopes(ee.oauth.SCOPES)
            ee.Initialize(credentials=credentials, project=configuration["project"])
        except Exception:
            raise AcquisitionError(
                "EARTH_ENGINE_AUTHENTICATION_FAILED",
                "Earth Engine authentication failed on the payload computer",
            ) from None


class PCPayloadRuntime(PayloadRuntime):
    """Reuse one PyTorch CUDA model stack across all four local workflows."""

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("The accelerated PC payload service requires CUDA")
        for name in ("VITA_CLOUD_BACKEND", "VITA_CROP_BACKEND"):
            backend = os.environ.get(name, "pytorch").strip().casefold()
            if backend != "pytorch":
                raise RuntimeError(f"{name}=pytorch is required by the PC-local service")

        started = time.perf_counter()
        super().__init__()
        self.processed_input_root = self.input_root
        self.raw_preprocess_cache_root = self.output_root.parent / "cache" / "raw-preprocess"
        self.raw_preprocess_cache_root.mkdir(parents=True, exist_ok=True)
        self.earth_engine_provider = PCEarthEngineAcquisitionProvider()
        alignment_warmup = _warm_alignment_cuda()
        self.acceleration = {
            **self.stack,
            "runtime_mode": "pc-pytorch-cuda",
            "raw_preprocess_cache": "content-addressed-hardlink-v2",
            "alignment_cuda_warmup": alignment_warmup,
            "models_preloaded": True,
        }
        self.startup_timing["alignment_cuda_warmup_seconds"] = alignment_warmup["seconds"]
        self.startup_timing["total_seconds"] = time.perf_counter() - started

    def _paths(self, scene_id: str):
        return RawPayloadRuntime._paths(self, scene_id)

    def health(self) -> dict[str, Any]:
        health = super().health()
        health.update(
            {
                "service": "vita-pc-payload",
                "acceleration": self.acceleration,
                "earth_engine": earth_engine_configuration(),
                "workflows": [
                    "sentinel-2",
                    "sentinel-2-live",
                    "balkan-1",
                    "balkan-1-raw",
                ],
            }
        )
        return health

    def _run_raw_cached(self, request: RawJobRequest) -> dict[str, Any]:
        paths = self._paths(request.input)
        cache_key, identity = _raw_cache_identity(paths)
        cache_directory = self.raw_preprocess_cache_root / cache_key
        prepared = _validated_raw_cache(cache_directory, cache_key=cache_key)
        response = run_raw_payload_job(
            **paths,
            output_root=self.output_root,
            job_id=request.job_id,
            region_id=request.region_id,
            condition_tile_size=int(os.environ.get("VITA_CONDITION_TILE_SIZE", "4096")),
            preloaded_cloud=self.cloud,
            preloaded_crop=self.crop,
            preloaded_acceleration={
                **self.acceleration,
                "raw_preprocess_cache_key": cache_key,
            },
            preprocessed_paths=prepared,
        )
        if prepared is None:
            _publish_raw_cache(
                cache_directory,
                cache_key=cache_key,
                identity=identity,
                response=response,
            )
        payload_seconds = float(response["timing_seconds"]["end_to_end_seconds"])
        pipeline_path = Path(response["artifacts"]["pipeline_result"])
        pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
        timings = _pipeline_timings(pipeline, payload_seconds=payload_seconds)
        timings.update(
            {
                "raw_cuda_alignment_seconds": float(
                    response["timing_seconds"]["cuda_alignment_seconds"]
                ),
                "raw_model_reconstruction_seconds": float(
                    response["timing_seconds"]["raw_model_reconstruction_seconds"]
                ),
                "accelerated_pipeline_seconds": float(
                    response["timing_seconds"]["accelerated_pipeline_seconds"]
                ),
                "model_load_and_warmup_seconds": 0.0,
            }
        )
        progress = [
            "reused_raw_preprocess"
            if response["acceleration"]["raw_preprocess_cache_hit"]
            else "aligned_raw_bands",
            "reconstructed_model_grid",
            "cloud_complete",
            "crop_complete",
            "condition_complete",
            "downlink_ready",
        ]
        return {
            **response,
            "payload_seconds": payload_seconds,
            "under_two_seconds": payload_seconds < 2.0,
            "under_five_seconds": payload_seconds < 5.0,
            "pipeline_timing_seconds": timings,
            "stack": response["acceleration"],
            "progress": progress,
        }

    def run(self, request: PCJobRequest) -> dict[str, Any]:
        if not SAFE_ID.fullmatch(request.job_id):
            raise ValueError("job_id must contain only letters, digits, dot, underscore or dash")
        if not SAFE_ID.fullmatch(request.region_id):
            raise ValueError("region_id must contain only letters, digits, dot, underscore or dash")

        if request.sensor in {"sentinel-2", "balkan-1"}:
            return super().run(
                JobRequest(
                    sensor=request.sensor,
                    input=request.input,
                    image=request.image,
                    region_id=request.region_id,
                    job_id=request.job_id,
                    acquired_at=request.acquired_at,
                    reflectance_scale=request.reflectance_scale,
                    crop_calibration=request.crop_calibration,
                )
            )

        raw_request = RawJobRequest(
            sensor=request.sensor,
            input=request.input,
            region_id=request.region_id,
            job_id=request.job_id,
            bbox_wgs84=request.bbox_wgs84,
            start_date=request.start_date,
            end_date=request.end_date,
        )
        if request.sensor == "sentinel-2-live":
            response = RawPayloadRuntime._run_live_sentinel(self, raw_request)
            timings = response.get("pipeline_timing_seconds", {})
            acquisition_seconds = timings.get("total_acquisition_seconds")
            orchestration_seconds = timings.get("orchestration_seconds")
            if isinstance(acquisition_seconds, (int, float)) and isinstance(
                orchestration_seconds, (int, float)
            ):
                timings["orchestration_seconds"] = max(
                    0.0,
                    float(orchestration_seconds) - float(acquisition_seconds),
                )
            return response
        return self._run_raw_cached(raw_request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    limiter = anyio.to_thread.current_default_thread_limiter()
    previous_tokens = limiter.total_tokens
    limiter.total_tokens = 1
    app.state.payload = await run_in_threadpool(PCPayloadRuntime)
    try:
        yield
    finally:
        limiter.total_tokens = previous_tokens


app = FastAPI(
    title="ViTA PC-local payload service",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)


@app.get("/healthz")
async def health() -> dict[str, Any]:
    try:
        return await run_in_threadpool(app.state.payload.health)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/v1/jobs")
async def run_job(request: PCJobRequest) -> dict[str, Any]:
    runtime: PCPayloadRuntime = app.state.payload
    if not runtime.job_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="The single PC payload worker is busy")
    try:
        return await run_in_threadpool(runtime.run, request)
    except FileExistsError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except AcquisitionError as error:
        raise HTTPException(status_code=422, detail=error.safe_record()) from error
    except (OSError, RuntimeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    finally:
        runtime.job_lock.release()


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the warm PC-local ViTA payload service.")
    parser.add_argument("--host", default=os.environ.get("VITA_PC_PAYLOAD_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("VITA_PC_PAYLOAD_PORT", "8090")),
    )
    args = parser.parse_args()
    uvicorn.run(
        "prithvi_payload.pc_service:app",
        host=args.host,
        port=args.port,
        workers=1,
        access_log=environment_flag("VITA_ACCESS_LOG", False),
    )


if __name__ == "__main__":
    main()

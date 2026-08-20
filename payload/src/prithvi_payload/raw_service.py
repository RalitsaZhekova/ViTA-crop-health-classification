"""Persistent warm-model service for isolated Balkan-1 raw payload jobs."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from datetime import date
from pathlib import Path
from typing import Any, Literal

import anyio
import numpy as np
import torch
import torch.nn.functional as functional
from fastapi import FastAPI, HTTPException
from prithvi_shared.files import sha256_file
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.concurrency import run_in_threadpool

from prithvi_payload.acquisition import (
    AcquisitionError,
    EarthEngineAcquisitionProvider,
    earth_engine_configuration,
)
from prithvi_payload.balkan_alignment import _batch_cross_correlate_shift_cuda
from prithvi_payload.pipeline import run_scene
from prithvi_payload.raw_payload_workload import (
    SAFE_ID,
    _load_warm_accelerated_models,
    run_raw_payload_job,
)
from prithvi_payload.runtime_config import environment_flag
from prithvi_payload.service import (
    BALKAN_BAND_ORDER,
    _pipeline_timings,
    _preload_pipeline_modules,
    _safe_relative,
)


class RawJobRequest(BaseModel):
    """Select one payload-local Balkan scene; no source pixels cross the uplink."""

    model_config = ConfigDict(extra="forbid")

    sensor: Literal["balkan-1", "balkan-1-raw", "sentinel-2-live"]
    input: str = Field(min_length=1, max_length=512)
    region_id: str = Field(min_length=1, max_length=80)
    job_id: str = Field(min_length=1, max_length=80)
    bbox_wgs84: tuple[float, float, float, float] | None = None
    start_date: date | None = None
    end_date: date | None = None

    @model_validator(mode="after")
    def validate_live_acquisition(self) -> RawJobRequest:
        live_values = (self.bbox_wgs84, self.start_date, self.end_date)
        if self.sensor == "sentinel-2-live":
            if any(value is None for value in live_values):
                raise ValueError("Live Sentinel analysis requires an area and date range")
            if self.input != "earth-engine":
                raise ValueError("Live Sentinel input must be earth-engine")
        elif any(value is not None for value in live_values):
            raise ValueError("Area and date range are only valid for live Sentinel analysis")
        return self


def _warm_alignment_cuda() -> dict[str, Any]:
    """Exercise the CUDA FFT and fused resampling primitives used by alignment."""

    row, column = np.indices((64, 64), dtype=np.float32)
    source = np.sin(row / 7.0) + np.cos(column / 9.0) + row * column * 1e-4
    target = np.roll(source, shift=(2, -3), axis=(0, 1))
    started = time.perf_counter()
    shifts = _batch_cross_correlate_shift_cuda(
        [source.astype(np.float32)],
        [target.astype(np.float32)],
        search_radius_px=8,
    )
    with torch.inference_mode():
        values = torch.ones((4, 1, 64, 64), dtype=torch.float32, device="cuda")
        coordinates = torch.linspace(-1.0, 1.0, 64, device="cuda")
        grid_y, grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=-1).expand(4, -1, -1, -1)
        functional.grid_sample(
            values,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        torch.cuda.synchronize()
    return {
        "cuda_batched_phase_correlation": True,
        "cuda_fused_four_band_warp": True,
        "seconds": time.perf_counter() - started,
        "probe_shift": [float(shifts[0][0]), float(shifts[0][1])],
    }


class RawPayloadRuntime:
    """Own one persistent CUDA context and one preloaded TensorRT model stack."""

    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("The warm raw payload service requires CUDA")
        self.input_root = Path(os.environ.get("VITA_INPUT_ROOT", "/data")).resolve()
        self.processed_input_root = Path(
            os.environ.get("VITA_PROCESSED_INPUT_ROOT", "/operational-data")
        ).resolve()
        self.output_root = Path(os.environ.get("VITA_OUTPUT_ROOT", "/runtime/runs")).resolve()
        if not self.input_root.is_dir():
            raise RuntimeError(f"Raw payload data root does not exist: {self.input_root}")
        if not self.processed_input_root.is_dir():
            raise RuntimeError(
                f"Processed payload data root does not exist: {self.processed_input_root}"
            )
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.job_lock = threading.Lock()

        started = time.perf_counter()
        import_seconds = _preload_pipeline_modules()
        alignment_warmup = _warm_alignment_cuda()
        model_started = time.perf_counter()
        self.cloud, self.crop, acceleration = _load_warm_accelerated_models()
        model_warmup_seconds = time.perf_counter() - model_started
        self.acceleration = {
            **acceleration,
            "alignment_cuda_warmup": alignment_warmup,
            "models_preloaded": True,
        }
        self.earth_engine_provider = EarthEngineAcquisitionProvider()
        self.startup_timing = {
            "pipeline_import_seconds": import_seconds,
            "alignment_cuda_warmup_seconds": alignment_warmup["seconds"],
            "model_load_and_warmup_seconds": model_warmup_seconds,
            "total_seconds": time.perf_counter() - started,
        }

    def health(self) -> dict[str, Any]:
        probe = torch.ones(1, device="cuda")
        torch.cuda.synchronize()
        return {
            "status": "ready",
            "service": "vita-balkan-payload",
            "cuda_probe": {"status": "ready", "device": probe.device.type},
            "acceleration": self.acceleration,
            "earth_engine": earth_engine_configuration(),
            "startup_timing_seconds": self.startup_timing,
        }

    def _paths(self, scene_id: str) -> dict[str, Path]:
        if not SAFE_ID.fullmatch(scene_id):
            raise ValueError(
                "Raw scene ID must contain only letters, digits, dot, underscore or dash"
            )
        paths = {
            "raw_path": self.input_root / "balkan1" / "raw" / scene_id / f"{scene_id}_Raw.tif",
            "metadata_path": (
                self.input_root / "balkan1" / "derived" / "l1a" / f"{scene_id}_L0R_manifest.json"
            ),
            "radiometric_diagnostics_path": (
                self.input_root
                / "balkan1"
                / "derived"
                / "l1a"
                / f"{scene_id}_L1A_reference_validation.json"
            ),
            "position_path": self.input_root / "balkan1" / "raw" / scene_id / "position.csv",
            "attitude_path": self.input_root / "balkan1" / "raw" / scene_id / "attitude.csv",
            "parent_calibration_path": (
                self.processed_input_root
                / "balkan1"
                / "preprocessed"
                / f"{scene_id}_L1ORT.crop_calibration.json"
            ),
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise ValueError(f"Raw scene {scene_id} is incomplete: missing {missing}")
        return paths

    @contextmanager
    def _processed_cloud_profile_selection(self, relative_input: str):
        """Use the reviewed dynamic profile and batch while a processed job runs."""

        backend = self.cloud.backend
        fixed_patch_size = getattr(backend, "raw_fixed_patch_size", None)
        if fixed_patch_size != backend.patch_size:
            raise RuntimeError("The raw cloud TensorRT profile is not active")
        raw_batch_size = int(backend.batch_size)
        if raw_batch_size != 1:
            raise RuntimeError("The raw cloud TensorRT batch-one contract is not active")
        normalized_input = relative_input.replace("\\", "/")
        scene_patch_sizes = self.acceleration.get("cloud_scene_patch_sizes", {})
        selected_patch_size = (
            scene_patch_sizes.get(normalized_input)
            if isinstance(scene_patch_sizes, dict)
            else None
        )
        processed_batch_size = raw_batch_size
        if isinstance(selected_patch_size, int):
            matching_profiles = [
                profile
                for profile in self.acceleration.get("cloud_profiles", [])
                if isinstance(profile, dict)
                and profile.get("patch_size") == selected_patch_size
            ]
            if len(matching_profiles) != 1:
                raise RuntimeError(
                    "The processed Balkan cloud profile is missing from the accepted manifest"
                )
            processed_batch_size = int(
                matching_profiles[0]["maximum_batch_size"]
            )
            if processed_batch_size < 1:
                raise RuntimeError(
                    "The processed Balkan cloud profile has an invalid batch contract"
                )
        backend.raw_fixed_patch_size = None
        backend.batch_size = processed_batch_size
        try:
            yield
        finally:
            backend.batch_size = raw_batch_size
            backend.raw_fixed_patch_size = fixed_patch_size

    def _run_processed(self, request: RawJobRequest) -> dict[str, Any]:
        source = _safe_relative(self.processed_input_root, request.input, name="input")
        if not source.is_file():
            raise ValueError(f"Balkan input GeoTIFF does not exist: {source}")
        calibration = source.with_name(f"{source.stem}.crop_calibration.json")
        if not calibration.is_file():
            raise ValueError(f"Balkan crop calibration does not exist: {calibration}")
        calibration_record = json.loads(calibration.read_text(encoding="utf-8"))
        acquired_at = calibration_record.get("acquired_at")
        if not isinstance(acquired_at, str) or not acquired_at:
            raise ValueError("Balkan acquisition time is missing")
        output = self.output_root / request.job_id
        if output.exists():
            raise FileExistsError(f"Balkan payload job already exists: {request.job_id}")
        progress: list[str] = []
        started = time.perf_counter()
        with self._processed_cloud_profile_selection(request.input):
            result = run_scene(
                source,
                sensor="balkan-1",
                output_root=output,
                acquired_at=acquired_at,
                scene_id=request.job_id,
                band_order=BALKAN_BAND_ORDER,
                crop_calibration_path=calibration,
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
        files = {
            name: {
                "bytes": (bundle / name).stat().st_size,
                "sha256": sha256_file(bundle / name),
            }
            for name in ("scene.json", "scene.webp", "condition.png")
        }
        return {
            "schema_version": "1.0",
            "status": "DOWNLINK_READY",
            "job_id": request.job_id,
            "scene_id": result["scene_id"],
            "sensor": "balkan-1",
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
            "stack": self.acceleration,
        }

    def _run_raw(self, request: RawJobRequest) -> dict[str, Any]:
        response = run_raw_payload_job(
            **self._paths(request.input),
            output_root=self.output_root,
            job_id=request.job_id,
            region_id=request.region_id,
            condition_tile_size=int(os.environ.get("VITA_CONDITION_TILE_SIZE", "4096")),
            preloaded_cloud=self.cloud,
            preloaded_crop=self.crop,
            preloaded_acceleration=self.acceleration,
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
        return {
            **response,
            "payload_seconds": payload_seconds,
            "under_two_seconds": payload_seconds < 2.0,
            "under_five_seconds": payload_seconds < 5.0,
            "pipeline_timing_seconds": timings,
            "stack": response["acceleration"],
            "progress": [
                "aligned_raw_bands",
                "reconstructed_model_grid",
                "cloud_complete",
                "crop_complete",
                "condition_complete",
                "downlink_ready",
            ],
        }

    def _run_live_sentinel(self, request: RawJobRequest) -> dict[str, Any]:
        if request.bbox_wgs84 is None or request.start_date is None or request.end_date is None:
            raise ValueError("Live Sentinel analysis requires an area and date range")
        output = self.output_root / request.job_id
        if output.exists():
            raise FileExistsError(f"Live Sentinel payload job already exists: {request.job_id}")
        started = time.perf_counter()
        acquired = self.earth_engine_provider.acquire(
            bbox_wgs84=request.bbox_wgs84,
            start_date=request.start_date,
            end_date=request.end_date,
            destination_root=output / "acquisition",
        )
        progress: list[str] = ["earth_engine_acquisition_complete"]
        pipeline_started = time.perf_counter()
        result = run_scene(
            acquired.local_tiff_path,
            sensor="sentinel-2",
            output_root=output,
            acquired_at=acquired.acquired_at,
            scene_id=request.job_id,
            band_order=("B02", "B03", "B04", "B08", "B8A"),
            reflectance_scale=10_000.0,
            stop_after="downlink",
            region_id=request.region_id,
            cloud_backend=self.cloud.backend,
            cloud_config=self.cloud.config,
            crop_model=self.crop,
            condition_tile_size=int(os.environ.get("VITA_CONDITION_TILE_SIZE", "4096")),
            progress_callback=progress.append,
            acquisition_metadata=acquired.safe_provenance(),
        )
        pipeline_seconds = time.perf_counter() - pipeline_started
        payload_seconds = time.perf_counter() - started
        if result.get("status") != "DOWNLINK_READY":
            raise RuntimeError(f"Pipeline stopped with status {result.get('status')}")
        bundle = output / "downlink"
        files = {
            name: {
                "bytes": (bundle / name).stat().st_size,
                "sha256": sha256_file(bundle / name),
            }
            for name in ("scene.json", "scene.webp", "condition.png")
        }
        timings = _pipeline_timings(result, payload_seconds=payload_seconds)
        timings.update(acquired.timing)
        timings["accelerated_pipeline_seconds"] = pipeline_seconds
        return {
            "schema_version": "1.0",
            "status": "DOWNLINK_READY",
            "job_id": request.job_id,
            "scene_id": result["scene_id"],
            "sensor": "sentinel-2",
            "region_id": request.region_id,
            "bundle_relative": f"runs/{request.job_id}/downlink",
            "files": files,
            "payload_seconds": payload_seconds,
            "under_two_seconds": payload_seconds < 2.0,
            "under_five_seconds": payload_seconds < 5.0,
            "pipeline_timing_seconds": timings,
            "summary": result.get("summary", {}),
            "progress": progress,
            "stack": {
                **self.acceleration,
                "earth_engine_acquisition": True,
                "earth_engine_collection": "COPERNICUS/S2_SR_HARMONIZED",
            },
        }

    def run(self, request: RawJobRequest) -> dict[str, Any]:
        if not SAFE_ID.fullmatch(request.job_id):
            raise ValueError("job_id must contain only letters, digits, dot, underscore or dash")
        if not SAFE_ID.fullmatch(request.region_id):
            raise ValueError("region_id must contain only letters, digits, dot, underscore or dash")
        if request.sensor == "sentinel-2-live":
            return self._run_live_sentinel(request)
        if request.sensor == "balkan-1-raw":
            return self._run_raw(request)
        return self._run_processed(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    limiter = anyio.to_thread.current_default_thread_limiter()
    previous_tokens = limiter.total_tokens
    limiter.total_tokens = 1
    app.state.payload = await run_in_threadpool(RawPayloadRuntime)
    try:
        yield
    finally:
        limiter.total_tokens = previous_tokens


app = FastAPI(
    title="ViTA warm raw payload service",
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
async def run_job(request: RawJobRequest) -> dict[str, Any]:
    runtime: RawPayloadRuntime = app.state.payload
    if not runtime.job_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="The single raw payload worker is busy")
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

    parser = argparse.ArgumentParser(description="Run the warm Balkan-1 payload service.")
    parser.add_argument("--host", default=os.environ.get("VITA_RAW_PAYLOAD_HOST", "0.0.0.0"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("VITA_RAW_PAYLOAD_PORT", "8091")),
    )
    args = parser.parse_args()
    uvicorn.run(
        "prithvi_payload.raw_service:app",
        host=args.host,
        port=args.port,
        workers=1,
        access_log=environment_flag("VITA_ACCESS_LOG", False),
    )


if __name__ == "__main__":
    main()

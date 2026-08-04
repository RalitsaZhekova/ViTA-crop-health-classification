"""Persistent model runtime and Earth Engine candidate orchestration."""

from __future__ import annotations

import hashlib
import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from prithvi_shared import PayloadAcquisitionCommand

from prithvi_payload.acquisition.earth_engine import (
    EARTH_ENGINE_PROJECT_ID,
    REFLECTANCE_SCALE,
    EarthEngineAcquisitionProvider,
)
from prithvi_payload.acquisition.errors import AcquisitionError
from prithvi_payload.cloud_classifier import (
    DEFAULT_CLOUD_CONFIG,
    CloudModel,
    load_cloud_model,
)
from prithvi_payload.inference import PayloadCropModel
from prithvi_payload.pipeline import continue_scene_from_cloud, run_scene

StatusCallback = Callable[[str, dict[str, Any]], None]


def _stage_seconds(value: Any, *keys: str) -> float:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return 0.0
        current = current.get(key)
    return float(current) if isinstance(current, (int, float)) else 0.0


def _notify(callback: StatusCallback, state: str, **fields: Any) -> None:
    callback(state, fields)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_attempt(
    *,
    candidate_scene_id: str,
    acquired_at: str,
    metadata_cloud_percentage: float | None,
    cloud: dict[str, Any] | None,
    accepted: bool,
    rejection_reason: str | None = None,
) -> dict[str, Any]:
    cloud = cloud or {}
    return {
        "candidate_scene_id": candidate_scene_id,
        "acquisition_time": acquired_at,
        "earth_engine_metadata_cloud_percentage": metadata_cloud_percentage,
        "payload_measured_thick_cloud_percentage": cloud.get("thick_cloud_percentage"),
        "payload_measured_thin_cloud_percentage": cloud.get("thin_cloud_percentage"),
        "payload_measured_shadow_percentage": cloud.get("cloud_shadow_percentage"),
        "payload_measured_cloud_percentage": cloud.get("total_cloud_percentage"),
        "payload_measured_unusable_percentage": cloud.get("unusable_percentage"),
        "accepted": accepted,
        "reason_rejected_for_demonstration": (
            None
            if accepted
            else rejection_reason or "payload cloud percentage outside requested target range"
        ),
    }


def _cleanup_rejected_candidate(candidate_root: Path) -> None:
    """Retain the safe acquisition record and remove rejected raster/model intermediates."""
    for filename in ("source_raw.tif", "scene.tif"):
        (candidate_root / filename).unlink(missing_ok=True)
    payload_root = candidate_root / "payload"
    if payload_root.is_dir() and payload_root.parent == candidate_root:
        shutil.rmtree(payload_root)


class PayloadRuntime:
    """One initialized Earth Engine, CUDA, cloud-model and crop-model runtime."""

    def __init__(
        self,
        *,
        provider: EarthEngineAcquisitionProvider | None = None,
        cloud_runtime: CloudModel | None = None,
        crop_model: PayloadCropModel | None = None,
        cloud_config_path: str | Path = DEFAULT_CLOUD_CONFIG,
        cuda_required: bool | None = None,
    ) -> None:
        self.provider = provider or EarthEngineAcquisitionProvider(
            project_id=os.environ.get("EE_PROJECT_ID", EARTH_ENGINE_PROJECT_ID)
        )
        self.cloud_runtime = cloud_runtime
        self.crop_model = crop_model
        self.cloud_config_path = Path(cloud_config_path)
        self.cuda_required = (
            os.environ.get("CUDA_REQUIRED", "1") == "1" if cuda_required is None else cuda_required
        )
        self.initialized = False

    def initialize(self) -> None:
        if self.initialized:
            return
        self.provider.initialize()
        if self.cuda_required and not torch.cuda.is_available():
            raise RuntimeError("CUDA_REQUIRED=1 but CUDA is unavailable")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.cloud_runtime is None:
            self.cloud_runtime = load_cloud_model(self.cloud_config_path)
        if self.crop_model is None:
            self.crop_model = PayloadCropModel.load(device=device)
        self._warm_models()
        self.initialized = True

    def _warm_models(self) -> None:
        if self.cloud_runtime is None or self.crop_model is None:
            raise RuntimeError("Models must be loaded before warmup")
        # Use non-uniform valid values so OmniCloudMask executes both ensemble
        # members during warmup instead of taking the all-nodata fast path.
        cloud_axis = np.linspace(0.05, 0.95, 1000, dtype=np.float32)
        cloud_tile = np.empty((4, 1000, 1000), dtype=np.float32)
        cloud_tile[0] = cloud_axis[:, None]
        cloud_tile[1] = cloud_axis[None, :]
        cloud_tile[2] = cloud_axis[::-1, None]
        cloud_tile[3] = cloud_axis[None, ::-1]
        self.cloud_runtime.backend.predict(cloud_tile)
        crop_tile = torch.zeros((1, 4, 1, 224, 224), dtype=torch.float32)
        self.crop_model.predict(
            crop_tile,
            temporal_coords=torch.tensor([[[2026.0, 1.0]]]),
            location_coords=torch.zeros((1, 2)),
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def process(
        self,
        command: PayloadAcquisitionCommand,
        job_directory: Path,
        status_callback: StatusCallback,
    ) -> dict[str, Any]:
        if not self.initialized or self.cloud_runtime is None or self.crop_model is None:
            raise RuntimeError("Payload runtime is not initialized")
        payload_started = time.perf_counter()
        _notify(status_callback, "searching_candidates")
        search_started = time.perf_counter()
        candidates, grid = self.provider.search_candidates(command)
        candidate_search_seconds = time.perf_counter() - search_started
        earth_engine_acquisition_seconds = candidate_search_seconds
        acquisition_root = job_directory / "acquisition"
        acquisition_root.mkdir(parents=True, exist_ok=True)
        attempts: list[dict[str, Any]] = []
        maximum_attempts = min(self.provider.max_scene_attempts, len(candidates))
        last_acquisition_error: AcquisitionError | None = None
        for attempt_number, candidate in enumerate(candidates[:maximum_attempts], 1):
            status_fields = {
                "current_candidate_number": attempt_number,
                "maximum_candidate_attempts": self.provider.max_scene_attempts,
                "safe_candidate_scene_id": candidate.system_index,
                "safe_metadata_cloud_percentage": candidate.metadata_cloud_percent,
                "safe_payload_measured_cloud_percentage": None,
                "candidate_attempts": attempts,
            }
            _notify(status_callback, "acquiring", **status_fields)
            acquisition_started = time.perf_counter()
            try:
                acquired = self.provider.acquire_candidate(
                    command,
                    candidate,
                    acquisition_root,
                    grid=grid,
                )
            except AcquisitionError as error:
                earth_engine_acquisition_seconds += time.perf_counter() - acquisition_started
                last_acquisition_error = error
                provider_reason = error.details.get("provider_reason")
                reason_suffix = f"; {provider_reason}" if isinstance(provider_reason, str) else ""
                attempt = _safe_attempt(
                    candidate_scene_id=candidate.system_index,
                    acquired_at=candidate.acquired_at,
                    metadata_cloud_percentage=candidate.metadata_cloud_percent,
                    cloud=None,
                    accepted=False,
                    rejection_reason=(
                        f"candidate acquisition failed ({error.code}{reason_suffix})"
                    ),
                )
                attempts.append(attempt)
                status_fields.update(candidate_attempts=attempts)
                _notify(status_callback, "acquiring", **status_fields)
                if error.code == "EARTH_ENGINE_DOWNLOAD_REQUEST_FAILED":
                    raise AcquisitionError(
                        error.code,
                        error.safe_message,
                        details={"candidate_attempts": attempts, **error.details},
                    ) from None
                continue
            earth_engine_acquisition_seconds += time.perf_counter() - acquisition_started
            candidate_root = acquired.local_tiff_path.parent
            payload_root = candidate_root / "payload"
            _notify(status_callback, "validating_input", **status_fields)
            _notify(status_callback, "evaluating_cloud", **status_fields)
            payload = run_scene(
                acquired.local_tiff_path,
                sensor="sentinel-2",
                output_root=payload_root,
                acquired_at=acquired.acquired_at,
                scene_id=command.job_id,
                reflectance_scale=REFLECTANCE_SCALE,
                stop_after="cloud",
                cloud_backend=self.cloud_runtime.backend,
                cloud_config=self.cloud_runtime.config,
            )
            if payload.get("status") != "CLOUD_COMPLETE":
                raise AcquisitionError(
                    "PAYLOAD_CLOUD_STAGE_FAILED", "Existing payload cloud stage did not complete"
                )
            cloud = payload["summary"]["cloud"]
            payload_cloud = float(cloud["total_cloud_percentage"])
            accepted = command.source.selection_policy == "least_cloudy" or (
                command.source.target_cloud_min_percent
                <= payload_cloud
                <= command.source.target_cloud_max_percent
            )
            attempt = _safe_attempt(
                candidate_scene_id=acquired.provider_scene_id,
                acquired_at=acquired.acquired_at,
                metadata_cloud_percentage=acquired.metadata_cloud_percent,
                cloud=cloud,
                accepted=accepted,
            )
            attempts.append(attempt)
            self.provider.record_candidate_evaluation(acquired, attempt)
            status_fields.update(
                safe_payload_measured_cloud_percentage=payload_cloud,
                candidate_attempts=attempts,
            )
            _notify(status_callback, "evaluating_cloud", **status_fields)
            if not accepted:
                _cleanup_rejected_candidate(candidate_root)
                continue

            target_range = (
                {
                    "minimum_percent": command.source.target_cloud_min_percent,
                    "maximum_percent": command.source.target_cloud_max_percent,
                    "ideal_percent": command.source.target_cloud_ideal_percent,
                }
                if command.source.selection_policy == "target_cloud_range"
                else None
            )
            provenance = acquired.safe_provenance(
                selection_policy=command.source.selection_policy,
                target_cloud_range=target_range,
                candidate_attempt_count=attempt_number,
            )

            def continuation_progress(
                state: str, current_fields: dict[str, Any] = status_fields
            ) -> None:
                _notify(status_callback, state, **current_fields)

            completed = continue_scene_from_cloud(
                payload_root / "result.json",
                region_id=command.region_id,
                crop_model=self.crop_model,
                acquisition_metadata=provenance,
                progress_callback=continuation_progress,
                overwrite=True,
            )
            if completed.get("status") != "DOWNLINK_READY":
                raise AcquisitionError(
                    "PAYLOAD_PROCESSING_STOPPED",
                    "Existing payload pipeline did not reach DOWNLINK_READY",
                    details={"payload_status": completed.get("status")},
                )
            downlink_root = payload_root / "downlink"
            entries = {path.name for path in downlink_root.iterdir()}
            if entries != {"scene.json", "scene.webp", "condition.png"}:
                raise AcquisitionError(
                    "PAYLOAD_BUNDLE_INVALID",
                    "Payload downlink does not contain exactly the three routine artifacts",
                )
            artifact_checksums = {
                filename: _sha256(downlink_root / filename) for filename in sorted(entries)
            }
            cloud_runtime = payload.get("stage_metadata", {}).get("cloud", {}).get("runtime", {})
            completed_stages = completed.get("stage_metadata", {})
            crop_runtime = completed_stages.get("crop", {}).get("runtime", {})
            condition_runtime = completed_stages.get("condition", {}).get("runtime", {})
            downlink_runtime = completed_stages.get("downlink", {}).get("runtime", {})
            timing = {
                "candidate_search_seconds": candidate_search_seconds,
                "earth_engine_acquisition_seconds": earth_engine_acquisition_seconds,
                "earth_engine_download_seconds": acquired.timing.get(
                    "earth_engine_download_seconds", 0.0
                ),
                "geotiff_validation_seconds": acquired.timing.get(
                    "geotiff_validation_seconds", 0.0
                ),
                "cloud_inference_seconds": _stage_seconds(cloud_runtime, "inference_seconds"),
                "mask_processing_seconds": _stage_seconds(cloud_runtime, "mask_processing_seconds"),
                "crop_inference_seconds": _stage_seconds(crop_runtime, "inference_seconds"),
                "condition_calculation_seconds": _stage_seconds(condition_runtime, "seconds"),
                "downlink_packaging_seconds": _stage_seconds(downlink_runtime, "seconds"),
            }
            timing["warm_science_seconds"] = sum(
                timing[name]
                for name in (
                    "cloud_inference_seconds",
                    "mask_processing_seconds",
                    "crop_inference_seconds",
                    "condition_calculation_seconds",
                    "downlink_packaging_seconds",
                )
            )
            timing["total_payload_processing_seconds"] = time.perf_counter() - payload_started
            return {
                "selected_scene": acquired.provider_scene_id,
                "metadata_cloud_percentage": acquired.metadata_cloud_percent,
                "payload_cloud_percentage": payload_cloud,
                "payload_shadow_percentage": cloud["cloud_shadow_percentage"],
                "payload_unusable_percentage": cloud["unusable_percentage"],
                "candidate_attempts": attempts,
                "condition": completed["summary"]["condition"],
                "artifact_directory": downlink_root.relative_to(job_directory).as_posix(),
                "artifacts": ["scene.json", "scene.webp", "condition.png"],
                "artifact_checksums": artifact_checksums,
                "payload_status": "DOWNLINK_READY",
                "timing": timing,
            }
        if command.source.selection_policy == "target_cloud_range":
            if attempts and all(
                attempt["payload_measured_cloud_percentage"] is None for attempt in attempts
            ):
                raise AcquisitionError(
                    "EARTH_ENGINE_ACQUISITION_FAILED",
                    "Earth Engine candidates could not be acquired",
                    details={"candidate_attempts": attempts},
                ) from None
            raise AcquisitionError(
                "PAYLOAD_NO_SCENE_IN_TARGET_CLOUD_RANGE",
                "No metadata-qualified candidate met the payload-measured cloud target",
                details={"candidate_attempts": attempts},
            )
        if last_acquisition_error is not None:
            raise AcquisitionError(
                "EARTH_ENGINE_ACQUISITION_FAILED",
                "Earth Engine candidates could not be acquired",
                details={"candidate_attempts": attempts},
            ) from None
        raise AcquisitionError(
            "PAYLOAD_PROCESSING_STOPPED",
            "The least-cloudy candidate did not complete payload processing",
            details={"candidate_attempts": attempts},
        )

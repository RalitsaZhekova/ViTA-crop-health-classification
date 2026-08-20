from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
from prithvi_payload import raw_service
from prithvi_payload.acquisition import AcquiredSentinelScene


def _raw_scene(root: Path, processed_root: Path, scene_id: str) -> None:
    paths = (
        root / "balkan1" / "raw" / scene_id / f"{scene_id}_Raw.tif",
        root / "balkan1" / "raw" / scene_id / "position.csv",
        root / "balkan1" / "raw" / scene_id / "attitude.csv",
        root / "balkan1" / "derived" / "l1a" / f"{scene_id}_L0R_manifest.json",
        root
        / "balkan1"
        / "derived"
        / "l1a"
        / f"{scene_id}_L1A_reference_validation.json",
        processed_root
        / "balkan1"
        / "preprocessed"
        / f"{scene_id}_L1ORT.crop_calibration.json",
    )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("input", encoding="utf-8")


def test_warm_raw_runtime_reuses_preloaded_models(tmp_path: Path, monkeypatch) -> None:
    scene_id = "3408"
    input_root = tmp_path / "data"
    processed_root = tmp_path / "operational-data"
    output_root = tmp_path / "runtime" / "runs"
    _raw_scene(input_root, processed_root, scene_id)
    output_root.mkdir(parents=True)
    pipeline_path = output_root / "web-raw-test" / "result.json"
    pipeline_path.parent.mkdir()
    pipeline_path.write_text(
        json.dumps({"stage_metadata": {}, "timing": {}}),
        encoding="utf-8",
    )
    cloud = object()
    crop = object()
    acceleration = {"cloud_backend": "tensorrt", "crop_backend": "tensorrt"}
    runtime = raw_service.RawPayloadRuntime.__new__(raw_service.RawPayloadRuntime)
    runtime.input_root = input_root
    runtime.processed_input_root = processed_root
    runtime.output_root = output_root
    runtime.cloud = cloud
    runtime.crop = crop
    runtime.acceleration = acceleration
    captured = {}

    def fake_run_raw_payload_job(**kwargs):
        captured.update(kwargs)
        return {
            "status": "DOWNLINK_READY",
            "job_id": kwargs["job_id"],
            "scene_id": kwargs["job_id"],
            "sensor": "balkan-1-raw",
            "region_id": kwargs["region_id"],
            "qualification": "UNQUALIFIED_ENGINEERING_EXPERIMENT",
            "bundle_relative": f"runs/{kwargs['job_id']}/downlink",
            "files": {},
            "artifacts": {"pipeline_result": str(pipeline_path)},
            "acceleration": {**acceleration, "models_preloaded": True},
            "summary": {},
            "timing_seconds": {
                "cuda_alignment_seconds": 8.0,
                "raw_model_reconstruction_seconds": 8.0,
                "accelerated_pipeline_seconds": 3.0,
                "end_to_end_seconds": 19.0,
            },
        }

    monkeypatch.setattr(raw_service, "run_raw_payload_job", fake_run_raw_payload_job)
    response = runtime.run(
        raw_service.RawJobRequest(
            sensor="balkan-1-raw",
            input=scene_id,
            region_id="raw-field",
            job_id="web-raw-test",
        )
    )

    assert response["status"] == "DOWNLINK_READY"
    assert response["payload_seconds"] == 19.0
    assert response["stack"]["models_preloaded"] is True
    assert captured["preloaded_cloud"] is cloud
    assert captured["preloaded_crop"] is crop
    assert captured["preloaded_acceleration"] is acceleration
    assert captured["raw_path"] == input_root / "balkan1" / "raw" / "3408" / "3408_Raw.tif"
    assert captured["parent_calibration_path"] == (
        processed_root / "balkan1" / "preprocessed" / "3408_L1ORT.crop_calibration.json"
    )


def test_warm_runtime_runs_processed_balkan_with_dynamic_accepted_profile(
    tmp_path: Path,
    monkeypatch,
) -> None:
    processed_root = tmp_path / "operational-data"
    source = processed_root / "balkan1" / "preprocessed" / "3408_L1ORT.tif"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"processed scene")
    calibration = source.with_name("3408_L1ORT.crop_calibration.json")
    calibration.write_text(
        json.dumps({"acquired_at": "2026-06-16T18:40:43.523000+00:00"}),
        encoding="utf-8",
    )
    output_root = tmp_path / "runtime" / "runs"
    output_root.mkdir(parents=True)
    backend = SimpleNamespace(
        patch_size=1000,
        raw_fixed_patch_size=1000,
        batch_size=1,
    )
    cloud = SimpleNamespace(backend=backend, config=object())
    crop = object()
    runtime = raw_service.RawPayloadRuntime.__new__(raw_service.RawPayloadRuntime)
    runtime.processed_input_root = processed_root
    runtime.output_root = output_root
    runtime.cloud = cloud
    runtime.crop = crop
    runtime.acceleration = {
        "cloud_backend": "tensorrt",
        "crop_backend": "tensorrt",
        "cloud_scene_patch_sizes": {
            "balkan1/preprocessed/3408_L1ORT.tif": 869,
        },
        "cloud_profiles": [
            {
                "patch_size": 869,
                "minimum_batch_size": 1,
                "maximum_batch_size": 4,
            }
        ],
    }
    captured = {}

    def fake_run_scene(input_path, **kwargs):
        captured["input"] = input_path
        captured.update(kwargs)
        assert backend.raw_fixed_patch_size is None
        assert backend.batch_size == 4
        bundle = kwargs["output_root"] / "downlink"
        bundle.mkdir(parents=True)
        for name in ("scene.json", "scene.webp", "condition.png"):
            (bundle / name).write_bytes(name.encode("utf-8"))
        return {
            "status": "DOWNLINK_READY",
            "scene_id": kwargs["scene_id"],
            "summary": {"crop": {"decision": "CLASSIFIED"}},
            "stage_metadata": {},
            "timing": {},
        }

    monkeypatch.setattr(raw_service, "run_scene", fake_run_scene)
    response = runtime.run(
        raw_service.RawJobRequest(
            sensor="balkan-1",
            input="balkan1/preprocessed/3408_L1ORT.tif",
            region_id="balkan-field",
            job_id="web-balkan-test",
        )
    )

    assert response["status"] == "DOWNLINK_READY"
    assert response["sensor"] == "balkan-1"
    assert captured["input"] == source
    assert captured["sensor"] == "balkan-1"
    assert captured["cloud_backend"] is backend
    assert captured["crop_model"] is crop
    assert backend.raw_fixed_patch_size == 1000
    assert backend.batch_size == 1
    assert set(response["files"]) == {"scene.json", "scene.webp", "condition.png"}


def test_processed_cloud_profile_restores_raw_contract_after_failure() -> None:
    backend = SimpleNamespace(
        patch_size=1000,
        raw_fixed_patch_size=1000,
        batch_size=1,
    )
    runtime = raw_service.RawPayloadRuntime.__new__(raw_service.RawPayloadRuntime)
    runtime.cloud = SimpleNamespace(backend=backend)
    runtime.acceleration = {
        "cloud_scene_patch_sizes": {
            "balkan1/preprocessed/3458_L1ORT.tif": 845,
        },
        "cloud_profiles": [
            {
                "patch_size": 845,
                "minimum_batch_size": 1,
                "maximum_batch_size": 4,
            }
        ],
    }

    with pytest.raises(
        RuntimeError,
        match="synthetic failure",
    ), runtime._processed_cloud_profile_selection(
        "balkan1/preprocessed/3458_L1ORT.tif"
    ):
        assert backend.raw_fixed_patch_size is None
        assert backend.batch_size == 4
        raise RuntimeError("synthetic failure")

    assert backend.raw_fixed_patch_size == 1000
    assert backend.batch_size == 1


def test_unqualified_processed_scene_retains_safe_batch_one() -> None:
    backend = SimpleNamespace(
        patch_size=1000,
        raw_fixed_patch_size=1000,
        batch_size=1,
    )
    runtime = raw_service.RawPayloadRuntime.__new__(raw_service.RawPayloadRuntime)
    runtime.cloud = SimpleNamespace(backend=backend)
    runtime.acceleration = {
        "cloud_scene_patch_sizes": {},
        "cloud_profiles": [],
    }

    with runtime._processed_cloud_profile_selection("balkan1/preprocessed/new.tif"):
        assert backend.raw_fixed_patch_size is None
        assert backend.batch_size == 1

    assert backend.raw_fixed_patch_size == 1000
    assert backend.batch_size == 1


def test_warm_runtime_acquires_live_sentinel_then_reuses_preloaded_models(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_root = tmp_path / "runtime" / "runs"
    output_root.mkdir(parents=True)
    source = tmp_path / "acquired-scene.tif"
    source.write_bytes(b"sentinel source")
    acquired = AcquiredSentinelScene(
        local_tiff_path=source,
        provider_scene_id="20260615T105621_20260615T110752_T31UFU",
        product_id="S2B_MSIL2A_20260615T105621_N0511_R094_T31UFU_20260615T110752",
        acquired_at="2026-06-15T10:56:21+00:00",
        metadata_cloud_percentage=4.2,
        requested_bbox_wgs84=(5.43, 52.50, 5.49, 52.54),
        output_crs="EPSG:32631",
        output_transform=(10.0, 0.0, 665000.0, 0.0, -10.0, 5820000.0),
        sha256="a" * 64,
        byte_size=15,
        candidate_rank=1,
        candidate_attempt_count=1,
        timing={
            "earth_engine_search_seconds": 0.2,
            "earth_engine_download_seconds": 0.8,
            "geotiff_validation_seconds": 0.1,
            "total_acquisition_seconds": 1.1,
        },
    )
    provider = SimpleNamespace(acquire=lambda **_kwargs: acquired)
    backend = object()
    config = object()
    cloud = SimpleNamespace(backend=backend, config=config)
    crop = object()
    runtime = raw_service.RawPayloadRuntime.__new__(raw_service.RawPayloadRuntime)
    runtime.output_root = output_root
    runtime.cloud = cloud
    runtime.crop = crop
    runtime.acceleration = {
        "cloud_backend": "tensorrt",
        "crop_backend": "tensorrt",
        "models_preloaded": True,
    }
    runtime.earth_engine_provider = provider
    captured = {}

    def fake_run_scene(input_path, **kwargs):
        captured["input"] = input_path
        captured.update(kwargs)
        bundle = kwargs["output_root"] / "downlink"
        bundle.mkdir(parents=True)
        for name in ("scene.json", "scene.webp", "condition.png"):
            (bundle / name).write_bytes(name.encode("utf-8"))
        return {
            "status": "DOWNLINK_READY",
            "scene_id": kwargs["scene_id"],
            "summary": {"crop": {"decision": "CLASSIFIED"}},
            "stage_metadata": {},
            "timing": {},
        }

    monkeypatch.setattr(raw_service, "run_scene", fake_run_scene)
    response = runtime.run(
        raw_service.RawJobRequest(
            sensor="sentinel-2-live",
            input="earth-engine",
            region_id="selected-farm",
            job_id="web-live-test",
            bbox_wgs84=(5.43, 52.50, 5.49, 52.54),
            start_date=date(2026, 6, 1),
            end_date=date(2026, 7, 1),
        )
    )

    assert response["status"] == "DOWNLINK_READY"
    assert response["sensor"] == "sentinel-2"
    assert response["stack"]["earth_engine_acquisition"] is True
    assert response["pipeline_timing_seconds"]["total_acquisition_seconds"] == 1.1
    assert captured["input"] == source
    assert captured["sensor"] == "sentinel-2"
    assert captured["cloud_backend"] is backend
    assert captured["cloud_config"] is config
    assert captured["crop_model"] is crop
    assert captured["reflectance_scale"] == 10_000.0
    assert captured["acquisition_metadata"]["provider"] == "earth_engine"


def test_live_request_rejects_missing_acquisition_fields() -> None:
    with pytest.raises(ValueError, match="requires an area and date range"):
        raw_service.RawJobRequest(
            sensor="sentinel-2-live",
            input="earth-engine",
            region_id="selected-farm",
            job_id="web-live-test",
        )

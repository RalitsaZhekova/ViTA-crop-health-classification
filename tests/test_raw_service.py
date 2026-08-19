from __future__ import annotations

import json
from pathlib import Path

from prithvi_payload import raw_service


def _raw_scene(root: Path, scene_id: str) -> None:
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
        root / "balkan1" / "preprocessed" / f"{scene_id}_L1ORT.crop_calibration.json",
    )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("input", encoding="utf-8")


def test_warm_raw_runtime_reuses_preloaded_models(tmp_path: Path, monkeypatch) -> None:
    scene_id = "3408"
    input_root = tmp_path / "data"
    output_root = tmp_path / "runtime" / "runs"
    _raw_scene(input_root, scene_id)
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

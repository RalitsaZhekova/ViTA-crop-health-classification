from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from prithvi_payload.job_store import JobNotFoundError, JobStore
from prithvi_payload.service import create_app
from prithvi_shared import PayloadAcquisitionCommand


def _command(job_id: str = "field_42_20260715_ab12cd34") -> PayloadAcquisitionCommand:
    return PayloadAcquisitionCommand.model_validate(
        {
            "schema_version": "1.0",
            "job_id": job_id,
            "region_id": "field_42",
            "source": {
                "provider": "earth_engine",
                "bbox_wgs84": [23.10, 42.50, 23.15, 42.55],
                "start_date": "2026-07-01",
                "end_date": "2026-07-29",
                "selection_policy": "target_cloud_range",
                "target_cloud_min_percent": 15,
                "target_cloud_max_percent": 35,
                "target_cloud_ideal_percent": 25,
            },
        }
    )


def test_job_store_persists_safe_progress_and_recovers_pending(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    command = _command()
    store.create(command)
    store.update(
        command.job_id,
        "evaluating_cloud",
        current_candidate_number=2,
        maximum_candidate_attempts=5,
        safe_candidate_scene_id="scene-safe",
        safe_metadata_cloud_percentage=24.0,
        safe_payload_measured_cloud_percentage=18.0,
    )

    recovered = JobStore(tmp_path / "jobs").recover_pending()
    status = store.get(command.job_id)

    assert recovered == [command.job_id]
    assert status["state"] == "queued"
    assert status["safe_candidate_scene_id"] == "scene-safe"
    assert status["safe_payload_measured_cloud_percentage"] == 18.0
    assert "url" not in json.dumps(status).lower()
    assert "credential" not in json.dumps(status).lower()


def test_completed_job_is_retained_and_artifacts_are_whitelisted(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    command = _command()
    store.create(command)
    artifact_root = store.job_directory(command.job_id) / "payload/downlink"
    artifact_root.mkdir(parents=True)
    for filename in ("scene.json", "scene.webp", "condition.png"):
        (artifact_root / filename).write_bytes(filename.encode())
    store.write_result(
        command.job_id,
        {"artifact_directory": "payload/downlink"},
    )
    store.update(command.job_id, "completed")

    assert JobStore(tmp_path / "jobs").recover_pending() == []
    assert store.artifact_path(command.job_id, "scene.json").is_file()
    with pytest.raises(ValueError, match="Unsupported"):
        store.artifact_path(command.job_id, "../../command.json")


class _Runtime:
    def health(self) -> dict[str, object]:
        return {
            "status": "ok",
            "earth_engine_project": "vita-503208",
            "cuda_required": True,
            "cuda_available": True,
            "cloud_model_loaded": True,
            "crop_model_loaded": True,
            "models_warmed": True,
        }


def test_service_exposes_only_fixed_artifact_routes(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    app = create_app(runtime=_Runtime(), store=store, start_service=False)  # type: ignore[arg-type]

    with TestClient(app) as client:
        assert client.get("/health").json()["earth_engine_project"] == "vita-503208"
        assert client.get("/v1/jobs/unknown/artifacts/scene.json").status_code == 404
        assert client.get("/v1/jobs/unknown/artifacts/result.json").status_code == 404
        assert client.get("/v1/jobs/../../command.json").status_code == 404

    with pytest.raises(JobNotFoundError):
        store.get("unknown")

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from prithvi_payload.acquisition.errors import AcquisitionError
from prithvi_payload.acquisition.models import AcquiredScene, CandidateMetadata
from prithvi_payload.crop_stage import DEFAULT_MAX_CLOUD_PERCENTAGE
from prithvi_payload.runtime import PayloadRuntime
from prithvi_shared import PayloadAcquisitionCommand


def _command(policy: str = "target_cloud_range") -> PayloadAcquisitionCommand:
    return PayloadAcquisitionCommand.model_validate(
        {
            "schema_version": "1.0",
            "job_id": "field_42_20260715_ab12cd34",
            "region_id": "field_42",
            "source": {
                "provider": "earth_engine",
                "bbox_wgs84": [23.10, 42.50, 23.15, 42.55],
                "start_date": "2026-07-01",
                "end_date": "2026-07-29",
                "selection_policy": policy,
                "target_cloud_min_percent": 15,
                "target_cloud_max_percent": 35,
                "target_cloud_ideal_percent": 25,
            },
        }
    )


def _candidate(index: int) -> CandidateMetadata:
    return CandidateMetadata(
        system_index=f"scene-{index}",
        acquired_at=f"2026-07-{index:02d}T10:00:00+00:00",
        acquired_at_millis=index,
        metadata_cloud_percent=20.0 + index,
        product_id=f"product-{index}",
        source_metadata={"MGRS_TILE": "34TFN"},
        candidate_rank=index,
    )


class _Provider:
    project_id = "vita-503208"
    max_scene_attempts = 5

    def __init__(self, count: int, *, failures: set[str] | None = None) -> None:
        self.candidates = [_candidate(index) for index in range(1, count + 1)]
        self.failures = failures or set()
        self.acquired_ids: list[str] = []
        self.evaluations: list[dict[str, Any]] = []
        self.initialize_calls = 0

    def initialize(self) -> None:
        self.initialize_calls += 1

    def search_candidates(self, _: object) -> tuple[list[CandidateMetadata], object]:
        return self.candidates, object()

    def acquire_candidate(
        self,
        command: PayloadAcquisitionCommand,
        candidate: CandidateMetadata,
        destination: Path,
        *,
        grid: object,
    ) -> AcquiredScene:
        del command, grid
        self.acquired_ids.append(candidate.system_index)
        candidate_root = destination / candidate.system_index
        candidate_root.mkdir(parents=True)
        record = candidate_root / "acquisition_record.json"
        record.write_text(
            json.dumps({"status": "acquired", "provider_scene_id": candidate.system_index}),
            encoding="utf-8",
        )
        if candidate.system_index in self.failures:
            raise AcquisitionError(
                "EARTH_ENGINE_DOWNLOAD_FAILED",
                "Earth Engine download failed after bounded retries",
            )
        scene = candidate_root / "scene.tif"
        raw = candidate_root / "source_raw.tif"
        scene.write_bytes(b"scene")
        raw.write_bytes(b"raw")
        return AcquiredScene(
            local_tiff_path=scene,
            provider="earth_engine",
            collection="COPERNICUS/S2_SR_HARMONIZED",
            provider_scene_id=candidate.system_index,
            product_id=candidate.product_id,
            acquired_at=candidate.acquired_at,
            metadata_cloud_percent=candidate.metadata_cloud_percent,
            requested_bbox_wgs84=(23.10, 42.50, 23.15, 42.55),
            output_crs="EPSG:32634",
            output_transform=(10, 0, 500_000, 0, -10, 4_700_000),
            width=32,
            height=32,
            band_names=("B02", "B03", "B04", "B08", "B8A"),
            reflectance_scale=10_000,
            source_metadata=candidate.source_metadata,
            sha256="a" * 64,
            byte_size=5,
            candidate_rank=candidate.candidate_rank,
        )

    def record_candidate_evaluation(
        self,
        acquired: AcquiredScene,
        evaluation: dict[str, Any],
    ) -> None:
        self.evaluations.append(evaluation)
        record_path = acquired.local_tiff_path.parent / "acquisition_record.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["payload_evaluation"] = evaluation
        record_path.write_text(json.dumps(record), encoding="utf-8")


def _runtime(provider: _Provider) -> PayloadRuntime:
    runtime = PayloadRuntime(
        provider=provider,  # type: ignore[arg-type]
        cloud_runtime=SimpleNamespace(backend=object(), cfg={}),  # type: ignore[arg-type]
        crop_model=object(),  # type: ignore[arg-type]
        cuda_required=False,
    )
    runtime.initialized = True
    return runtime


def _cloud_result(cloud_percentage: float, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    mask = output_root / "authoritative_unusable_mask.tif"
    mask.write_bytes(b"preserved-mask")
    result = {
        "status": "CLOUD_COMPLETE",
        "summary": {
            "cloud": {
                "thick_cloud_percentage": cloud_percentage,
                "thin_cloud_percentage": 0.0,
                "cloud_shadow_percentage": 3.0,
                "total_cloud_percentage": cloud_percentage,
                "unusable_percentage": cloud_percentage + 3.0,
            }
        },
        "mask_token": str(mask),
    }
    (output_root / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return result


def _completed_continuation(result_path: Path) -> dict[str, Any]:
    cloud_result = json.loads(result_path.read_text(encoding="utf-8"))
    assert Path(cloud_result["mask_token"]).read_bytes() == b"preserved-mask"
    downlink = result_path.parent / "downlink"
    downlink.mkdir()
    for filename in ("scene.json", "scene.webp", "condition.png"):
        (downlink / filename).write_bytes(filename.encode())
    return {
        "status": "DOWNLINK_READY",
        "summary": {"condition": {"label": "Watch", "score": 72.0}},
    }


def test_payload_cloud_rejection_then_acceptance_reuses_selected_cloud_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _Provider(3)
    runtime = _runtime(provider)
    cloud_values = iter((8.0, 24.0))
    cloud_calls: list[Path] = []
    continuation_calls: list[Path] = []

    def run_scene(_: Path, *, output_root: Path, **__: Any) -> dict[str, Any]:
        cloud_calls.append(output_root)
        return _cloud_result(next(cloud_values), output_root)

    def continue_scene(result_path: Path, **kwargs: Any) -> dict[str, Any]:
        continuation_calls.append(result_path)
        assert kwargs["acquisition_metadata"]["provider_scene_id"] == "scene-2"
        assert kwargs["overwrite"] is True
        assert "max_cloud_percentage" not in kwargs
        for state in ("running_crop", "running_condition", "packaging"):
            kwargs["progress_callback"](state)
        return _completed_continuation(result_path)

    monkeypatch.setattr("prithvi_payload.runtime.run_scene", run_scene)
    monkeypatch.setattr("prithvi_payload.runtime.continue_scene_from_cloud", continue_scene)
    statuses: list[tuple[str, dict[str, Any]]] = []

    result = runtime.process(
        _command(),
        tmp_path / "job",
        lambda state, fields: statuses.append((state, fields)),
    )

    assert provider.acquired_ids == ["scene-1", "scene-2"]
    assert len(cloud_calls) == 2
    assert continuation_calls == [
        tmp_path / "job/acquisition/scene-2/payload/result.json"
    ]
    assert result["selected_scene"] == "scene-2"
    assert result["payload_cloud_percentage"] == 24.0
    assert result["artifacts"] == ["scene.json", "scene.webp", "condition.png"]
    assert [item["accepted"] for item in result["candidate_attempts"]] == [False, True]
    assert {state for state, _ in statuses} >= {
        "searching_candidates",
        "acquiring",
        "validating_input",
        "evaluating_cloud",
        "running_crop",
        "running_condition",
        "packaging",
    }
    assert any(
        fields.get("safe_payload_measured_cloud_percentage") == 24.0
        for _, fields in statuses
    )
    rejected_record = json.loads(
        (tmp_path / "job/acquisition/scene-1/acquisition_record.json").read_text()
    )
    assert rejected_record["payload_evaluation"]["payload_measured_cloud_percentage"] == 8.0
    assert not (tmp_path / "job/acquisition/scene-1/scene.tif").exists()


def test_maximum_five_payload_cloud_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _Provider(6)
    runtime = _runtime(provider)
    monkeypatch.setattr(
        "prithvi_payload.runtime.run_scene",
        lambda _path, *, output_root, **_kwargs: _cloud_result(2.0, output_root),
    )

    with pytest.raises(AcquisitionError) as caught:
        runtime.process(_command(), tmp_path / "job", lambda *_: None)

    assert caught.value.code == "PAYLOAD_NO_SCENE_IN_TARGET_CLOUD_RANGE"
    assert provider.acquired_ids == [f"scene-{index}" for index in range(1, 6)]
    assert len(caught.value.details["candidate_attempts"]) == 5


def test_candidate_acquisition_failure_advances_safely(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _Provider(2, failures={"scene-1"})
    runtime = _runtime(provider)
    monkeypatch.setattr(
        "prithvi_payload.runtime.run_scene",
        lambda _path, *, output_root, **_kwargs: _cloud_result(20.0, output_root),
    )
    monkeypatch.setattr(
        "prithvi_payload.runtime.continue_scene_from_cloud",
        lambda result_path, **_kwargs: _completed_continuation(result_path),
    )

    result = runtime.process(_command(), tmp_path / "job", lambda *_: None)

    assert result["selected_scene"] == "scene-2"
    failed = result["candidate_attempts"][0]
    assert failed["payload_measured_cloud_percentage"] is None
    assert failed["reason_rejected_for_demonstration"] == (
        "candidate acquisition failed (EARTH_ENGINE_DOWNLOAD_FAILED)"
    )
    assert "credential" not in json.dumps(failed).lower()


def test_download_request_failure_stops_with_safe_provider_reason(tmp_path: Path) -> None:
    provider = _Provider(2)
    runtime = _runtime(provider)

    def reject_request(*_args: Any, **_kwargs: Any) -> None:
        raise AcquisitionError(
            "EARTH_ENGINE_DOWNLOAD_REQUEST_FAILED",
            "Earth Engine rejected the fixed GeoTIFF download request",
            details={
                "provider_reason": "invalid_request",
                "provider_error_type": "EEException",
            },
        )

    provider.acquire_candidate = reject_request  # type: ignore[method-assign]

    with pytest.raises(AcquisitionError) as caught:
        runtime.process(_command(), tmp_path / "job", lambda *_: None)

    assert caught.value.code == "EARTH_ENGINE_DOWNLOAD_REQUEST_FAILED"
    assert caught.value.details["provider_reason"] == "invalid_request"
    assert len(caught.value.details["candidate_attempts"]) == 1
    assert "invalid_request" in caught.value.details["candidate_attempts"][0][
        "reason_rejected_for_demonstration"
    ]


def test_all_download_failures_are_not_reported_as_cloud_range_rejection(
    tmp_path: Path,
) -> None:
    failed_scenes = {f"scene-{index}" for index in range(1, 6)}
    provider = _Provider(5, failures=failed_scenes)
    runtime = _runtime(provider)

    with pytest.raises(AcquisitionError) as caught:
        runtime.process(_command(), tmp_path / "job", lambda *_: None)

    assert caught.value.code == "EARTH_ENGINE_ACQUISITION_FAILED"
    assert len(caught.value.details["candidate_attempts"]) == 5
    assert all(
        attempt["payload_measured_cloud_percentage"] is None
        for attempt in caught.value.details["candidate_attempts"]
    )


def test_existing_scientific_rejection_threshold_is_unchanged() -> None:
    assert DEFAULT_MAX_CLOUD_PERCENTAGE == 60.0


def test_persistent_runtime_initializes_and_warms_models_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _Provider(1)
    cloud_backend = SimpleNamespace(calls=0)

    def cloud_predict(_: object) -> None:
        cloud_backend.calls += 1

    cloud_backend.predict = cloud_predict
    cloud_runtime = SimpleNamespace(backend=cloud_backend, cfg={})
    crop_model = SimpleNamespace(calls=0)

    def crop_predict(*_: object, **__: object) -> None:
        crop_model.calls += 1

    crop_model.predict = crop_predict
    cloud_loads = 0
    crop_loads = 0

    def load_cloud(_: object) -> object:
        nonlocal cloud_loads
        cloud_loads += 1
        return cloud_runtime

    def load_crop(*, device: str) -> object:
        nonlocal crop_loads
        assert device in {"cpu", "cuda"}
        crop_loads += 1
        return crop_model

    monkeypatch.setattr(
        "prithvi_payload.runtime.CloudDetectionPipeline.from_yaml", load_cloud
    )
    monkeypatch.setattr("prithvi_payload.runtime.PayloadCropModel.load", load_crop)
    runtime = PayloadRuntime(provider=provider, cuda_required=False)  # type: ignore[arg-type]

    runtime.initialize()
    runtime.initialize()

    assert provider.initialize_calls == 1
    assert cloud_loads == 1
    assert crop_loads == 1
    assert cloud_backend.calls == 1
    assert crop_model.calls == 1

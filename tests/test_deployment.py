from __future__ import annotations

from pathlib import Path

import pytest
from prithvi_payload.deployment_check import _paths_from_environment
from prithvi_payload.inference import _environment_flag
from prithvi_payload.service import (
    JobRequest,
    _balkan_prepare_inputs,
    _pipeline_timings,
    _safe_relative,
)
from pydantic import ValidationError
from vita_integration.ingest import parser as ingest_parser


def test_payload_job_contract_forbids_uplink_content() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        JobRequest.model_validate(
            {
                "sensor": "sentinel-2",
                "input": "sentinel2",
                "image": "scene.tif",
                "region_id": "region-1",
                "job_id": "job-1",
                "image_bytes": "not-allowed",
            }
        )


def test_payload_paths_cannot_escape_data_mount(tmp_path: Path) -> None:
    scene = tmp_path / "sentinel2" / "scene.tif"
    scene.parent.mkdir()
    scene.touch()

    assert _safe_relative(tmp_path.resolve(), "sentinel2/scene.tif", name="input") == scene
    with pytest.raises(ValueError, match="relative path"):
        _safe_relative(tmp_path.resolve(), "../secret", name="input")


def test_ingest_command_has_explicit_bundle_and_store() -> None:
    args = ingest_parser().parse_args(["bundle", "--store", "runtime/ground"])
    assert args.bundle == Path("bundle")
    assert args.store == Path("runtime/ground")


def test_boolean_environment_parser_is_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VITA_TEST_FLAG", "yes")
    assert _environment_flag("VITA_TEST_FLAG", False)
    monkeypatch.setenv("VITA_TEST_FLAG", "sometimes")
    with pytest.raises(ValueError, match="must be one of"):
        _environment_flag("VITA_TEST_FLAG", False)


def test_payload_response_flattens_stage_timings() -> None:
    result = {
        "timing": {"intake_seconds": 0.1, "cloud_plan_seconds": 0.2},
        "stage_metadata": {
            "cloud": {"runtime": {"seconds": 1.0, "inference_seconds": 0.7}},
            "crop": {"runtime": {"seconds": 0.5, "inference_seconds": 0.25}},
            "condition": {"runtime": {"seconds": 0.3}},
            "downlink": {"runtime": {"seconds": 0.2}},
        },
    }

    timing = _pipeline_timings(result, payload_seconds=2.5)

    assert timing["cloud_stage_seconds"] == 1.0
    assert timing["cloud_inference_seconds"] == 0.7
    assert timing["crop_inference_seconds"] == 0.25
    assert timing["downlink_packaging_seconds"] == 0.2
    assert timing["reported_stage_total_seconds"] == pytest.approx(2.3)
    assert timing["orchestration_seconds"] == pytest.approx(0.2)


def test_balkan_startup_accepts_two_fixed_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "VITA_BALKAN_PREPARE_INPUTS",
        "balkan1/preprocessed/3370_L1ORT.tif,balkan1/preprocessed/3408_L1ORT.tif",
    )
    assert _balkan_prepare_inputs() == (
        "balkan1/preprocessed/3370_L1ORT.tif",
        "balkan1/preprocessed/3408_L1ORT.tif",
    )


def test_balkan_startup_rejects_duplicate_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VITA_BALKAN_PREPARE_INPUTS", "same.tif,same.tif")
    with pytest.raises(RuntimeError, match="distinct"):
        _balkan_prepare_inputs()


def test_deployment_requires_exactly_two_distinct_sensor_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VITA_TEST_INPUTS", "first.tif,second.tif")
    assert _paths_from_environment("VITA_TEST_INPUTS") == ("first.tif", "second.tif")
    monkeypatch.setenv("VITA_TEST_INPUTS", "first.tif")
    with pytest.raises(RuntimeError, match="exactly two distinct"):
        _paths_from_environment("VITA_TEST_INPUTS")


def test_payload_demo_manifest_contains_only_four_scenes_and_two_calibrations() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    manifest = repository_root / "deploy" / "payload" / "demo-assets.sha256"
    paths = [line.split("  ", 1)[1] for line in manifest.read_text().splitlines()]

    assert len(paths) == 6
    assert sum(path.endswith(".tif") and "/sentinel2/" in path for path in paths) == 2
    assert sum(path.endswith(".tif") and "/balkan1/" in path for path in paths) == 2
    assert sum(path.endswith(".crop_calibration.json") for path in paths) == 2

    model_manifest = repository_root / "deploy" / "payload" / "model-assets.sha256"
    model_paths = [line.split("  ", 1)[1] for line in model_manifest.read_text().splitlines()]
    assert len(model_paths) == 3
    assert all(path.startswith("payload/models/") for path in model_paths)

from __future__ import annotations

from pathlib import Path

import pytest
from prithvi_payload.inference import _environment_flag
from prithvi_payload.service import JobRequest, _pipeline_timings, _safe_relative
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

    timing = _pipeline_timings(result)

    assert timing["cloud_stage_seconds"] == 1.0
    assert timing["cloud_inference_seconds"] == 0.7
    assert timing["crop_inference_seconds"] == 0.25
    assert timing["downlink_packaging_seconds"] == 0.2

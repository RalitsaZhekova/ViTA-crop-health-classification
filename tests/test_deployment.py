from __future__ import annotations

from pathlib import Path

import pytest
from cloud_detection.config import load_config
from prithvi_payload.cloud_classifier import DEFAULT_CLOUD_CONFIG
from prithvi_payload.deployment_check import _paths_from_environment
from prithvi_payload.inference import _environment_flag
from prithvi_payload.service import (
    JobRequest,
    _balkan_cloud_profile_inputs,
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
            "cloud": {
                "runtime": {
                    "seconds": 1.0,
                    "inference_seconds": 0.7,
                    "input_preparation_seconds": 0.08,
                }
            },
            "crop": {
                "runtime": {
                    "seconds": 0.5,
                    "inference_seconds": 0.25,
                    "overlapped_product_preparation_seconds": 0.12,
                    "tile_preparation_seconds": 0.1,
                }
            },
            "condition": {
                "runtime": {"seconds": 0.3, "metric_summary_seconds": 0.04}
            },
            "downlink": {"runtime": {"seconds": 0.2}},
        },
    }

    timing = _pipeline_timings(result, payload_seconds=2.5)

    assert timing["cloud_stage_seconds"] == 1.0
    assert timing["cloud_inference_seconds"] == 0.7
    assert timing["cloud_input_preparation_seconds"] == 0.08
    assert timing["crop_inference_seconds"] == 0.25
    assert timing["crop_overlapped_product_preparation_seconds"] == 0.12
    assert timing["crop_tile_preparation_seconds"] == 0.1
    assert timing["condition_metric_summary_seconds"] == 0.04
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


def test_balkan_cloud_qualification_input_is_separate_from_crop_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "VITA_BALKAN_PREPARE_INPUTS",
        "balkan1/preprocessed/3370_L1ORT.tif,balkan1/preprocessed/3408_L1ORT.tif",
    )
    monkeypatch.setenv(
        "VITA_BALKAN_CLOUD_PROFILE_INPUTS",
        "balkan1/preprocessed/3458_L1ORT.tif",
    )
    assert _balkan_cloud_profile_inputs() == (
        "balkan1/preprocessed/3458_L1ORT.tif",
    )


def test_balkan_cloud_qualification_rejects_crop_parity_duplicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VITA_BALKAN_PREPARE_INPUTS", "same.tif")
    monkeypatch.setenv("VITA_BALKAN_CLOUD_PROFILE_INPUTS", "same.tif")
    with pytest.raises(RuntimeError, match="must not duplicate"):
        _balkan_cloud_profile_inputs()


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


def test_default_cloud_configuration_is_installed_package_data() -> None:
    assert DEFAULT_CLOUD_CONFIG.is_file()
    assert DEFAULT_CLOUD_CONFIG.parts[-3:] == (
        "cloud_detection",
        "configs",
        "cloud_detector.yaml",
    )
    assert load_config(DEFAULT_CLOUD_CONFIG)["model"]["name"] == "omnicloudmask_v4"


def test_direct_tensorrt_is_offline_and_does_not_import_torch_tensorrt() -> None:
    root = Path(__file__).resolve().parents[1]
    runtime_sources = [
        root / "payload/src/prithvi_payload/inference.py",
        root / "payload/src/prithvi_payload/service.py",
        root / "payload/src/prithvi_payload/tensorrt_runtime.py",
        root / "payload/src/cloud_detection/tensorrt_backend.py",
    ]
    assert all("torch_tensorrt" not in path.read_text(encoding="utf-8") for path in runtime_sources)

    builder = (root / "payload/src/prithvi_payload/tensorrt_builder.py").read_text(
        encoding="utf-8"
    )
    assert "torch.onnx.export" in builder
    assert '"--stronglyTyped"' in builder
    assert 'arguments.append("--fp16")' in builder
    assert '"--builderOptimizationLevel"' in builder
    assert '"--skipInference"' in builder
    assert "_parse_onnx_with_tensorrt" in builder
    assert "zero-sized initializer" in builder
    assert "zero-sized Constant" in builder
    assert "_normalize_onnxscript_integer_attributes(program)" in builder
    assert "_canonicalize_crop_onnx(path)" in builder
    assert "_canonicalize_cloud_onnx(path)" in builder
    build_body = builder.split("def build(", maxsplit=1)[1]
    assert build_body.index("_release_cuda_memory()") < build_body.index(
        "built_crop = _build_plan"
    )
    assert build_body.index("_validate_crop_parity(") < build_body.index(
        "built_cloud = ["
    )
    assert build_body.index("_validate_crop_parity(") < build_body.index(
        "write_manifest_atomic("
    )
    assert build_body.index("_validate_cloud_parity(") < build_body.index(
        "write_manifest_atomic("
    )

    service = (root / "payload/src/prithvi_payload/service.py").read_text(encoding="utf-8")
    assert "torch.onnx.export" not in service
    assert "trtexec" not in service


def test_payload_env_matches_the_production_crop_acceleration_contract() -> None:
    environment = {}
    path = Path(__file__).resolve().parents[1] / "deploy/payload.env.example"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            name, value = line.split("=", maxsplit=1)
            environment[name] = value

    assert environment["VITA_CROP_BACKEND"] == "tensorrt"
    assert environment["VITA_REPLACE_TRT_ARTIFACTS"] == "0"
    assert environment["VITA_CROP_BATCH_SIZE"] == "16"
    assert environment["VITA_CROP_TRT_PRECISION"] == "mixed-fp16"
    assert environment["VITA_CLOUD_BACKEND"] == "tensorrt"
    assert environment["VITA_CLOUD_INFERENCE_DTYPE"] == "fp16"
    assert environment["VITA_CLOUD_TRT_PRECISION"] == "fp16"
    assert environment["VITA_CLOUD_SENTINEL_TRT_PRECISION"] == "fp32"
    assert environment["VITA_TRT_BUILDER_OPTIMIZATION_LEVEL"] == "5"
    assert environment["VITA_CLOUD_WARMUP_PATCH_SIZES"] == "845,869,891,1000"
    assert environment["VITA_SKIP_PERFORMANCE_ACCEPTANCE"] == "0"
    compose = (
        Path(__file__).resolve().parents[1] / "deploy/compose.payload.yaml"
    ).read_text(encoding="utf-8")
    assert "VITA_PERFORMANCE_REPORT: /runtime/performance-acceptance.json" in compose


def test_deploy_builds_plans_before_starting_the_service() -> None:
    deploy = (
        Path(__file__).resolve().parents[1] / "deploy/payload/deploy.sh"
    ).read_text(encoding="utf-8")
    builder = "python payload -m prithvi_payload.tensorrt_builder"
    assert builder in deploy
    stop = 'docker compose "${compose_args[@]}" stop payload'
    assert stop in deploy
    assert deploy.index(stop) < deploy.index(builder)
    replacement = "replace_direct_vita_tensorrt_artifacts"
    assert replacement in deploy
    assert deploy.index(stop) < deploy.rindex(replacement) < deploy.index(builder)
    remove_container = 'docker compose "${compose_args[@]}" rm --force --stop payload'
    assert remove_container in deploy
    assert deploy.index(stop) < deploy.index(remove_container) < deploy.index(builder)
    assert deploy.index(builder) < deploy.index('docker compose "${compose_args[@]}" up -d')
    assert 'fail_startup "payload acceleration acceptance failed"' in deploy
    assert 'fail_startup "payload two-second performance acceptance failed"' in deploy


def test_existing_payload_image_uses_the_small_code_overlay() -> None:
    root = Path(__file__).resolve().parents[1]
    deploy = (root / "deploy/payload/deploy.sh").read_text(encoding="utf-8")
    overlay = root / "deploy/Dockerfile.payload-overlay"

    assert overlay.is_file()
    assert "FROM ${VITA_PAYLOAD_OVERLAY_BASE_IMAGE}" in overlay.read_text(
        encoding="utf-8"
    )
    assert "docker image inspect" in deploy
    assert "--file deploy/Dockerfile.payload-overlay" in deploy


def test_ground_demo_reports_payload_stage_timings() -> None:
    wrapper = (
        Path(__file__).resolve().parents[1]
        / "scripts/ground/Invoke-VitaPayload.ps1"
    ).read_text(encoding="utf-8")

    assert "pipeline_timing_seconds = $response.pipeline_timing_seconds" in wrapper

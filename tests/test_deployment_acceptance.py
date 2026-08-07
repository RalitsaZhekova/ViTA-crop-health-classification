from __future__ import annotations

import pytest
from prithvi_payload.deployment_acceptance import validate_health


def _health() -> dict:
    return {
        "status": "ready",
        "stack": {
            "cuda_available": True,
            "gpu": "Orin",
            "crop_backend": "tensorrt",
            "tensorrt_cudagraphs": True,
            "crop_tensorrt_engine_count": 2,
            "crop_tensorrt_parity": {
                "class_mismatch_fraction": 0.0,
                "mean_absolute_probability_error": 0.0,
            },
            "crop_batch_size": 16,
            "cloud_backend": "omnicloudmask_tensorrt_fp16",
            "cloud_batch_size": 4,
            "cloud_tensorrt_engine_count": 4,
            "cloud_tensorrt_profiles": [
                {"engine_count": 1, "class_mismatch_fraction": 0.0}
            ],
            "cloud_scene_warmup_profiles": [
                {"input": f"scene-{index}.tif", "prediction_retained": False}
                for index in range(4)
            ],
            "balkan_analysis_caches": [{"input": "3370.tif"}, {"input": "3408.tif"}],
        },
    }


def _production_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "VITA_CLOUD_TRT_REQUIRE_FULL",
        "VITA_CLOUD_TRT_VALIDATE_WARMUP_CALLS",
        "VITA_COMPACT_PAYLOAD_PIPELINE",
        "VITA_CONDITION_EXACT_PERCENTILES",
        "VITA_CROP_IN_MEMORY",
        "VITA_DOWNLINK_GRID_IN_MEMORY",
        "VITA_TRT_CUDAGRAPHS",
    ):
        monkeypatch.setenv(name, "1")


def test_jetson_acceptance_requires_both_tensorrt_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _production_flags(monkeypatch)
    accepted = validate_health(_health())
    assert accepted["status"] == "JETSON_ACCELERATION_READY"
    assert accepted["warmed_scene_count"] == 4


def test_jetson_acceptance_rejects_cloud_pytorch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _production_flags(monkeypatch)
    health = _health()
    health["stack"]["cloud_backend"] = "omnicloudmask_cuda_fp16"
    with pytest.raises(RuntimeError, match="Cloud inference"):
        validate_health(health)


def test_jetson_acceptance_rejects_crop_parity_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _production_flags(monkeypatch)
    health = _health()
    health["stack"]["crop_tensorrt_parity"]["class_mismatch_fraction"] = 0.01
    with pytest.raises(RuntimeError, match="Crop TensorRT failed class mismatch"):
        validate_health(health)

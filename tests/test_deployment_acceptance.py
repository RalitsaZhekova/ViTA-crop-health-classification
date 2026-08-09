from __future__ import annotations

import pytest
from prithvi_payload.deployment_acceptance import validate_health


def _health() -> dict:
    return {
        "status": "ready",
        "stack": {
            "cuda_available": True,
            "gpu": "Orin",
            "crop_backend": "pytorch",
            "crop_device": "cuda",
            "crop_inference_dtype": "fp32",
            "crop_tf32": False,
            "tensorrt_cudagraphs": False,
            "crop_tensorrt_engine_count": 0,
            "crop_tensorrt_precision": None,
            "crop_tensorrt_tf32": None,
            "crop_tensorrt_parity": {},
            "crop_batch_size": 16,
            "cloud_backend": "omnicloudmask_cuda_fp16",
            "cloud_batch_size": 4,
            "cloud_tensorrt_engine_count": 0,
            "cloud_tensorrt_profiles": [],
            "cloud_warmup_profiles": [
                {"batch_size": 1, "patch_size": 869},
                {"batch_size": 1, "patch_size": 1000},
                {"batch_size": 4, "patch_size": 869},
            ],
            "cloud_scene_warmup_profiles": [
                {
                    "kind": "fixed_input_profile",
                    "input": f"scene-{index}.tif",
                    "prediction_retained": False,
                }
                for index in range(4)
            ],
            "balkan_analysis_caches": [{"input": "3370.tif"}, {"input": "3408.tif"}],
        },
    }


def _production_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "VITA_COMPACT_PAYLOAD_PIPELINE",
        "VITA_CONDITION_EXACT_PERCENTILES",
        "VITA_CROP_IN_MEMORY",
        "VITA_DOWNLINK_GRID_IN_MEMORY",
    ):
        monkeypatch.setenv(name, "1")


def test_jetson_acceptance_requires_native_cuda_for_both_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _production_flags(monkeypatch)
    accepted = validate_health(_health())
    assert accepted["status"] == "JETSON_ACCELERATION_READY"
    assert accepted["warmed_scene_count"] == 4


def test_jetson_acceptance_rejects_cloud_tensorrt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _production_flags(monkeypatch)
    health = _health()
    health["stack"]["cloud_backend"] = "omnicloudmask_tensorrt_fp16"
    with pytest.raises(RuntimeError, match="Cloud inference"):
        validate_health(health)


def test_jetson_acceptance_rejects_cloud_tensorrt_engines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _production_flags(monkeypatch)
    health = _health()
    health["stack"]["cloud_tensorrt_engine_count"] = 1
    health["stack"]["cloud_tensorrt_profiles"] = [{"engine_count": 1}]
    with pytest.raises(RuntimeError, match="engine partitions"):
        validate_health(health)


def test_jetson_acceptance_rejects_tensorrt_cuda_graphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _production_flags(monkeypatch)
    health = _health()
    health["stack"]["tensorrt_cudagraphs"] = True
    with pytest.raises(RuntimeError, match="CUDA graph replay"):
        validate_health(health)


def test_jetson_acceptance_requires_explicit_empty_tensorrt_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _production_flags(monkeypatch)
    health = _health()
    del health["stack"]["cloud_tensorrt_profiles"]
    with pytest.raises(RuntimeError, match="profiles are still active"):
        validate_health(health)


def test_jetson_acceptance_rejects_crop_tensorrt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _production_flags(monkeypatch)
    health = _health()
    health["stack"]["crop_backend"] = "tensorrt"
    health["stack"]["crop_tensorrt_engine_count"] = 1
    with pytest.raises(RuntimeError, match="accepted PyTorch graph"):
        validate_health(health)


def test_jetson_acceptance_rejects_cpu_crop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _production_flags(monkeypatch)
    health = _health()
    health["stack"]["crop_device"] = "cpu"
    with pytest.raises(RuntimeError, match="not running on CUDA"):
        validate_health(health)


def test_jetson_acceptance_rejects_tf32_crop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _production_flags(monkeypatch)
    health = _health()
    health["stack"]["crop_tf32"] = True
    with pytest.raises(RuntimeError, match="TF32 enabled"):
        validate_health(health)

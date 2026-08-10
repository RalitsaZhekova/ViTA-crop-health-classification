from __future__ import annotations

import pytest
from prithvi_payload.deployment_acceptance import validate_health


def _health() -> dict:
    return {
        "status": "ready",
        "stack": {
            "cuda_available": True,
            "gpu": "Orin",
            "tensorrt_runtime": "native-python",
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
            "cloud_tensorrt_parity": {},
            "tensorrt_manifest_sha256": None,
            "cloud_warmup_profiles": [
                {"batch_size": 1, "patch_size": 700},
                {"batch_size": 1, "patch_size": 869},
                {"batch_size": 1, "patch_size": 891},
                {"batch_size": 1, "patch_size": 1000},
                {"batch_size": 4, "patch_size": 700},
                {"batch_size": 4, "patch_size": 869},
                {"batch_size": 4, "patch_size": 891},
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
    with pytest.raises(RuntimeError, match="engines are active"):
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
    with pytest.raises(RuntimeError, match="engines are active"):
        validate_health(health)


def test_jetson_acceptance_rejects_crop_tensorrt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _production_flags(monkeypatch)
    health = _health()
    health["stack"]["crop_backend"] = "tensorrt"
    health["stack"]["crop_tensorrt_engine_count"] = 1
    with pytest.raises(RuntimeError, match="requested accepted backend"):
        validate_health(health)


def test_jetson_acceptance_accepts_checksum_bound_direct_tensorrt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _production_flags(monkeypatch)
    monkeypatch.setenv("VITA_CROP_BACKEND", "tensorrt")
    monkeypatch.setenv("VITA_CLOUD_BACKEND", "tensorrt")
    monkeypatch.setenv("VITA_CROP_TRT_PRECISION", "mixed-fp16")
    monkeypatch.setenv("VITA_CLOUD_INFERENCE_DTYPE", "fp16")
    health = _health()
    stack = health["stack"]
    stack.update(
        {
            "crop_backend": "tensorrt",
            "crop_inference_dtype": "mixed-fp16",
            "crop_tensorrt_engine_count": 1,
            "crop_tensorrt_precision": "mixed-fp16",
            "crop_tensorrt_tf32": False,
            "crop_tensorrt_parity": {
                "class_mismatch_fraction": 0.0005,
                "mean_absolute_probability_error": 0.001,
            },
            "cloud_backend": "omnicloudmask_tensorrt_fp16",
            "cloud_tensorrt_engine_count": 3,
            "cloud_tensorrt_profiles": [
                {
                    "patch_size": 700,
                    "minimum_batch_size": 1,
                    "maximum_batch_size": 1,
                },
                {
                    "patch_size": 869,
                    "minimum_batch_size": 1,
                    "maximum_batch_size": 4,
                },
                {
                    "patch_size": 891,
                    "minimum_batch_size": 1,
                    "maximum_batch_size": 4,
                },
            ],
            "cloud_tensorrt_parity": {
                "class_mismatch_fraction": 0.0002,
                "scenes": [{"input": f"scene-{index}.tif"} for index in range(4)],
            },
            "tensorrt_manifest_sha256": "a" * 64,
            "cloud_warmup_profiles": [
                {"batch_size": 1, "patch_size": 700},
                {"batch_size": 1, "patch_size": 869},
                {"batch_size": 1, "patch_size": 891},
                {"batch_size": 4, "patch_size": 869},
                {"batch_size": 4, "patch_size": 891},
            ],
        }
    )

    accepted = validate_health(health)

    assert accepted["crop_backend"] == "tensorrt"
    assert accepted["cloud_tensorrt_engine_count"] == 3
    assert accepted["cloud_profile_count"] == 3


def test_jetson_acceptance_rejects_direct_tensorrt_parity_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _production_flags(monkeypatch)
    monkeypatch.setenv("VITA_CROP_BACKEND", "tensorrt")
    monkeypatch.setenv("VITA_CLOUD_BACKEND", "tensorrt")
    health = _health()
    health["stack"]["crop_backend"] = "tensorrt"
    health["stack"]["crop_inference_dtype"] = "mixed-fp16"
    health["stack"]["crop_tensorrt_engine_count"] = 1
    health["stack"]["crop_tensorrt_precision"] = "mixed-fp16"
    health["stack"]["crop_tensorrt_tf32"] = False
    health["stack"]["cloud_backend"] = "omnicloudmask_tensorrt_fp16"
    health["stack"]["crop_tensorrt_parity"] = {
        "class_mismatch_fraction": 0.003,
        "mean_absolute_probability_error": 0.001,
    }

    with pytest.raises(RuntimeError, match="decision parity"):
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

"""Fail-closed readiness acceptance for the production Jetson payload."""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any
from urllib.request import urlopen

from prithvi_payload.runtime_config import environment_flag


def validate_health(health: dict[str, Any]) -> dict[str, Any]:
    if health.get("status") != "ready":
        raise RuntimeError("Payload health status is not ready")
    stack = health.get("stack")
    if not isinstance(stack, dict):
        raise RuntimeError("Payload health has no stack record")
    if stack.get("cuda_available") is not True:
        raise RuntimeError("CUDA is not available in the payload service")
    if stack.get("crop_backend") != "tensorrt":
        raise RuntimeError("Crop inference is not using TensorRT")
    if stack.get("tensorrt_cudagraphs") is not True:
        raise RuntimeError("TensorRT CUDA graph replay is not enabled")
    if int(stack.get("crop_tensorrt_engine_count", 0)) < 1:
        raise RuntimeError("Crop inference has no TensorRT engine partitions")
    expected_crop_precision = os.environ.get(
        "VITA_CROP_TRT_PRECISION", "fp32"
    ).strip().casefold()
    if stack.get("crop_tensorrt_precision") != expected_crop_precision:
        raise RuntimeError("Crop TensorRT engine uses the wrong precision")
    if int(stack.get("crop_batch_size", 0)) != int(
        os.environ.get("VITA_CROP_BATCH_SIZE", "16")
    ):
        raise RuntimeError("Crop TensorRT engine uses the wrong fixed batch size")
    crop_parity = stack.get("crop_tensorrt_parity")
    if not isinstance(crop_parity, dict):
        raise RuntimeError("Crop TensorRT has no PyTorch parity record")
    crop_mismatch = crop_parity.get("class_mismatch_fraction")
    crop_mean_error = crop_parity.get("mean_absolute_probability_error")
    maximum_crop_mismatch = float(
        os.environ.get("VITA_CROP_TRT_MAX_CLASS_MISMATCH", "0.002")
    )
    maximum_crop_mean_error = float(
        os.environ.get("VITA_CROP_TRT_MAX_MEAN_PROBABILITY_ERROR", "0.005")
    )
    for value, maximum, label in (
        (crop_mismatch, maximum_crop_mismatch, "class mismatch"),
        (crop_mean_error, maximum_crop_mean_error, "mean probability error"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) > maximum
        ):
            raise RuntimeError(f"Crop TensorRT failed {label} parity")
    if stack.get("cloud_backend") != "omnicloudmask_tensorrt_fp16":
        raise RuntimeError("Cloud inference is not using FP16 TensorRT")
    if int(stack.get("cloud_batch_size", 0)) != int(
        os.environ.get("VITA_CLOUD_BATCH_SIZE", "4")
    ):
        raise RuntimeError("Cloud TensorRT uses the wrong fixed batch size")
    if int(stack.get("cloud_tensorrt_engine_count", 0)) < 1:
        raise RuntimeError("Cloud inference has no TensorRT engines")

    profiles = stack.get("cloud_tensorrt_profiles")
    if not isinstance(profiles, list) or not profiles:
        raise RuntimeError("Cloud TensorRT has no validated static profiles")
    maximum_mismatch = float(
        os.environ.get("VITA_CLOUD_TRT_MAX_CLASS_MISMATCH", "0.001")
    )
    for profile in profiles:
        if not isinstance(profile, dict) or int(profile.get("engine_count", 0)) < 1:
            raise RuntimeError("A cloud profile has no TensorRT engine")
        mismatch = profile.get("class_mismatch_fraction")
        if (
            isinstance(mismatch, bool)
            or not isinstance(mismatch, (int, float))
            or not math.isfinite(float(mismatch))
            or float(mismatch) > maximum_mismatch
        ):
            raise RuntimeError("A cloud TensorRT profile failed class parity")

    scene_profiles = stack.get("cloud_scene_warmup_profiles")
    if not isinstance(scene_profiles, list) or len(scene_profiles) != 4:
        raise RuntimeError("Exactly four payload-local demo scenes must be warmed")
    scene_inputs = [profile.get("input") for profile in scene_profiles]
    if len(set(scene_inputs)) != 4 or any(not value for value in scene_inputs):
        raise RuntimeError("Demo-scene warmup inputs must be distinct")
    if any(profile.get("prediction_retained") is not False for profile in scene_profiles):
        raise RuntimeError("Startup must not retain cached demo predictions")

    balkan_caches = stack.get("balkan_analysis_caches")
    if not isinstance(balkan_caches, list) or len(balkan_caches) != 2:
        raise RuntimeError("Exactly two Balkan analysis grids must be prepared")

    required_flags = (
        "VITA_CLOUD_TRT_REQUIRE_FULL",
        "VITA_CLOUD_TRT_VALIDATE_WARMUP_CALLS",
        "VITA_COMPACT_PAYLOAD_PIPELINE",
        "VITA_CONDITION_EXACT_PERCENTILES",
        "VITA_CROP_IN_MEMORY",
        "VITA_DOWNLINK_GRID_IN_MEMORY",
        "VITA_TRT_CUDAGRAPHS",
    )
    for name in required_flags:
        if not environment_flag(name, False):
            raise RuntimeError(f"Production acceptance requires {name}=1")

    return {
        "status": "JETSON_ACCELERATION_READY",
        "gpu": stack.get("gpu"),
        "crop_tensorrt_engine_count": stack["crop_tensorrt_engine_count"],
        "crop_tensorrt_precision": stack["crop_tensorrt_precision"],
        "crop_tensorrt_parity": crop_parity,
        "cloud_tensorrt_engine_count": stack["cloud_tensorrt_engine_count"],
        "cloud_profile_count": len(profiles),
        "warmed_scene_count": len(scene_profiles),
        "prepared_balkan_scene_count": len(balkan_caches),
        "target_payload_seconds": 2.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8090/healthz")
    args = parser.parse_args()
    with urlopen(args.url, timeout=10) as response:  # noqa: S310 - fixed loopback URL
        health = json.load(response)
    print(json.dumps(validate_health(health), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

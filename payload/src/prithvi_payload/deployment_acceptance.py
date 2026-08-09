"""Fail-closed readiness acceptance for the production Jetson payload."""

from __future__ import annotations

import argparse
import json
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
    if stack.get("crop_backend") != "pytorch":
        raise RuntimeError("Crop inference is not using the accepted PyTorch graph")
    if stack.get("crop_device") != "cuda":
        raise RuntimeError("Crop inference is not running on CUDA")
    if stack.get("crop_inference_dtype") != "fp32":
        raise RuntimeError("Crop inference is not using the accepted FP32 graph")
    if stack.get("crop_tf32") is not False:
        raise RuntimeError("Crop inference has TF32 enabled")
    if stack.get("crop_tensorrt_engine_count") != 0:
        raise RuntimeError("Rejected crop TensorRT engine partitions are still active")
    if stack.get("tensorrt_cudagraphs") is not False:
        raise RuntimeError("Rejected TensorRT CUDA graph replay is still enabled")
    if int(stack.get("crop_batch_size", 0)) != int(
        os.environ.get("VITA_CROP_BATCH_SIZE", "16")
    ):
        raise RuntimeError("Crop CUDA inference uses the wrong fixed batch size")
    if stack.get("crop_tensorrt_parity") != {}:
        raise RuntimeError("Rejected crop TensorRT calibration is still active")
    if stack.get("cloud_backend") != "omnicloudmask_cuda_fp16":
        raise RuntimeError("Cloud inference is not using the accepted FP16 CUDA graph")
    if int(stack.get("cloud_batch_size", 0)) != int(
        os.environ.get("VITA_CLOUD_BATCH_SIZE", "4")
    ):
        raise RuntimeError("Cloud CUDA inference uses the wrong fixed batch size")
    if stack.get("cloud_tensorrt_engine_count") != 0:
        raise RuntimeError("Rejected cloud TensorRT engine partitions are still active")

    profiles = stack.get("cloud_tensorrt_profiles")
    if profiles != []:
        raise RuntimeError("Rejected cloud TensorRT profiles are still active")

    warmup_profiles = stack.get("cloud_warmup_profiles")
    expected_cloud_warmups = {(1, 1000), (1, 869), (stack["cloud_batch_size"], 869)}
    if not isinstance(warmup_profiles, list) or any(
        not isinstance(profile, dict) for profile in warmup_profiles
    ):
        raise RuntimeError("Cloud CUDA warmup profiles do not match production shapes")
    if {
        (profile.get("batch_size"), profile.get("patch_size"))
        for profile in warmup_profiles
    } != expected_cloud_warmups:
        raise RuntimeError("Cloud CUDA warmup profiles do not match production shapes")

    scene_profiles = stack.get("cloud_scene_warmup_profiles")
    if not isinstance(scene_profiles, list) or len(scene_profiles) != 4:
        raise RuntimeError("Exactly four payload-local demo scenes must be warmed")
    scene_inputs = [profile.get("input") for profile in scene_profiles]
    if len(set(scene_inputs)) != 4 or any(not value for value in scene_inputs):
        raise RuntimeError("Demo-scene warmup inputs must be distinct")
    if any(profile.get("kind") != "fixed_input_profile" for profile in scene_profiles):
        raise RuntimeError("Demo-scene warmups must exercise fixed input profiles")
    if any(profile.get("prediction_retained") is not False for profile in scene_profiles):
        raise RuntimeError("Startup must not retain cached demo predictions")

    balkan_caches = stack.get("balkan_analysis_caches")
    if not isinstance(balkan_caches, list) or len(balkan_caches) != 2:
        raise RuntimeError("Exactly two Balkan analysis grids must be prepared")

    required_flags = (
        "VITA_COMPACT_PAYLOAD_PIPELINE",
        "VITA_CONDITION_EXACT_PERCENTILES",
        "VITA_CROP_IN_MEMORY",
        "VITA_DOWNLINK_GRID_IN_MEMORY",
    )
    for name in required_flags:
        if not environment_flag(name, False):
            raise RuntimeError(f"Production acceptance requires {name}=1")

    return {
        "status": "JETSON_ACCELERATION_READY",
        "gpu": stack.get("gpu"),
        "crop_backend": stack["crop_backend"],
        "crop_device": stack["crop_device"],
        "crop_inference_dtype": stack["crop_inference_dtype"],
        "crop_tf32": stack["crop_tf32"],
        "crop_batch_size": stack["crop_batch_size"],
        "cloud_backend": stack["cloud_backend"],
        "cloud_batch_size": stack["cloud_batch_size"],
        "cloud_tensorrt_engine_count": stack["cloud_tensorrt_engine_count"],
        "cloud_profile_count": 0,
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

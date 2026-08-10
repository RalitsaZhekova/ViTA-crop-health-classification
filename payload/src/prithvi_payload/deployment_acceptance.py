"""Fail-closed readiness acceptance for the production Jetson payload."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any
from urllib.request import urlopen

from cloud_detection.tensorrt_backend import (
    CLOUD_MAX_AGGREGATE_CLASS_MISMATCH,
    CLOUD_MAX_SCENE_CLASS_MISMATCH,
    REVIEWED_CLOUD_BASE_PATCH_SIZE,
    REVIEWED_CLOUD_SCENE_BATCH_SIZES,
    REVIEWED_CLOUD_SCENE_PATCH_SIZES,
)

from prithvi_payload.runtime_config import environment_flag


def validate_health(health: dict[str, Any]) -> dict[str, Any]:
    if health.get("status") != "ready":
        raise RuntimeError("Payload health status is not ready")
    stack = health.get("stack")
    if not isinstance(stack, dict):
        raise RuntimeError("Payload health has no stack record")
    if stack.get("cuda_available") is not True:
        raise RuntimeError("CUDA is not available in the payload service")
    requested_backend = os.environ.get("VITA_CROP_BACKEND", "pytorch").strip().casefold()
    if requested_backend not in {"pytorch", "tensorrt"}:
        raise RuntimeError("VITA_CROP_BACKEND must be pytorch or tensorrt")
    if stack.get("crop_backend") != requested_backend:
        raise RuntimeError("Crop inference is not using the requested accepted backend")
    if stack.get("crop_device") != "cuda":
        raise RuntimeError("Crop inference is not running on CUDA")
    expected_crop_precision = (
        "fp32"
        if requested_backend == "pytorch"
        else os.environ.get(
            "VITA_CROP_TRT_PRECISION", "mixed-fp16"
        ).strip().casefold()
    )
    if stack.get("crop_inference_dtype") != expected_crop_precision:
        raise RuntimeError("Crop inference is not using the accepted precision")
    if stack.get("crop_tf32") is not False:
        raise RuntimeError("Crop inference has TF32 enabled")
    if stack.get("tensorrt_cudagraphs") is not False:
        raise RuntimeError("TensorRT CUDA graph replay must remain disabled")
    if int(stack.get("crop_batch_size", 0)) != int(
        os.environ.get("VITA_CROP_BATCH_SIZE", "16")
    ):
        raise RuntimeError("Crop CUDA inference uses the wrong fixed batch size")
    requested_cloud_backend = os.environ.get(
        "VITA_CLOUD_BACKEND", "pytorch"
    ).strip().casefold()
    if requested_cloud_backend != requested_backend:
        raise RuntimeError("Crop and cloud execution modes do not match")
    cloud_precision = os.environ.get(
        "VITA_CLOUD_INFERENCE_DTYPE", "fp16"
    ).strip().casefold()
    expected_cloud_backend = (
        f"omnicloudmask_tensorrt_{cloud_precision}"
        if requested_backend == "tensorrt"
        else f"omnicloudmask_cuda_{cloud_precision}"
    )
    if stack.get("cloud_backend") != expected_cloud_backend:
        raise RuntimeError("Cloud inference is not using the requested accepted backend")
    if int(stack.get("cloud_batch_size", 0)) != int(
        os.environ.get("VITA_CLOUD_BATCH_SIZE", "4")
    ):
        raise RuntimeError("Cloud CUDA inference uses the wrong fixed batch size")
    profiles = stack.get("cloud_tensorrt_profiles")
    if requested_backend == "pytorch":
        if stack.get("crop_tensorrt_engine_count") != 0:
            raise RuntimeError("Crop TensorRT engines are active in PyTorch mode")
        if stack.get("crop_tensorrt_parity") != {}:
            raise RuntimeError("Crop TensorRT parity is active in PyTorch mode")
        if stack.get("cloud_tensorrt_engine_count") != 0 or profiles != []:
            raise RuntimeError("Cloud TensorRT engines are active in PyTorch mode")
    else:
        if stack.get("tensorrt_runtime") != "native-python":
            raise RuntimeError("Direct TensorRT is not using the native Python runtime")
        if stack.get("crop_tensorrt_engine_count") != 1:
            raise RuntimeError("Direct crop TensorRT must load exactly one engine")
        if stack.get("crop_tensorrt_precision") != expected_crop_precision:
            raise RuntimeError("Direct crop TensorRT precision does not match acceptance")
        if stack.get("crop_tensorrt_tf32") is not False:
            raise RuntimeError("Direct crop TensorRT must disable TF32")
        crop_parity = stack.get("crop_tensorrt_parity")
        if not isinstance(crop_parity, dict):
            raise RuntimeError("Direct crop TensorRT parity record is missing")
        if float(crop_parity.get("class_mismatch_fraction", 1.0)) > 0.002:
            raise RuntimeError("Direct crop TensorRT exceeds the decision parity gate")
        if float(crop_parity.get("mean_absolute_probability_error", 1.0)) > 0.005:
            raise RuntimeError("Direct crop TensorRT exceeds the probability parity gate")
        expected_profile_count = len(REVIEWED_CLOUD_SCENE_PATCH_SIZES)
        if stack.get("cloud_tensorrt_engine_count") != expected_profile_count:
            raise RuntimeError(
                f"Direct cloud TensorRT must load exactly {expected_profile_count} engines"
            )
        if not isinstance(profiles, list) or len(profiles) != expected_profile_count:
            raise RuntimeError("Direct cloud TensorRT profile record is incomplete")
        profile_contract = {
            (
                profile.get("patch_size"),
                profile.get("minimum_batch_size"),
                profile.get("maximum_batch_size"),
            )
            for profile in profiles
            if isinstance(profile, dict)
        }
        expected_profile_contract = {
            (patch_size, 1, batch_size)
            for patch_size, batch_size in REVIEWED_CLOUD_SCENE_BATCH_SIZES
        }
        if profile_contract != expected_profile_contract:
            raise RuntimeError("Direct cloud TensorRT profiles do not match payload shapes")
        cloud_parity = stack.get("cloud_tensorrt_parity")
        if not isinstance(cloud_parity, dict) or float(
            cloud_parity.get("class_mismatch_fraction", 1.0)
        ) > CLOUD_MAX_AGGREGATE_CLASS_MISMATCH:
            raise RuntimeError("Direct cloud TensorRT exceeds the class parity gate")
        scenes = cloud_parity.get("scenes")
        if not isinstance(scenes, list) or len(scenes) != 4:
            raise RuntimeError("Direct cloud TensorRT was not accepted on four scenes")
        if any(
            not isinstance(scene, dict)
            or float(scene.get("class_mismatch_fraction", 1.0))
            > CLOUD_MAX_SCENE_CLASS_MISMATCH
            for scene in scenes
        ):
            raise RuntimeError(
                "Direct cloud TensorRT exceeds the per-scene class parity gate"
            )
        manifest_digest = stack.get("tensorrt_manifest_sha256")
        if not isinstance(manifest_digest, str) or len(manifest_digest) != 64:
            raise RuntimeError("Direct TensorRT accepted manifest is missing")

    warmup_profiles = stack.get("cloud_warmup_profiles")
    if requested_backend == "tensorrt":
        expected_cloud_warmups: set[tuple[int, int]] = set()
        for patch_size, batch_size in REVIEWED_CLOUD_SCENE_BATCH_SIZES:
            expected_cloud_warmups.add((1, patch_size))
            expected_cloud_warmups.add((batch_size, patch_size))
    else:
        expected_cloud_warmups = {(1, REVIEWED_CLOUD_BASE_PATCH_SIZE)} | {
            (batch_size, patch_size)
            for patch_size in REVIEWED_CLOUD_SCENE_PATCH_SIZES
            for batch_size in (1, stack["cloud_batch_size"])
        }
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

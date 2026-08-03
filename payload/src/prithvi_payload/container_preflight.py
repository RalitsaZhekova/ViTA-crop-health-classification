"""Safe container startup checks that do not initialize Earth Engine or load models."""

from __future__ import annotations

import argparse
import json
import os
import platform
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import torch
from prithvi_shared import SELECTED_CHECKPOINT_NAME, SELECTED_CHECKPOINT_SHA256

from prithvi_payload.cloud_classifier import (
    CLOUD_MODEL_SHA256,
    cloud_checkpoint_sha256,
    default_cloud_weights_directory,
)
from prithvi_payload.inference import checkpoint_sha256, default_model_directory


def _application_version() -> str:
    try:
        return version("prithvi-crop-payload")
    except PackageNotFoundError:
        return "source-checkout"


def verify_model_artifacts() -> dict[str, str]:
    """Verify the two selected model artifacts through canonical hash functions."""
    crop_directory = default_model_directory()
    architecture = crop_directory / "architecture.yaml"
    crop_checkpoint = crop_directory / SELECTED_CHECKPOINT_NAME
    cloud_directory = default_cloud_weights_directory()
    for artifact in (architecture, crop_checkpoint):
        if not artifact.is_file():
            raise RuntimeError("A required payload model artifact is missing")
    if checkpoint_sha256(crop_checkpoint) != SELECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Selected crop-model checksum mismatch")
    if cloud_checkpoint_sha256(cloud_directory) != CLOUD_MODEL_SHA256:
        raise RuntimeError("Selected cloud-model checksum mismatch")
    return {
        "crop_model_sha256": SELECTED_CHECKPOINT_SHA256,
        "cloud_model_sha256": CLOUD_MODEL_SHA256,
    }


def safe_diagnostics(*, models_only: bool = False) -> dict[str, Any]:
    """Return diagnostics that contain no credentials, tokens, URLs or paths."""
    model_hashes = verify_model_artifacts()
    if models_only:
        return {"status": "ok", **model_hashes}

    cuda_required = os.environ.get("CUDA_REQUIRED", "1") == "1"
    cuda_available = torch.cuda.is_available()
    if cuda_required and not cuda_available:
        raise RuntimeError("CUDA_REQUIRED=1 but PyTorch cannot access CUDA")
    concurrency = os.environ.get("VITA_MAX_CONCURRENT_JOBS", "1")
    if concurrency != "1":
        raise RuntimeError("VITA_MAX_CONCURRENT_JOBS must be 1 for the payload GPU service")
    return {
        "status": "ok",
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_build_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "payload_application_version": _application_version(),
        **model_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the payload container runtime.")
    parser.add_argument("--models-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(safe_diagnostics(models_only=args.models_only), sort_keys=True))


if __name__ == "__main__":
    main()

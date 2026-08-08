"""Canonical cloud-model identity and artifact verification."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cloud_detection.backend import (
    OMNICLOUDMASK_ENSEMBLE_SHA256,
    CloudBackend,
    OmniCloudMaskBackend,
)
from cloud_detection.config import load_config

CLOUD_MODEL_NAME = "omnicloudmask_v4"
CLOUD_MODEL_SHA256 = OMNICLOUDMASK_ENSEMBLE_SHA256

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLOUD_CONFIG = PACKAGE_ROOT / "cloud_detection" / "configs" / "cloud_detector.yaml"


@dataclass(frozen=True)
class CloudModel:
    """Loaded backend plus the exact masking configuration used with it."""

    backend: CloudBackend
    config: dict[str, Any]


def load_cloud_model(config_path: str | Path = DEFAULT_CLOUD_CONFIG) -> CloudModel:
    path = Path(config_path).resolve()
    config = load_config(path)
    configured_weights = Path(config["model"]["weights_folder"])
    if not configured_weights.is_absolute():
        configured_weights = (path.parent / configured_weights).resolve()
    model = config["model"]
    inference_dtype = os.environ.get(
        "VITA_CLOUD_INFERENCE_DTYPE",
        model.get("inference_dtype", "fp32"),
    )
    try:
        batch_size = int(
            os.environ.get(
                "VITA_CLOUD_BATCH_SIZE",
                model.get("batch_size", 1),
            )
        )
    except (TypeError, ValueError) as error:
        raise ValueError("VITA_CLOUD_BATCH_SIZE must be a positive integer") from error
    if batch_size < 1:
        raise ValueError("VITA_CLOUD_BATCH_SIZE must be a positive integer")
    model["inference_dtype"] = inference_dtype
    model["batch_size"] = batch_size
    backend = OmniCloudMaskBackend(
        name=model["name"],
        weights_folder=os.environ.get("OMNICLOUDMASK_MODEL_DIR", str(configured_weights)),
        device=model.get("device", "auto"),
        expected_sha256=model.get("expected_sha256"),
        inference_dtype=inference_dtype,
        patch_size=int(model["patch_size"]),
        patch_overlap=int(model["patch_overlap"]),
        batch_size=batch_size,
    )
    return CloudModel(backend=backend, config=config)

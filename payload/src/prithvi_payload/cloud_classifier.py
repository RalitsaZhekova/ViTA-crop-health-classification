"""Canonical cloud-model identity and artifact verification."""

from __future__ import annotations

import os
from pathlib import Path

from cloud_detection.backend import (
    OMNICLOUDMASK_ENSEMBLE_SHA256,
    ensemble_sha256,
)

CLOUD_MODEL_NAME = "omnicloudmask_v4"
CLOUD_MODEL_SHA256 = OMNICLOUDMASK_ENSEMBLE_SHA256

PAYLOAD_ROOT = Path(__file__).resolve().parents[2]


def default_cloud_weights_directory() -> Path:
    configured = os.environ.get("OMNICLOUDMASK_MODEL_DIR")
    if configured:
        return Path(configured)
    source_checkout = PAYLOAD_ROOT / "models" / "omnicloudmask"
    if source_checkout.parent.is_dir():
        return source_checkout
    return Path.cwd() / "models" / "omnicloudmask"


def cloud_checkpoint_sha256(directory: Path) -> str:
    """Verify and fingerprint the two-component OmniCloudMask ensemble."""
    return ensemble_sha256(directory)

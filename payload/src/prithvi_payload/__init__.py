"""Flight-side crop and cloud classification inference."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from prithvi_payload.cloud_classifier import (
        CLOUD_BANDS,
        CLOUD_CLASS_NAMES,
        CloudClassification,
        PayloadCloudClassifier,
    )
    from prithvi_payload.inference import InferenceOutput, PayloadCropModel

__all__ = [
    "CLOUD_BANDS",
    "CLOUD_CLASS_NAMES",
    "CloudClassification",
    "InferenceOutput",
    "PayloadCloudClassifier",
    "PayloadCropModel",
]


def __getattr__(name: str) -> Any:
    """Load crop and cloud dependencies only when their public API is used."""
    if name in {
        "CLOUD_BANDS",
        "CLOUD_CLASS_NAMES",
        "CloudClassification",
        "PayloadCloudClassifier",
    }:
        from prithvi_payload import cloud_classifier

        return getattr(cloud_classifier, name)
    if name in {"InferenceOutput", "PayloadCropModel"}:
        from prithvi_payload import inference

        return getattr(inference, name)
    raise AttributeError(name)

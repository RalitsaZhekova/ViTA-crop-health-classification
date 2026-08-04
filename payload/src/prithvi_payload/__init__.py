"""Flight-side crop inference public API."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from prithvi_payload.inference import InferenceOutput, PayloadCropModel

__all__ = [
    "InferenceOutput",
    "PayloadCropModel",
]


def __getattr__(name: str) -> Any:
    """Load crop-model dependencies only when their public API is used."""
    if name in {"InferenceOutput", "PayloadCropModel"}:
        from prithvi_payload import inference

        return getattr(inference, name)
    raise AttributeError(name)

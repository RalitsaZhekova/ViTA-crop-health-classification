"""Standalone cloud classification and masking pipeline."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .pipeline import CloudDetectionPipeline
    from .types import CloudDetectionResult

__all__ = ["CloudDetectionPipeline", "CloudDetectionResult"]


def __getattr__(name: str) -> Any:
    """Avoid loading GeoTIFF and preview dependencies for classifier-only use."""
    if name == "CloudDetectionPipeline":
        from .pipeline import CloudDetectionPipeline

        return CloudDetectionPipeline
    if name == "CloudDetectionResult":
        from .types import CloudDetectionResult

        return CloudDetectionResult
    raise AttributeError(name)

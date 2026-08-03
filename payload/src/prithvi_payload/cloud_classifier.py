"""Standalone cloud classification; it does not apply a cloud mask."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from cloud_detection.backend import (
    OMNICLOUDMASK_ENSEMBLE_SHA256,
    CloudBackend,
    OmniCloudMaskBackend,
    ensemble_sha256,
)
from cloud_detection.preprocessing import normalize_reflectance, strict_valid_mask
from cloud_detection.tiling import reconstruct, split_tiles

CLOUD_MODEL_NAME = "omnicloudmask_v4"
CLOUD_MODEL_SHA256 = OMNICLOUDMASK_ENSEMBLE_SHA256
CLOUD_BANDS = ("B08", "B04", "B03", "B02")
CLOUD_CLASS_NAMES = ("clear", "thick_cloud", "thin_cloud", "cloud_shadow")
DEFAULT_TILE_SIZE = 1000
DEFAULT_OVERLAP = 300
DEFAULT_REFLECTANCE_SCALE = 10_000.0

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


@dataclass(frozen=True)
class CloudClassification:
    """Raw semantic classification, deliberately separate from masking."""

    semantic_class: np.ndarray
    scores: np.ndarray
    invalid_input: np.ndarray
    class_fractions: dict[str, float]
    score_kind: str
    band_order: tuple[str, ...]
    device: str


class PayloadCloudClassifier:
    """Classify Sentinel-2 or Balkan-1 pixels as clear/cloud/thin-cloud/shadow."""

    def __init__(
        self,
        backend: CloudBackend,
        *,
        tile_size: int = DEFAULT_TILE_SIZE,
        overlap: int = DEFAULT_OVERLAP,
        device: str = "unknown",
    ) -> None:
        if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
            raise ValueError("Invalid tile size or overlap.")
        self.backend = backend
        self.tile_size = tile_size
        self.overlap = overlap
        self.device = device

    @classmethod
    def load(
        cls,
        weights_directory: Path | None = None,
        *,
        device: str = "auto",
        tile_size: int = DEFAULT_TILE_SIZE,
        overlap: int = DEFAULT_OVERLAP,
    ) -> PayloadCloudClassifier:
        weights_directory = weights_directory or default_cloud_weights_directory()
        actual_digest = cloud_checkpoint_sha256(weights_directory)
        if actual_digest != CLOUD_MODEL_SHA256:
            raise RuntimeError(
                "Cloud checkpoint checksum mismatch: "
                f"expected {CLOUD_MODEL_SHA256}, got {actual_digest}"
            )
        backend = OmniCloudMaskBackend(
            CLOUD_MODEL_NAME,
            weights_directory,
            device=device,
            expected_sha256=CLOUD_MODEL_SHA256,
        )
        return cls(
            backend,
            tile_size=tile_size,
            overlap=overlap,
            device=str(backend.device),
        )

    def classify(
        self,
        image: np.ndarray,
        *,
        band_order: tuple[str, ...] | list[str] = CLOUD_BANDS,
        reflectance_scale: float = DEFAULT_REFLECTANCE_SCALE,
        nodata_value: int | float | None = 0,
    ) -> CloudClassification:
        """Classify a co-registered ``[NIR,Red,Green,Blue]`` array.

        The default names describe the retained Sentinel-2 adapter. Balkan-1
        uses the same spectral order through the stage planner. The model then
        applies its own per-patch dynamic normalization.
        """
        supplied_bands = tuple(band_order)
        if supplied_bands != CLOUD_BANDS:
            raise ValueError(
                f"Cloud classifier requires exact band order {CLOUD_BANDS}; "
                f"received {supplied_bands}."
            )
        if image.ndim != 3 or image.shape[0] != len(CLOUD_BANDS):
            raise ValueError(
                f"Expected image shaped (4,height,width), received {image.shape}."
            )
        if image.shape[1] == 0 or image.shape[2] == 0:
            raise ValueError("Image height and width must be non-zero.")

        normalized, invalid = normalize_reflectance(
            image,
            scale=reflectance_scale,
            nodata_value=nodata_value,
        )
        invalid |= ~strict_valid_mask(normalized[[1, 2, 0]])
        normalized[:, invalid] = 0.0
        tiles, windows, original_shape, padded_shape = split_tiles(
            normalized,
            size=self.tile_size,
            overlap=self.overlap,
            padding_mode="reflect",
        )
        predictions = [self.backend.predict(tile) for tile in tiles]
        score_kinds = {prediction.score_kind for prediction in predictions}
        if len(score_kinds) != 1:
            raise RuntimeError(f"Backend returned mixed score kinds: {score_kinds}.")
        scores = reconstruct(
            [prediction.scores for prediction in predictions],
            windows,
            original_shape,
            padded_shape,
        )
        semantic = scores.argmax(axis=0).astype(np.uint8)
        valid = ~invalid
        valid_count = int(valid.sum())
        fractions = {
            name: (
                float(np.count_nonzero((semantic == class_id) & valid) / valid_count)
                if valid_count
                else 0.0
            )
            for class_id, name in enumerate(CLOUD_CLASS_NAMES)
        }
        return CloudClassification(
            semantic_class=semantic,
            scores=scores,
            invalid_input=invalid,
            class_fractions=fractions,
            score_kind=score_kinds.pop(),
            band_order=CLOUD_BANDS,
            device=self.device,
        )

"""Standalone cloud classification; it does not apply a cloud mask."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from cloud_detection.backend import CloudBackend, CloudSEN12Backend
from cloud_detection.preprocessing import normalize_reflectance
from cloud_detection.tiling import reconstruct, split_tiles

CLOUD_MODEL_NAME = "dtacs4bands"
CLOUD_MODEL_SHA256 = "37205adce72fbbb65a3cfa8f47676c84ebf9b1555a27a3838d584072c954b22d"
CLOUD_BANDS = ("B08", "B04", "B03", "B02")
CLOUD_CLASS_NAMES = ("clear", "thick_cloud", "thin_cloud", "cloud_shadow")
DEFAULT_TILE_SIZE = 512
DEFAULT_OVERLAP = 64
DEFAULT_REFLECTANCE_SCALE = 10_000.0

PAYLOAD_ROOT = Path(__file__).resolve().parents[2]


def default_cloud_weights_directory() -> Path:
    configured = os.environ.get("CLOUDSEN12_MODEL_DIR")
    if configured:
        return Path(configured)
    source_checkout = PAYLOAD_ROOT / "models" / "cloudsen12"
    if source_checkout.parent.is_dir():
        return source_checkout
    return Path.cwd() / "models" / "cloudsen12"


def cloud_checkpoint_sha256(path: Path) -> str:
    """Hash the cloud checkpoint without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    """Classify Sentinel-2 L1C pixels as clear/cloud/thin-cloud/shadow."""

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
        model_path = weights_directory / f"{CLOUD_MODEL_NAME}.pt"
        if not model_path.is_file():
            raise FileNotFoundError(
                f"{model_path} is missing. Run "
                "payload/scripts/download_cloud_weights.py while online."
            )
        actual_digest = cloud_checkpoint_sha256(model_path)
        if actual_digest != CLOUD_MODEL_SHA256:
            raise RuntimeError(
                "Cloud checkpoint checksum mismatch: "
                f"expected {CLOUD_MODEL_SHA256}, got {actual_digest}"
            )
        backend = CloudSEN12Backend(
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
        """Classify a co-registered ``[B08,B04,B03,B02]`` L1C array.

        ``reflectance_scale=10000`` accepts Sentinel-2 L1C digital numbers.
        Use ``reflectance_scale=1`` only for values already in TOA reflectance.
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

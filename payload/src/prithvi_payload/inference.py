"""Load the pinned model and run training-free tensor inference."""

from __future__ import annotations

import hashlib
import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from prithvi_shared import (
    CROP_CLASSIFICATION_THRESHOLD,
    MODEL_BANDS,
    NORMALIZATION_MEANS,
    NORMALIZATION_STDS,
    SELECTED_CHECKPOINT_NAME,
    SELECTED_CHECKPOINT_SHA256,
    TIME_STEPS,
)
from terratorch.registry import MODEL_FACTORY_REGISTRY
from torch import Tensor, nn

PAYLOAD_ROOT = Path(__file__).resolve().parents[2]


def default_model_directory() -> Path:
    """Resolve models from an explicit deployment setting or source checkout."""
    configured = os.environ.get("PRITHVI_MODEL_DIR")
    if configured:
        return Path(configured)
    source_checkout = PAYLOAD_ROOT / "models"
    if source_checkout.is_dir():
        return source_checkout
    return Path.cwd() / "models"


@dataclass(frozen=True)
class InferenceOutput:
    """Per-pixel crop/non-crop outputs for one batch of model tiles."""

    crop_probability: Tensor
    crop_binary: Tensor
    crop_confidence: Tensor


def checkpoint_sha256(path: Path) -> str:
    """Hash a checkpoint without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_state(checkpoint: dict[str, Any]) -> dict[str, Tensor]:
    state = checkpoint.get("state_dict")
    if isinstance(state, dict):
        model_state = {
            name.removeprefix("model."): value
            for name, value in state.items()
            if name.startswith("model.")
        }
    elif checkpoint and all(
        isinstance(name, str) and isinstance(value, Tensor) for name, value in checkpoint.items()
    ):
        model_state = checkpoint
    else:
        raise ValueError("Artifact does not contain a model state dictionary")
    if not model_state:
        raise ValueError("Artifact does not contain model weights")
    return model_state


class PayloadCropModel:
    """Pinned single-image Prithvi model with calibrated binary outputs."""

    def __init__(
        self,
        model: nn.Module,
        *,
        device: torch.device,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self._means = torch.tensor(
            NORMALIZATION_MEANS,
            dtype=torch.float32,
            device=device,
        ).view(1, len(MODEL_BANDS), 1, 1, 1)
        self._stds = torch.tensor(
            NORMALIZATION_STDS,
            dtype=torch.float32,
            device=device,
        ).view(1, len(MODEL_BANDS), 1, 1, 1)

    @classmethod
    def load(
        cls,
        model_directory: Path | None = None,
        *,
        device: str | torch.device = "cuda",
    ) -> PayloadCropModel:
        """Build the architecture and load only the selected checkpoint."""
        model_directory = model_directory or default_model_directory()
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")

        architecture_path = model_directory / "architecture.yaml"
        checkpoint_path = model_directory / SELECTED_CHECKPOINT_NAME
        if not architecture_path.is_file():
            raise FileNotFoundError(architecture_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        actual_digest = checkpoint_sha256(checkpoint_path)
        if actual_digest != SELECTED_CHECKPOINT_SHA256:
            raise RuntimeError(
                "Selected checkpoint checksum mismatch: "
                f"expected {SELECTED_CHECKPOINT_SHA256}, got {actual_digest}"
            )

        architecture = yaml.safe_load(architecture_path.read_text(encoding="utf-8"))
        factory = MODEL_FACTORY_REGISTRY.build(architecture["model_factory"])
        model = factory.build_model(
            task=architecture["task"],
            **architecture["model_args"],
        )
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        model.load_state_dict(_model_state(checkpoint), strict=True)
        return cls(model, device=requested_device)

    def _validate_inputs(
        self,
        image: Tensor,
        temporal_coords: Tensor,
        location_coords: Tensor,
    ) -> None:
        if image.ndim != 5:
            raise ValueError("image must have shape [batch, bands, time, height, width]")
        if image.shape[1] != len(MODEL_BANDS) or image.shape[2] != TIME_STEPS:
            raise ValueError(
                f"Expected {len(MODEL_BANDS)} bands and {TIME_STEPS} dates, "
                f"got shape {tuple(image.shape)}"
            )
        batch_size = image.shape[0]
        if temporal_coords.shape != (batch_size, TIME_STEPS, 2):
            raise ValueError("temporal_coords must have shape [batch, time, 2] as year/day-of-year")
        if location_coords.shape != (batch_size, 2):
            raise ValueError("location_coords must have shape [batch, 2] as latitude/longitude")

    def predict(
        self,
        image: Tensor,
        *,
        temporal_coords: Tensor,
        location_coords: Tensor,
    ) -> InferenceOutput:
        """Predict one or more tiles supplied on the training numeric scale."""
        self._validate_inputs(image, temporal_coords, location_coords)
        image = image.to(self.device, dtype=torch.float32, non_blocking=True)
        temporal_coords = temporal_coords.to(
            self.device,
            dtype=torch.float32,
            non_blocking=True,
        )
        location_coords = location_coords.to(
            self.device,
            dtype=torch.float32,
            non_blocking=True,
        )
        normalized = (image - self._means) / self._stds
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            logits = self.model(
                normalized,
                temporal_coords=temporal_coords,
                location_coords=location_coords,
            ).output
            probabilities = logits.softmax(dim=1)
            crop_probability = probabilities[:, 1]
            crop_binary = (crop_probability >= CROP_CLASSIFICATION_THRESHOLD).to(dtype=torch.uint8)
            crop_confidence = torch.maximum(crop_probability, 1 - crop_probability)
        return InferenceOutput(
            crop_probability=crop_probability,
            crop_binary=crop_binary,
            crop_confidence=crop_confidence,
        )

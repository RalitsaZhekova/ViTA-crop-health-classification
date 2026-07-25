"""Crop/non-crop decisions derived from the existing fine-grained logits."""

from __future__ import annotations

import torch
from torch import Tensor

from prithvi_crop.calibration import CROP_CLASSIFICATION_THRESHOLD
from prithvi_crop.europe import (
    PROJECT_BINARY_IGNORE_CLASSES,
    PROJECT_CROP_CLASSES,
    PROJECT_NON_CROP_CLASSES,
)


def binary_logits_from_fine_logits(logits: Tensor) -> Tensor:
    """Collapse fine-grained logits into non-crop and crop logits."""
    if logits.ndim < 2:
        raise ValueError("Fine-grained logits must include a class dimension")
    required_classes = max(*PROJECT_NON_CROP_CLASSES, *PROJECT_CROP_CLASSES) + 1
    if logits.shape[1] < required_classes:
        raise ValueError(
            f"Expected at least {required_classes} fine-grained classes, "
            f"got {logits.shape[1]}"
        )
    return torch.stack(
        (
            torch.logsumexp(logits[:, PROJECT_NON_CROP_CLASSES], dim=1),
            torch.logsumexp(logits[:, PROJECT_CROP_CLASSES], dim=1),
        ),
        dim=1,
    )


def crop_probability_from_fine_logits(logits: Tensor) -> Tensor:
    """Return the combined probability assigned to all crop classes."""
    return binary_logits_from_fine_logits(logits).softmax(dim=1)[:, 1]


def binary_predictions_from_fine_logits(
    logits: Tensor,
    *,
    crop_threshold: float = CROP_CLASSIFICATION_THRESHOLD,
) -> Tensor:
    """Return 1 for crop and 0 for non-crop at a calibrated threshold."""
    if not 0 < crop_threshold < 1:
        raise ValueError("crop_threshold must be strictly between 0 and 1")
    return (
        crop_probability_from_fine_logits(logits) >= crop_threshold
    ).to(dtype=torch.long)


def binary_targets_from_fine_targets(
    targets: Tensor,
    *,
    ignore_index: int = -1,
) -> Tensor:
    """Map supported fine labels to binary targets and ignore ambiguous labels."""
    binary = torch.full_like(targets, ignore_index)
    for class_index in PROJECT_NON_CROP_CLASSES:
        binary[targets == class_index] = 0
    for class_index in PROJECT_CROP_CLASSES:
        if class_index not in PROJECT_BINARY_IGNORE_CLASSES:
            binary[targets == class_index] = 1
    return binary

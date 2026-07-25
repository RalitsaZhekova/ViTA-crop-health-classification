import pytest
import torch

from prithvi_crop.binary import (
    CROP_CLASSIFICATION_THRESHOLD,
    binary_logits_from_fine_logits,
    binary_predictions_from_fine_logits,
    binary_targets_from_fine_targets,
    crop_probability_from_fine_logits,
)


def test_binary_logits_group_existing_fine_classes() -> None:
    logits = torch.full((1, 13, 1, 1), -20.0)
    logits[:, 2] = 2.0
    logits[:, 0] = 1.0

    binary = binary_logits_from_fine_logits(logits)

    assert binary.shape == (1, 2, 1, 1)
    assert binary[0, 1, 0, 0] > binary[0, 0, 0, 0]
    assert crop_probability_from_fine_logits(logits).item() > 0.5


def test_binary_targets_ignore_other_and_nodata() -> None:
    targets = torch.tensor([[-1, 0, 1, 2, 9, 12]])

    binary = binary_targets_from_fine_targets(targets)

    assert binary.tolist() == [[-1, 0, 0, 1, 1, -1]]


def test_binary_prediction_uses_calibrated_threshold() -> None:
    logits = torch.zeros((1, 13, 1, 1))

    low_threshold = binary_predictions_from_fine_logits(
        logits,
        crop_threshold=0.5,
    )
    high_threshold = binary_predictions_from_fine_logits(
        logits,
        crop_threshold=0.9,
    )

    assert low_threshold.item() == 1
    assert high_threshold.item() == 0
    assert pytest.approx(0.615) == CROP_CLASSIFICATION_THRESHOLD
    with pytest.raises(ValueError, match="strictly"):
        binary_predictions_from_fine_logits(logits, crop_threshold=1.0)

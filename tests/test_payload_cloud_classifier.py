import sys
from pathlib import Path

import numpy as np
import pytest

PAYLOAD_SOURCE = Path("payload/src").resolve()
if str(PAYLOAD_SOURCE) not in sys.path:
    sys.path.insert(0, str(PAYLOAD_SOURCE))

from prithvi_payload._cloudsen12 import TestBackend  # noqa: E402
from prithvi_payload.cloud_classifier import (  # noqa: E402
    CLOUD_BANDS,
    CLOUD_CLASS_NAMES,
    PayloadCloudClassifier,
)


def _classifier() -> PayloadCloudClassifier:
    return PayloadCloudClassifier(
        TestBackend(),
        tile_size=16,
        overlap=4,
        device="test",
    )


def test_cloud_classification_is_semantic_only() -> None:
    image = np.full((4, 19, 21), 1_000, dtype=np.uint16)
    image[1:, 4:10, 5:12] = 8_000
    image[:, 0, 0] = 0

    result = _classifier().classify(image)

    assert result.semantic_class.shape == (19, 21)
    assert result.scores.shape == (4, 19, 21)
    assert result.invalid_input.shape == (19, 21)
    assert result.invalid_input[0, 0]
    assert result.score_kind == "synthetic_probability"
    assert result.band_order == CLOUD_BANDS
    assert set(result.class_fractions) == set(CLOUD_CLASS_NAMES)
    assert sum(result.class_fractions.values()) == pytest.approx(1.0)
    assert not hasattr(result, "unusable_mask")


def test_cloud_classifier_rejects_silent_band_reordering() -> None:
    image = np.ones((4, 8, 8), dtype=np.float32)

    with pytest.raises(ValueError, match="exact band order"):
        _classifier().classify(
            image,
            band_order=("B02", "B03", "B04", "B08"),
            reflectance_scale=1,
        )


def test_cloud_classifier_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="Expected image shaped"):
        _classifier().classify(np.ones((3, 8, 8), dtype=np.float32))

from __future__ import annotations

import numpy as np
from cloud_detection.preview import (
    _semantic_display,
    _semantic_legend,
    _semantic_overlay,
)
from prithvi_payload.crop_executor import (
    _crop_display,
    _crop_legend,
    _crop_probability_overlay,
)


def test_cloud_preview_preserves_and_distinguishes_every_class() -> None:
    semantic = np.array([[0, 1, 2], [3, 255, 99]], dtype=np.uint8)
    original = semantic.copy()

    display = _semantic_display(semantic)
    overlay = _semantic_overlay(semantic)
    labels = [handle.get_label() for handle in _semantic_legend(semantic)]

    np.testing.assert_array_equal(semantic, original)
    np.testing.assert_array_equal(display, [[0, 1, 2], [3, 4, 4]])
    assert overlay.shape == (2, 3, 4)
    assert overlay[0, 0, 3] == 0.0
    assert np.all(overlay[0, 1:, 3] > 0.0)
    assert overlay[1, 0, 3] > 0.0
    assert overlay[1, 1, 3] == overlay[1, 2, 3]
    assert labels == [
        "Clear: 16.7%",
        "Thick cloud: 16.7%",
        "Thin cloud: 16.7%",
        "Cloud shadow: 16.7%",
        "Invalid / nodata: 33.3%",
    ]


def test_crop_preview_keeps_exact_mask_separate_from_smooth_overlay() -> None:
    probability = np.array([[0.0, 0.5, 1.0, -9999.0]], dtype=np.float32)
    binary = np.array([[0, 1, 1, 255]], dtype=np.uint8)
    original_probability = probability.copy()
    original_binary = binary.copy()

    display = _crop_display(binary)
    overlay = _crop_probability_overlay(probability)
    labels = [handle.get_label() for handle in _crop_legend(binary)]

    np.testing.assert_array_equal(probability, original_probability)
    np.testing.assert_array_equal(binary, original_binary)
    np.testing.assert_array_equal(display, [[0, 1, 1, 2]])
    assert overlay[0, 0, 3] < overlay[0, 1, 3] < overlay[0, 2, 3]
    assert overlay[0, 3, 3] == 0.0
    assert labels == ["Non-crop", "Crop: 66.7% of usable", "Excluded: 25.0%"]

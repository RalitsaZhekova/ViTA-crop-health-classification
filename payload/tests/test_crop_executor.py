from __future__ import annotations

import numpy as np
import pytest
from prithvi_payload.crop_executor import (
    _accumulate_prediction,
    _tile_blend_weights,
)


def test_overlap_weighted_stitching_smooths_tile_transition() -> None:
    probability_sum = np.zeros((4, 10), dtype=np.float32)
    probability_weight = np.zeros_like(probability_sum)
    weights = _tile_blend_weights(tile_size=8, halo=2)

    _accumulate_prediction(
        probability_sum,
        probability_weight,
        np.zeros((8, 8), dtype=np.float32),
        weights,
        requested_y=-2,
        requested_x=-2,
    )
    _accumulate_prediction(
        probability_sum,
        probability_weight,
        np.ones((8, 8), dtype=np.float32),
        weights,
        requested_y=-2,
        requested_x=2,
    )

    blended = probability_sum / probability_weight
    middle_row = blended[2]
    assert np.all(np.diff(middle_row) >= 0)
    assert np.max(np.diff(middle_row)) < 0.5
    assert middle_row[2:6] == pytest.approx([0.2, 0.4, 0.6, 0.8])


def test_tile_blend_weights_reject_invalid_halo() -> None:
    with pytest.raises(ValueError, match="Invalid tile size"):
        _tile_blend_weights(tile_size=8, halo=4)

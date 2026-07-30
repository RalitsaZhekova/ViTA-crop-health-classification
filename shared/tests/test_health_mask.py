from __future__ import annotations

import numpy as np
import pytest
from prithvi_shared.calibration import HEALTH_ANALYSIS_CROP_THRESHOLD
from prithvi_shared.health import build_analysis_mask


def test_build_analysis_mask_combines_every_quality_gate() -> None:
    crop_binary = np.array(
        [[1, 1, 1, 1], [0, 255, 1, 1]],
        dtype=np.uint8,
    )
    unusable = np.array(
        [[0, 1, 0, 0], [0, 0, 0, 0]],
        dtype=np.uint8,
    )
    probability = np.array(
        [
            [0.90, 0.90, HEALTH_ANALYSIS_CROP_THRESHOLD - 0.01, np.nan],
            [0.99, 0.99, HEALTH_ANALYSIS_CROP_THRESHOLD, 0.80],
        ],
        dtype=np.float32,
    )
    nodata = np.array(
        [[0, 0, 0, 0], [0, 0, 0, 1]],
        dtype=np.uint8,
    )

    result = build_analysis_mask(
        crop_binary,
        unusable,
        crop_probability=probability,
        nodata=nodata,
    )

    expected = np.array(
        [[True, False, False, False], [False, False, True, False]],
        dtype=bool,
    )
    np.testing.assert_array_equal(result, expected)


def test_build_analysis_mask_accepts_boolean_masks() -> None:
    crop_binary = np.array([[True, False], [True, True]])
    unusable = np.array([[False, False], [True, False]])
    probability = np.ones((2, 2), dtype=np.float32)

    result = build_analysis_mask(
        crop_binary,
        unusable,
        crop_probability=probability,
    )

    np.testing.assert_array_equal(
        result,
        np.array([[True, False], [False, True]]),
    )


@pytest.mark.parametrize(
    ("crop_binary", "unusable", "probability", "message"),
    [
        (
            np.zeros((2, 2, 1), dtype=np.uint8),
            np.zeros((2, 2, 1), dtype=np.uint8),
            np.zeros((2, 2, 1), dtype=np.float32),
            "2D",
        ),
        (
            np.zeros((2, 2), dtype=np.uint8),
            np.zeros((2, 3), dtype=np.uint8),
            np.zeros((2, 2), dtype=np.float32),
            "Unusable mask shape",
        ),
        (
            np.zeros((2, 2), dtype=np.uint8),
            np.zeros((2, 2), dtype=np.uint8),
            np.zeros((3, 2), dtype=np.float32),
            "Crop probability shape",
        ),
    ],
)
def test_build_analysis_mask_rejects_incompatible_shapes(
    crop_binary: np.ndarray,
    unusable: np.ndarray,
    probability: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_analysis_mask(
            crop_binary,
            unusable,
            crop_probability=probability,
        )


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("crop", np.array([[0, 2]], dtype=np.uint8), "outside 0, 1 and 255"),
        ("unusable", np.array([[0, 2]], dtype=np.uint8), "outside 0 and 1"),
        ("nodata", np.array([[0, 2]], dtype=np.uint8), "outside 0 and 1"),
    ],
)
def test_build_analysis_mask_rejects_invalid_mask_values(
    argument: str,
    value: np.ndarray,
    message: str,
) -> None:
    crop = np.array([[0, 1]], dtype=np.uint8)
    unusable = np.zeros((1, 2), dtype=np.uint8)
    nodata = np.zeros((1, 2), dtype=np.uint8)
    if argument == "crop":
        crop = value
    elif argument == "unusable":
        unusable = value
    else:
        nodata = value

    with pytest.raises(ValueError, match=message):
        build_analysis_mask(
            crop,
            unusable,
            crop_probability=np.ones((1, 2), dtype=np.float32),
            nodata=nodata,
        )


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_build_analysis_mask_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        build_analysis_mask(
            np.ones((1, 1), dtype=np.uint8),
            np.zeros((1, 1), dtype=np.uint8),
            crop_probability=np.ones((1, 1), dtype=np.float32),
            minimum_crop_probability=threshold,
        )


@pytest.mark.parametrize(
    ("argument", "dtype", "message"),
    [
        ("crop", np.float32, "Crop binary mask"),
        ("unusable", np.float32, "Unusable mask"),
        ("nodata", np.float32, "No-data mask"),
    ],
)
def test_build_analysis_mask_rejects_non_integral_masks(
    argument: str,
    dtype: type[np.generic],
    message: str,
) -> None:
    crop = np.ones((1, 1), dtype=np.uint8)
    unusable = np.zeros((1, 1), dtype=np.uint8)
    nodata = np.zeros((1, 1), dtype=np.uint8)
    if argument == "crop":
        crop = crop.astype(dtype)
    elif argument == "unusable":
        unusable = unusable.astype(dtype)
    else:
        nodata = nodata.astype(dtype)

    with pytest.raises(TypeError, match=message):
        build_analysis_mask(
            crop,
            unusable,
            crop_probability=np.ones((1, 1), dtype=np.float32),
            nodata=nodata,
        )

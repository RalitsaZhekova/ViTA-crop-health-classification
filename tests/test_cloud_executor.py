from __future__ import annotations

import numpy as np
from cloud_detection.backend import BackendPrediction
from cloud_detection.preprocessing import normalize_reflectance, strict_valid_mask
from prithvi_payload.cloud_executor import _predict_semantic, _prepare_semantic_input


class _SemanticBackend:
    def predict_semantic(self, image: np.ndarray) -> np.ndarray:
        return np.full(image.shape[1:], 3, dtype=np.uint8)


class _ConfidenceBackend:
    def predict(self, image: np.ndarray) -> BackendPrediction:
        scores = np.zeros((4, *image.shape[1:]), dtype=np.float32)
        scores[2] = 1.0
        return BackendPrediction(scores=scores, score_kind="softmax_confidence")


def test_executor_prefers_semantic_only_cloud_output() -> None:
    semantic, score_kind = _predict_semantic(
        _SemanticBackend(),  # type: ignore[arg-type]
        np.ones((4, 32, 48), dtype=np.float32),
    )

    assert score_kind == "semantic_class"
    assert semantic.shape == (32, 48)
    assert np.all(semantic == 3)


def test_executor_retains_confidence_backend_fallback() -> None:
    semantic, score_kind = _predict_semantic(
        _ConfidenceBackend(),  # type: ignore[arg-type]
        np.ones((4, 32, 48), dtype=np.float32),
    )

    assert score_kind == "softmax_confidence"
    assert np.all(semantic == 2)


def test_compact_cloud_preparation_matches_the_original_double_validation() -> None:
    rng = np.random.default_rng(14)
    raw = rng.uniform(-500, 12_000, (4, 47, 61)).astype(np.float32)
    raw[:, 0, 0] = 0.0
    raw[3, 1, 2] = np.nan
    raw[0, 3, 4] = np.inf
    raw[1, 5, 6] = -100.0

    normalized, source_invalid = normalize_reflectance(
        raw,
        scale=10_000.0,
        clip_min=-0.2,
        clip_max=1.0,
        nodata_value=0.0,
    )
    strict_invalid = ~strict_valid_mask(normalized[[1, 2, 0]])
    expected_invalid = source_invalid | strict_invalid
    normalized[:, expected_invalid] = 0.0
    expected_rgn = normalized[[1, 2, 0]].astype(np.float32, copy=True)
    expected_model_valid = np.all(np.isfinite(expected_rgn), axis=0) & np.all(
        expected_rgn > np.finfo(np.float32).tiny,
        axis=0,
    )
    expected_rgn[:, ~expected_model_valid] = 0.0

    prepared, invalid, has_model_input = _prepare_semantic_input(
        raw,
        scale=10_000.0,
        clip_min=-0.2,
        clip_max=1.0,
        nodata_value=0.0,
        strict_positive_rgn=True,
    )

    np.testing.assert_array_equal(prepared, expected_rgn)
    np.testing.assert_array_equal(invalid, expected_invalid)
    assert has_model_input


def test_compact_cloud_preparation_preserves_non_strict_output_mask() -> None:
    raw = np.ones((4, 32, 32), dtype=np.float32)
    raw[1, 2, 3] = -1.0
    _, expected_invalid = normalize_reflectance(raw, scale=10_000.0)

    prepared, invalid, has_model_input = _prepare_semantic_input(
        raw,
        scale=10_000.0,
        clip_min=None,
        clip_max=None,
        nodata_value=0.0,
        strict_positive_rgn=False,
    )

    assert np.all(prepared[:, 2, 3] == 0.0)
    np.testing.assert_array_equal(invalid, expected_invalid)
    assert has_model_input


def test_experimental_raw_cloud_detail_restoration_is_model_input_only() -> None:
    raw = np.full((4, 33, 33), 0.4, dtype=np.float32)
    raw[:3, 16, 16] = 0.5
    raw[:, 0, 0] = 0.0

    prepared, invalid, has_model_input = _prepare_semantic_input(
        raw,
        scale=1.0,
        clip_min=None,
        clip_max=None,
        nodata_value=0.0,
        strict_positive_rgn=True,
        spatial_detail_restoration={
            "method": "nodata_aware_unsharp_mask",
            "sigma_pixels": 1.2,
            "amount": 4.0,
            "bands": ["RED", "GREEN", "NIR_BROAD"],
        },
    )

    assert prepared.shape == (3, 33, 33)
    assert np.all(prepared[:, 16, 16] > 0.5)
    assert np.all(prepared[:, 0, 0] == 0.0)
    assert invalid[0, 0]
    assert has_model_input
    np.testing.assert_allclose(raw[:, 16, 16], np.array([0.5, 0.5, 0.5, 0.4]))

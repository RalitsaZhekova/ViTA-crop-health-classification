from __future__ import annotations

import numpy as np
from cloud_detection.backend import BackendPrediction
from prithvi_payload.cloud_executor import _predict_semantic


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

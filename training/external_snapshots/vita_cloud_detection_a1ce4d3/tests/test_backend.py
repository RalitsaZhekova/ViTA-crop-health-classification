import numpy as np

from cloud_detection.backend import CloudSEN12Backend, TestBackend


def test_softmax_sums_to_one():
    logits = np.array(
        [
            [[0.0, 1.0]],
            [[1.0, 0.0]],
            [[-1.0, -1.0]],
            [[0.5, 0.5]],
        ],
        dtype=np.float32,
    )
    probabilities = CloudSEN12Backend._softmax(logits)
    assert np.allclose(probabilities.sum(axis=0), 1.0)
    assert np.all(probabilities >= 0)


def test_test_backend_contract():
    prediction = TestBackend().predict(np.zeros((4, 32, 32), dtype=np.float32))
    assert prediction.scores.shape == (4, 32, 32)
    assert prediction.score_kind == "synthetic_probability"

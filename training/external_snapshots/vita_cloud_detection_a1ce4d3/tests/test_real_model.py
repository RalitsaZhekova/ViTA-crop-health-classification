import numpy as np
import pytest

from cloud_detection.backend import CloudSEN12Backend


@pytest.mark.integration
def test_real_model() -> None:
    backend = CloudSEN12Backend("dtacs4bands", "models/pretrained")
    prediction = backend.predict(np.zeros((4, 64, 64), dtype=np.float32))
    assert prediction.scores.shape == (4, 64, 64)
    assert prediction.score_kind in {"softmax_probability", "hard_one_hot"}

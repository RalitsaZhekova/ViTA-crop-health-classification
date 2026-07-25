from pathlib import Path

import numpy as np
import pytest
from cloud_detection.backend import CloudSEN12Backend
from prithvi_payload.cloud_classifier import CLOUD_MODEL_SHA256


@pytest.mark.integration
def test_real_model() -> None:
    backend = CloudSEN12Backend(
        "dtacs4bands",
        Path("payload/models/cloudsen12"),
        expected_sha256=CLOUD_MODEL_SHA256,
    )
    prediction = backend.predict(np.zeros((4, 64, 64), dtype=np.float32))
    assert prediction.scores.shape == (4, 64, 64)
    assert prediction.score_kind in {"softmax_confidence", "hard_one_hot"}

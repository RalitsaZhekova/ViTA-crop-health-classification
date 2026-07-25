import numpy as np
from cloud_detection.metrics import binary_metrics


def test_metrics() -> None:
    prediction = np.array([[1, 1], [0, 0]])
    reference = np.array([[1, 0], [1, 0]])
    metrics = binary_metrics(prediction, reference)
    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["iou"] == 1 / 3

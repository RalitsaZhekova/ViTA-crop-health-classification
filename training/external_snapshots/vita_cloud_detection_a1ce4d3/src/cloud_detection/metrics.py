from __future__ import annotations

import numpy as np


def binary_metrics(
    predicted: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float | int]:
    prediction = np.asarray(predicted).astype(bool)
    truth = np.asarray(reference).astype(bool)
    if prediction.shape != truth.shape:
        raise ValueError("Prediction/reference shape mismatch.")

    true_positive = int((prediction & truth).sum())
    false_positive = int((prediction & ~truth).sum())
    true_negative = int((~prediction & ~truth).sum())
    false_negative = int((~prediction & truth).sum())

    def divide(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else 0.0

    precision = divide(true_positive, true_positive + false_positive)
    recall = divide(true_positive, true_positive + false_negative)

    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "accuracy": divide(
            true_positive + true_negative,
            true_positive + false_positive + true_negative + false_negative,
        ),
        "precision": precision,
        "recall": recall,
        "specificity": divide(true_negative, true_negative + false_positive),
        "f1": divide(2 * precision * recall, precision + recall),
        "iou": divide(true_positive, true_positive + false_positive + false_negative),
    }

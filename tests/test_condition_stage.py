from __future__ import annotations

import math

import numpy as np
from prithvi_payload.condition_stage import StreamingMetric


def _rounded(value: float) -> float:
    return float(round(value, 12))


def test_exact_streaming_metric_preserves_reference_results(
    monkeypatch,
) -> None:
    monkeypatch.setenv("VITA_CONDITION_EXACT_PERCENTILES", "1")
    rng = np.random.default_rng(31)
    values = rng.normal(50.0, 12.0, (73, 59)).astype(np.float32)
    values[4, 7] = np.nan
    valid = rng.random(values.shape) > 0.23
    reference = np.asarray(values[valid & np.isfinite(values)], dtype=np.float64)
    metric = StreamingMetric("exact-reference")

    metric.update(values, valid_mask=valid)
    result = metric.summary()

    mean = float(np.sum(reference, dtype=np.float64)) / reference.size
    total_squared = float(np.sum(reference * reference, dtype=np.float64))
    variance = max(0.0, total_squared / reference.size - mean * mean)
    percentiles = np.percentile(reference, (10, 25, 50, 90))
    assert result == {
        "valid_pixels": int(reference.size),
        "mean": _rounded(mean),
        "standard_deviation": _rounded(math.sqrt(variance)),
        "minimum": _rounded(float(np.min(reference))),
        "percentile_10": _rounded(float(percentiles[0])),
        "lower_quartile": _rounded(float(percentiles[1])),
        "median": _rounded(float(percentiles[2])),
        "percentile_90": _rounded(float(percentiles[3])),
        "maximum": _rounded(float(np.max(reference))),
        "percentile_sample_pixels": int(reference.size),
        "percentile_method": "exact_all_valid_pixels",
    }


def test_exact_mad_matches_the_previous_float32_deviation_pass(monkeypatch) -> None:
    monkeypatch.setenv("VITA_CONDITION_EXACT_PERCENTILES", "1")
    values = np.array(
        [[10.25, np.nan, 30.75], [41.125, 52.5, 63.875]],
        dtype=np.float32,
    )
    metric = StreamingMetric("condition_score")
    metric.update(values)
    center = metric.summary()["median"]

    expected = float(
        round(
            float(np.percentile(np.abs(values[np.isfinite(values)] - center), 50)),
            12,
        )
    )

    assert metric.exact_median_absolute_deviation(center) == expected

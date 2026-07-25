from __future__ import annotations

import statistics
import time
from pathlib import Path

import psutil

from .pipeline import CloudDetectionPipeline


def benchmark_file(
    pipeline: CloudDetectionPipeline,
    input_path: str | Path,
    output_root: str | Path,
    runs: int = 3,
) -> dict[str, float | int]:
    """Measure wall-clock time and sampled resident memory after each run.

    The RSS value is sampled after each completed run. It is not presented as a guaranteed
    operating-system-level peak measurement.
    """
    if runs <= 0:
        raise ValueError("runs must be positive.")

    process = psutil.Process()
    runtimes: list[float] = []
    rss_before = process.memory_info().rss
    observed_rss_max = rss_before

    for run_index in range(runs):
        started = time.perf_counter()
        pipeline.predict_file(
            input_path,
            Path(output_root) / f"run_{run_index + 1}",
        )
        runtimes.append(time.perf_counter() - started)
        observed_rss_max = max(observed_rss_max, process.memory_info().rss)

    return {
        "runs": runs,
        "runtime_mean_seconds": statistics.mean(runtimes),
        "runtime_median_seconds": statistics.median(runtimes),
        "runtime_min_seconds": min(runtimes),
        "runtime_max_seconds": max(runtimes),
        "rss_before_bytes": rss_before,
        "observed_rss_after_run_max_bytes": observed_rss_max,
        "observed_rss_delta_bytes": observed_rss_max - rss_before,
    }

"""Report internal timing from one completed payload job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TIMING_FIELDS = (
    "earth_engine_acquisition_seconds",
    "geotiff_validation_seconds",
    "cloud_inference_seconds",
    "mask_processing_seconds",
    "crop_inference_seconds",
    "condition_calculation_seconds",
    "downlink_packaging_seconds",
    "warm_science_seconds",
    "total_payload_processing_seconds",
)


def benchmark_record(job_directory: str | Path) -> dict[str, Any]:
    directory = Path(job_directory).resolve()
    result_path = directory / "job_result.json"
    status_path = directory / "status.json"
    if not result_path.is_file() or not status_path.is_file():
        raise ValueError("Benchmark requires a persisted payload job directory")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("state") != "completed" or result.get("payload_status") != "DOWNLINK_READY":
        raise ValueError("Benchmark requires a completed DOWNLINK_READY payload job")
    timing = result.get("timing")
    if not isinstance(timing, dict):
        raise ValueError("Payload job does not contain benchmark timing")
    values: dict[str, float] = {}
    for field in TIMING_FIELDS:
        value = timing.get(field)
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"Payload timing is missing {field}")
        values[field] = float(value)
    return {
        "schema_version": "1.0",
        "job_id": status.get("job_id"),
        "selected_scene": result.get("selected_scene"),
        "timing": values,
        "warm_timing_excludes": [
            "earth_engine_network_acquisition",
            "ssh",
            "container_startup",
            "python_startup",
            "model_loading",
            "model_warmup",
            "ground_ingestion",
            "dashboard_rendering",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Report one completed payload benchmark.")
    parser.add_argument("job_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(benchmark_record(args.job_directory), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

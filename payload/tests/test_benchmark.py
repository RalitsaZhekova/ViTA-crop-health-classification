from __future__ import annotations

import json
from pathlib import Path

import pytest

from prithvi_payload.benchmark import TIMING_FIELDS, benchmark_record


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_benchmark_reports_separate_stages_and_exclusions(tmp_path: Path) -> None:
    timing = {field: float(index) / 10 for index, field in enumerate(TIMING_FIELDS, 1)}
    _write_json(
        tmp_path / "job_result.json",
        {
            "payload_status": "DOWNLINK_READY",
            "selected_scene": "provider-scene",
            "timing": timing,
        },
    )
    _write_json(
        tmp_path / "status.json",
        {"state": "completed", "job_id": "field_20260701_abcd"},
    )

    report = benchmark_record(tmp_path)

    assert report["timing"] == timing
    assert report["selected_scene"] == "provider-scene"
    assert set(report["warm_timing_excludes"]) == {
        "earth_engine_network_acquisition",
        "ssh",
        "container_startup",
        "python_startup",
        "model_loading",
        "model_warmup",
        "ground_ingestion",
        "dashboard_rendering",
    }


def test_benchmark_rejects_incomplete_or_legacy_job(tmp_path: Path) -> None:
    _write_json(tmp_path / "job_result.json", {"payload_status": "DOWNLINK_READY"})
    _write_json(tmp_path / "status.json", {"state": "completed"})
    with pytest.raises(ValueError, match="timing"):
        benchmark_record(tmp_path)

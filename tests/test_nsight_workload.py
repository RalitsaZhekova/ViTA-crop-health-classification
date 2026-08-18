from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from prithvi_payload import nsight_workload
from prithvi_payload.nvtx import annotate


def _configure_scenes(monkeypatch) -> None:
    monkeypatch.setenv("VITA_DEMO_SENTINEL_IMAGES", "sentinel/a.tif,sentinel/b.tif")
    monkeypatch.setenv("VITA_BALKAN_PREPARE_INPUTS", "balkan/a.tif,balkan/b.tif")


def test_nsight_workload_builds_exact_four_scene_contract(monkeypatch) -> None:
    _configure_scenes(monkeypatch)

    requests = nsight_workload._job_requests("trace1")

    assert [request.sensor for request in requests] == [
        "sentinel-2",
        "sentinel-2",
        "balkan-1",
        "balkan-1",
    ]
    assert [request.job_id for request in requests] == [
        "nsight-trace1-1",
        "nsight-trace1-2",
        "nsight-trace1-3",
        "nsight-trace1-4",
    ]


def test_nsight_workload_controls_capture_and_removes_disposable_runs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _configure_scenes(monkeypatch)
    capture_calls: list[str] = []

    class FakeRuntime:
        def __init__(self) -> None:
            self.output_root = tmp_path / "runs"
            self.output_root.mkdir()
            self.stack = {"crop_backend": "tensorrt", "cloud_backend": "tensorrt"}

        def run(self, request) -> dict[str, Any]:
            (self.output_root / request.job_id).mkdir()
            return {
                "job_id": request.job_id,
                "sensor": request.sensor,
                "payload_seconds": 0.5,
                "pipeline_timing_seconds": {"cloud_stage_seconds": 0.1},
            }

    monkeypatch.setattr(nsight_workload, "PayloadRuntime", FakeRuntime)
    monkeypatch.setattr(nsight_workload, "_cuda_profiler_call", capture_calls.append)
    monkeypatch.setattr(nsight_workload.torch.cuda, "synchronize", lambda: None)
    report_path = tmp_path / "profile" / "workload.json"

    report = nsight_workload.run_workload(
        run_id="trace2",
        report_path=report_path,
        control_capture=True,
        keep_runs=False,
    )

    assert capture_calls == ["cudaProfilerStart", "cudaProfilerStop"]
    assert report["scene_count"] == 4
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == (
        "CAPTURE_WORKLOAD_COMPLETE"
    )
    assert not list((tmp_path / "runs").iterdir())


def test_nvtx_annotation_is_a_noop_when_disabled(monkeypatch) -> None:
    monkeypatch.delenv("VITA_NVTX_ENABLED", raising=False)

    @annotate("test.range")
    def add(left: int, right: int) -> int:
        return left + right

    assert add(2, 3) == 5

"""Run the four fixed payload scenes inside one controlled Nsight capture."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from prithvi_payload.nvtx import range as nvtx_range
from prithvi_payload.performance_acceptance import _requests
from prithvi_payload.service import JobRequest, PayloadRuntime

_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,39}$")


def _job_requests(run_id: str) -> list[JobRequest]:
    """Build exactly the same two-Sentinel/two-Balkan qualification workload."""

    if not _SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError("run_id must be a short filename-safe identifier")
    configured = _requests(run_id)
    if [item["sensor"] for item in configured] != [
        "sentinel-2",
        "sentinel-2",
        "balkan-1",
        "balkan-1",
    ]:
        raise RuntimeError("Nsight workload requires two Sentinel and two Balkan scenes")
    requests: list[JobRequest] = []
    for index, item in enumerate(configured, start=1):
        body = {
            key: value
            for key, value in item.items()
            if key not in {"job_prefix", "sensor"}
        }
        body["sensor"] = item["sensor"]
        body["job_id"] = f"nsight-{run_id}-{index}"
        requests.append(JobRequest(**body))
    return requests


def _cuda_profiler_call(name: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Nsight capture control requires CUDA")
    function = getattr(torch.cuda.cudart(), name)
    status = int(function())
    if status != 0:
        raise RuntimeError(f"{name} failed with CUDA status {status}")


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _remove_run(runtime: PayloadRuntime, job_id: str) -> None:
    if not job_id.startswith("nsight-") or any(value in job_id for value in ("/", "\\")):
        raise ValueError("Refusing to remove a non-Nsight payload run")
    path = runtime.output_root / job_id
    if path.parent != runtime.output_root:
        raise ValueError("Nsight output escaped the payload run root")
    if path.is_symlink():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def run_workload(
    *,
    run_id: str,
    report_path: Path,
    control_capture: bool,
    keep_runs: bool,
) -> dict[str, Any]:
    """Warm once, capture four full pipeline jobs, and persist a compact manifest."""

    runtime = PayloadRuntime()
    requests = _job_requests(run_id)
    responses: list[dict[str, Any]] = []
    created_job_ids: list[str] = []
    capture_started = False
    torch.cuda.synchronize()
    if control_capture:
        _cuda_profiler_call("cudaProfilerStart")
        capture_started = True
    suite_started = time.perf_counter()
    try:
        with nvtx_range("vita.profile.four_scene_suite"):
            for index, request in enumerate(requests, start=1):
                label = f"vita.scene.{index}.{request.sensor}.{Path(request.input).stem}"
                created_job_ids.append(request.job_id)
                with nvtx_range(label):
                    response = runtime.run(request)
                    torch.cuda.synchronize()
                responses.append(response)
    finally:
        if capture_started:
            torch.cuda.synchronize()
            _cuda_profiler_call("cudaProfilerStop")
        if not keep_runs:
            for job_id in created_job_ids:
                _remove_run(runtime, job_id)

    report = {
        "schema_version": "1.0",
        "status": "CAPTURE_WORKLOAD_COMPLETE",
        "run_id": run_id,
        "scene_count": len(responses),
        "suite_seconds": time.perf_counter() - suite_started,
        "capture_control": "cudaProfilerApi" if control_capture else "none",
        "nvtx_enabled": os.environ.get("VITA_NVTX_ENABLED", "0"),
        "stack": runtime.stack,
        "scenes": [
            {
                "job_id": response["job_id"],
                "sensor": response["sensor"],
                "payload_seconds": response["payload_seconds"],
                "pipeline_timing_seconds": response["pipeline_timing_seconds"],
            }
            for response in responses
        ],
    }
    _write_report(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the fixed four-scene payload workload under Nsight Systems."
    )
    parser.add_argument(
        "--run-id",
        default=f"{time.time_ns()}",
        help="Short identifier used for disposable payload run directories.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            os.environ.get(
                "VITA_NSYS_WORKLOAD_REPORT",
                "/profiles/vita-four-scene-workload.json",
            )
        ),
    )
    parser.add_argument(
        "--no-capture-control",
        action="store_true",
        help="Do not call cudaProfilerStart/Stop (useful for local smoke tests).",
    )
    parser.add_argument("--keep-runs", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run_workload(
                run_id=args.run_id,
                report_path=args.report.resolve(),
                control_capture=not args.no_capture_control,
                keep_runs=args.keep_runs,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

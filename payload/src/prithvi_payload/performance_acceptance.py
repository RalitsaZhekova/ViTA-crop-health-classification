"""Run the four fixed MVP scenes and enforce the warm two-second payload SLO."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import PurePosixPath
from typing import Any
from urllib.request import Request, urlopen


def _paths(name: str) -> tuple[str, str]:
    values = tuple(value.strip() for value in os.environ.get(name, "").split(","))
    values = tuple(value for value in values if value)
    if len(values) != 2 or len(set(values)) != 2:
        raise RuntimeError(f"{name} must contain exactly two distinct paths")
    return values  # type: ignore[return-value]


def _requests(run_stamp: str) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for index, value in enumerate(_paths("VITA_DEMO_SENTINEL_IMAGES"), start=1):
        path = PurePosixPath(value)
        requests.append(
            {
                "sensor": "sentinel-2",
                "input": path.parent.as_posix(),
                "image": path.name,
                "region_id": f"accept-sentinel-{index}",
                "job_prefix": f"accept-s{index}-{run_stamp}",
            }
        )
    for index, value in enumerate(_paths("VITA_BALKAN_PREPARE_INPUTS"), start=1):
        requests.append(
            {
                "sensor": "balkan-1",
                "input": value,
                "region_id": f"accept-balkan-{index}",
                "job_prefix": f"accept-b{index}-{run_stamp}",
            }
        )
    return requests


def _submit(url: str, body: dict[str, Any]) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=1800) as response:  # noqa: S310 - loopback URL
        value = json.load(response)
    if not isinstance(value, dict) or value.get("status") != "DOWNLINK_READY":
        raise RuntimeError("Payload performance request did not produce a downlink bundle")
    return value


def run_acceptance(
    *,
    url: str,
    repetitions: int,
    target_seconds: float,
) -> dict[str, Any]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    if target_seconds <= 0:
        raise ValueError("target_seconds must be positive")
    stamp = f"{time.time_ns()}"
    results = []
    for scene in _requests(stamp):
        timings = []
        runs = []
        for repetition in range(1, repetitions + 1):
            body = {
                key: value
                for key, value in scene.items()
                if key != "job_prefix"
            }
            body["job_id"] = f"{scene['job_prefix']}-r{repetition}"
            response = _submit(url, body)
            elapsed = float(response["payload_seconds"])
            timings.append(elapsed)
            runs.append(
                {
                    "job_id": body["job_id"],
                    "payload_seconds": elapsed,
                    "under_two_seconds": response.get("under_two_seconds"),
                    "pipeline_timing_seconds": response.get(
                        "pipeline_timing_seconds", {}
                    ),
                }
            )
        results.append(
            {
                "sensor": scene["sensor"],
                "input": scene["input"],
                "image": scene.get("image"),
                "seconds": timings,
                "runs": runs,
                "minimum_seconds": min(timings),
                "median_seconds": statistics.median(timings),
                "maximum_seconds": max(timings),
                "all_under_target": all(value < target_seconds for value in timings),
            }
        )
    accepted = all(item["all_under_target"] for item in results)
    report = {
        "status": "UNDER_TWO_SECONDS" if accepted else "PERFORMANCE_SLO_FAILED",
        "target_payload_seconds": target_seconds,
        "repetitions_per_scene": repetitions,
        "scenes": results,
    }
    if not accepted:
        raise RuntimeError(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8090/v1/jobs")
    parser.add_argument(
        "--repetitions",
        type=int,
        default=int(os.environ.get("VITA_PERFORMANCE_REPETITIONS", "3")),
    )
    parser.add_argument("--target-seconds", type=float, default=2.0)
    args = parser.parse_args()
    print(
        json.dumps(
            run_acceptance(
                url=args.url,
                repetitions=args.repetitions,
                target_seconds=args.target_seconds,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

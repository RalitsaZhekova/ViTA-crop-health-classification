"""Coordinate-driven payload mission command and canonical ground ingestion."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
import uuid
from pathlib import Path

from prithvi_ground.catalog import SceneCatalog
from prithvi_shared import PayloadAcquisitionCommand

from vita_integration.ground_client import GroundClient, GroundClientError
from vita_integration.payload_client import PayloadClient, PayloadClientError

TERMINAL_STATES = {"completed", "rejected", "failed"}


def _bbox(value: str) -> tuple[float, float, float, float]:
    try:
        parsed = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "bbox must contain four comma-separated numbers"
        ) from error
    if len(parsed) != 4:
        raise argparse.ArgumentTypeError("bbox must contain west,south,east,north")
    return parsed


def _job_id(region_id: str, start_date: str) -> str:
    region_prefix = region_id[:48]
    compact_date = start_date.replace("-", "")
    return f"{region_prefix}_{compact_date}_{uuid.uuid4().hex[:16]}"


def build_command(args: argparse.Namespace) -> PayloadAcquisitionCommand:
    return PayloadAcquisitionCommand.model_validate(
        {
            "schema_version": "1.0",
            "job_id": _job_id(args.region_id, args.start),
            "region_id": args.region_id,
            "source": {
                "provider": "earth_engine",
                "bbox_wgs84": args.bbox,
                "start_date": args.start,
                "end_date": args.end,
                "selection_policy": args.selection_policy,
                "target_cloud_min_percent": args.target_cloud_min,
                "target_cloud_max_percent": args.target_cloud_max,
                "target_cloud_ideal_percent": args.target_cloud_ideal,
            },
        }
    )


def run_region(args: argparse.Namespace) -> int:
    command = build_command(args)
    client = PayloadClient(args.payload_url)
    health = client.health()
    if health.get("status") != "ok":
        raise PayloadClientError("Payload service is not ready")
    client.submit(command)
    previous_progress: tuple[object, ...] | None = None
    while True:
        status = client.status(command.job_id)
        progress = (
            status.get("state"),
            status.get("current_candidate_number"),
            status.get("safe_candidate_scene_id"),
            status.get("safe_metadata_cloud_percentage"),
            status.get("safe_payload_measured_cloud_percentage"),
        )
        if progress != previous_progress:
            print(
                "payload progress:",
                json.dumps(
                    {
                        "state": progress[0],
                        "candidate": progress[1],
                        "scene": progress[2],
                        "metadata_cloud_percent": progress[3],
                        "payload_cloud_percent": progress[4],
                    },
                    sort_keys=True,
                ),
            )
            previous_progress = progress
        if status.get("state") in TERMINAL_STATES:
            break
        time.sleep(args.poll_interval)
    if status.get("state") != "completed":
        print(json.dumps({"job_id": command.job_id, "status": status}, indent=2))
        return 2

    with tempfile.TemporaryDirectory(prefix="vita-downlink-") as temporary:
        bundle = client.download_bundle(
            command.job_id,
            Path(temporary) / "bundle",
            expected_checksums=status["artifact_checksums"],
        )
        scene, created = SceneCatalog(args.ground_store).ingest(bundle)
        ground_response = GroundClient(args.dashboard_url).ingest_bundle(bundle)
        ground_api_scene = ground_response["scene"]
        if ground_api_scene.get("scene_id") != scene["scene_id"]:
            raise GroundClientError("Ground service returned a different scene identifier")
    condition = status.get("condition", {})
    print(
        json.dumps(
            {
                "job_id": command.job_id,
                "selected_scene": status.get("selected_scene"),
                "metadata_cloud_percentage": status.get("metadata_cloud_percentage"),
                "payload_measured_cloud_percentage": status.get(
                    "payload_measured_cloud_percentage"
                ),
                "payload_measured_shadow_percentage": status.get(
                    "payload_measured_shadow_percentage"
                ),
                "payload_measured_unusable_percentage": status.get(
                    "payload_measured_unusable_percentage"
                ),
                "condition": condition,
                "ground_created": created,
                "ground_api_created": ground_response.get("created"),
                "ground_scene_id": scene["scene_id"],
                "dashboard_url": args.dashboard_url,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Run coordinate-driven ViTA payload missions.")
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run-region")
    run.add_argument("--bbox", type=_bbox, required=True)
    run.add_argument("--start", required=True)
    run.add_argument("--end", required=True)
    run.add_argument("--region-id", required=True)
    run.add_argument(
        "--selection-policy",
        choices=("target_cloud_range", "least_cloudy"),
        default="target_cloud_range",
    )
    run.add_argument("--target-cloud-min", type=float, default=15.0)
    run.add_argument("--target-cloud-max", type=float, default=35.0)
    run.add_argument("--target-cloud-ideal", type=float, default=25.0)
    run.add_argument("--payload-url", required=True)
    run.add_argument("--ground-store", type=Path, required=True)
    run.add_argument("--dashboard-url", default="http://127.0.0.1:8000/")
    run.add_argument("--poll-interval", type=float, default=2.0)
    run.set_defaults(handler=run_region)
    return root


def main() -> None:
    args = parser().parse_args()
    try:
        raise SystemExit(args.handler(args))
    except (GroundClientError, PayloadClientError, ValueError) as error:
        print(f"Mission failed: {error}")
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()

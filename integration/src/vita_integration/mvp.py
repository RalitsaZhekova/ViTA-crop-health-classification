"""Sentinel payload-to-ground MVP orchestration."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import uvicorn
from cloud_detection.backend import CloudBackend
from cloud_detection.cli import DEFAULT_CONFIG
from prithvi_ground.api import create_app
from prithvi_ground.catalog import SceneCatalog

from .pipeline import run_sentinel_end_to_end

MVP_SCHEMA_VERSION = "1.0"
MVP_ALGORITHM_VERSION = "sentinel-ground-mvp-v1"


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _scene_links(scene_id: str, region_id: str) -> dict[str, str]:
    scene_root = f"/api/v1/scenes/{scene_id}"
    region_root = f"/api/v1/regions/{region_id}"
    return {
        "application": "/",
        "scene": scene_root,
        "manifest": f"{scene_root}/manifest",
        "preview": f"{scene_root}/preview",
        "condition_overlay": f"{scene_root}/condition-overlay",
        "region_history": f"{region_root}/history",
        "openapi": "/docs",
    }


def run_sentinel_mvp(
    input_path: str | Path,
    *,
    output_root: str | Path,
    ground_store: str | Path,
    acquired_at: str,
    scene_id: str | None = None,
    region_id: str | None = None,
    reflectance_scale: float | None = None,
    max_crop_cloud_percentage: float = 60.0,
    condition_tile_size: int = 512,
    downlink_max_image_dimension: int = 1600,
    downlink_grid_size: int = 16,
    cloud_config_path: str | Path = DEFAULT_CONFIG,
    cloud_backend: CloudBackend | None = None,
    cloud_config: dict[str, Any] | None = None,
    crop_model: Any | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run payload processing, receive its compact bundle, and make it API-ready."""
    started = time.perf_counter()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "mvp_result.json"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"MVP result already exists: {result_path}")

    payload = run_sentinel_end_to_end(
        input_path,
        output_root=output,
        acquired_at=acquired_at,
        scene_id=scene_id,
        region_id=region_id,
        reflectance_scale=reflectance_scale,
        max_crop_cloud_percentage=max_crop_cloud_percentage,
        condition_tile_size=condition_tile_size,
        downlink_max_image_dimension=downlink_max_image_dimension,
        downlink_grid_size=downlink_grid_size,
        cloud_config_path=cloud_config_path,
        cloud_backend=cloud_backend,
        cloud_config=cloud_config,
        crop_model=crop_model,
        overwrite=overwrite,
    )
    if payload["status"] != "COMPLETE":
        result = {
            "schema_version": MVP_SCHEMA_VERSION,
            "algorithm_version": MVP_ALGORITHM_VERSION,
            "status": "PAYLOAD_STOPPED",
            "completed_stages": ["payload"],
            "scene_id": payload.get("scene_id"),
            "payload_status": payload.get("payload_status"),
            "end_to_end_result": "end_to_end_result.json",
            "ground": None,
            "runtime_seconds": time.perf_counter() - started,
        }
        _write_json_atomic(result_path, result)
        return result

    manifest_path = output / payload["downlink_manifest"]
    scene, created = SceneCatalog(ground_store).ingest(manifest_path.parent)
    if scene["scene_id"] != payload["scene_id"]:
        raise RuntimeError("Ground catalog scene does not match the payload result")
    links = _scene_links(scene["scene_id"], scene["region_id"])
    result = {
        "schema_version": MVP_SCHEMA_VERSION,
        "algorithm_version": MVP_ALGORITHM_VERSION,
        "status": "MVP_READY",
        "completed_stages": ["payload", "downlink", "ground_catalog", "api_ready"],
        "scene_id": scene["scene_id"],
        "region_id": scene["region_id"],
        "sensor": scene["sensor"],
        "end_to_end_result": "end_to_end_result.json",
        "ground": {
            "created": created,
            "scene": scene,
            "links": links,
        },
        "summary": payload["summary"],
        "runtime_seconds": time.perf_counter() - started,
    }
    _write_json_atomic(result_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Sentinel intake through the interactive ViTA ground MVP."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ground-store", type=Path, required=True)
    parser.add_argument("--acquired-at", required=True)
    parser.add_argument("--scene-id")
    parser.add_argument("--region-id")
    parser.add_argument("--reflectance-scale", type=float)
    parser.add_argument("--max-crop-cloud-percentage", type=float, default=60.0)
    parser.add_argument("--condition-tile-size", type=int, default=512)
    parser.add_argument("--downlink-max-image-dimension", type=int, default=1600)
    parser.add_argument("--downlink-grid-size", type=int, default=16)
    parser.add_argument("--cloud-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_sentinel_mvp(
        args.input,
        output_root=args.output,
        ground_store=args.ground_store,
        acquired_at=args.acquired_at,
        scene_id=args.scene_id,
        region_id=args.region_id,
        reflectance_scale=args.reflectance_scale,
        max_crop_cloud_percentage=args.max_crop_cloud_percentage,
        condition_tile_size=args.condition_tile_size,
        downlink_max_image_dimension=args.downlink_max_image_dimension,
        downlink_grid_size=args.downlink_grid_size,
        cloud_config_path=args.cloud_config,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if result["status"] != "MVP_READY":
        raise SystemExit(2)
    if args.serve:
        print(
            f"ViTA MVP ready at http://{args.host}:{args.port}",
            file=sys.stderr,
            flush=True,
        )
        uvicorn.run(create_app(args.ground_store), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

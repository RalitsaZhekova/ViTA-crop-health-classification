"""Validate the fixed four-scene payload package without loading either model."""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
from typing import Any

import rasterio
from vita_integration.cli import SENTINEL_BAND_ORDER, _sentinel_contract

from prithvi_payload.scene_intake import inspect_scene

BALKAN_BAND_ORDER = ("BLUE", "GREEN", "RED", "NIR", "PAN")


def _paths_from_environment(name: str) -> tuple[str, ...]:
    values = tuple(value.strip() for value in os.environ.get(name, "").split(",") if value.strip())
    if len(values) != 2 or len(set(values)) != 2:
        raise RuntimeError(f"{name} must contain exactly two distinct comma-separated paths")
    return values


def _safe_data_path(root: Path, relative_value: str) -> Path:
    relative = PurePosixPath(relative_value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise RuntimeError(f"Unsafe payload input path: {relative_value}")
    path = root.joinpath(*relative.parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"Payload input escapes the data root: {relative_value}") from error
    if not path.is_file():
        raise RuntimeError(f"Payload input is missing: {relative_value}")
    return path


def _require_ready(record: dict[str, Any], relative_value: str) -> None:
    readiness = record.get("readiness", {})
    rejected = {name: value for name, value in readiness.items() if value != "READY"}
    if rejected:
        raise RuntimeError(f"Payload input is not model-ready: {relative_value}: {rejected}")


def validate() -> dict[str, Any]:
    data_root = Path(os.environ.get("VITA_INPUT_ROOT", "/data")).resolve()
    sentinel_inputs = _paths_from_environment("VITA_DEMO_SENTINEL_IMAGES")
    balkan_inputs = _paths_from_environment("VITA_BALKAN_PREPARE_INPUTS")
    records: list[dict[str, Any]] = []

    for index, relative_value in enumerate(sentinel_inputs, start=1):
        source = _safe_data_path(data_root, relative_value)
        acquired_at, reflectance_scale = _sentinel_contract(
            source,
            acquired_at=None,
            reflectance_scale=None,
        )
        record = inspect_scene(
            source,
            sensor="sentinel-2",
            acquired_at=acquired_at,
            scene_id=f"deployment-sentinel-{index}",
            band_order=SENTINEL_BAND_ORDER,
        )
        _require_ready(record, relative_value)
        records.append(
            {
                "sensor": "sentinel-2",
                "input": relative_value,
                "width": record["raster"]["width"],
                "height": record["raster"]["height"],
                "reflectance_scale": reflectance_scale,
            }
        )

    for index, relative_value in enumerate(balkan_inputs, start=1):
        source = _safe_data_path(data_root, relative_value)
        calibration = source.with_name(f"{source.stem}.crop_calibration.json")
        if not calibration.is_file():
            raise RuntimeError(f"Balkan calibration is missing: {calibration}")
        calibration_record = json.loads(calibration.read_text(encoding="utf-8"))
        acquired_at = calibration_record.get("acquired_at")
        record = inspect_scene(
            source,
            sensor="balkan-1",
            acquired_at=acquired_at,
            scene_id=f"deployment-balkan-{index}",
            band_order=BALKAN_BAND_ORDER,
            crop_calibration_path=calibration,
        )
        _require_ready(record, relative_value)
        source_indices = record["model_band_routes"]["cloud_detection"]["source_band_indices"]
        with rasterio.open(source) as dataset:
            common_overviews = set(dataset.overviews(source_indices[0]))
            for source_index in source_indices[1:]:
                common_overviews &= set(dataset.overviews(source_index))
        if 4 not in common_overviews:
            raise RuntimeError(f"Balkan factor-4 overview is missing: {relative_value}")
        records.append(
            {
                "sensor": "balkan-1",
                "input": relative_value,
                "width": record["raster"]["width"],
                "height": record["raster"]["height"],
                "overview_factors": sorted(common_overviews),
                "calibration": calibration.name,
            }
        )

    return {"status": "ready", "data_root": str(data_root), "scenes": records}


def main() -> None:
    print(json.dumps(validate(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

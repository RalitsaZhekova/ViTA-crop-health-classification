"""Bounded-memory execution of a planned cloud-detection stage."""

from __future__ import annotations

import json
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from cloud_detection.backend import CloudBackend
from cloud_detection.postprocessing import postprocess
from cloud_detection.preprocessing import normalize_reflectance
from rasterio.windows import Window as RasterWindow


def _percentage(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def _output_profile(source: dict[str, Any]) -> dict[str, Any]:
    profile = source.copy()
    profile.update(
        count=1,
        dtype="uint8",
        nodata=255,
        compress="deflate",
        BIGTIFF="IF_SAFER",
    )
    return profile


def _read_padded_tile(
    dataset: rasterio.DatasetReader,
    indices: list[int],
    *,
    y: int,
    x: int,
    tile_size: int,
    halo: int,
) -> np.ndarray:
    requested_y = y - halo
    requested_x = x - halo
    read_y_start = max(0, requested_y)
    read_x_start = max(0, requested_x)
    read_y_end = min(dataset.height, requested_y + tile_size)
    read_x_end = min(dataset.width, requested_x + tile_size)
    window = RasterWindow(
        read_x_start,
        read_y_start,
        read_x_end - read_x_start,
        read_y_end - read_y_start,
    )
    values = dataset.read(indices, window=window)
    top = read_y_start - requested_y
    left = read_x_start - requested_x
    bottom = requested_y + tile_size - read_y_end
    right = requested_x + tile_size - read_x_end
    padding_mode = "reflect" if values.shape[-2] > 1 and values.shape[-1] > 1 else "edge"
    return np.pad(
        values,
        ((0, 0), (top, bottom), (left, right)),
        mode=padding_mode,
    )


def execute_cloud_stage(
    plan: dict[str, Any],
    *,
    output_root: str | Path,
    backend: CloudBackend,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Execute a ready cloud plan without materialising a full scene array."""
    if plan.get("readiness") != "READY":
        raise ValueError("Cloud execution requires a READY cloud-stage plan")
    if plan.get("execution", {}).get("mode") != "WINDOWED_GEOTIFF":
        raise ValueError("Cloud plan does not request windowed GeoTIFF execution")

    source_path = Path(plan["source_path"])
    indices = plan["input"]["source_band_indices_1_based"]
    scale = float(plan["input"]["reflectance_scale"])
    tile_size = int(plan["execution"]["tile_size"])
    halo = int(plan["execution"]["overlap"])
    core_size = tile_size - 2 * halo
    if core_size <= 0:
        raise ValueError("Cloud tile overlap leaves no writable core")

    classes = config["classes"]
    expected_classes = {"clear": 0, "thick_cloud": 1, "thin_cloud": 2, "cloud_shadow": 3}
    if classes != expected_classes:
        raise ValueError(f"Unexpected cloud class mapping: {classes}")

    output_root = Path(output_root)
    stem = str(plan["scene_id"])
    semantic_path = output_root / "cloud_masks" / f"{stem}_semantic.tif"
    unusable_path = output_root / "cloud_masks" / f"{stem}_unusable.tif"
    invalid_path = output_root / "cloud_masks" / f"{stem}_invalid.tif"
    metadata_path = output_root / "metadata" / f"{stem}.json"
    for path in (semantic_path, unusable_path, invalid_path, metadata_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    class_counts = np.zeros(4, dtype=np.int64)
    invalid_count = 0
    unusable_count = 0
    score_kind: str | None = None
    tile_count = 0

    with rasterio.open(source_path) as source, ExitStack() as stack:
        if max(indices) > source.count or len(set(indices)) != 4:
            raise ValueError("Cloud source-band indices do not match the GeoTIFF")
        profile = _output_profile(source.profile)
        semantic_output = stack.enter_context(rasterio.open(semantic_path, "w", **profile))
        unusable_output = stack.enter_context(rasterio.open(unusable_path, "w", **profile))
        invalid_output = stack.enter_context(rasterio.open(invalid_path, "w", **profile))
        semantic_output.set_band_description(1, "0 clear, 1 thick, 2 thin, 3 shadow, 255 invalid")
        unusable_output.set_band_description(1, "0 usable, 1 unusable")
        invalid_output.set_band_description(1, "0 valid input, 1 invalid input")

        nodata_value = plan["input"].get("nodata_value")
        if nodata_value is None:
            nodata_value = config["input"].get("nodata_value", source.nodata)
        for y in range(0, source.height, core_size):
            core_height = min(core_size, source.height - y)
            for x in range(0, source.width, core_size):
                core_width = min(core_size, source.width - x)
                raw_tile = _read_padded_tile(
                    source,
                    indices,
                    y=y,
                    x=x,
                    tile_size=tile_size,
                    halo=halo,
                )
                image, invalid = normalize_reflectance(
                    raw_tile,
                    scale=scale,
                    clip_min=config["input"].get("clip_min"),
                    clip_max=config["input"].get("clip_max"),
                    nodata_value=nodata_value,
                )
                prediction = backend.predict(image.astype(np.float32, copy=False))
                if prediction.scores.shape != (4, tile_size, tile_size):
                    raise ValueError(
                        "Cloud backend returned unexpected score shape: "
                        f"{prediction.scores.shape}"
                    )
                if score_kind is None:
                    score_kind = prediction.score_kind
                elif score_kind != prediction.score_kind:
                    raise ValueError("Cloud backend returned mixed score kinds")

                semantic_tile = prediction.scores.argmax(axis=0).astype(np.uint8)
                unusable_tile = postprocess(
                    semantic_tile,
                    classes,
                    config["postprocessing"],
                    invalid,
                )
                core_slice = (
                    slice(halo, halo + core_height),
                    slice(halo, halo + core_width),
                )
                semantic_core = semantic_tile[core_slice].copy()
                unusable_core = unusable_tile[core_slice]
                invalid_core = invalid[core_slice]
                valid_core = ~invalid_core
                class_counts += np.bincount(
                    semantic_core[valid_core],
                    minlength=4,
                )[:4]
                invalid_count += int(invalid_core.sum())
                unusable_count += int(unusable_core.sum())
                semantic_core[invalid_core] = 255

                output_window = RasterWindow(x, y, core_width, core_height)
                semantic_output.write(semantic_core, 1, window=output_window)
                unusable_output.write(unusable_core.astype(np.uint8), 1, window=output_window)
                invalid_output.write(invalid_core.astype(np.uint8), 1, window=output_window)
                tile_count += 1

        width = source.width
        height = source.height

    total_pixels = width * height
    valid_pixels = total_pixels - invalid_count
    class_fractions = {
        name: float(class_counts[class_index] / valid_pixels) if valid_pixels else 0.0
        for name, class_index in classes.items()
    }
    unusable_percentage = _percentage(unusable_count, total_pixels)
    decision_config = config["decision"]
    if unusable_percentage >= float(
        decision_config["reject_min_unusable_percentage"]
    ):
        decision = "REJECT"
    elif unusable_percentage > float(
        decision_config["process_max_unusable_percentage"]
    ):
        decision = "PROCESS_CLEAR_AREAS"
    else:
        decision = "PROCESS"

    metadata = {
        "schema_version": "0.1-draft",
        "stage": "cloud_detection",
        "scene_id": plan["scene_id"],
        "sensor": plan["sensor"],
        "compatibility": plan["compatibility"],
        "validation_status": plan["validation_status"],
        "score_kind": score_kind,
        "raster": {"width": width, "height": height, "total_pixels": total_pixels},
        "runtime": {
            "seconds": time.perf_counter() - started,
            "tile_count": tile_count,
            "tile_size": tile_size,
            "halo": halo,
        },
        "class_fractions": class_fractions,
        "cloud_percentage": 100.0
        * (class_fractions["thick_cloud"] + class_fractions["thin_cloud"]),
        "shadow_percentage": 100.0 * class_fractions["cloud_shadow"],
        "invalid_percentage": _percentage(invalid_count, total_pixels),
        "usable_percentage": 100.0 - unusable_percentage,
        "unusable_percentage": unusable_percentage,
        "decision": decision,
        "output_files": {
            "semantic_mask": str(semantic_path.resolve()),
            "unusable_mask": str(unusable_path.resolve()),
            "invalid_mask": str(invalid_path.resolve()),
            "metadata": str(metadata_path.resolve()),
        },
        "warnings": plan.get("warnings", []),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata

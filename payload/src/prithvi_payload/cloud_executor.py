"""Bounded-memory execution of a planned cloud-detection stage."""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from cloud_detection.backend import CloudBackend
from cloud_detection.postprocessing import postprocess
from cloud_detection.preprocessing import normalize_reflectance, strict_valid_mask
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject, transform_bounds
from rasterio.windows import Window as RasterWindow

GDAL_WARP_THREADS = max(1, min(4, os.cpu_count() or 1))


def _percentage(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def _output_profile(source: dict[str, Any]) -> dict[str, Any]:
    profile = source.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=255,
        compress="deflate",
        zlevel=1,
        num_threads="ALL_CPUS",
        BIGTIFF="IF_SAFER",
    )
    return profile


def _utm_crs(source_crs: CRS, bounds: rasterio.coords.BoundingBox) -> CRS:
    west, south, east, north = transform_bounds(
        source_crs,
        "EPSG:4326",
        *bounds,
        densify_pts=21,
    )
    longitude = (west + east) / 2.0
    latitude = (south + north) / 2.0
    zone = max(1, min(60, int(math.floor((longitude + 180.0) / 6.0) + 1)))
    return CRS.from_epsg((32600 if latitude >= 0 else 32700) + zone)


def _reproject_mask_to_source(
    analysis_path: Path,
    source: rasterio.DatasetReader,
    destination_path: Path,
    *,
    description: str,
) -> None:
    profile = _output_profile(source.profile)
    with (
        rasterio.open(analysis_path) as analysis,
        rasterio.open(destination_path, "w", **profile) as destination,
    ):
        reproject(
            source=rasterio.band(analysis, 1),
            destination=rasterio.band(destination, 1),
            src_transform=analysis.transform,
            src_crs=analysis.crs,
            src_nodata=analysis.nodata,
            dst_transform=source.transform,
            dst_crs=source.crs,
            dst_nodata=255,
            resampling=Resampling.nearest,
            init_dest_nodata=True,
            num_threads=GDAL_WARP_THREADS,
        )
        destination.set_band_description(1, description)


def _synchronize_cuda(backend: CloudBackend) -> None:
    device = getattr(backend, "device", None)
    if getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


def _predict_semantic(
    backend: CloudBackend,
    image: np.ndarray,
) -> tuple[np.ndarray, str]:
    semantic_predictor = getattr(backend, "predict_semantic", None)
    if callable(semantic_predictor):
        semantic = np.asarray(semantic_predictor(image), dtype=np.uint8)
        expected_shape = image.shape[1:]
        if semantic.shape != expected_shape or np.any(semantic > 3):
            raise ValueError(f"Cloud backend returned invalid semantic classes: {semantic.shape}")
        return semantic, "semantic_class"

    prediction = backend.predict(image)
    expected_shape = (4, *image.shape[1:])
    if prediction.scores.shape != expected_shape:
        raise ValueError(
            f"Cloud backend returned unexpected score shape: {prediction.scores.shape}"
        )
    return prediction.scores.argmax(axis=0).astype(np.uint8), prediction.score_kind


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
    preview_path = output_root / "visualisations" / f"{stem}_cloud.png"
    metadata_path = output_root / "metadata" / f"{stem}.json"
    save_diagnostic_preview = bool(config.get("output", {}).get("save_preview", False))
    output_paths = [semantic_path, unusable_path, invalid_path, metadata_path]
    if save_diagnostic_preview:
        output_paths.append(preview_path)
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    class_counts = np.zeros(4, dtype=np.int64)
    invalid_count = 0
    unusable_count = 0
    score_kind: str | None = None
    tile_count = 0
    inference_seconds = 0.0
    mask_processing_seconds = 0.0

    reprojection_seconds = 0.0
    analysis_grid_preparation_seconds = 0.0
    shared_balkan_grid = bool(plan["input"].get("analysis_grid_ready"))
    analysis_grid_mode = (
        "balkan_1_shared_utm_10m" if shared_balkan_grid else "source_grid"
    )
    execution_strategy = (
        "full_shared_analysis_grid" if shared_balkan_grid else "windowed_source_grid"
    )
    with rasterio.open(source_path) as source, ExitStack() as resources:
        if max(indices) > source.count or len(set(indices)) != 4:
            raise ValueError("Cloud source-band indices do not match the GeoTIFF")
        if source.crs is None:
            raise ValueError("Cloud source GeoTIFF must have a CRS")
        source_width = source.width
        source_height = source.height
        nodata_value = plan["input"].get("nodata_value")
        if nodata_value is None:
            nodata_value = config["input"].get("nodata_value", source.nodata)

        analysis_source: rasterio.DatasetReader = source
        analysis_indices = indices
        analysis_semantic_path = semantic_path
        analysis_unusable_path = unusable_path
        analysis_invalid_path = invalid_path
        if plan["sensor"] == "balkan-1" and not shared_balkan_grid:
            target_resolution = float(config["input"].get("balkan_target_resolution_m", 10.0))
            if target_resolution <= 0:
                raise ValueError("Balkan cloud target resolution must be positive")
            target_crs = _utm_crs(source.crs, source.bounds)
            target_transform, target_width, target_height = calculate_default_transform(
                source.crs,
                target_crs,
                source.width,
                source.height,
                *source.bounds,
                resolution=target_resolution,
            )
            temporary_directory = Path(
                resources.enter_context(
                    tempfile.TemporaryDirectory(
                        prefix=f".{stem}_model_grid_",
                        dir=semantic_path.parent,
                    )
                )
            )
            analysis_input_path = temporary_directory / "input_rgnb.tif"
            preparation_started = time.perf_counter()
            with rasterio.open(
                analysis_input_path,
                "w",
                driver="GTiff",
                width=target_width,
                height=target_height,
                count=4,
                dtype="float32",
                crs=target_crs,
                transform=target_transform,
                nodata=nodata_value,
                BIGTIFF="IF_SAFER",
            ) as analysis_output:
                for output_index, source_index in enumerate(indices, start=1):
                    reproject(
                        source=rasterio.band(source, source_index),
                        destination=rasterio.band(analysis_output, output_index),
                        src_transform=source.transform,
                        src_crs=source.crs,
                        src_nodata=nodata_value,
                        dst_transform=target_transform,
                        dst_crs=target_crs,
                        dst_nodata=nodata_value,
                        resampling=Resampling.bilinear,
                        num_threads=GDAL_WARP_THREADS,
                        init_dest_nodata=True,
                    )
                    analysis_output.set_band_description(
                        output_index,
                        plan["input"]["logical_band_order"][output_index - 1],
                    )
            analysis_grid_preparation_seconds = time.perf_counter() - preparation_started
            analysis_source = resources.enter_context(rasterio.open(analysis_input_path))
            analysis_indices = [1, 2, 3, 4]
            analysis_semantic_path = temporary_directory / "semantic.tif"
            analysis_unusable_path = temporary_directory / "unusable.tif"
            analysis_invalid_path = temporary_directory / "invalid.tif"
            analysis_grid_mode = "balkan_1_utm_10m"
            execution_strategy = "full_resampled_analysis_grid"

        analysis_width = analysis_source.width
        analysis_height = analysis_source.height
        analysis_crs = str(analysis_source.crs)
        analysis_resolution = [
            abs(float(analysis_source.res[0])),
            abs(float(analysis_source.res[1])),
        ]
        profile = _output_profile(analysis_source.profile)
        with ExitStack() as outputs:
            semantic_output = outputs.enter_context(
                rasterio.open(analysis_semantic_path, "w", **profile)
            )
            unusable_output = outputs.enter_context(
                rasterio.open(analysis_unusable_path, "w", **profile)
            )
            invalid_output = outputs.enter_context(
                rasterio.open(analysis_invalid_path, "w", **profile)
            )
            semantic_output.set_band_description(
                1, "0 clear, 1 thick, 2 thin, 3 shadow, 255 invalid"
            )
            unusable_output.set_band_description(1, "0 usable, 1 unusable")
            invalid_output.set_band_description(1, "0 valid input, 1 invalid input")

            use_full_analysis_grid = plan["sensor"] == "balkan-1"
            if use_full_analysis_grid:
                maximum_pixels = int(config["input"].get("balkan_max_analysis_pixels", 25_000_000))
                if analysis_width * analysis_height > maximum_pixels:
                    raise ValueError(
                        "Balkan 10 m cloud-analysis grid exceeds the reviewed "
                        f"{maximum_pixels}-pixel memory bound"
                    )
                raw_image = analysis_source.read(analysis_indices)
                image, invalid = normalize_reflectance(
                    raw_image,
                    scale=scale,
                    clip_min=config["input"].get("clip_min"),
                    clip_max=config["input"].get("clip_max"),
                    nodata_value=nodata_value,
                )
                if bool(config["input"].get("strict_positive_rgn", True)):
                    invalid |= ~strict_valid_mask(image[[1, 2, 0]])
                    image[:, invalid] = 0.0
                _synchronize_cuda(backend)
                inference_started = time.perf_counter()
                semantic, score_kind = _predict_semantic(
                    backend,
                    image.astype(np.float32, copy=False),
                )
                _synchronize_cuda(backend)
                inference_seconds += time.perf_counter() - inference_started
                mask_started = time.perf_counter()
                unusable = postprocess(
                    semantic,
                    classes,
                    config["postprocessing"],
                    invalid,
                )
                mask_processing_seconds += time.perf_counter() - mask_started
                valid = ~invalid
                class_counts += np.bincount(semantic[valid], minlength=4)[:4]
                invalid_count += int(invalid.sum())
                unusable_count += int(unusable.sum())
                semantic[invalid] = 255
                semantic_output.write(semantic, 1)
                unusable_output.write(unusable.astype(np.uint8), 1)
                invalid_output.write(invalid.astype(np.uint8), 1)
                tile_count = 1
            else:
                for y in range(0, analysis_height, core_size):
                    core_height = min(core_size, analysis_height - y)
                    for x in range(0, analysis_width, core_size):
                        core_width = min(core_size, analysis_width - x)
                        raw_tile = _read_padded_tile(
                            analysis_source,
                            analysis_indices,
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
                        if bool(config["input"].get("strict_positive_rgn", True)):
                            invalid |= ~strict_valid_mask(image[[1, 2, 0]])
                            image[:, invalid] = 0.0
                        _synchronize_cuda(backend)
                        inference_started = time.perf_counter()
                        semantic_tile, prediction_score_kind = _predict_semantic(
                            backend,
                            image.astype(np.float32, copy=False),
                        )
                        _synchronize_cuda(backend)
                        inference_seconds += time.perf_counter() - inference_started
                        if score_kind is None:
                            score_kind = prediction_score_kind
                        elif score_kind != prediction_score_kind:
                            raise ValueError("Cloud backend returned mixed score kinds")

                        mask_started = time.perf_counter()
                        unusable_tile = postprocess(
                            semantic_tile,
                            classes,
                            config["postprocessing"],
                            invalid,
                        )
                        mask_processing_seconds += time.perf_counter() - mask_started
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
                        unusable_output.write(
                            unusable_core.astype(np.uint8), 1, window=output_window
                        )
                        invalid_output.write(invalid_core.astype(np.uint8), 1, window=output_window)
                        tile_count += 1

        if analysis_grid_mode == "balkan_1_utm_10m":
            reprojection_started = time.perf_counter()
            _reproject_mask_to_source(
                analysis_semantic_path,
                source,
                semantic_path,
                description="0 clear, 1 thick, 2 thin, 3 shadow, 255 invalid",
            )
            _reproject_mask_to_source(
                analysis_unusable_path,
                source,
                unusable_path,
                description="0 usable, 1 unusable",
            )
            _reproject_mask_to_source(
                analysis_invalid_path,
                source,
                invalid_path,
                description="0 valid input, 1 invalid input",
            )
            reprojection_seconds = time.perf_counter() - reprojection_started

    width = source_width
    height = source_height
    analysis_total_pixels = analysis_width * analysis_height
    valid_pixels = analysis_total_pixels - invalid_count
    class_fractions = {
        name: float(class_counts[class_index] / valid_pixels) if valid_pixels else 0.0
        for name, class_index in classes.items()
    }
    class_percentages = {name: 100.0 * fraction for name, fraction in class_fractions.items()}
    unusable_percentage = _percentage(unusable_count, analysis_total_pixels)
    decision_config = config["decision"]
    if unusable_percentage >= float(decision_config["reject_min_unusable_percentage"]):
        decision = "REJECT"
    elif unusable_percentage > float(decision_config["process_max_unusable_percentage"]):
        decision = "PROCESS_CLEAR_AREAS"
    else:
        decision = "PROCESS"

    preview_seconds = 0.0
    if save_diagnostic_preview:
        preview_started = time.perf_counter()
        from cloud_detection.preview import save_preview

        preview_scale = min(1.0, 1200.0 / max(width, height))
        preview_height = max(1, round(height * preview_scale))
        preview_width = max(1, round(width * preview_scale))
        with rasterio.open(source_path) as source:
            preview_raw = source.read(
                indices,
                out_shape=(4, preview_height, preview_width),
                resampling=Resampling.bilinear,
            )
        preview_image, _ = normalize_reflectance(
            preview_raw,
            scale=scale,
            clip_min=config["input"].get("clip_min"),
            clip_max=config["input"].get("clip_max"),
            nodata_value=nodata_value,
        )
        with (
            rasterio.open(semantic_path) as semantic_source,
            rasterio.open(unusable_path) as unusable_source,
        ):
            semantic_preview = semantic_source.read(
                1,
                out_shape=(preview_height, preview_width),
                resampling=Resampling.nearest,
            )
            unusable_preview = unusable_source.read(
                1,
                out_shape=(preview_height, preview_width),
                resampling=Resampling.nearest,
            )
        save_preview(
            preview_path,
            preview_image,
            semantic_preview,
            unusable_preview,
            channelwise_rgb=plan["sensor"] == "balkan-1",
        )
        preview_seconds = time.perf_counter() - preview_started

    metadata = {
        "schema_version": "0.1-draft",
        "stage": "cloud_detection",
        "scene_id": plan["scene_id"],
        "sensor": plan["sensor"],
        "compatibility": plan["compatibility"],
        "validation_status": plan["validation_status"],
        "model": {
            "name": config["model"]["name"],
            "package_version": str(config["model"]["package_version"]),
            "model_version": str(config["model"].get("model_version", "unknown")),
            "ensemble_sha256": config["model"].get("expected_sha256"),
        },
        "score_kind": score_kind,
        "raster": {
            "width": width,
            "height": height,
            "total_pixels": width * height,
        },
        "analysis_grid": {
            "mode": analysis_grid_mode,
            "crs": analysis_crs,
            "resolution": analysis_resolution,
            "width": analysis_width,
            "height": analysis_height,
            "total_pixels": analysis_total_pixels,
            "output_masks_reprojected_to_source_grid": (
                analysis_grid_mode == "balkan_1_utm_10m"
            ),
        },
        "runtime": {
            "seconds": time.perf_counter() - started,
            "inference_seconds": inference_seconds,
            "mask_processing_seconds": mask_processing_seconds,
            "analysis_grid_preparation_seconds": analysis_grid_preparation_seconds,
            "mask_reprojection_seconds": reprojection_seconds,
            "preview_seconds": preview_seconds,
            "device": str(getattr(backend, "device", "unknown")),
            "inference_dtype": str(getattr(backend, "inference_dtype", "unknown")),
            "execution_strategy": execution_strategy,
            "tile_count": tile_count,
            "tile_size": tile_size,
            "halo": halo,
        },
        "class_fractions": class_fractions,
        "class_percentages": class_percentages,
        "cloud_percentage": 100.0
        * (class_fractions["thick_cloud"] + class_fractions["thin_cloud"]),
        "shadow_percentage": 100.0 * class_fractions["cloud_shadow"],
        "invalid_percentage": _percentage(invalid_count, analysis_total_pixels),
        "usable_percentage": 100.0 - unusable_percentage,
        "unusable_percentage": unusable_percentage,
        "decision": decision,
        "output_files": {
            "semantic_mask": str(semantic_path.resolve()),
            "unusable_mask": str(unusable_path.resolve()),
            "invalid_mask": str(invalid_path.resolve()),
            "metadata": str(metadata_path.resolve()),
            **(
                {"preview": str(preview_path.resolve())}
                if save_diagnostic_preview
                else {}
            ),
        },
        "warnings": plan.get("warnings", []),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata

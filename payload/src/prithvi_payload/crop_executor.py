"""Bounded-memory crop segmentation with cloud-mask exclusion."""

from __future__ import annotations

import json
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from rasterio.enums import Resampling
from rasterio.warp import transform as transform_coordinates
from rasterio.windows import Window as RasterWindow

from prithvi_payload.inference import PayloadCropModel
from prithvi_shared import NORMALIZATION_MEANS

FLOAT_NODATA = -9999.0
BYTE_NODATA = 255


def _read_padded(
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
    mode = "reflect" if values.shape[-2] > 1 and values.shape[-1] > 1 else "edge"
    return np.pad(values, ((0, 0), (top, bottom), (left, right)), mode=mode)


def _location_coordinate(
    source: rasterio.DatasetReader,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
) -> list[float]:
    center_x, center_y = source.xy(
        y + (height - 1) / 2,
        x + (width - 1) / 2,
    )
    longitudes, latitudes = transform_coordinates(
        source.crs,
        "EPSG:4326",
        [center_x],
        [center_y],
    )
    return [float(latitudes[0]), float(longitudes[0])]


def _probability_profile(source: rasterio.DatasetReader) -> dict[str, Any]:
    profile = source.profile.copy()
    profile.update(
        count=1,
        dtype="float32",
        nodata=FLOAT_NODATA,
        compress="deflate",
        BIGTIFF="IF_SAFER",
    )
    return profile


def _binary_profile(source: rasterio.DatasetReader) -> dict[str, Any]:
    profile = source.profile.copy()
    profile.update(
        count=1,
        dtype="uint8",
        nodata=BYTE_NODATA,
        compress="deflate",
        BIGTIFF="IF_SAFER",
    )
    return profile


def _preview_rgb(values: np.ndarray) -> np.ndarray:
    rgb = values.transpose(1, 2, 0).astype(np.float32, copy=False)
    finite = rgb[np.isfinite(rgb)]
    if not finite.size:
        return np.zeros_like(rgb)
    low, high = np.percentile(finite, [2, 98])
    return np.clip((rgb - low) / max(float(high - low), 1e-6), 0, 1)


def _save_preview(
    path: Path,
    rgb: np.ndarray,
    unusable: np.ndarray,
    probability: np.ndarray,
    binary: np.ndarray,
) -> None:
    figure = Figure(figsize=(19, 5.5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 4)
    axes[0].imshow(rgb)
    axes[0].set_title("RGB composite")
    axes[1].imshow(unusable, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Cloud/invalid exclusion mask")
    probability_image = np.ma.masked_equal(probability, FLOAT_NODATA)
    display = axes[2].imshow(probability_image, cmap="viridis", vmin=0, vmax=1)
    axes[2].set_title("Crop probability")
    figure.colorbar(display, ax=axes[2], fraction=0.046, pad=0.04)
    binary_map = ListedColormap(["#3d3d3d", "#43a047"])
    binary_image = np.ma.masked_equal(binary, BYTE_NODATA)
    axes[3].imshow(binary_image, cmap=binary_map, vmin=0, vmax=1)
    axes[3].set_title("Crop mask (green = crop)")
    for axis in axes:
        axis.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)


def execute_crop_stage(
    plan: dict[str, Any],
    *,
    output_root: str | Path,
    model: PayloadCropModel | None = None,
) -> dict[str, Any]:
    """Execute a ready crop plan and mask every unusable output pixel."""
    if plan.get("readiness") != "READY":
        raise ValueError("Crop execution requires a READY crop-stage plan")

    output_root = Path(output_root)
    stem = str(plan["scene_id"])
    probability_path = output_root / "crop_maps" / f"{stem}_probability.tif"
    binary_path = output_root / "crop_maps" / f"{stem}_binary.tif"
    confidence_path = output_root / "crop_maps" / f"{stem}_confidence.tif"
    preview_path = output_root / "visualisations" / f"{stem}_crop.png"
    metadata_path = output_root / "metadata" / f"{stem}_crop.json"
    for path in (
        probability_path,
        binary_path,
        confidence_path,
        preview_path,
        metadata_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    source_path = Path(plan["source_path"])
    unusable_path = Path(plan["input"]["unusable_mask"])
    indices = list(plan["input"]["source_band_indices_1_based"])
    multiplier = float(plan["input"]["training_scale_multiplier"])
    temporal_coordinate = plan["input"]["temporal_coordinate_year_doy"]
    tile_size = int(plan["execution"]["tile_size"])
    halo = int(plan["execution"]["halo"])
    batch_size = int(plan["execution"]["batch_size"])
    core_size = tile_size - 2 * halo
    if core_size <= 0:
        raise ValueError("Crop tile halo leaves no writable core")

    if model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = PayloadCropModel.load(device=device)
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    started = time.perf_counter()

    usable_count = 0
    crop_count = 0
    probability_sum = 0.0
    confidence_sum = 0.0
    tile_count = 0
    inferred_tile_count = 0
    skipped_tile_count = 0

    with (
        rasterio.open(source_path) as source,
        rasterio.open(unusable_path) as unusable_source,
        ExitStack() as stack,
    ):
        if max(indices) > source.count or len(set(indices)) != 4:
            raise ValueError("Crop source-band indices do not match the GeoTIFF")
        if (
            source.width != unusable_source.width
            or source.height != unusable_source.height
            or source.crs != unusable_source.crs
            or source.transform != unusable_source.transform
        ):
            raise ValueError("Crop input and unusable mask are not on the same grid")

        probability_output = stack.enter_context(
            rasterio.open(probability_path, "w", **_probability_profile(source))
        )
        binary_output = stack.enter_context(
            rasterio.open(binary_path, "w", **_binary_profile(source))
        )
        confidence_output = stack.enter_context(
            rasterio.open(confidence_path, "w", **_probability_profile(source))
        )
        probability_output.set_band_description(1, "crop probability; -9999 unusable")
        binary_output.set_band_description(1, "0 non-crop, 1 crop, 255 unusable")
        confidence_output.set_band_description(1, "winning-class confidence; -9999 unusable")

        pending: list[dict[str, Any]] = []

        def flush_pending() -> None:
            nonlocal crop_count, probability_sum, confidence_sum, inferred_tile_count
            if not pending:
                return
            images = torch.from_numpy(np.stack([item["image"] for item in pending]))
            images = images.unsqueeze(2)
            temporal = torch.tensor(
                [[temporal_coordinate] for _ in pending],
                dtype=torch.float32,
            )
            locations = torch.tensor(
                [item["location"] for item in pending],
                dtype=torch.float32,
            )
            prediction = model.predict(
                images,
                temporal_coords=temporal,
                location_coords=locations,
            )
            probabilities = prediction.crop_probability.float().cpu().numpy()
            binaries = prediction.crop_binary.cpu().numpy()
            confidences = prediction.crop_confidence.float().cpu().numpy()
            for index, item in enumerate(pending):
                core_slice = item["core_slice"]
                unusable_core = item["unusable_core"]
                probability_core = probabilities[index][core_slice].copy()
                binary_core = binaries[index][core_slice].copy()
                confidence_core = confidences[index][core_slice].copy()
                usable_core = ~unusable_core
                crop_count += int(np.count_nonzero(binary_core[usable_core] == 1))
                probability_sum += float(probability_core[usable_core].sum(dtype=np.float64))
                confidence_sum += float(confidence_core[usable_core].sum(dtype=np.float64))
                probability_core[unusable_core] = FLOAT_NODATA
                binary_core[unusable_core] = BYTE_NODATA
                confidence_core[unusable_core] = FLOAT_NODATA
                probability_output.write(probability_core, 1, window=item["window"])
                binary_output.write(binary_core, 1, window=item["window"])
                confidence_output.write(confidence_core, 1, window=item["window"])
            inferred_tile_count += len(pending)
            pending.clear()

        means = np.asarray(NORMALIZATION_MEANS, dtype=np.float32)[:, None, None]
        for y in range(0, source.height, core_size):
            core_height = min(core_size, source.height - y)
            for x in range(0, source.width, core_size):
                core_width = min(core_size, source.width - x)
                unusable_tile = _read_padded(
                    unusable_source,
                    [1],
                    y=y,
                    x=x,
                    tile_size=tile_size,
                    halo=halo,
                )[0].astype(bool)
                core_slice = (
                    slice(halo, halo + core_height),
                    slice(halo, halo + core_width),
                )
                unusable_core = unusable_tile[core_slice]
                output_window = RasterWindow(x, y, core_width, core_height)
                usable_in_core = int(np.count_nonzero(~unusable_core))
                usable_count += usable_in_core
                tile_count += 1
                if usable_in_core == 0:
                    probability_output.write(
                        np.full((core_height, core_width), FLOAT_NODATA, dtype=np.float32),
                        1,
                        window=output_window,
                    )
                    binary_output.write(
                        np.full((core_height, core_width), BYTE_NODATA, dtype=np.uint8),
                        1,
                        window=output_window,
                    )
                    confidence_output.write(
                        np.full((core_height, core_width), FLOAT_NODATA, dtype=np.float32),
                        1,
                        window=output_window,
                    )
                    skipped_tile_count += 1
                    continue

                raw_tile = _read_padded(
                    source,
                    indices,
                    y=y,
                    x=x,
                    tile_size=tile_size,
                    halo=halo,
                ).astype(np.float32, copy=False)
                image = raw_tile * np.float32(multiplier)
                invalid = ~np.isfinite(image).all(axis=0)
                if source.nodata is not None:
                    invalid |= np.all(raw_tile == source.nodata, axis=0)
                image = image.copy()
                image[:, invalid] = means[:, 0, 0][:, None]
                pending.append(
                    {
                        "image": image,
                        "location": _location_coordinate(
                            source,
                            x=x,
                            y=y,
                            width=core_width,
                            height=core_height,
                        ),
                        "core_slice": core_slice,
                        "unusable_core": unusable_core,
                        "window": output_window,
                    }
                )
                if len(pending) >= batch_size:
                    flush_pending()
        flush_pending()
        width, height = source.width, source.height

        preview_scale = min(1.0, 1200.0 / max(width, height))
        preview_height = max(1, round(height * preview_scale))
        preview_width = max(1, round(width * preview_scale))
        rgb_indices = [indices[2], indices[1], indices[0]]
        rgb = source.read(
            rgb_indices,
            out_shape=(3, preview_height, preview_width),
            resampling=Resampling.bilinear,
        )
        unusable_preview = unusable_source.read(
            1,
            out_shape=(preview_height, preview_width),
            resampling=Resampling.nearest,
        )

    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    runtime_seconds = time.perf_counter() - started
    with (
        rasterio.open(probability_path) as probability_source,
        rasterio.open(binary_path) as binary_source,
    ):
        probability_preview = probability_source.read(
            1,
            out_shape=(preview_height, preview_width),
            resampling=Resampling.bilinear,
        )
        binary_preview = binary_source.read(
            1,
            out_shape=(preview_height, preview_width),
            resampling=Resampling.nearest,
        )
    _save_preview(
        preview_path,
        _preview_rgb(rgb),
        unusable_preview,
        probability_preview,
        binary_preview,
    )

    total_pixels = width * height
    metadata = {
        "schema_version": "0.1-draft",
        "stage": "crop_classification",
        "scene_id": plan["scene_id"],
        "sensor": plan["sensor"],
        "model": plan["model"],
        "gate": plan["gate"],
        "mask_application": {
            "source": str(unusable_path.resolve()),
            "semantics": "0 usable, 1 unusable",
            "unusable_pixels_excluded_from_inference_output": True,
            "cloud_pixels_preserved_during_inference": True,
            "invalid_input_pixels_replaced_with_training_mean": True,
        },
        "raster": {"width": width, "height": height, "total_pixels": total_pixels},
        "runtime": {
            "seconds": runtime_seconds,
            "device": str(model.device),
            "tile_count": tile_count,
            "inferred_tile_count": inferred_tile_count,
            "fully_masked_tile_count": skipped_tile_count,
            "tile_size": tile_size,
            "halo": halo,
            "batch_size": batch_size,
        },
        "usable_pixels": usable_count,
        "usable_percentage": 100.0 * usable_count / total_pixels if total_pixels else 0.0,
        "crop_pixels": crop_count,
        "crop_fraction_usable": crop_count / usable_count if usable_count else None,
        "crop_percentage_usable": 100.0 * crop_count / usable_count if usable_count else None,
        "mean_crop_probability_usable": probability_sum / usable_count if usable_count else None,
        "mean_confidence_usable": confidence_sum / usable_count if usable_count else None,
        "output_files": {
            "crop_probability": str(probability_path.resolve()),
            "crop_binary": str(binary_path.resolve()),
            "crop_confidence": str(confidence_path.resolve()),
            "preview": str(preview_path.resolve()),
            "metadata": str(metadata_path.resolve()),
        },
        "warnings": plan.get("warnings", []),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata

"""Bounded-memory crop segmentation with cloud-mask exclusion."""

from __future__ import annotations

import json
import os
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from prithvi_shared import NORMALIZATION_MEANS
from rasterio.enums import Resampling
from rasterio.warp import (
    transform as transform_coordinates,
)
from rasterio.windows import Window as RasterWindow

from prithvi_payload.balkan_crop_calibration import (
    ADAPTER_MODE,
    apply_calibration,
    load_calibration,
)
from prithvi_payload.inference import PayloadCropModel
from prithvi_payload.raster_ops import read_padded_array, read_padded_tile
from prithvi_payload.runtime_config import environment_flag

FLOAT_NODATA = -9999.0
BYTE_NODATA = 255


def _tile_blend_weights(tile_size: int, halo: int) -> np.ndarray:
    """Return deterministic edge-tapered weights for overlapping model tiles."""
    overlap = 2 * halo
    if tile_size <= 0 or halo < 0 or overlap >= tile_size:
        raise ValueError("Invalid tile size or halo for probability blending")
    axis = np.ones(tile_size, dtype=np.float32)
    if overlap:
        ramp = np.arange(1, overlap + 1, dtype=np.float32) / np.float32(overlap + 1)
        axis[:overlap] = ramp
        axis[-overlap:] = ramp[::-1]
    return np.multiply.outer(axis, axis)


def _accumulate_prediction(
    probability_sum: np.ndarray,
    probability_weight: np.ndarray,
    tile_probability: np.ndarray,
    tile_weight: np.ndarray,
    *,
    requested_y: int,
    requested_x: int,
) -> None:
    """Blend the in-bounds part of one prediction tile into scene accumulators."""
    source_y_start = max(0, requested_y)
    source_x_start = max(0, requested_x)
    source_y_end = min(probability_sum.shape[0], requested_y + tile_probability.shape[0])
    source_x_end = min(probability_sum.shape[1], requested_x + tile_probability.shape[1])
    if source_y_start >= source_y_end or source_x_start >= source_x_end:
        return
    tile_y_start = source_y_start - requested_y
    tile_x_start = source_x_start - requested_x
    tile_y_end = tile_y_start + source_y_end - source_y_start
    tile_x_end = tile_x_start + source_x_end - source_x_start
    source_slice = np.s_[source_y_start:source_y_end, source_x_start:source_x_end]
    tile_slice = np.s_[tile_y_start:tile_y_end, tile_x_start:tile_x_end]
    weights = tile_weight[tile_slice]
    probability_sum[source_slice] += tile_probability[tile_slice] * weights
    probability_weight[source_slice] += weights


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
        BIGTIFF="IF_SAFER",
    )
    if environment_flag("VITA_FAST_INTERMEDIATE_RASTERS", False):
        for option in ("compress", "predictor", "zlevel", "num_threads"):
            profile.pop(option, None)
    else:
        profile.update(
            compress="deflate",
            predictor=3,
            zlevel=1,
            num_threads="ALL_CPUS",
        )
    return profile


def _binary_profile(source: rasterio.DatasetReader) -> dict[str, Any]:
    profile = source.profile.copy()
    profile.update(
        count=1,
        dtype="uint8",
        nodata=BYTE_NODATA,
        BIGTIFF="IF_SAFER",
    )
    if environment_flag("VITA_FAST_INTERMEDIATE_RASTERS", False):
        for option in ("compress", "predictor", "zlevel", "num_threads"):
            profile.pop(option, None)
    else:
        profile.update(
            compress="deflate",
            predictor=2,
            zlevel=1,
            num_threads="ALL_CPUS",
        )
    return profile


def _preview_rgb(values: np.ndarray) -> np.ndarray:
    rgb = values.transpose(1, 2, 0).astype(np.float32, copy=False)
    finite = rgb[np.isfinite(rgb)]
    if not finite.size:
        return np.zeros_like(rgb)
    low, high = np.percentile(finite, [2, 98])
    return np.clip((rgb - low) / max(float(high - low), 1e-6), 0, 1)


def _crop_probability_overlay(probability: np.ndarray) -> np.ndarray:
    """Create a display-only smooth green overlay from crop probability."""
    values = np.asarray(probability, dtype=np.float32)
    valid = np.isfinite(values) & (values >= 0.0) & (values <= 1.0)
    clipped = np.clip(values, 0.0, 1.0)
    overlay = np.zeros((*values.shape, 4), dtype=np.float32)
    overlay[..., :3] = (0.0, 200.0 / 255.0, 83.0 / 255.0)
    overlay[..., 3] = np.where(valid, 0.08 + 0.62 * clipped, 0.0)
    return overlay


def _crop_display(binary: np.ndarray) -> np.ndarray:
    """Map exact binary/nodata values to consecutive display categories."""
    values = np.asarray(binary)
    display = np.full(values.shape, 2, dtype=np.uint8)
    display[values == 0] = 0
    display[values == 1] = 1
    return display


def _crop_legend(binary: np.ndarray) -> list[Any]:
    from matplotlib.patches import Patch

    values = np.asarray(binary)
    usable = np.isin(values, (0, 1))
    usable_count = int(np.count_nonzero(usable))
    crop_count = int(np.count_nonzero(values == 1))
    total = max(values.size, 1)
    crop_percentage = 100.0 * crop_count / usable_count if usable_count else 0.0
    excluded_percentage = 100.0 * np.count_nonzero(~usable) / total
    return [
        Patch(facecolor="#30343b", label="Non-crop"),
        Patch(facecolor="#43a047", label=f"Crop: {crop_percentage:.1f}% of usable"),
        Patch(
            facecolor="#f4f6f8", edgecolor="#777777", label=f"Excluded: {excluded_percentage:.1f}%"
        ),
    ]


def _crop_threshold(plan: dict[str, Any]) -> float:
    value = plan.get("model", {}).get("crop_probability_threshold")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError("Crop plan has an invalid probability threshold")
    return float(value)


def _save_preview(
    path: Path,
    rgb: np.ndarray,
    unusable: np.ndarray,
    probability: np.ndarray,
    binary: np.ndarray,
    crop_threshold: float,
) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import ListedColormap
    from matplotlib.figure import Figure

    figure = Figure(figsize=(14, 11), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2).ravel()
    axes[0].imshow(rgb)
    axes[0].set_title("RGB composite")
    excluded = unusable == 1
    display_probability = np.where(excluded, FLOAT_NODATA, probability)
    display_binary = np.where(excluded, BYTE_NODATA, binary)
    probability_image = np.ma.masked_equal(display_probability, FLOAT_NODATA)
    display = axes[1].imshow(
        probability_image,
        cmap="viridis",
        vmin=0,
        vmax=1,
        interpolation="bilinear",
    )
    axes[1].set_title("Continuous crop probability")
    colorbar = figure.colorbar(display, ax=axes[1], fraction=0.046, pad=0.04)
    colorbar.set_label("0 = low crop likelihood\n1 = high crop likelihood")
    axes[2].imshow(rgb)
    axes[2].imshow(_crop_probability_overlay(display_probability), interpolation="bilinear")
    accepted = display_binary == 1
    if np.any(accepted) and np.any(~accepted):
        axes[2].contour(accepted.astype(np.uint8), levels=[0.5], colors="#00e676", linewidths=0.8)
    axes[2].set_title("Crop-likelihood overlay\ngreen edge = accepted crop")
    axes[3].imshow(
        _crop_display(display_binary),
        cmap=ListedColormap(["#30343b", "#43a047", "#f4f6f8"]),
        vmin=-0.5,
        vmax=2.5,
        interpolation="nearest",
    )
    axes[3].set_title(f"Exact crop mask at p >= {crop_threshold:.2f}")
    for axis in axes:
        axis.axis("off")
    figure.legend(
        handles=_crop_legend(display_binary),
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        "Crop detail preview - exact mask uses nearest-neighbor display; "
        "probability overlay smoothing is visual only",
        fontsize=13,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)


def _execute_native_crop_stage(
    plan: dict[str, Any],
    *,
    output_root: str | Path,
    model: PayloadCropModel | None = None,
    persist_rasters: bool = True,
    cloud_products: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Execute a ready crop plan and mask every unusable output pixel."""
    if plan.get("readiness") != "READY":
        raise ValueError("Crop execution requires a READY crop-stage plan")

    output_root = Path(output_root)
    stem = str(plan["scene_id"])
    probability_path = output_root / "crop_maps" / f"{stem}_probability.tif"
    binary_path = output_root / "crop_maps" / f"{stem}_binary.tif"
    confidence_path = output_root / "crop_maps" / f"{stem}_confidence.tif"
    blend_sum_path = output_root / "crop_maps" / f".{stem}_blend_sum.partial"
    blend_weight_path = output_root / "crop_maps" / f".{stem}_blend_weight.partial"
    preview_path = output_root / "visualisations" / f"{stem}_crop.png"
    metadata_path = output_root / "metadata" / f"{stem}_crop.json"
    save_diagnostic_preview = bool(plan.get("execution", {}).get("save_preview", False))
    if save_diagnostic_preview and not persist_rasters:
        raise ValueError("Diagnostic crop previews require persisted crop rasters")
    output_paths = (
        [probability_path, binary_path, confidence_path, metadata_path]
        if persist_rasters
        else [metadata_path]
    )
    if save_diagnostic_preview:
        output_paths.append(preview_path)
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    blend_sum_path.unlink(missing_ok=True)
    blend_weight_path.unlink(missing_ok=True)

    source_path = Path(plan["source_path"])
    unusable_value = plan["input"].get("unusable_mask")
    unusable_path = Path(unusable_value) if isinstance(unusable_value, str) else None
    if unusable_path is None and cloud_products is None:
        raise ValueError("Crop execution requires a persisted or in-memory unusable mask")
    indices = list(plan["input"]["source_band_indices_1_based"])
    multiplier = float(plan["input"]["training_scale_multiplier"])
    adapter = plan["input"].get("spectral_adapter")
    calibration = None
    if (
        plan.get("sensor") == "balkan-1"
        and isinstance(adapter, dict)
        and adapter.get("mode") == ADAPTER_MODE
    ):
        calibration_source_path = plan["input"].get("calibration_source_path")
        if not isinstance(calibration_source_path, str) or not calibration_source_path:
            raise ValueError("Balkan crop calibration source is missing")
        calibration = load_calibration(
            adapter["calibration_path"],
            source_path=calibration_source_path,
        )
    temporal_coordinate = plan["input"]["temporal_coordinate_year_doy"]
    tile_size = int(plan["execution"]["tile_size"])
    halo = int(plan["execution"]["halo"])
    batch_size = int(plan["execution"]["batch_size"])
    crop_threshold = _crop_threshold(plan)
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
    probability_total = 0.0
    confidence_total = 0.0
    tile_count = 0
    inferred_tile_count = 0
    skipped_tile_count = 0
    inference_seconds = 0.0

    with rasterio.open(source_path) as source, ExitStack() as stack:
        unusable_source = (
            stack.enter_context(rasterio.open(unusable_path))
            if unusable_path is not None
            else None
        )
        if max(indices) > source.count or len(set(indices)) != 4:
            raise ValueError("Crop source-band indices do not match the GeoTIFF")
        if unusable_source is not None:
            if (
                source.width != unusable_source.width
                or source.height != unusable_source.height
                or source.crs != unusable_source.crs
                or source.transform != unusable_source.transform
            ):
                raise ValueError("Crop input and unusable mask are not on the same grid")
            cloud_unusable_product = None
        else:
            assert cloud_products is not None
            cloud_unusable_product = np.asarray(
                cloud_products["unusable_mask"], dtype=np.uint8
            )
            if cloud_unusable_product.shape != (source.height, source.width):
                raise ValueError("In-memory unusable mask is not on the crop grid")

        probability_output = None
        binary_output = None
        confidence_output = None
        if persist_rasters:
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
            confidence_output.set_band_description(
                1, "winning-class confidence; -9999 unusable"
            )
            probability_product = None
            binary_product = None
        else:
            probability_product = np.full(
                (source.height, source.width), FLOAT_NODATA, dtype=np.float32
            )
            binary_product = np.full(
                (source.height, source.width), BYTE_NODATA, dtype=np.uint8
            )
            if cloud_products is not None and "source_bands" in cloud_products:
                cloud_indices = tuple(
                    int(value) for value in cloud_products["source_band_indices"]
                )
                try:
                    source_positions = [cloud_indices.index(index) for index in indices]
                except ValueError as error:
                    raise ValueError(
                        "In-memory cloud source does not provide all crop bands"
                    ) from error
                source_band_product = np.asarray(
                    cloud_products["source_bands"], dtype=np.float32
                )[source_positions]
                source_valid_product = np.asarray(
                    cloud_products["source_valid_mask"], dtype=bool
                )
            else:
                source_band_product = source.read(
                    indices,
                    out_dtype="float32",
                )
                source_valid_product = np.all(
                    source.read_masks(indices) > 0,
                    axis=0,
                )
            if (
                source_band_product.shape != (4, source.height, source.width)
                or source_valid_product.shape != (source.height, source.width)
            ):
                raise ValueError("In-memory source bands are not on the crop grid")
            model_band_product = source_band_product * np.float32(multiplier)
            if calibration is not None:
                model_band_product = apply_calibration(
                    model_band_product,
                    calibration,
                )
            unusable_product = np.ones(
                (source.height, source.width), dtype=np.uint8
            )

        blend_bytes = source.height * source.width * np.dtype(np.float32).itemsize * 2
        in_memory_limit = int(
            os.environ.get("VITA_CROP_IN_MEMORY_MAX_BYTES", str(1024**3))
        )
        use_in_memory_blend = (
            environment_flag("VITA_CROP_IN_MEMORY", True)
            and blend_bytes <= in_memory_limit
        )
        if use_in_memory_blend:
            probability_sum = np.zeros((source.height, source.width), dtype=np.float32)
            probability_weight = np.zeros_like(probability_sum)
        else:
            probability_sum = np.memmap(
                blend_sum_path,
                mode="w+",
                dtype=np.float32,
                shape=(source.height, source.width),
            )
            probability_sum[:] = 0.0
            stack.callback(blend_sum_path.unlink, missing_ok=True)
            stack.callback(probability_sum._mmap.close)
            probability_weight = np.memmap(
                blend_weight_path,
                mode="w+",
                dtype=np.float32,
                shape=(source.height, source.width),
            )
            probability_weight[:] = 0.0
            stack.callback(blend_weight_path.unlink, missing_ok=True)
            stack.callback(probability_weight._mmap.close)
        tile_weight = _tile_blend_weights(tile_size, halo)

        pending: list[dict[str, Any]] = []

        def flush_pending() -> None:
            nonlocal inference_seconds, inferred_tile_count
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
            if model.device.type == "cuda":
                torch.cuda.synchronize(model.device)
            inference_started = time.perf_counter()
            prediction = model.predict(
                images,
                temporal_coords=temporal,
                location_coords=locations,
            )
            if model.device.type == "cuda":
                torch.cuda.synchronize(model.device)
            inference_seconds += time.perf_counter() - inference_started
            probabilities = prediction.crop_probability.float().cpu().numpy()
            for index, item in enumerate(pending):
                _accumulate_prediction(
                    probability_sum,
                    probability_weight,
                    probabilities[index],
                    tile_weight,
                    requested_y=item["requested_y"],
                    requested_x=item["requested_x"],
                )
            inferred_tile_count += len(pending)
            pending.clear()

        means = np.asarray(NORMALIZATION_MEANS, dtype=np.float32)[:, None, None]
        for y in range(0, source.height, core_size):
            core_height = min(core_size, source.height - y)
            for x in range(0, source.width, core_size):
                core_width = min(core_size, source.width - x)
                unusable_tile = (
                    read_padded_tile(
                        unusable_source,
                        [1],
                        y=y,
                        x=x,
                        tile_size=tile_size,
                        halo=halo,
                        out_dtype="float32",
                    )[0].astype(bool)
                    if unusable_source is not None
                    else read_padded_array(
                        cloud_unusable_product[np.newaxis, ...],
                        y=y,
                        x=x,
                        tile_size=tile_size,
                        halo=halo,
                    )[0].astype(bool)
                )
                core_slice = (
                    slice(halo, halo + core_height),
                    slice(halo, halo + core_width),
                )
                unusable_core = unusable_tile[core_slice]
                output_window = RasterWindow(x, y, core_width, core_height)
                if not persist_rasters:
                    assert unusable_product is not None
                    unusable_product[
                        y : y + core_height,
                        x : x + core_width,
                    ] = unusable_core.astype(np.uint8)
                usable_in_core = int(np.count_nonzero(~unusable_core))
                tile_count += 1
                if usable_in_core == 0:
                    if persist_rasters:
                        assert probability_output is not None
                        assert binary_output is not None
                        assert confidence_output is not None
                        probability_output.write(
                            np.full(
                                (core_height, core_width),
                                FLOAT_NODATA,
                                dtype=np.float32,
                            ),
                            1,
                            window=output_window,
                        )
                        binary_output.write(
                            np.full(
                                (core_height, core_width),
                                BYTE_NODATA,
                                dtype=np.uint8,
                            ),
                            1,
                            window=output_window,
                        )
                        confidence_output.write(
                            np.full(
                                (core_height, core_width),
                                FLOAT_NODATA,
                                dtype=np.float32,
                            ),
                            1,
                            window=output_window,
                        )
                    skipped_tile_count += 1
                    continue

                if persist_rasters:
                    raw_tile = read_padded_tile(
                        source,
                        indices,
                        y=y,
                        x=x,
                        tile_size=tile_size,
                        halo=halo,
                        out_dtype="float32",
                    )
                    image = raw_tile * np.float32(multiplier)
                    if calibration is not None:
                        image = apply_calibration(image, calibration)
                else:
                    assert source_band_product is not None
                    assert model_band_product is not None
                    raw_tile = read_padded_array(
                        source_band_product,
                        y=y,
                        x=x,
                        tile_size=tile_size,
                        halo=halo,
                    )
                    image = read_padded_array(
                        model_band_product,
                        y=y,
                        x=x,
                        tile_size=tile_size,
                        halo=halo,
                    )
                invalid = ~np.isfinite(image).all(axis=0)
                if source.nodata is not None:
                    if calibration is not None:
                        invalid |= np.any(raw_tile == source.nodata, axis=0)
                    else:
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
                        "requested_y": y - halo,
                        "requested_x": x - halo,
                    }
                )
                if len(pending) >= batch_size:
                    flush_pending()
        flush_pending()
        if isinstance(probability_sum, np.memmap):
            probability_sum.flush()
        if isinstance(probability_weight, np.memmap):
            probability_weight.flush()

        for _, window in source.block_windows(1):
            row_slice, column_slice = window.toslices()
            weights = np.asarray(probability_weight[row_slice, column_slice])
            sums = np.asarray(probability_sum[row_slice, column_slice])
            unusable = (
                unusable_source.read(1, window=window).astype(bool)
                if unusable_source is not None
                else cloud_unusable_product[row_slice, column_slice].astype(bool)
            )
            has_prediction = weights > 0
            if np.any(~unusable & ~has_prediction):
                raise RuntimeError("Crop probability blending left usable pixels uncovered")
            probability = np.divide(
                sums,
                weights,
                out=np.zeros_like(sums),
                where=has_prediction,
            )
            binary = (probability >= crop_threshold).astype(np.uint8)
            confidence = np.maximum(probability, 1.0 - probability)
            usable = ~unusable & has_prediction
            usable_count += int(np.count_nonzero(usable))
            crop_count += int(np.count_nonzero(binary[usable] == 1))
            probability_total += float(probability[usable].sum(dtype=np.float64))
            confidence_total += float(confidence[usable].sum(dtype=np.float64))
            probability[~usable] = FLOAT_NODATA
            binary[~usable] = BYTE_NODATA
            confidence[~usable] = FLOAT_NODATA
            if persist_rasters:
                assert probability_output is not None
                assert binary_output is not None
                assert confidence_output is not None
                probability_output.write(probability, 1, window=window)
                binary_output.write(binary, 1, window=window)
                confidence_output.write(confidence, 1, window=window)
            else:
                assert probability_product is not None
                assert binary_product is not None
                probability_product[row_slice, column_slice] = probability
                binary_product[row_slice, column_slice] = binary
        width, height = source.width, source.height

        if save_diagnostic_preview:
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
    if save_diagnostic_preview:
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
            crop_threshold,
        )

    total_pixels = width * height
    metadata = {
        "schema_version": "0.1-draft",
        "stage": "crop_classification",
        "scene_id": plan["scene_id"],
        "sensor": plan["sensor"],
        "model": plan["model"],
        "gate": plan["gate"],
        "spectral_adapter": (
            {
                "mode": calibration["adapter_mode"],
                "calibration_path": adapter["calibration_path"],
                "source_sha256": calibration["source"]["sha256"],
                "reference": calibration["reference"],
                "validation": calibration["validation"],
                "output_grid": "shared_10m_analysis_grid",
            }
            if calibration is not None
            else None
        ),
        "mask_application": {
            "source": (
                str(unusable_path.resolve())
                if unusable_path is not None
                else "ephemeral_memory"
            ),
            "semantics": "0 usable, 1 unusable",
            "unusable_pixels_excluded_from_inference_output": True,
            "cloud_pixels_preserved_during_inference": True,
            "invalid_input_pixels_replaced_with_training_mean": True,
        },
        "raster": {"width": width, "height": height, "total_pixels": total_pixels},
        "runtime": {
            "seconds": runtime_seconds,
            "inference_seconds": inference_seconds,
            "device": str(model.device),
            "inference_backend": model.backend,
            "tensorrt_engine_count": model.tensorrt_engine_count,
            "tile_count": tile_count,
            "inferred_tile_count": inferred_tile_count,
            "fully_masked_tile_count": skipped_tile_count,
            "tile_size": tile_size,
            "halo": halo,
            "stitching_policy": "linear_overlap_weighted_probability",
            "overlap_pixels": 2 * halo,
            "batch_size": batch_size,
            "blend_storage": "memory" if use_in_memory_blend else "disk_memmap",
            "blend_buffer_bytes": blend_bytes,
        },
        "usable_pixels": usable_count,
        "usable_percentage": 100.0 * usable_count / total_pixels if total_pixels else 0.0,
        "crop_pixels": crop_count,
        "crop_fraction_usable": crop_count / usable_count if usable_count else None,
        "crop_percentage_usable": 100.0 * crop_count / usable_count if usable_count else None,
        "mean_crop_probability_usable": (
            probability_total / usable_count if usable_count else None
        ),
        "mean_confidence_usable": confidence_total / usable_count if usable_count else None,
        "output_files": {
            **(
                {
                    "crop_probability": str(probability_path.resolve()),
                    "crop_binary": str(binary_path.resolve()),
                    "crop_confidence": str(confidence_path.resolve()),
                }
                if persist_rasters
                else {}
            ),
            "metadata": str(metadata_path.resolve()),
            **(
                {"preview": str(preview_path.resolve())}
                if save_diagnostic_preview
                else {}
            ),
        },
        "warnings": plan.get("warnings", []),
    }
    if not persist_rasters:
        metadata["runtime"]["product_storage"] = "ephemeral_memory"
        metadata["_products"] = {
            "crop_probability": probability_product,
            "crop_binary": binary_product,
            "source_bands": source_band_product,
            "model_bands": model_band_product,
            "model_scale_multiplier": multiplier,
            "source_band_indices": tuple(indices),
            "source_valid_mask": source_valid_product,
            "unusable_mask": unusable_product,
            **(
                {
                    "semantic_mask": cloud_products["semantic_mask"],
                    "invalid_mask": cloud_products["invalid_mask"],
                }
                if cloud_products is not None
                else {}
            ),
        }
    serializable_metadata = {key: value for key, value in metadata.items() if key != "_products"}
    metadata_path.write_text(
        json.dumps(serializable_metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def execute_crop_stage(
    plan: dict[str, Any],
    *,
    output_root: str | Path,
    model: PayloadCropModel | None = None,
    persist_rasters: bool = True,
    cloud_products: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Execute crop inference on the plan's already prepared science grid."""
    return _execute_native_crop_stage(
        plan,
        output_root=output_root,
        model=model,
        persist_rasters=persist_rasters,
        cloud_products=cloud_products,
    )

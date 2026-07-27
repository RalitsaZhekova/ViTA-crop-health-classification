"""Windowed Phase 2 processing for one completed payload result."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import zlib
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

from prithvi_ground.condition import (
    CONDITION_ALGORITHM_VERSION,
    SCORED_INDEX_NAMES,
    ConditionConfig,
    build_condition_assessment,
    calculate_condition_score_layers,
    calculate_spatial_condition_layers,
)
from prithvi_ground.health import (
    ALGORITHM_VERSION as INDEX_ALGORITHM_VERSION,
)
from prithvi_ground.health import build_analysis_mask, calculate_health_layers

GROUND_SCENE_ALGORITHM_VERSION = "ground-scene-v1"
FLOAT_NODATA = -9999.0
BYTE_NODATA = 255
HEALTH_LAYER_NAMES = (
    "ndvi",
    "gndvi",
    "evi",
    "savi",
    "cvi",
    "vari",
    "excess_green",
    "rgb_brightness",
)


@dataclass
class StreamingMetric:
    """Exact moments plus deterministic priority-reservoir percentiles."""

    name: str
    sample_limit: int = 50_000

    def __post_init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.total_squared = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf
        self._values = np.empty(0, dtype=np.float64)
        self._priorities = np.empty(0, dtype=np.float64)
        self._seed = np.uint64(zlib.crc32(self.name.encode("utf-8")))
        self._next_sample_id = 0
        if self.sample_limit <= 0:
            raise ValueError("sample_limit must be positive")

    def update(
        self,
        values: np.ndarray,
        *,
        sample_ids: np.ndarray | None = None,
    ) -> None:
        flattened = np.asarray(values, dtype=np.float64).ravel()
        if sample_ids is None:
            identifiers = np.arange(
                self._next_sample_id,
                self._next_sample_id + flattened.size,
                dtype=np.uint64,
            )
            self._next_sample_id += flattened.size
        else:
            identifiers = np.asarray(sample_ids, dtype=np.uint64).ravel()
            if identifiers.shape != flattened.shape:
                raise ValueError("sample_ids must match the flattened metric values")
        finite_mask = np.isfinite(flattened)
        finite = flattened[finite_mask]
        identifiers = identifiers[finite_mask]
        if finite.size == 0:
            return
        self.count += int(finite.size)
        self.total += float(np.sum(finite, dtype=np.float64))
        self.total_squared += float(np.sum(finite * finite, dtype=np.float64))
        self.minimum = min(self.minimum, float(np.min(finite)))
        self.maximum = max(self.maximum, float(np.max(finite)))

        priorities = self._priorities_for_ids(identifiers)
        values_combined = np.concatenate((self._values, finite))
        priorities_combined = np.concatenate((self._priorities, priorities))
        if values_combined.size > self.sample_limit:
            selected = np.argpartition(priorities_combined, -self.sample_limit)[
                -self.sample_limit :
            ]
            values_combined = values_combined[selected]
            priorities_combined = priorities_combined[selected]
        self._values = values_combined
        self._priorities = priorities_combined

    def _priorities_for_ids(self, identifiers: np.ndarray) -> np.ndarray:
        """Return stable SplitMix64 priorities for spatial sample identifiers."""
        mixed = identifiers.astype(np.uint64, copy=True) + self._seed
        mixed ^= mixed >> np.uint64(30)
        mixed *= np.uint64(0xBF58476D1CE4E5B9)
        mixed ^= mixed >> np.uint64(27)
        mixed *= np.uint64(0x94D049BB133111EB)
        mixed ^= mixed >> np.uint64(31)
        return (mixed >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))

    def summary(self) -> dict[str, Any]:
        if self.count == 0:
            return {
                "valid_pixels": 0,
                "mean": None,
                "standard_deviation": None,
                "minimum": None,
                "percentile_10": None,
                "lower_quartile": None,
                "median": None,
                "percentile_90": None,
                "maximum": None,
                "percentile_sample_pixels": 0,
                "percentile_method": "deterministic_priority_reservoir",
            }
        mean = self.total / self.count
        variance = max(0.0, self.total_squared / self.count - mean * mean)
        percentiles = np.percentile(self._values, (10, 25, 50, 90))
        def rounded(value: float) -> float:
            return float(round(value, 12))

        return {
            "valid_pixels": self.count,
            "mean": rounded(mean),
            "standard_deviation": rounded(math.sqrt(variance)),
            "minimum": rounded(self.minimum),
            "percentile_10": rounded(float(percentiles[0])),
            "lower_quartile": rounded(float(percentiles[1])),
            "median": rounded(float(percentiles[2])),
            "percentile_90": rounded(float(percentiles[3])),
            "maximum": rounded(self.maximum),
            "percentile_sample_pixels": int(self._values.size),
            "percentile_method": "deterministic_priority_reservoir",
        }


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)


def _resolve_asset(value: Any, result_root: Path, *, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Payload result is missing {name}")
    path = Path(value)
    if not path.is_absolute():
        path = result_root / path
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path.resolve()


def _load_payload_result(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Payload result must contain a JSON object")
    if value.get("status") != "CROP_COMPLETE":
        raise ValueError("Ground processing requires payload status CROP_COMPLETE")
    completed = value.get("completed_stages")
    if not isinstance(completed, list) or "crop" not in completed:
        raise ValueError("Payload result does not record a completed crop stage")
    return value


def _validate_identifier(value: str, *, name: str) -> str:
    if not value or value in {".", ".."} or any(char in value for char in ("/", "\\")):
        raise ValueError(f"{name} must be a non-empty filename-safe identifier")
    return value


def _iter_windows(width: int, height: int, tile_size: int) -> Iterator[Window]:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    for row in range(0, height, tile_size):
        for column in range(0, width, tile_size):
            yield Window(
                column,
                row,
                min(tile_size, width - column),
                min(tile_size, height - row),
            )


def _raster_profile(
    source: rasterio.DatasetReader,
    *,
    dtype: str,
    nodata: float | int,
) -> dict[str, Any]:
    profile = source.profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype=dtype,
        nodata=nodata,
        compress="deflate",
        predictor=3 if dtype == "float32" else 2,
        BIGTIFF="IF_SAFER",
    )
    if source.width >= 16 and source.height >= 16:
        block_width = min(256, (source.width // 16) * 16)
        block_height = min(256, (source.height // 16) * 16)
        profile.update(
            tiled=True,
            blockxsize=block_width,
            blockysize=block_height,
        )
    else:
        profile.update(tiled=False)
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
    return profile


def _validate_same_grid(
    source: rasterio.DatasetReader,
    other: rasterio.DatasetReader,
    *,
    name: str,
) -> None:
    if (
        source.width != other.width
        or source.height != other.height
        or source.crs != other.crs
        or source.transform != other.transform
    ):
        raise ValueError(f"{name} is not on the source-scene grid")


def _window_pixel_ids(window: Window, width: int) -> np.ndarray:
    rows = np.arange(
        round(window.row_off),
        round(window.row_off + window.height),
        dtype=np.uint64,
    )[:, np.newaxis]
    columns = np.arange(
        round(window.col_off),
        round(window.col_off + window.width),
        dtype=np.uint64,
    )[np.newaxis, :]
    return rows * np.uint64(width) + columns


def _write_float_tile(
    destination: rasterio.DatasetWriter,
    values: np.ndarray,
    window: Window,
) -> None:
    encoded = np.where(np.isfinite(values), values, FLOAT_NODATA).astype(np.float32)
    destination.write(encoded, 1, window=window)


def _preview_rgb(values: np.ndarray) -> np.ndarray:
    rgb = np.moveaxis(values, 0, -1).astype(np.float32)
    output = np.zeros_like(rgb)
    for channel in range(3):
        data = rgb[..., channel]
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            continue
        low, high = np.percentile(finite, (2, 98))
        if high <= low:
            high = low + 1.0
        output[..., channel] = np.clip((data - low) / (high - low), 0, 1)
    return output


def _save_quicklook(
    path: Path,
    *,
    source_path: Path,
    rgb_indices: list[int],
    reflectance_scale: float,
    condition_path: Path,
    valid_mask_path: Path,
    alert_path: Path,
    label: str,
    score: float | None,
) -> None:
    cache_root = path.parent.parent / ".cache" / "matplotlib"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import ListedColormap
    from matplotlib.figure import Figure

    with rasterio.open(source_path) as source:
        scale = min(1.0, 1200.0 / max(source.width, source.height))
        height = max(1, round(source.height * scale))
        width = max(1, round(source.width * scale))
        rgb = source.read(
            rgb_indices,
            out_shape=(3, height, width),
            resampling=Resampling.bilinear,
        ).astype(np.float32)
        rgb /= reflectance_scale
    with (
        rasterio.open(condition_path) as condition_source,
        rasterio.open(valid_mask_path) as valid_source,
        rasterio.open(alert_path) as alert_source,
    ):
        condition = condition_source.read(
            1,
            out_shape=(height, width),
            masked=True,
            resampling=Resampling.bilinear,
        )
        valid = valid_source.read(
            1,
            out_shape=(height, width),
            resampling=Resampling.nearest,
        )
        alert = alert_source.read(
            1,
            out_shape=(height, width),
            resampling=Resampling.nearest,
        )

    figure = Figure(figsize=(15, 4.5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 4)
    axes[0].imshow(_preview_rgb(rgb))
    axes[0].set_title("RGB")
    axes[1].imshow(valid, cmap=ListedColormap(["black", "#32cd32"]), vmin=0, vmax=1)
    axes[1].set_title("Valid confident crop")
    condition_image = axes[2].imshow(condition, cmap="RdYlGn", vmin=0, vmax=100)
    axes[2].set_title("Spectral condition score")
    figure.colorbar(condition_image, ax=axes[2], fraction=0.046, pad=0.04)
    axes[3].imshow(alert, cmap=ListedColormap(["black", "#ff3b30"]), vmin=0, vmax=1)
    rendered_score = "n/a" if score is None else f"{score:.1f}/100"
    axes[3].set_title(f"Alerts\n{label}: {rendered_score}")
    for axis in axes:
        axis.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)


def run_ground_scene(
    payload_result_path: str | Path,
    *,
    output_root: str | Path,
    region_id: str | None = None,
    tile_size: int = 512,
    config: ConditionConfig | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Process one payload result without loading the complete scene into RAM."""
    started = time.perf_counter()
    cfg = config or ConditionConfig()
    result_path = Path(payload_result_path).resolve()
    payload = _load_payload_result(result_path)
    result_root = result_path.parent
    scene_id = _validate_identifier(str(payload.get("scene_id", "")), name="scene_id")
    resolved_region_id = _validate_identifier(region_id or scene_id, name="region_id")
    sensor = str(payload.get("sensor", ""))
    if sensor not in {"sentinel-2", "balkan-1"}:
        raise ValueError("Payload result has an unsupported sensor")

    stage_metadata = payload.get("stage_metadata")
    if not isinstance(stage_metadata, dict):
        raise ValueError("Payload result is missing stage metadata")
    intake = stage_metadata.get("intake")
    cloud_plan = stage_metadata.get("cloud_plan")
    if not isinstance(intake, dict) or not isinstance(cloud_plan, dict):
        raise ValueError("Payload result is missing intake or cloud-plan metadata")
    acquired_at = intake.get("acquired_at")
    if not isinstance(acquired_at, str) or not acquired_at:
        raise ValueError("Payload result is missing acquisition time")

    source_path = _resolve_asset(intake.get("source_path"), result_root, name="source scene")
    cloud_artifacts = payload.get("artifacts", {}).get("cloud", {})
    crop_artifacts = payload.get("artifacts", {}).get("crop", {})
    unusable_path = _resolve_asset(
        cloud_artifacts.get("unusable_mask"), result_root, name="unusable mask"
    )
    crop_binary_path = _resolve_asset(
        crop_artifacts.get("crop_binary"), result_root, name="crop binary mask"
    )
    crop_probability_path = _resolve_asset(
        crop_artifacts.get("crop_probability"), result_root, name="crop probability"
    )

    band_mapping = intake.get("logical_band_mapping")
    if not isinstance(band_mapping, dict):
        raise ValueError("Payload intake metadata is missing logical band mapping")
    warnings: list[str] = []
    nir_role = "NIR_NARROW"
    if nir_role not in band_mapping:
        nir_role = "NIR_BROAD"
        warnings.append(
            "NIR_NARROW was unavailable; health analysis used NIR_BROAD and requires "
            "sensor-specific validation."
        )
    roles = ("BLUE", "GREEN", "RED", nir_role)
    if any(role not in band_mapping for role in roles):
        raise ValueError(f"Source scene does not provide required ground roles: {roles}")
    band_indices = [int(band_mapping[role]["index"]) for role in roles]

    cloud_input = cloud_plan.get("input")
    if not isinstance(cloud_input, dict):
        raise ValueError("Cloud plan is missing the radiometric input contract")
    reflectance_scale = cloud_input.get("reflectance_scale")
    if not isinstance(reflectance_scale, (int, float)) or not np.isfinite(
        reflectance_scale
    ):
        raise ValueError("A finite reflectance scale is required for ground analysis")
    reflectance_scale = float(reflectance_scale)
    if reflectance_scale <= 0:
        raise ValueError("Reflectance scale must be positive")

    output = Path(output_root).resolve()
    report_path = output / "crop_condition_report.json"
    if report_path.exists() and not overwrite:
        raise FileExistsError(f"Ground result already exists: {report_path}")
    health_root = output / "health_layers"
    condition_root = output / "condition"
    visual_root = output / "visualisations"
    for directory in (health_root, condition_root, visual_root):
        directory.mkdir(parents=True, exist_ok=True)

    health_paths = {
        name: health_root / f"{scene_id}_{name}.tif" for name in HEALTH_LAYER_NAMES
    }
    condition_score_path = condition_root / f"{scene_id}_condition_score.tif"
    valid_mask_path = condition_root / f"{scene_id}_valid_crop.tif"
    deficit_path = condition_root / f"{scene_id}_robust_deficit_z.tif"
    relative_anomaly_path = condition_root / f"{scene_id}_relative_anomaly.tif"
    low_vigor_path = condition_root / f"{scene_id}_low_vigor.tif"
    alert_path = condition_root / f"{scene_id}_alert.tif"
    quicklook_path = visual_root / f"{scene_id}_crop_condition.png"

    metric_accumulators = {
        name: StreamingMetric(f"metric:{name}") for name in HEALTH_LAYER_NAMES
    }
    component_accumulators = {
        name: StreamingMetric(f"component:{name}") for name in SCORED_INDEX_NAMES
    }
    condition_accumulator = StreamingMetric("condition_score")
    probability_accumulator = StreamingMetric("crop_probability")
    candidate_crop_pixels = 0
    window_count = 0

    with ExitStack() as stack:
        source = stack.enter_context(rasterio.open(source_path))
        unusable_source = stack.enter_context(rasterio.open(unusable_path))
        crop_source = stack.enter_context(rasterio.open(crop_binary_path))
        probability_source = stack.enter_context(rasterio.open(crop_probability_path))
        for item, name in (
            (unusable_source, "Unusable mask"),
            (crop_source, "Crop binary mask"),
            (probability_source, "Crop probability"),
        ):
            _validate_same_grid(source, item, name=name)
        if source.crs is None:
            raise ValueError("Source scene has no CRS")
        if max(band_indices) > source.count:
            raise ValueError("Ground band mapping references a missing source band")

        float_profile = _raster_profile(source, dtype="float32", nodata=FLOAT_NODATA)
        byte_profile = _raster_profile(source, dtype="uint8", nodata=BYTE_NODATA)
        health_outputs = {
            name: stack.enter_context(rasterio.open(path, "w", **float_profile))
            for name, path in health_paths.items()
        }
        for name, destination in health_outputs.items():
            destination.set_band_description(1, f"{name}; -9999 outside valid crop")
        condition_output = stack.enter_context(
            rasterio.open(condition_score_path, "w", **float_profile)
        )
        condition_output.set_band_description(1, "spectral condition score 0..100")
        valid_output = stack.enter_context(
            rasterio.open(valid_mask_path, "w", **byte_profile)
        )
        valid_output.set_band_description(1, "0 excluded, 1 valid confident crop")

        windows = _iter_windows(source.width, source.height, tile_size)
        for window in windows:
            pixel_ids = _window_pixel_ids(window, source.width)
            raw_bands = source.read(band_indices, window=window).astype(np.float32)
            reflectance = raw_bands / reflectance_scale
            source_valid = np.all(
                source.read_masks(band_indices, window=window) > 0,
                axis=0,
            )
            crop_binary = crop_source.read(1, window=window)
            unusable = unusable_source.read(1, window=window)
            crop_probability = probability_source.read(1, window=window).astype(np.float32)
            candidate_crop_pixels += int(
                np.count_nonzero((crop_binary == 1) & (unusable == 0))
            )
            requested_mask = build_analysis_mask(
                crop_binary,
                unusable,
                crop_probability=crop_probability,
                nodata=~source_valid,
            )
            health_layers = calculate_health_layers(
                reflectance[0],
                reflectance[1],
                reflectance[2],
                reflectance[3],
                requested_mask,
            )
            score_layers = calculate_condition_score_layers(health_layers, config=cfg)

            for name, values in health_layers.values.items():
                _write_float_tile(health_outputs[name], values, window)
                metric_accumulators[name].update(values, sample_ids=pixel_ids)
            _write_float_tile(condition_output, score_layers.condition_score, window)
            valid_output.write(
                score_layers.valid_score_mask.astype(np.uint8),
                1,
                window=window,
            )
            condition_accumulator.update(
                score_layers.condition_score,
                sample_ids=pixel_ids,
            )
            for name, values in score_layers.component_scores.items():
                component_accumulators[name].update(values, sample_ids=pixel_ids)
            probability_accumulator.update(
                np.where(score_layers.valid_score_mask, crop_probability, np.nan),
                sample_ids=pixel_ids,
            )
            window_count += 1

        width = source.width
        height = source.height
        crs = str(source.crs)
        transform = list(source.transform)[:6]
        bounds = [float(value) for value in source.bounds]
        resolution = [abs(float(source.res[0])), abs(float(source.res[1]))]

    total_pixels = width * height
    condition_summary = condition_accumulator.summary()
    analysis_pixels = int(condition_summary["valid_pixels"])
    analysis_percentage = 100.0 * analysis_pixels / total_pixels if total_pixels else 0.0
    sufficient = (
        analysis_pixels >= cfg.minimum_analysis_pixels
        and analysis_percentage >= cfg.minimum_analysis_percentage
    )

    median_score = condition_summary["median"] if sufficient else None
    lower_quartile_score = condition_summary["lower_quartile"] if sufficient else None
    median_absolute_deviation = 0.0
    if sufficient:
        deviation_accumulator = StreamingMetric("condition_absolute_deviation")
        with rasterio.open(condition_score_path) as condition_source:
            for window in _iter_windows(width, height, tile_size):
                pixel_ids = _window_pixel_ids(window, width)
                score = condition_source.read(1, window=window).astype(np.float32)
                score[score == FLOAT_NODATA] = np.nan
                deviation_accumulator.update(
                    np.abs(score - float(median_score)),
                    sample_ids=pixel_ids,
                )
        median_absolute_deviation = float(deviation_accumulator.summary()["median"])

    relative_anomaly_pixels = 0
    low_vigor_pixels = 0
    with ExitStack() as stack:
        condition_source = stack.enter_context(rasterio.open(condition_score_path))
        float_profile = _raster_profile(condition_source, dtype="float32", nodata=FLOAT_NODATA)
        byte_profile = _raster_profile(condition_source, dtype="uint8", nodata=BYTE_NODATA)
        deficit_output = stack.enter_context(rasterio.open(deficit_path, "w", **float_profile))
        relative_output = stack.enter_context(
            rasterio.open(relative_anomaly_path, "w", **byte_profile)
        )
        low_output = stack.enter_context(rasterio.open(low_vigor_path, "w", **byte_profile))
        alert_output = stack.enter_context(rasterio.open(alert_path, "w", **byte_profile))
        deficit_output.set_band_description(1, "robust deficit z; -9999 excluded")
        relative_output.set_band_description(1, "0 normal, 1 relative anomaly")
        low_output.set_band_description(1, "0 above, 1 below absolute low-vigor threshold")
        alert_output.set_band_description(1, "0 no alert, 1 spectral condition alert")

        for window in _iter_windows(width, height, tile_size):
            score = condition_source.read(1, window=window).astype(np.float32)
            valid = np.isfinite(score) & (score != FLOAT_NODATA)
            if sufficient:
                spatial = calculate_spatial_condition_layers(
                    score,
                    valid,
                    median_score=float(median_score),
                    median_absolute_deviation=median_absolute_deviation,
                    config=cfg,
                )
                relative_anomaly_pixels += int(
                    np.count_nonzero(spatial.relative_anomaly_mask)
                )
                low_vigor_pixels += int(np.count_nonzero(spatial.low_vigor_mask))
                _write_float_tile(deficit_output, spatial.robust_deficit_z, window)
                relative_output.write(
                    spatial.relative_anomaly_mask.astype(np.uint8), 1, window=window
                )
                low_output.write(spatial.low_vigor_mask.astype(np.uint8), 1, window=window)
                alert_output.write(spatial.alert_mask.astype(np.uint8), 1, window=window)
            else:
                shape = (round(window.height), round(window.width))
                _write_float_tile(
                    deficit_output,
                    np.full(shape, np.nan, dtype=np.float32),
                    window,
                )
                zeros = np.zeros(shape, dtype=np.uint8)
                relative_output.write(zeros, 1, window=window)
                low_output.write(zeros, 1, window=window)
                alert_output.write(zeros, 1, window=window)

    component_medians = {
        name: component_accumulators[name].summary()["median"]
        for name in SCORED_INDEX_NAMES
    }
    probability_summary = probability_accumulator.summary()
    assessment = build_condition_assessment(
        analysis_pixels=analysis_pixels,
        total_pixels=total_pixels,
        median_score=median_score,
        lower_quartile_score=lower_quartile_score,
        relative_anomaly_pixels=relative_anomaly_pixels,
        low_vigor_pixels=low_vigor_pixels,
        component_median_scores=component_medians,
        mean_crop_probability=probability_summary["mean"],
        config=cfg,
    )

    _save_quicklook(
        quicklook_path,
        source_path=source_path,
        rgb_indices=band_indices[:3],
        reflectance_scale=reflectance_scale,
        condition_path=condition_score_path,
        valid_mask_path=valid_mask_path,
        alert_path=alert_path,
        label=assessment.label,
        score=assessment.condition_score,
    )

    assets = {
        **{name: health_paths[name] for name in HEALTH_LAYER_NAMES},
        "condition_score": condition_score_path,
        "valid_crop_mask": valid_mask_path,
        "robust_deficit_z": deficit_path,
        "relative_anomaly_mask": relative_anomaly_path,
        "low_vigor_mask": low_vigor_path,
        "alert_mask": alert_path,
        "quicklook": quicklook_path,
    }
    raster_assets = {
        name: path.relative_to(output).as_posix() for name, path in sorted(assets.items())
    }
    warnings.append(
        "Percentiles use a deterministic priority-reservoir sample of at most "
        "50,000 valid pixels per metric; means and standard deviations use all pixels."
    )
    warnings.append(
        "Single-scene condition labels are screening priorities, not disease diagnoses."
    )
    report = {
        "schema_version": "1.0",
        "algorithm_version": GROUND_SCENE_ALGORITHM_VERSION,
        "index_algorithm_version": INDEX_ALGORITHM_VERSION,
        "condition_algorithm_version": CONDITION_ALGORITHM_VERSION,
        "scene_id": scene_id,
        "region_id": resolved_region_id,
        "sensor": sensor,
        "acquired_at": acquired_at,
        "status": assessment.status,
        "quality": {
            "total_pixels": total_pixels,
            "candidate_crop_pixels": candidate_crop_pixels,
            "analysis_pixels": analysis_pixels,
            "analysis_percentage": analysis_percentage,
            "mean_crop_probability": probability_summary["mean"],
        },
        "metrics": {
            name: accumulator.summary()
            for name, accumulator in sorted(metric_accumulators.items())
        },
        "condition": assessment.to_dict(),
        "raster_assets": raster_assets,
        "geospatial": {
            "crs": crs,
            "transform": transform,
            "bounds": bounds,
            "resolution": resolution,
            "width": width,
            "height": height,
        },
        "radiometry": {
            "input_scale_divisor": reflectance_scale,
            "analysis_units": "scaled_reflectance",
            "nir_role": nir_role,
            "source": cloud_input.get("reflectance_scale_source"),
        },
        "provenance": {
            "payload_result": str(result_path),
            "source_scene": str(source_path),
            "unusable_mask": str(unusable_path),
            "crop_binary": str(crop_binary_path),
            "crop_probability": str(crop_probability_path),
        },
        "runtime": {
            "seconds": time.perf_counter() - started,
            "tile_size": tile_size,
            "first_pass_windows": window_count,
            "passes": 3 if sufficient else 2,
        },
        "warnings": warnings,
    }
    _write_json_atomic(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run windowed Phase 2 crop-condition analysis from payload result.json."
    )
    parser.add_argument("payload_result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--region-id")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = run_ground_scene(
        args.payload_result,
        output_root=args.output,
        region_id=args.region_id,
        tile_size=args.tile_size,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

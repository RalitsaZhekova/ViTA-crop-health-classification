"""Representative, deterministic inputs for crop TensorRT parity validation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from prithvi_shared import (
    CROP_CLASSIFICATION_THRESHOLD,
    HEALTH_ANALYSIS_CROP_THRESHOLD,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    NORMALIZATION_MEANS,
    NORMALIZATION_STDS,
)
from rasterio.warp import transform as transform_coordinates
from rasterio.windows import Window
from torch import Tensor

from prithvi_payload.balkan_crop_calibration import (
    BALKAN_CROP_CLASSIFICATION_THRESHOLD,
    BALKAN_HEALTH_ANALYSIS_CROP_THRESHOLD,
    apply_calibration,
    load_calibration,
)


def _temporal_coordinate(acquired_at: str) -> list[float]:
    parsed = datetime.fromisoformat(acquired_at.replace("Z", "+00:00"))
    return [float(parsed.year), float(parsed.timetuple().tm_yday)]


def _tile_origin(length: int, tile_size: int, fraction: float) -> int:
    if length < tile_size:
        raise ValueError(f"TensorRT parity source dimension {length} is smaller than {tile_size}")
    return round((length - tile_size) * fraction)


def _location_coordinate(
    source: rasterio.DatasetReader,
    *,
    x: int,
    y: int,
) -> list[float]:
    center_x, center_y = source.xy(
        y + (INPUT_HEIGHT - 1) / 2,
        x + (INPUT_WIDTH - 1) / 2,
    )
    longitudes, latitudes = transform_coordinates(
        source.crs,
        "EPSG:4326",
        [center_x],
        [center_y],
    )
    return [float(latitudes[0]), float(longitudes[0])]


def _profile_calibration(profile: dict[str, Any]) -> dict[str, Any] | None:
    calibration_path = profile.get("calibration_path")
    calibration_source_path = profile.get("calibration_source_path")
    if calibration_path is None and calibration_source_path is None:
        return None
    if not isinstance(calibration_path, str) or not isinstance(calibration_source_path, str):
        raise ValueError("Crop parity calibration profile is incomplete")
    return load_calibration(
        calibration_path,
        source_path=calibration_source_path,
    )


def build_crop_parity_inputs(
    profiles: list[dict[str, Any]],
    *,
    batch_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Build one balanced model batch from the packaged payload scenes.

    The returned image is already normalized because it is supplied directly to
    the exported Prithvi graph, below ``PayloadCropModel.predict``.
    """
    if not profiles:
        raise ValueError("Crop TensorRT parity requires representative scene profiles")
    if batch_size < len(profiles) or batch_size % len(profiles):
        raise ValueError(
            "Crop TensorRT parity batch must contain an equal number of tiles per scene"
        )

    # Four well-separated locations exercise the actual scene radiometry without
    # depending on random noise or pixels at padded image edges.
    origins = ((0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8))
    images: list[np.ndarray] = []
    temporal_coordinates: list[list[list[float]]] = []
    location_coordinates: list[list[float]] = []
    decision_thresholds: list[list[float]] = []
    valid_masks: list[np.ndarray] = []
    means = np.asarray(NORMALIZATION_MEANS, dtype=np.float32)[:, None, None]
    stds = np.asarray(NORMALIZATION_STDS, dtype=np.float32)[:, None, None]
    tiles_per_scene = batch_size // len(profiles)

    for profile in profiles:
        source_path = Path(str(profile.get("source_path", ""))).resolve()
        indices = profile.get("source_band_indices")
        acquired_at = profile.get("acquired_at")
        multiplier = profile.get("training_scale_multiplier")
        sensor = profile.get("sensor")
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        if (
            not isinstance(indices, list)
            or len(indices) != 4
            or any(not isinstance(index, int) or index <= 0 for index in indices)
        ):
            raise ValueError("Crop parity profile requires four source-band indices")
        if not isinstance(acquired_at, str) or not acquired_at:
            raise ValueError("Crop parity profile requires an acquisition timestamp")
        if not isinstance(multiplier, (int, float)) or float(multiplier) <= 0:
            raise ValueError("Crop parity profile requires a positive scale multiplier")
        if sensor not in {"sentinel-2", "balkan-1"}:
            raise ValueError("Crop parity profile has an unsupported sensor")
        calibration = _profile_calibration(profile)
        temporal = _temporal_coordinate(acquired_at)

        with rasterio.open(source_path) as source:
            if source.crs is None:
                raise ValueError(f"Crop parity source has no CRS: {source_path}")
            if max(indices) > source.count or len(set(indices)) != 4:
                raise ValueError("Crop parity source-band indices do not match the GeoTIFF")
            for tile_index in range(tiles_per_scene):
                x_fraction, y_fraction = origins[tile_index % len(origins)]
                x = _tile_origin(source.width, INPUT_WIDTH, x_fraction)
                y = _tile_origin(source.height, INPUT_HEIGHT, y_fraction)
                window = Window(x, y, INPUT_WIDTH, INPUT_HEIGHT)
                raw = source.read(indices, window=window, out_dtype="float32")
                valid = np.all(source.read_masks(indices, window=window) > 0, axis=0)
                image = raw * np.float32(multiplier)
                if calibration is not None:
                    image = apply_calibration(image, calibration)
                invalid = ~np.isfinite(image).all(axis=0) | ~valid
                if source.nodata is not None:
                    if calibration is not None:
                        invalid |= np.any(raw == source.nodata, axis=0)
                    else:
                        invalid |= np.all(raw == source.nodata, axis=0)
                image = image.astype(np.float32, copy=True)
                image[:, invalid] = means[:, 0, 0][:, None]
                images.append((image - means) / stds)
                temporal_coordinates.append([temporal])
                location_coordinates.append(_location_coordinate(source, x=x, y=y))
                decision_thresholds.append(
                    [
                        BALKAN_CROP_CLASSIFICATION_THRESHOLD,
                        BALKAN_HEALTH_ANALYSIS_CROP_THRESHOLD,
                    ]
                    if sensor == "balkan-1"
                    else [
                        CROP_CLASSIFICATION_THRESHOLD,
                        HEALTH_ANALYSIS_CROP_THRESHOLD,
                    ]
                )
                valid_masks.append(~invalid)

    image_tensor = torch.from_numpy(np.stack(images)).unsqueeze(2)
    temporal_tensor = torch.tensor(temporal_coordinates, dtype=torch.float32)
    location_tensor = torch.tensor(location_coordinates, dtype=torch.float32)
    threshold_tensor = torch.tensor(decision_thresholds, dtype=torch.float32)
    valid_tensor = torch.from_numpy(np.stack(valid_masks))
    return (
        image_tensor,
        temporal_tensor,
        location_tensor,
        threshold_tensor,
        valid_tensor,
    )

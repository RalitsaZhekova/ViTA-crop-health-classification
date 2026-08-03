from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from cloud_detection.backend import TestBackend
from cloud_detection.cli import DEFAULT_CONFIG
from cloud_detection.config import load_config
from prithvi_payload.cloud_stage import build_cloud_stage_plan
from prithvi_payload.crop_stage import build_crop_stage_plan
from prithvi_payload.inference import InferenceOutput
from prithvi_payload.pipeline import run_scene
from prithvi_payload.scene_intake import inspect_scene
from rasterio.transform import from_origin

BALKAN_ORDER = ("BLUE", "GREEN", "RED", "NIR", "PAN")


class FakeCropModel:
    device = torch.device("cpu")

    def predict(
        self,
        image: torch.Tensor,
        *,
        temporal_coords: torch.Tensor,
        location_coords: torch.Tensor,
    ) -> InferenceOutput:
        del temporal_coords, location_coords
        shape = (image.shape[0], image.shape[-2], image.shape[-1])
        probability = torch.full(shape, 0.90, dtype=torch.float32)
        return InferenceOutput(
            crop_probability=probability,
            crop_binary=torch.ones(shape, dtype=torch.uint8),
            crop_confidence=probability,
        )


def _write_balkan_scene(path: Path) -> None:
    values = np.stack(
        [
            np.full((16, 16), 0.10, dtype=np.float32),
            np.full((16, 16), 0.20, dtype=np.float32),
            np.full((16, 16), 0.30, dtype=np.float32),
            np.full((16, 16), 0.60, dtype=np.float32),
            np.full((16, 16), 0.40, dtype=np.float32),
        ]
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=16,
        height=16,
        count=5,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(23.0, 43.0, 0.00002, 0.00002),
        nodata=0.0,
    ) as destination:
        destination.write(values)


def test_explicit_band_order_adapts_undescribed_l1ort_without_mutating_it(
    tmp_path: Path,
) -> None:
    scene = tmp_path / "3036_L1ORT.tif"
    _write_balkan_scene(scene)

    rejected = inspect_scene(scene, sensor="balkan-1")
    accepted = inspect_scene(scene, sensor="balkan-1", band_order=BALKAN_ORDER)

    assert rejected["readiness"]["intake"] == "REJECT"
    assert "Every input band must have a description" in rejected["errors"]
    assert accepted["readiness"]["intake"] == "READY"
    assert accepted["sensor_contract"]["band_order_source"] == "explicit_override"
    assert accepted["model_band_routes"]["cloud_detection"][
        "source_band_indices"
    ] == [4, 3, 2, 1]
    assert accepted["readiness"]["crop"] == "BALKAN_1_SPECTRAL_HARMONISATION_REQUIRED"
    assert accepted["model_band_routes"]["crop_classification"][
        "source_band_indices"
    ] is None
    with rasterio.open(scene) as unchanged:
        assert unchanged.descriptions == (None, None, None, None, None)


def test_provisional_crop_transfer_is_explicit_and_auditable(tmp_path: Path) -> None:
    scene = tmp_path / "3036_L1ORT.tif"
    _write_balkan_scene(scene)
    intake = inspect_scene(
        scene,
        sensor="balkan-1",
        band_order=BALKAN_ORDER,
        acquired_at="2026-01-01T12:00:00Z",
        allow_provisional_balkan_crop=True,
    )

    route = intake["model_band_routes"]["crop_classification"]
    assert intake["readiness"]["crop"] == "READY_PROVISIONAL"
    assert route["source_band_indices"] == [1, 2, 3, 4]
    assert route["source_logical_order"] == ["BLUE", "GREEN", "RED", "NIR_BROAD"]
    assert route["spectral_adapter"] == {
        "mode": "PROVISIONAL_BALKAN_1_NIR_TRANSFER",
        "source_nir_role": "NIR_BROAD",
        "model_nir_role": "NIR_NARROW",
        "validation_status": "EXECUTION_ONLY_UNVALIDATED",
    }

    cloud_plan = build_cloud_stage_plan(intake, reflectance_scale=1)
    unusable = tmp_path / "unusable.tif"
    unusable.touch()
    crop_plan = build_crop_stage_plan(
        intake,
        cloud_plan,
        {
            "cloud_percentage": 10.0,
            "output_files": {"unusable_mask": str(unusable)},
        },
    )
    assert crop_plan["readiness"] == "READY"
    assert crop_plan["compatibility"] == "PROVISIONAL_EXECUTION_ONLY"
    assert crop_plan["input"]["training_scale_multiplier"] == 10_000.0
    assert any("software execution only" in warning for warning in crop_plan["warnings"])


def test_provisional_crop_adapter_rejects_other_sensors(tmp_path: Path) -> None:
    scene = tmp_path / "scene.tif"
    _write_balkan_scene(scene)

    with pytest.raises(ValueError, match="only valid for balkan-1"):
        inspect_scene(
            scene,
            sensor="sentinel-2",
            allow_provisional_balkan_crop=True,
        )


def test_provisional_balkan_route_reaches_crop_execution(tmp_path: Path) -> None:
    scene = tmp_path / "3036_L1ORT.tif"
    _write_balkan_scene(scene)

    result = run_scene(
        scene,
        sensor="balkan-1",
        output_root=tmp_path / "run",
        acquired_at="2026-01-01T12:00:00Z",
        band_order=BALKAN_ORDER,
        allow_provisional_balkan_crop=True,
        reflectance_scale=1,
        stop_after="crop",
        cloud_backend=TestBackend(),
        cloud_config=load_config(DEFAULT_CONFIG),
        crop_model=FakeCropModel(),
    )

    assert result["status"] == "CROP_COMPLETE"
    assert result["completed_stages"] == ["intake", "cloud", "crop"]
    assert result["stage_metadata"]["crop_plan"]["compatibility"] == (
        "PROVISIONAL_EXECUTION_ONLY"
    )
    assert result["stage_metadata"]["cloud"]["runtime"]["device"] == "unknown"
    assert result["stage_metadata"]["cloud"]["analysis_grid"]["mode"] == (
        "balkan_1_utm_10m"
    )
    for mask_name in ("semantic_mask", "unusable_mask", "invalid_mask"):
        with rasterio.open(result["artifacts"]["cloud"][mask_name]) as mask:
            assert mask.shape == (16, 16)
    assert result["summary"]["crop"]["device"] == "cpu"

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from cloud_detection.backend import BackendPrediction, TestBackend
from cloud_detection.cli import DEFAULT_CONFIG
from cloud_detection.config import load_config
from prithvi_payload.inference import InferenceOutput
from rasterio.transform import from_origin
from vita_integration.pipeline import run_sentinel_end_to_end


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


class AllCloudBackend:
    def predict(self, tile: np.ndarray) -> BackendPrediction:
        scores = np.zeros((4, tile.shape[1], tile.shape[2]), dtype=np.float32)
        scores[1] = 1.0
        return BackendPrediction(scores=scores, score_kind="test_all_cloud")


def _write_sentinel_scene(path: Path) -> None:
    shape = (64, 64)
    blue = np.full(shape, 0.05, dtype=np.float32)
    green = np.full(shape, 0.10, dtype=np.float32)
    red = np.full(shape, 0.05, dtype=np.float32)
    nir_broad = np.full(shape, 0.75, dtype=np.float32)
    nir_narrow = np.full(shape, 0.80, dtype=np.float32)
    blue[:16, :16] = 0.20
    green[:16, :16] = 0.25
    red[:16, :16] = 0.30
    nir_broad[:16, :16] = 0.34
    nir_narrow[:16, :16] = 0.35
    values = (10_000 * np.stack((red, green, blue, nir_broad, nir_narrow))).astype(np.uint16)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=shape[1],
        height=shape[0],
        count=5,
        dtype="uint16",
        crs="EPSG:32631",
        transform=from_origin(500_000, 4_600_000, 10, 10),
        nodata=0,
    ) as destination:
        destination.write(values)
        for index, description in enumerate(("B4", "B3", "B2", "B8", "B8A"), 1):
            destination.set_band_description(index, description)


def test_sentinel_end_to_end_completes_payload_and_ground(tmp_path: Path) -> None:
    source = tmp_path / "sentinel.tif"
    _write_sentinel_scene(source)
    output = tmp_path / "run"

    result = run_sentinel_end_to_end(
        source,
        output_root=output,
        acquired_at="2026-07-27T12:00:00Z",
        reflectance_scale=10_000,
        cloud_backend=TestBackend(),
        cloud_config=load_config(DEFAULT_CONFIG),
        crop_model=FakeCropModel(),
        ground_tile_size=16,
    )

    assert result["status"] == "COMPLETE"
    assert result["completed_stages"] == ["payload", "ground"]
    assert result["payload_status"] == "CROP_COMPLETE"
    assert result["ground_status"] == "MEASURED"
    assert result["summary"]["condition"]["label"] == "Watch"
    assert result["summary"]["crop"]["crop_percentage_usable"] == 100.0
    assert not Path(result["payload_result"]).is_absolute()
    assert not Path(result["ground_report"]).is_absolute()
    assert (output / result["payload_result"]).is_file()
    assert (output / result["ground_report"]).is_file()
    saved = json.loads((output / "end_to_end_result.json").read_text())
    assert saved["summary"] == result["summary"]

    payload = json.loads((output / result["payload_result"]).read_text())
    ground_report_path = output / result["ground_report"]
    ground = json.loads(ground_report_path.read_text())
    with rasterio.open(source) as source_dataset:
        expected_grid = (
            source_dataset.width,
            source_dataset.height,
            source_dataset.crs,
            source_dataset.transform,
        )

    payload_rasters = [
        *(
            Path(path)
            for path in payload["artifacts"]["cloud"].values()
            if Path(path).suffix == ".tif"
        ),
        *(
            Path(path)
            for path in payload["artifacts"]["crop"].values()
            if Path(path).suffix == ".tif"
        ),
    ]
    ground_assets = [ground_report_path.parent / path for path in ground["raster_assets"].values()]
    assert len(payload_rasters) == 6
    assert len(ground_assets) == 15
    for path in (*payload_rasters, *ground_assets):
        assert path.is_file()
        if path.suffix == ".tif":
            with rasterio.open(path) as dataset:
                assert dataset.count == 1
                assert (
                    dataset.width,
                    dataset.height,
                    dataset.crs,
                    dataset.transform,
                ) == expected_grid


def test_sentinel_end_to_end_records_cloud_gate_stop(tmp_path: Path) -> None:
    source = tmp_path / "sentinel.tif"
    _write_sentinel_scene(source)
    output = tmp_path / "run"

    result = run_sentinel_end_to_end(
        source,
        output_root=output,
        acquired_at="2026-07-27T12:00:00Z",
        reflectance_scale=10_000,
        cloud_backend=AllCloudBackend(),
        cloud_config=load_config(DEFAULT_CONFIG),
    )

    assert result["status"] == "PAYLOAD_STOPPED"
    assert result["payload_status"] == "CROP_SKIPPED_CLOUD_GATE"
    assert result["ground_report"] is None
    assert not (output / "ground").exists()


def test_sentinel_end_to_end_protects_existing_result(tmp_path: Path) -> None:
    source = tmp_path / "sentinel.tif"
    _write_sentinel_scene(source)
    output = tmp_path / "run"
    arguments = {
        "output_root": output,
        "acquired_at": "2026-07-27T12:00:00Z",
        "reflectance_scale": 10_000,
        "cloud_backend": TestBackend(),
        "cloud_config": load_config(DEFAULT_CONFIG),
        "crop_model": FakeCropModel(),
    }
    run_sentinel_end_to_end(source, **arguments)

    with pytest.raises(FileExistsError, match="already exists"):
        run_sentinel_end_to_end(source, **arguments)

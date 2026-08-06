from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from prithvi_payload.balkan_analysis import materialize_balkan_analysis_grid
from prithvi_payload.balkan_crop_calibration import sha256_file
from rasterio.transform import from_origin


def test_balkan_analysis_grid_is_a_shared_four_band_10m_product(tmp_path: Path) -> None:
    source_path = tmp_path / "balkan.tif"
    values = np.ones((5, 48, 64), dtype=np.float32)
    for index in range(5):
        values[index] *= np.float32(1000 + index * 100)
    with rasterio.open(
        source_path,
        "w",
        driver="GTiff",
        width=64,
        height=48,
        count=5,
        dtype="float32",
        crs="EPSG:32631",
        transform=from_origin(500_000.0, 4_500_000.0, 1.0, 1.0),
        nodata=0.0,
    ) as destination:
        destination.write(values)

    intake = {
        "scene_id": "balkan-test",
        "sensor": "balkan-1",
        "source_path": str(source_path),
        "logical_band_mapping": {
            role: {"index": index}
            for index, role in enumerate(
                ("BLUE", "GREEN", "RED", "NIR_BROAD", "PAN"), start=1
            )
        },
        "model_band_routes": {"crop_classification": {}},
    }

    result = materialize_balkan_analysis_grid(intake, output_root=tmp_path / "run")

    with rasterio.open(result["source_path"]) as analysis:
        assert analysis.count == 4
        assert analysis.descriptions == ("BLUE", "GREEN", "RED", "NIR_BROAD")
        assert analysis.crs == rasterio.crs.CRS.from_epsg(32631)
        assert analysis.res == (10.0, 10.0)
        assert analysis.width < 64
        assert analysis.height < 48
    assert result["model_band_routes"]["cloud_detection"]["source_band_indices"] == [
        4,
        3,
        2,
        1,
    ]
    assert result["model_band_routes"]["crop_classification"]["source_band_indices"] == [
        1,
        2,
        3,
        4,
    ]
    assert len(result["display"]["native_source_channel_limits"]) == 3


def test_balkan_analysis_grid_reuses_verified_source_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "balkan.tif"
    with rasterio.open(
        source_path,
        "w",
        driver="GTiff",
        width=64,
        height=48,
        count=5,
        dtype="float32",
        crs="EPSG:32631",
        transform=from_origin(500_000.0, 4_500_000.0, 1.0, 1.0),
        nodata=0.0,
    ) as destination:
        destination.write(np.ones((5, 48, 64), dtype=np.float32))
    intake = {
        "scene_id": "cached-balkan",
        "sensor": "balkan-1",
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "logical_band_mapping": {
            role: {"index": index}
            for index, role in enumerate(
                ("BLUE", "GREEN", "RED", "NIR_BROAD", "PAN"), start=1
            )
        },
        "model_band_routes": {"crop_classification": {}},
    }
    monkeypatch.setenv("VITA_BALKAN_ANALYSIS_CACHE_DIR", str(tmp_path / "cache"))

    first = materialize_balkan_analysis_grid(intake, output_root=tmp_path / "runs" / "one")
    second = materialize_balkan_analysis_grid(intake, output_root=tmp_path / "runs" / "two")

    assert first["runtime"]["cache_hit"] is False
    assert second["runtime"]["cache_hit"] is True
    assert first["source_path"] == second["source_path"]
    assert second["cache"]["key"] == first["cache"]["key"]

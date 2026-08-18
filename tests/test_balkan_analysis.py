from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from prithvi_payload.balkan_analysis import materialize_balkan_analysis_grid
from prithvi_payload.balkan_crop_calibration import sha256_file
from prithvi_payload.balkan_raw_proxy import RAW_PROXY_ALGORITHM
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


def test_single_pass_raw_proxy_is_not_resampled_again(tmp_path: Path) -> None:
    source_path = tmp_path / "raw-proxy-final-grid.tif"
    values = np.ones((5, 48, 64), dtype=np.float32)
    with rasterio.open(
        source_path,
        "w",
        driver="GTiff",
        width=64,
        height=48,
        count=5,
        dtype="float32",
        crs="EPSG:32631",
        transform=from_origin(500_000.0, 4_500_000.0, 10.0, 10.0),
        nodata=0.0,
    ) as destination:
        destination.write(values)
        for index, role in enumerate(
            ("BLUE", "GREEN", "RED", "NIR_BROAD", "PANCHROMATIC"),
            start=1,
        ):
            destination.set_band_description(index, role)
        destination.update_tags(
            PROCESSING_LEVEL="EXPERIMENTAL_RAW_MODEL_PROXY",
            RAW_PROXY_ALGORITHM=RAW_PROXY_ALGORITHM,
            RAW_TO_MODEL_RESAMPLING_PASSES="1",
        )
    intake = {
        "scene_id": "single-pass-raw",
        "sensor": "balkan-1",
        "source_path": str(source_path),
        "logical_band_mapping": {
            role: {"index": index}
            for index, role in enumerate(
                ("BLUE", "GREEN", "RED", "NIR_BROAD", "PANCHROMATIC"),
                start=1,
            )
        },
        "model_band_routes": {"crop_classification": {}},
    }

    result = materialize_balkan_analysis_grid(intake, output_root=tmp_path / "run")

    assert result["source_path"] == str(source_path.resolve())
    assert result["preprocessing"]["mode"] == "single_pass_raw_proxy_passthrough"
    assert result["preprocessing"]["raw_to_model_resampling_passes"] == 1
    assert result["preprocessing"]["additional_analysis_resampling_passes"] == 0
    assert result["runtime"]["warp_seconds"] == 0
    assert not list((tmp_path / "run" / "analysis").glob("*.tif"))


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


def test_balkan_analysis_uses_common_overview_without_changing_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "overview-balkan.tif"
    values = np.empty((5, 480, 640), dtype=np.float32)
    for index in range(5):
        values[index] = np.float32(0.1 + index * 0.05)
    with rasterio.open(
        source_path,
        "w",
        driver="GTiff",
        width=640,
        height=480,
        count=5,
        dtype="float32",
        crs="EPSG:32631",
        transform=from_origin(500_000.0, 4_500_000.0, 1.0, 1.0),
        nodata=0.0,
        tiled=True,
        blockxsize=128,
        blockysize=128,
    ) as destination:
        destination.write(values)
        destination.build_overviews([2, 4], rasterio.enums.Resampling.average)
    intake = {
        "scene_id": "overview-balkan",
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
    monkeypatch.setenv("VITA_BALKAN_OVERVIEW_REQUIRED", "1")
    monkeypatch.setenv("VITA_BALKAN_ANALYSIS_CACHE_DIR", str(tmp_path / "cache"))

    result = materialize_balkan_analysis_grid(intake, output_root=tmp_path / "run")

    assert result["preprocessing"]["mode"] == "embedded_overview_then_average"
    assert result["preprocessing"]["overview_factor"] == 4
    assert result["preprocessing"]["radiometry_modified"] is False
    assert result["runtime"]["overview_read_seconds"] >= 0
    with rasterio.open(result["source_path"]) as analysis:
        assert analysis.count == 4
        assert analysis.descriptions == ("BLUE", "GREEN", "RED", "NIR_BROAD")
        assert analysis.res == (10.0, 10.0)
        assert analysis.tags()["ANALYSIS_STRATEGY"] == "embedded_overview_then_average"
        assert analysis.tags()["SOURCE_OVERVIEW_FACTOR"] == "4"
        for index, expected in enumerate((0.1, 0.15, 0.2, 0.25), start=1):
            band = analysis.read(index, masked=True)
            assert float(band.mean()) == pytest.approx(expected, abs=1e-6)

    monkeypatch.setenv("VITA_BALKAN_PREP_MAX_BYTES", "1")
    with pytest.raises(ValueError, match="estimated working set"):
        materialize_balkan_analysis_grid(intake, output_root=tmp_path / "memory-limited-run")


def test_balkan_analysis_required_overview_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "no-overview.tif"
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
        "scene_id": "no-overview",
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
    monkeypatch.setenv("VITA_BALKAN_OVERVIEW_REQUIRED", "1")

    with pytest.raises(ValueError, match="overview fast path is required but unavailable"):
        materialize_balkan_analysis_grid(intake, output_root=tmp_path / "run")

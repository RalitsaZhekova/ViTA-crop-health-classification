from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from prithvi_payload.acquisition.earth_engine import (
    EARTH_ENGINE_BANDS,
    AcquiredSentinelScene,
    EarthEngineAcquisitionProvider,
    _normalise_geotiff,
    _parse_candidates,
)
from prithvi_payload.acquisition.errors import AcquisitionError
from prithvi_payload.acquisition.grid import calculate_target_grid


def test_live_area_grid_is_ten_metre_utm_and_bounded() -> None:
    grid = calculate_target_grid((5.43, 52.50, 5.49, 52.54))

    assert grid.crs == "EPSG:32631"
    assert 64 <= grid.width <= 1000
    assert 64 <= grid.height <= 1000
    assert grid.transform.a == 10
    assert grid.transform.e == -10
    assert grid.estimated_uncompressed_bytes == grid.width * grid.height * 5 * 2


@pytest.mark.parametrize(
    ("bounds", "code"),
    [
        ((5.0, 52.0, 6.0, 53.0), "REGION_TOO_LARGE"),
        ((5.4300, 52.5000, 5.4301, 52.5001), "REGION_TOO_SMALL"),
        ((5.5, 52.5, 5.4, 52.6), "INVALID_REGION"),
    ],
)
def test_live_area_grid_rejects_unsafe_dimensions(bounds, code: str) -> None:
    with pytest.raises(AcquisitionError) as error:
        calculate_target_grid(bounds)

    assert error.value.code == code


def test_candidates_are_ranked_by_cloud_then_recency() -> None:
    candidates = _parse_candidates(
        [
            {
                "id": "collection/older-clear",
                "properties": {
                    "system:time_start": 1_700_000_000_000,
                    "CLOUDY_PIXEL_PERCENTAGE": 3.0,
                },
            },
            {
                "id": "collection/cloudy",
                "properties": {
                    "system:time_start": 1_800_000_000_000,
                    "CLOUDY_PIXEL_PERCENTAGE": 20.0,
                },
            },
            {
                "id": "collection/newer-clear",
                "properties": {
                    "system:time_start": 1_750_000_000_000,
                    "CLOUDY_PIXEL_PERCENTAGE": 3.0,
                },
            },
        ]
    )

    assert [candidate.system_index for candidate in candidates] == [
        "newer-clear",
        "older-clear",
        "cloudy",
    ]
    assert [candidate.rank for candidate in candidates] == [1, 2, 3]


def test_download_parameters_pin_the_qualified_grid() -> None:
    grid = calculate_target_grid((5.43, 52.50, 5.49, 52.54))
    parameters = EarthEngineAcquisitionProvider._download_parameters(grid)

    assert parameters["bands"] == list(EARTH_ENGINE_BANDS)
    assert parameters["crs"] == grid.crs
    assert parameters["crs_transform"] == list(grid.transform)[:6]
    assert parameters["dimensions"] == [grid.width, grid.height]
    assert parameters["format"] == "GEO_TIFF"
    assert parameters["filePerBand"] is False


def _write_source(path: Path, *, empty: bool = False) -> tuple:
    grid = calculate_target_grid((5.43, 52.50, 5.44, 52.51))
    values = np.zeros((5, grid.height, grid.width), dtype=np.uint16)
    if not empty:
        values[:] = 2_500
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=grid.width,
        height=grid.height,
        count=5,
        dtype="uint16",
        crs=grid.crs,
        transform=grid.transform,
    ) as destination:
        destination.write(values)
    return grid, values


def test_downloaded_geotiff_is_validated_and_tagged(tmp_path: Path) -> None:
    source = tmp_path / "source.tif"
    destination = tmp_path / "scene.tif"
    grid, values = _write_source(source)

    _normalise_geotiff(
        source,
        destination,
        grid=grid,
        acquired_at="2026-06-15T10:56:21+00:00",
    )

    with rasterio.open(destination) as dataset:
        assert dataset.descriptions == EARTH_ENGINE_BANDS
        assert dataset.tags()["REFLECTANCE_SCALE"] == "10000"
        assert dataset.tags()["SENSOR"] == "sentinel-2"
        assert np.array_equal(dataset.read(), values)


def test_downloaded_geotiff_rejects_incomplete_tile_coverage(tmp_path: Path) -> None:
    source = tmp_path / "empty-source.tif"
    destination = tmp_path / "scene.tif"
    grid, _ = _write_source(source, empty=True)

    with pytest.raises(AcquisitionError) as error:
        _normalise_geotiff(
            source,
            destination,
            grid=grid,
            acquired_at="2026-06-15T10:56:21+00:00",
        )

    assert error.value.code == "EARTH_ENGINE_INCOMPLETE_COVERAGE"
    assert not destination.exists()


def test_safe_provenance_contains_no_credentials_or_signed_urls(tmp_path: Path) -> None:
    scene = AcquiredSentinelScene(
        local_tiff_path=tmp_path / "scene.tif",
        provider_scene_id="scene-id",
        product_id=None,
        acquired_at="2026-06-15T10:56:21+00:00",
        metadata_cloud_percentage=4.2,
        requested_bbox_wgs84=(5.43, 52.50, 5.49, 52.54),
        output_crs="EPSG:32631",
        output_transform=(10.0, 0.0, 665000.0, 0.0, -10.0, 5820000.0),
        sha256="a" * 64,
        byte_size=1024,
        candidate_rank=1,
        candidate_attempt_count=1,
        timing={},
    )

    provenance = scene.safe_provenance()

    assert provenance["provider"] == "earth_engine"
    assert provenance["source_scale"] == 10_000
    assert provenance["selection_policy"] == "least_cloudy"
    assert provenance["target_cloud_range"] is None
    assert not ({"credentials", "token", "url"} & set(provenance))

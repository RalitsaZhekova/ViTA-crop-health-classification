from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine
from prithvi_payload.acquisition.earth_engine import (
    EARTH_ENGINE_BANDS,
    EarthEngineAcquisitionProvider,
    normalize_and_validate_geotiff,
)
from prithvi_payload.acquisition.grid import TargetGrid


def test_earth_engine_provider_is_bounded_to_three_candidates() -> None:
    assert EarthEngineAcquisitionProvider(max_candidates=2, max_scene_attempts=2)
    with pytest.raises(ValueError, match="within 1..3"):
        EarthEngineAcquisitionProvider(max_candidates=4, max_scene_attempts=2)
    with pytest.raises(ValueError, match="cannot exceed"):
        EarthEngineAcquisitionProvider(max_candidates=2, max_scene_attempts=3)


def test_normalized_earth_engine_tiff_has_clean_band_metadata(tmp_path: Path) -> None:
    transform = Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_500_000.0)
    grid = TargetGrid(
        crs="EPSG:32614",
        transform=transform,
        width=16,
        height=12,
        bounds=(500_000.0, 4_499_880.0, 500_160.0, 4_500_000.0),
        estimated_uncompressed_bytes=16 * 12 * 5 * 2,
    )
    raw = tmp_path / "raw.tif"
    normalized = tmp_path / "scene.tif"
    pixels = np.arange(5 * 12 * 16, dtype=np.uint16).reshape(5, 12, 16)
    with rasterio.open(
        raw,
        "w",
        driver="GTiff",
        width=16,
        height=12,
        count=5,
        dtype="uint16",
        crs=grid.crs,
        transform=grid.transform,
        interleave="pixel",
    ) as destination:
        destination.write(pixels)

    normalize_and_validate_geotiff(raw, normalized, grid=grid)

    with rasterio.open(normalized) as source:
        np.testing.assert_array_equal(source.read(), pixels)
        assert source.descriptions == EARTH_ENGINE_BANDS
        assert source.profile["interleave"] == "band"
        assert source.profile["compress"] == "deflate"
        assert source.tags()["REFLECTANCE_SCALE"] == "10000"

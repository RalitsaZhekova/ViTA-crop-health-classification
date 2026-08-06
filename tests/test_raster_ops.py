from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from prithvi_payload.raster_ops import read_padded_tile, utm_crs_for_bounds
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
from rasterio.transform import from_origin


def test_utm_selection_uses_the_raster_geographic_centre() -> None:
    selected = utm_crs_for_bounds(
        CRS.from_epsg(4326),
        BoundingBox(left=-111.5, bottom=32.8, right=-111.3, top=33.0),
    )

    assert selected == CRS.from_epsg(32612)


def test_padded_tile_preserves_the_existing_reflect_padding_contract(tmp_path: Path) -> None:
    source_path = tmp_path / "tile.tif"
    values = np.arange(6, dtype=np.uint16).reshape(1, 2, 3)
    with rasterio.open(
        source_path,
        "w",
        driver="GTiff",
        width=3,
        height=2,
        count=1,
        dtype="uint16",
        crs="EPSG:32631",
        transform=from_origin(500_000.0, 4_500_000.0, 10.0, 10.0),
    ) as destination:
        destination.write(values)

    with rasterio.open(source_path) as source:
        native = read_padded_tile(source, [1], y=0, x=0, tile_size=4, halo=1)
        float_values = read_padded_tile(
            source,
            [1],
            y=0,
            x=0,
            tile_size=4,
            halo=1,
            out_dtype="float32",
        )

    expected = np.pad(values, ((0, 0), (1, 1), (1, 0)), mode="reflect")
    np.testing.assert_array_equal(native, expected)
    np.testing.assert_array_equal(float_values, expected.astype(np.float32))
    assert native.dtype == np.uint16
    assert float_values.dtype == np.float32

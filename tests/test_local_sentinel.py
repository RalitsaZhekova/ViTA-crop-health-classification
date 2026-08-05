from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from vita_integration.cli import SENTINEL_BAND_ORDER, _sentinel_contract


def _write_sentinel(path: Path) -> None:
    pixels = np.arange(5 * 12 * 16, dtype=np.uint16).reshape(5, 12, 16)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=16,
        height=12,
        count=5,
        dtype="uint16",
        crs="EPSG:32614",
        transform=from_origin(500_000.0, 4_500_000.0, 10.0, 10.0),
    ) as destination:
        destination.write(pixels)
        destination.descriptions = SENTINEL_BAND_ORDER
        destination.update_tags(
            ACQUIRED_AT="2026-07-12T17:22:30.106000+00:00",
            REFLECTANCE_SCALE="10000",
            SENSOR="sentinel-2",
        )


def test_sentinel_contract_comes_from_the_local_geotiff(tmp_path: Path) -> None:
    source = tmp_path / "sentinel.tif"
    _write_sentinel(source)

    acquired_at, scale = _sentinel_contract(
        source,
        acquired_at=None,
        reflectance_scale=None,
    )

    assert acquired_at == "2026-07-12T17:22:30.106000+00:00"
    assert scale == 10_000.0


def test_sentinel_timestamp_can_fall_back_to_the_filename(tmp_path: Path) -> None:
    source = tmp_path / "S2_20260720T171859_T14TPL.tif"
    _write_sentinel(source)
    with rasterio.open(source, "r+") as destination:
        tags = destination.tags()
        tags.pop("ACQUIRED_AT")
        destination.update_tags(**tags, ACQUIRED_AT="")

    acquired_at, _ = _sentinel_contract(
        source,
        acquired_at=None,
        reflectance_scale=None,
    )

    assert acquired_at == "2026-07-20T17:18:59+00:00"

from pathlib import Path

import numpy as np
import pytest
import rasterio
from cloud_detection.io import RasterValidationError, read_geotiff
from rasterio.transform import from_origin


def test_rejects_partial_band_descriptions(tmp_path: Path) -> None:
    path = tmp_path / "partial_descriptions.tif"
    profile = {
        "driver": "GTiff",
        "height": 8,
        "width": 8,
        "count": 4,
        "dtype": "uint16",
        "crs": "EPSG:32632",
        "transform": from_origin(500000, 5000000, 10, 10),
    }
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(np.ones((4, 8, 8), dtype=np.uint16))
        dataset.set_band_description(1, "B08")

    with pytest.raises(RasterValidationError, match="band order"):
        read_geotiff(path, ["B08", "B04", "B03", "B02"])

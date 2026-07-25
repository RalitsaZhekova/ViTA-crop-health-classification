from pathlib import Path

import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    config = yaml.safe_load(Path("configs/cloud_detector.yaml").read_text(encoding="utf-8"))
    config["tiling"]["size"] = 64
    config["tiling"]["overlap"] = 16
    config["postprocessing"]["dilation_pixels"] = 0
    config["postprocessing"]["minimum_region_pixels"] = 1
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


@pytest.fixture
def synthetic_tif(tmp_path: Path) -> Path:
    array = np.full((4, 96, 112), 1200, dtype=np.uint16)
    array[:, 20:70, 30:90] = 8000
    path = tmp_path / "synthetic.tif"
    profile = {
        "driver": "GTiff",
        "height": 96,
        "width": 112,
        "count": 4,
        "dtype": "uint16",
        "crs": "EPSG:32632",
        "transform": from_origin(500000, 5000000, 10, 10),
    }
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(array)
        for index, name in enumerate(["B08", "B04", "B03", "B02"], start=1):
            dataset.set_band_description(index, name)
    return path

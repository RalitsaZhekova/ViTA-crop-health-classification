import json

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from prithvi_crop.europe import (
    PASTIS_FINE_CLASS_MAP,
    PastisReplayDataset,
    european_sample_count,
    map_pastis_masks,
    select_seasonal_dates,
)


def test_pastis_mapping_adds_no_classes_and_preserves_exact_crops() -> None:
    source = np.arange(20, dtype=np.int64).reshape(4, 5)
    fine, crop = map_pastis_masks(source)

    assert set(np.unique(fine)).issubset({-1, 2, 3, 7, 11})
    for source_class, project_class in PASTIS_FINE_CLASS_MAP.items():
        assert fine[source == source_class].item() == project_class
    assert crop[source == 0].item() == 0
    assert np.all(crop[(source >= 1) & (source <= 18)] == 1)
    assert crop[source == 19].item() == -1


def test_twenty_percent_replay_keeps_every_original_sample() -> None:
    assert european_sample_count(2775, 0.2) == 694


def test_seasonal_date_selection_matches_original_phenology() -> None:
    dates = [20190220, 20190306, 20190717, 20190927, 20191020]
    indices, coordinates = select_seasonal_dates(dates)
    assert indices == [1, 2, 3]
    assert coordinates[:, 1].tolist() == [65.0, 198.0, 270.0]


def test_pastis_mapping_rejects_unknown_labels() -> None:
    with pytest.raises(ValueError, match="unknown"):
        map_pastis_masks(np.asarray([[20]]))


def test_pastis_replay_sample_matches_prithvi_contract(tmp_path) -> None:
    root = tmp_path / "PASTIS"
    (root / "DATA_S2").mkdir(parents=True)
    (root / "ANNOTATIONS").mkdir()
    dates = {"0": 20190306, "1": 20190717, "2": 20190927}
    metadata = gpd.GeoDataFrame(
        {
            "ID_PATCH": [42],
            "Fold": [1],
            "dates-S2": [json.dumps(dates)],
        },
        geometry=[Point(2.0, 48.0)],
        crs=4326,
    )
    metadata.to_file(root / "metadata.geojson", driver="GeoJSON")
    image = np.zeros((3, 10, 128, 128), dtype=np.float32)
    image[:, 7] = 8000
    np.save(root / "DATA_S2/S2_42.npy", image)
    mask = np.full((1, 128, 128), 3, dtype=np.int64)
    np.save(root / "ANNOTATIONS/TARGET_42.npy", mask)

    sample = PastisReplayDataset(
        tmp_path,
        folds=(1,),
        augment=False,
    )[0]

    assert sample["image"].shape == (4, 3, 224, 224)
    assert sample["mask"].shape == (224, 224)
    assert sample["crop_mask"].shape == (224, 224)
    assert sample["mask"][112, 112].item() == 2
    assert sample["crop_mask"][112, 112].item() == 1
    assert sample["mask"][0, 0].item() == -1
    assert sample["location_coords"].tolist() == [48.0, 2.0]

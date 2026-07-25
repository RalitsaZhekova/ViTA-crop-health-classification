import json

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

from prithvi_crop import pastis_validation
from prithvi_crop.pastis_validation import validate_pastis


def _write_patch(root, patch_id: int) -> None:
    (root / "DATA_S2").mkdir(exist_ok=True)
    (root / "ANNOTATIONS").mkdir(exist_ok=True)
    np.save(
        root / "DATA_S2" / f"S2_{patch_id}.npy",
        np.zeros((3, 10, 128, 128), dtype=np.int16),
    )
    np.save(
        root / "ANNOTATIONS" / f"TARGET_{patch_id}.npy",
        np.zeros((3, 128, 128), dtype=np.uint8),
    )


def test_validation_follows_metadata_and_ignores_unlabelled_images(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(pastis_validation, "EXPECTED_METADATA_PATCHES", 5)
    rows = []
    for patch_id, fold in enumerate(range(1, 6), start=10):
        _write_patch(tmp_path, patch_id)
        rows.append(
            {
                "ID_PATCH": patch_id,
                "Fold": fold,
                "dates-S2": json.dumps(
                    {"0": 20190301, "1": 20190701, "2": 20191001}
                ),
                "geometry": Point(700000 + patch_id, 6600000),
            }
        )
    np.save(
        tmp_path / "DATA_S2" / "S2_999.npy",
        np.zeros((3, 10, 128, 128), dtype=np.int16),
    )
    gpd.GeoDataFrame(rows, crs=2154).to_file(
        tmp_path / "metadata.geojson",
        driver="GeoJSON",
    )

    report = validate_pastis(tmp_path)

    assert report["status"] == "ready"
    assert report["metadata_patches"] == 5
    assert report["extra_unlabelled_images_ignored"] == 1


def test_validation_rejects_missing_metadata_target(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pastis_validation, "EXPECTED_METADATA_PATCHES", 5)
    rows = []
    for patch_id, fold in enumerate(range(1, 6), start=10):
        _write_patch(tmp_path, patch_id)
        rows.append(
            {
                "ID_PATCH": patch_id,
                "Fold": fold,
                "dates-S2": {
                    "0": 20190301,
                    "1": 20190701,
                    "2": 20191001,
                },
                "geometry": Point(700000 + patch_id, 6600000),
            }
        )
    (tmp_path / "ANNOTATIONS" / "TARGET_12.npy").unlink()
    gpd.GeoDataFrame(rows, crs=2154).to_file(
        tmp_path / "metadata.geojson",
        driver="GeoJSON",
    )

    with pytest.raises(ValueError, match="missing targets"):
        validate_pastis(tmp_path)

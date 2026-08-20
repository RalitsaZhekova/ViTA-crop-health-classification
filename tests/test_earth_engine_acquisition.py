from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import google.auth
import numpy as np
import pytest
import rasterio
from prithvi_ground.catalog import SOURCE_MEASUREMENT_FIELDS, _validate_source
from prithvi_payload.acquisition.earth_engine import (
    EARTH_ENGINE_BANDS,
    AcquiredSentinelScene,
    Candidate,
    EarthEngineAcquisitionProvider,
    _normalise_geotiff,
    _parse_candidates,
    _unique_datatake_candidates,
    initialize_earth_engine,
)
from prithvi_payload.acquisition.errors import AcquisitionError
from prithvi_payload.acquisition.grid import calculate_target_grid


def test_initialize_earth_engine_preserves_adc_scopes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    credentials_path = tmp_path / "credentials.json"
    credentials_path.write_text("{}", encoding="utf-8")
    credential = object()
    captured: dict[str, object] = {}

    def load_credentials(path: str, **kwargs: object) -> tuple[object, None]:
        captured["path"] = path
        captured["load_kwargs"] = kwargs
        return credential, None

    def initialize(**kwargs: object) -> None:
        captured["initialize_kwargs"] = kwargs

    monkeypatch.setenv("VITA_EE_PROJECT", "vita-503208")
    monkeypatch.setenv("VITA_EE_CREDENTIALS", str(credentials_path))
    monkeypatch.setattr(google.auth, "load_credentials_from_file", load_credentials)
    monkeypatch.setitem(sys.modules, "ee", SimpleNamespace(Initialize=initialize))

    initialize_earth_engine()

    assert captured["path"] == str(credentials_path)
    assert captured["load_kwargs"] == {}
    assert captured["initialize_kwargs"] == {
        "credentials": credential,
        "project": "vita-503208",
    }


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
                    "DATATAKE_IDENTIFIER": "take-older",
                },
            },
            {
                "id": "collection/cloudy",
                "properties": {
                    "system:time_start": 1_800_000_000_000,
                    "CLOUDY_PIXEL_PERCENTAGE": 20.0,
                    "DATATAKE_IDENTIFIER": "take-cloudy",
                },
            },
            {
                "id": "collection/newer-clear",
                "properties": {
                    "system:time_start": 1_750_000_000_000,
                    "CLOUDY_PIXEL_PERCENTAGE": 3.0,
                    "DATATAKE_IDENTIFIER": "take-newer",
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


def test_candidates_are_limited_to_unique_datatakes() -> None:
    candidates = [
        Candidate("tile-a", "2026-06-01T10:00:00+00:00", 1, 1.0, None, "take-1"),
        Candidate("tile-b", "2026-06-01T10:00:01+00:00", 2, 2.0, None, "take-1"),
        Candidate("tile-c", "2026-06-06T10:00:00+00:00", 3, 3.0, None, "take-2"),
    ]

    selected = _unique_datatake_candidates(candidates, 2)

    assert [candidate.system_index for candidate in selected] == ["tile-a", "tile-c"]
    assert [candidate.rank for candidate in selected] == [1, 2]


def test_candidate_image_mosaics_every_granule_in_the_datatake(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []

    class Collection:
        def filterBounds(self, region):
            calls.append(("filterBounds", region))
            return self

        def filter(self, condition):
            calls.append(("filter", condition))
            return self

        def select(self, selectors, names):
            calls.append(("select", selectors, names))
            return self

        def mosaic(self):
            calls.append(("mosaic",))
            return "mosaic-image"

    collection = Collection()
    fake_ee = SimpleNamespace(
        Geometry=SimpleNamespace(
            Rectangle=lambda bounds, geodesic: ("rectangle", bounds, geodesic)
        ),
        Filter=SimpleNamespace(eq=lambda name, value: ("eq", name, value)),
        ImageCollection=lambda name: (
            calls.append(("collection", name)) or collection
        ),
    )
    monkeypatch.setitem(sys.modules, "ee", fake_ee)
    candidate = Candidate(
        system_index="tile-a",
        acquired_at="2026-06-01T10:00:00+00:00",
        acquired_at_millis=1_780_307_200_000,
        metadata_cloud_percentage=1.0,
        product_id=None,
        datatake_identifier="take-1",
    )

    image = EarthEngineAcquisitionProvider._candidate_image(
        candidate, (5.43, 52.50, 5.49, 52.54)
    )

    assert image == "mosaic-image"
    assert ("filter", ("eq", "DATATAKE_IDENTIFIER", "take-1")) in calls
    assert calls[-2:] == [
        ("select", ["B2", "B3", "B4", "B8", "B8A"], list(EARTH_ENGINE_BANDS)),
        ("mosaic",),
    ]


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
        datatake_identifier="take-id",
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
    assert provenance["selection_policy"] == "least_cloudy_acquisition_pass"
    assert provenance["spatial_assembly"] == "same_acquisition_pass_mosaic"
    assert provenance["target_cloud_range"] is None
    assert not ({"credentials", "token", "url"} & set(provenance))

    provenance.update({field: 0.0 for field in SOURCE_MEASUREMENT_FIELDS})
    _validate_source(provenance)

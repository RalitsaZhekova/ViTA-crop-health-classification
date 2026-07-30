from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from affine import Affine
from prithvi_payload.acquisition.earth_engine import (
    EARTH_ENGINE_BANDS,
    EARTH_ENGINE_COLLECTION,
    EARTH_ENGINE_PROJECT_ID,
    REFLECTANCE_SCALE,
    EarthEngineAcquisitionProvider,
    initialize_earth_engine,
    normalize_and_validate_geotiff,
    order_candidates,
    parse_candidate_features,
)
from prithvi_payload.acquisition.errors import AcquisitionError
from prithvi_payload.acquisition.grid import TargetGrid, calculate_target_grid
from prithvi_payload.acquisition.models import CandidateMetadata
from prithvi_shared import PayloadAcquisitionCommand


def _command(policy: str = "target_cloud_range") -> PayloadAcquisitionCommand:
    return PayloadAcquisitionCommand.model_validate(
        {
            "schema_version": "1.0",
            "job_id": "field_42_20260715_ab12cd34",
            "region_id": "field_42",
            "source": {
                "provider": "earth_engine",
                "bbox_wgs84": [23.10, 42.50, 23.15, 42.55],
                "start_date": "2026-07-01",
                "end_date": "2026-07-29",
                "selection_policy": policy,
                "target_cloud_min_percent": 15,
                "target_cloud_max_percent": 35,
                "target_cloud_ideal_percent": 25,
            },
        }
    )


def _candidate(
    scene_id: str,
    cloud: float | None,
    timestamp: int,
) -> CandidateMetadata:
    return CandidateMetadata(
        system_index=scene_id,
        acquired_at=f"2026-07-{timestamp:02d}T10:00:00+00:00",
        acquired_at_millis=timestamp,
        metadata_cloud_percent=cloud,
        product_id=f"product-{scene_id}",
        source_metadata={},
    )


def test_target_cloud_ordering_filters_then_chooses_closest_to_ideal() -> None:
    ordered = order_candidates(
        [
            _candidate("outside", 14.9, 10),
            _candidate("far", 35.0, 30),
            _candidate("newer", 24.0, 40),
            _candidate("older", 26.0, 20),
            _candidate("no-metadata", None, 50),
            _candidate("ideal", 25.0, 10),
        ],
        _command(),
    )

    assert [candidate.system_index for candidate in ordered] == [
        "ideal",
        "newer",
        "older",
        "far",
    ]
    assert [candidate.candidate_rank for candidate in ordered] == [1, 2, 3, 4]


def test_target_cloud_ordering_is_newest_then_lexical_for_equal_distance() -> None:
    ordered = order_candidates(
        [
            _candidate("b", 24.0, 10),
            _candidate("a", 26.0, 20),
            _candidate("c", 24.0, 20),
        ],
        _command(),
    )

    assert [candidate.system_index for candidate in ordered] == ["a", "c", "b"]


def test_no_metadata_comes_last_for_least_cloudy() -> None:
    ordered = order_candidates(
        [
            _candidate("missing", None, 30),
            _candidate("cloudy", 40.0, 20),
            _candidate("clear", 5.0, 10),
        ],
        _command("least_cloudy"),
    )

    assert [candidate.system_index for candidate in ordered] == [
        "clear",
        "cloudy",
        "missing",
    ]


def test_no_candidate_in_target_metadata_range_has_fixed_error() -> None:
    with pytest.raises(AcquisitionError) as caught:
        order_candidates(
            [_candidate("clear", 2.0, 10), _candidate("missing", None, 20)],
            _command(),
        )

    assert caught.value.code == "EARTH_ENGINE_NO_TARGET_CLOUD_SCENE"


def test_candidate_parser_keeps_only_safe_metadata() -> None:
    parsed = parse_candidate_features(
        [
            {
                "id": f"{EARTH_ENGINE_COLLECTION}/scene-a",
                "properties": {
                    "system:time_start": 1_785_321_000_000,
                    "CLOUDY_PIXEL_PERCENTAGE": 24.5,
                    "MGRS_TILE": "34TFN",
                    "PRODUCT_ID": "safe-product",
                    "private_key": "must-not-survive",
                },
            }
        ]
    )

    assert parsed[0].system_index == "scene-a"
    assert parsed[0].metadata_cloud_percent == 24.5
    assert parsed[0].source_metadata == {"MGRS_TILE": "34TFN"}
    assert "private_key" not in parsed[0].safe_record()["source_metadata"]


def test_target_grid_is_fixed_ten_metre_utm_and_bounded() -> None:
    grid = calculate_target_grid((23.10, 42.50, 23.15, 42.55))

    assert grid.crs == "EPSG:32634"
    assert grid.transform.a == 10
    assert grid.transform.e == -10
    assert 0 < grid.width <= 1024
    assert 0 < grid.height <= 1024
    assert grid.estimated_uncompressed_bytes == grid.width * grid.height * 5 * 2

    with pytest.raises(AcquisitionError) as caught:
        calculate_target_grid((23.0, 42.0, 24.0, 43.0))
    assert caught.value.code == "REGION_TOO_LARGE"


def _small_grid() -> TargetGrid:
    return TargetGrid(
        crs="EPSG:32634",
        transform=Affine(10, 0, 500_000, 0, -10, 4_700_000),
        width=3,
        height=2,
        bounds=(500_000, 4_699_980, 500_030, 4_700_000),
        estimated_uncompressed_bytes=60,
    )


def _write_raw_tiff(path: Path, grid: TargetGrid) -> np.ndarray:
    values = np.arange(5 * grid.height * grid.width, dtype=np.uint16).reshape(
        5, grid.height, grid.width
    )
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
    ) as dataset:
        dataset.write(values)
    return values


def test_normalization_assigns_metadata_without_changing_pixels(tmp_path: Path) -> None:
    grid = _small_grid()
    raw = tmp_path / "source_raw.tif"
    normalized = tmp_path / "scene.tif"
    expected = _write_raw_tiff(raw, grid)

    normalize_and_validate_geotiff(raw, normalized, grid=grid)

    with rasterio.open(raw) as raw_dataset, rasterio.open(normalized) as normalized_dataset:
        np.testing.assert_array_equal(raw_dataset.read(), normalized_dataset.read())
        np.testing.assert_array_equal(normalized_dataset.read(), expected)
        assert normalized_dataset.descriptions == EARTH_ENGINE_BANDS
        assert normalized_dataset.tags()["REFLECTANCE_SCALE"] == str(int(REFLECTANCE_SCALE))


def test_download_parameters_are_fixed_and_contain_no_resampling_expression() -> None:
    provider = EarthEngineAcquisitionProvider(max_candidates=50, max_scene_attempts=5)
    grid = _small_grid()

    parameters = provider._download_parameters(grid)

    assert parameters["bands"] == ["B02", "B03", "B04", "B08", "B8A"]
    assert parameters["format"] == "GEO_TIFF"
    assert parameters["filePerBand"] is False
    assert parameters["crs"] == grid.crs
    assert parameters["crs_transform"] == list(grid.transform)[:6]
    assert parameters["dimensions"] == [grid.width, grid.height]
    # Earth Engine rejects region + crs_transform + dimensions. The target
    # transform and dimensions already describe the bbox-derived grid exactly.
    assert "region" not in parameters
    assert "resampling" not in parameters


class _Response:
    def __init__(self, status: int, body: bytes = b"") -> None:
        self.status_code = status
        self.body = body
        self.headers = {
            "Content-Type": "image/tiff",
            "Content-Length": str(len(body)),
        }

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def iter_content(self, chunk_size: int) -> list[bytes]:
        del chunk_size
        return [self.body]


class _Session:
    def __init__(self, responses: list[_Response]) -> None:
        self.responses = responses
        self.urls: list[str] = []

    def get(self, url: str, **_: Any) -> _Response:
        self.urls.append(url)
        return self.responses.pop(0)


class _Image:
    def __init__(self, url: str) -> None:
        self.url = url
        self.calls = 0

    def getDownloadURL(self, _: dict[str, Any]) -> str:  # noqa: N802
        self.calls += 1
        return self.url


class _RejectedImage:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def getDownloadURL(self, _: dict[str, Any]) -> str:  # noqa: N802
        self.calls += 1
        raise self.error


def test_candidate_download_retries_without_logging_signed_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    grid = _small_grid()
    raw = tmp_path / "raw.tif"
    _write_raw_tiff(raw, grid)
    signed_url = "https://provider.invalid/download?access_token=top-secret"
    session = _Session([_Response(503), _Response(200, raw.read_bytes())])
    provider = EarthEngineAcquisitionProvider(session=session)
    image = _Image(signed_url)
    monkeypatch.setattr("prithvi_payload.acquisition.earth_engine.time.sleep", lambda _: None)

    with caplog.at_level(logging.DEBUG):
        provider._stream_candidate(image, {}, tmp_path / "download.partial")

    assert image.calls == 2
    assert len(session.urls) == 2
    assert signed_url not in caplog.text
    assert "top-secret" not in caplog.text


def test_download_request_failure_exposes_only_safe_category(tmp_path: Path) -> None:
    secret = "https://provider.invalid/download?access_token=top-secret"
    provider = EarthEngineAcquisitionProvider(session=_Session([]))
    image = _RejectedImage(RuntimeError(f"Permission denied for {secret}"))

    with pytest.raises(AcquisitionError) as caught:
        provider._stream_candidate(image, {}, tmp_path / "download.partial")

    assert caught.value.code == "EARTH_ENGINE_DOWNLOAD_REQUEST_FAILED"
    assert caught.value.details == {
        "provider_reason": "permission_denied",
        "provider_error_type": "RuntimeError",
    }
    serialized = str(caught.value.safe_record())
    assert secret not in serialized
    assert "top-secret" not in serialized


def test_transient_download_request_failure_retries_safely(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _EventuallyAvailableImage:
        calls = 0

        def getDownloadURL(self, _: dict[str, Any]) -> str:  # noqa: N802
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("Service temporarily unavailable")
            return "https://provider.invalid/safe"

    grid = _small_grid()
    raw = tmp_path / "raw.tif"
    _write_raw_tiff(raw, grid)
    image = _EventuallyAvailableImage()
    provider = EarthEngineAcquisitionProvider(
        session=_Session([_Response(200, raw.read_bytes())])
    )
    monkeypatch.setattr("prithvi_payload.acquisition.earth_engine.time.sleep", lambda _: None)

    provider._stream_candidate(image, {}, tmp_path / "download.partial")

    assert image.calls == 2


def test_authentication_failure_is_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    ee_module = types.ModuleType("ee")
    ee_module.Initialize = lambda **_: None  # type: ignore[attr-defined]
    google_module = types.ModuleType("google")
    google_module.__path__ = []  # type: ignore[attr-defined]
    auth_module = types.ModuleType("google.auth")

    def fail_default(**_: Any) -> None:
        raise RuntimeError("private_key=top-secret access_token=also-secret")

    auth_module.default = fail_default  # type: ignore[attr-defined]
    google_module.auth = auth_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ee", ee_module)
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.auth", auth_module)

    with pytest.raises(AcquisitionError) as caught:
        initialize_earth_engine(EARTH_ENGINE_PROJECT_ID)

    assert caught.value.code == "EARTH_ENGINE_AUTHENTICATION_FAILED"
    assert caught.value.__cause__ is None
    assert "top-secret" not in str(caught.value)
    assert "access_token" not in str(caught.value)


def test_provider_limits_reject_zero_instead_of_using_environment_default() -> None:
    with pytest.raises(ValueError, match="1..50"):
        EarthEngineAcquisitionProvider(max_candidates=0)
    with pytest.raises(ValueError, match="1..5"):
        EarthEngineAcquisitionProvider(max_scene_attempts=0)

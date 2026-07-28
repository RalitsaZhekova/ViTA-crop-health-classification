from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from prithvi_payload.condition_stage import (
    FLOAT_NODATA,
    StreamingMetric,
    run_payload_condition,
)
from rasterio.transform import from_origin


def _write_raster(
    path: Path,
    values: np.ndarray,
    *,
    descriptions: tuple[str, ...] | None = None,
    transform_shift: float = 0.0,
    nodata: float | int | None = None,
) -> None:
    data = values[np.newaxis, ...] if values.ndim == 2 else values
    profile = {
        "driver": "GTiff",
        "width": data.shape[2],
        "height": data.shape[1],
        "count": data.shape[0],
        "dtype": data.dtype,
        "crs": "EPSG:32631",
        "transform": from_origin(500_000 + transform_shift, 4_600_000, 10, 10),
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(data)
        if descriptions is not None:
            for index, description in enumerate(descriptions, start=1):
                destination.set_band_description(index, description)


def _build_payload_fixture(
    root: Path,
    *,
    crop_size: int = 32,
    transform_shift: float = 0.0,
) -> Path:
    shape = (32, 32)
    blue = np.full(shape, 0.05, dtype=np.float32)
    green = np.full(shape, 0.10, dtype=np.float32)
    red = np.full(shape, 0.05, dtype=np.float32)
    nir_broad = np.full(shape, 0.75, dtype=np.float32)
    nir_narrow = np.full(shape, 0.80, dtype=np.float32)
    blue[:8, :8] = 0.20
    green[:8, :8] = 0.25
    red[:8, :8] = 0.30
    nir_broad[:8, :8] = 0.34
    nir_narrow[:8, :8] = 0.35
    source_values = 10_000.0 * np.stack((red, green, blue, nir_broad, nir_narrow))

    source_path = root / "sentinel_scene.tif"
    unusable_path = root / "unusable.tif"
    crop_path = root / "crop_binary.tif"
    probability_path = root / "crop_probability.tif"
    _write_raster(
        source_path,
        source_values.astype(np.float32),
        descriptions=("B4", "B3", "B2", "B8", "B8A"),
    )
    _write_raster(unusable_path, np.zeros(shape, dtype=np.uint8), nodata=255)
    crop = np.zeros(shape, dtype=np.uint8)
    crop.flat[:crop_size] = 1
    if crop_size == shape[0] * shape[1]:
        crop[:] = 1
    _write_raster(
        crop_path,
        crop,
        transform_shift=transform_shift,
        nodata=255,
    )
    _write_raster(
        probability_path,
        np.full(shape, 0.90, dtype=np.float32),
        nodata=-9999.0,
    )

    payload_result = {
        "schema_version": "0.1-draft",
        "scene_id": "sentinel_scene",
        "sensor": "sentinel-2",
        "status": "CROP_COMPLETE",
        "completed_stages": ["intake", "cloud", "crop"],
        "artifacts": {
            "cloud": {"unusable_mask": str(unusable_path)},
            "crop": {
                "crop_binary": str(crop_path),
                "crop_probability": str(probability_path),
            },
        },
        "stage_metadata": {
            "intake": {
                "source_path": str(source_path),
                "acquired_at": "2026-07-27T12:00:00+00:00",
                "logical_band_mapping": {
                    "RED": {"index": 1, "description": "B4"},
                    "GREEN": {"index": 2, "description": "B3"},
                    "BLUE": {"index": 3, "description": "B2"},
                    "NIR_BROAD": {"index": 4, "description": "B8"},
                    "NIR_NARROW": {"index": 5, "description": "B8A"},
                },
            },
            "cloud_plan": {
                "input": {
                    "reflectance_scale": 10_000.0,
                    "reflectance_scale_source": "test_fixture",
                }
            },
        },
    }
    result_path = root / "result.json"
    result_path.write_text(json.dumps(payload_result), encoding="utf-8")
    return result_path


def test_run_payload_condition_writes_complete_geospatial_result(tmp_path: Path) -> None:
    result_path = _build_payload_fixture(tmp_path, crop_size=32 * 32)
    output = tmp_path / "ground"

    report = run_payload_condition(
        result_path,
        output_root=output,
        region_id="test-region",
        tile_size=8,
    )

    assert report["status"] == "MEASURED"
    assert report["condition"]["label"] == "Watch"
    assert report["condition"]["relative_anomaly_percentage"] == pytest.approx(6.25)
    assert report["condition"]["low_vigor_percentage"] == pytest.approx(6.25)
    assert report["quality"]["analysis_pixels"] == 32 * 32
    assert report["runtime"]["first_pass_windows"] == 16
    assert report["radiometry"]["nir_role"] == "NIR_NARROW"
    assert report["region_id"] == "test-region"

    saved_report = json.loads((output / "crop_condition_report.json").read_text())
    assert saved_report["condition"] == report["condition"]
    repository_root = Path(__file__).parents[2]
    schema = json.loads(
        (repository_root / "shared/schemas/health_observation.schema.json").read_text()
    )
    assert set(schema["required"]) <= set(saved_report)
    assert set(saved_report) <= set(schema["properties"])
    assert set(schema["properties"]["condition"]["required"]) <= set(saved_report["condition"])
    assert len(report["raster_assets"]) == 15
    for relative_path in report["raster_assets"].values():
        assert not Path(relative_path).is_absolute()
        assert (output / relative_path).is_file()

    alert_path = output / report["raster_assets"]["alert_mask"]
    with rasterio.open(alert_path) as alert:
        assert alert.crs.to_string() == "EPSG:32631"
        assert alert.width == 32
        assert alert.height == 32
        assert alert.compression.name == "deflate"
        assert np.count_nonzero(alert.read(1) == 1) == 64

    ndvi_path = output / report["raster_assets"]["ndvi"]
    with rasterio.open(ndvi_path) as ndvi:
        values = ndvi.read(1)
        assert values[16, 16] > values[0, 0]
        assert ndvi.nodata == FLOAT_NODATA


def test_payload_condition_is_deterministic_across_tile_sizes(tmp_path: Path) -> None:
    result_path = _build_payload_fixture(tmp_path, crop_size=32 * 32)

    first = run_payload_condition(result_path, output_root=tmp_path / "a", tile_size=7)
    second = run_payload_condition(result_path, output_root=tmp_path / "b", tile_size=16)

    assert first["condition"] == second["condition"]
    assert first["metrics"] == second["metrics"]
    assert first["quality"] == second["quality"]


def test_payload_condition_returns_insufficient_data_without_alerts(tmp_path: Path) -> None:
    result_path = _build_payload_fixture(tmp_path, crop_size=16)
    output = tmp_path / "ground"

    report = run_payload_condition(result_path, output_root=output, tile_size=8)

    assert report["status"] == "INSUFFICIENT_DATA"
    assert report["condition"]["label"] == "Insufficient data"
    assert report["condition"]["condition_score"] is None
    with rasterio.open(output / report["raster_assets"]["alert_mask"]) as alert:
        assert not np.any(alert.read(1) == 1)


def test_payload_condition_rejects_misaligned_payload_mask(tmp_path: Path) -> None:
    result_path = _build_payload_fixture(
        tmp_path,
        crop_size=32 * 32,
        transform_shift=5.0,
    )

    with pytest.raises(ValueError, match="Crop binary mask.*grid"):
        run_payload_condition(result_path, output_root=tmp_path / "ground")


def test_payload_condition_requires_completed_crop_stage(tmp_path: Path) -> None:
    result_path = _build_payload_fixture(tmp_path)
    value = json.loads(result_path.read_text())
    value["status"] = "CLOUD_COMPLETE"
    result_path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="CROP_COMPLETE"):
        run_payload_condition(result_path, output_root=tmp_path / "ground")


def test_payload_condition_requires_explicit_reflectance_scale(tmp_path: Path) -> None:
    result_path = _build_payload_fixture(tmp_path)
    value = json.loads(result_path.read_text())
    del value["stage_metadata"]["cloud_plan"]["input"]["reflectance_scale"]
    result_path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="reflectance scale"):
        run_payload_condition(result_path, output_root=tmp_path / "ground")


def test_payload_condition_records_broad_nir_fallback(tmp_path: Path) -> None:
    result_path = _build_payload_fixture(tmp_path, crop_size=32 * 32)
    value = json.loads(result_path.read_text())
    del value["stage_metadata"]["intake"]["logical_band_mapping"]["NIR_NARROW"]
    result_path.write_text(json.dumps(value), encoding="utf-8")

    report = run_payload_condition(result_path, output_root=tmp_path / "ground")

    assert report["radiometry"]["nir_role"] == "NIR_BROAD"
    assert any("NIR_NARROW was unavailable" in item for item in report["warnings"])


def test_payload_condition_protects_existing_result(tmp_path: Path) -> None:
    result_path = _build_payload_fixture(tmp_path, crop_size=32 * 32)
    output = tmp_path / "ground"
    run_payload_condition(result_path, output_root=output)

    with pytest.raises(FileExistsError, match="already exists"):
        run_payload_condition(result_path, output_root=output)


def test_streaming_metric_uses_all_values_for_moments_and_is_deterministic() -> None:
    first = StreamingMetric("test", sample_limit=100)
    second = StreamingMetric("test", sample_limit=100)
    values = np.arange(1000, dtype=np.float64)
    for chunk in np.array_split(values, 7):
        first.update(chunk)
        second.update(chunk)

    first_summary = first.summary()
    second_summary = second.summary()
    assert first_summary == second_summary
    assert first_summary["valid_pixels"] == 1000
    assert first_summary["mean"] == pytest.approx(np.mean(values))
    assert first_summary["standard_deviation"] == pytest.approx(np.std(values))
    assert first_summary["minimum"] == 0
    assert first_summary["maximum"] == 999
    assert first_summary["percentile_sample_pixels"] == 100

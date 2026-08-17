from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from prithvi_payload.balkan_alignment import (
    AlignmentConfig,
    BalkanAlignmentError,
    align_balkan_geotiff,
    initial_row_shifts,
    load_band_start_rows,
)
from prithvi_payload.balkan_analysis import materialize_balkan_analysis_grid
from prithvi_payload.balkan_crop_calibration import (
    ADAPTER_MODE,
    CALIBRATION_SCHEMA_VERSION,
    MODEL_BAND_ORDER,
    SOURCE_BAND_ORDER,
    CalibrationError,
    load_calibration,
    sha256_file,
)
from prithvi_payload.cloud_stage import build_cloud_stage_plan
from prithvi_payload.scene_intake import inspect_scene
from rasterio.transform import Affine, from_origin
from scipy.ndimage import gaussian_filter, shift

RAW_DESCRIPTIONS = ("Band_1", "Band_2", "Band_3", "Band_7", "Band_0")


def _base_scene(size: int = 256) -> np.ndarray:
    random = np.random.default_rng(7)
    image = gaussian_filter(random.normal(size=(size, size)).astype(np.float32), sigma=2.0)
    image -= float(image.min())
    image *= np.float32(700.0 / float(image.max()))
    for row, column, height, width, value in (
        (35, 28, 38, 54, 1200.0),
        (96, 132, 72, 31, 1800.0),
        (181, 52, 29, 90, 900.0),
    ):
        image[row : row + height, column : column + width] += np.float32(value)
    return image + np.float32(400.0)


def _write_scene(
    path: Path,
    shifts: tuple[tuple[float, float], ...],
    *,
    georeferenced: bool,
) -> np.ndarray:
    pan = _base_scene()
    visible = pan * np.float32(0.8) + np.float32(120.0)
    nir = pan * np.float32(1.15) + np.float32(80.0)
    unshifted = (visible, visible * 0.95, visible * 1.05, nir, pan)
    values = np.stack(
        [
            shift(
                band,
                shift=band_shift,
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
            for band, band_shift in zip(unshifted, shifts, strict=True)
        ]
    ).astype(np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=pan.shape[1],
        height=pan.shape[0],
        count=5,
        dtype="float32",
        crs="EPSG:32631" if georeferenced else None,
        transform=(
            from_origin(500_000.0, 4_500_000.0, 2.0, 2.0)
            if georeferenced
            else Affine.identity()
        ),
        nodata=0.0,
    ) as destination:
        destination.write(values)
        for index, description in enumerate(RAW_DESCRIPTIONS, start=1):
            destination.set_band_description(index, description)
    return pan


def _test_config(*, search_radius: int = 12) -> AlignmentConfig:
    return AlignmentConfig(
        grid_rows=4,
        grid_cols=4,
        measurement_tile_size=64,
        local_search_radius_px=search_radius,
        global_search_radius_px=search_radius,
        minimum_confidence=0.15,
        write_block_size=64,
    )


def _write_parent_calibration(path: Path, source: Path) -> None:
    calibration = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "adapter_mode": ADAPTER_MODE,
        "sensor": "balkan-1",
        "acquired_at": "2026-03-31T12:00:00+00:00",
        "source_band_order": list(SOURCE_BAND_ORDER),
        "model_band_order": list(MODEL_BAND_ORDER),
        "source_band_indices_1_based": [1, 2, 3, 4],
        "source_scale_to_model_units": 1.0,
        "analysis_resolution_metres": 10.0,
        "source": {
            "filename": source.name,
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        },
        "curves": [
            {
                "band": band,
                "source_knots": [0.0, 3_000.0],
                "target_values": [0.0, 10_000.0],
            }
            for band in SOURCE_BAND_ORDER
        ],
        "reference": {"sensor": "sentinel-2", "sha256": "a" * 64},
        "validation": {
            "status": "PASS",
            "held_out_valid_pixels": 10_000,
            "held_out_band_correlations": [0.9, 0.9, 0.9, 0.9],
        },
    }
    path.write_text(json.dumps(calibration), encoding="utf-8")


def test_alignment_writes_pipeline_compatible_geotiff(tmp_path: Path) -> None:
    source = tmp_path / "source.tif"
    output = tmp_path / "aligned.tif"
    pan = _write_scene(
        source,
        ((5.0, -3.0), (-4.0, 2.0), (3.0, 4.0), (-6.0, -5.0), (0.0, 0.0)),
        georeferenced=True,
    )

    report = align_balkan_geotiff(source, output, config=_test_config())

    assert report["output"]["pipeline_ready"] is True
    assert report["output"]["radiometry_modified"] is False
    assert report["registrations"]["PANCHROMATIC"]["method"] == "reference"
    assert all(item["aligned"] for item in report["registrations"].values())
    with rasterio.open(output) as aligned:
        assert aligned.descriptions == (
            "BLUE",
            "GREEN",
            "RED",
            "NIR_BROAD",
            "PANCHROMATIC",
        )
        assert aligned.crs == rasterio.crs.CRS.from_epsg(32631)
        assert aligned.transform == from_origin(500_000.0, 4_500_000.0, 2.0, 2.0)
        assert aligned.tags()["PIPELINE_READY"] == "TRUE"
        assert aligned.tags()["RADIOMETRY_MODIFIED"] == "FALSE"
        assert aligned.overviews(1) == [2]
        registered = aligned.read(masked=True)
    core = (slice(24, -24), slice(24, -24))
    for index in range(4):
        band = np.asarray(registered[index][core])
        assert np.corrcoef(band.ravel(), pan[core].ravel())[0, 1] > 0.98

    intake = inspect_scene(
        output,
        sensor="balkan-1",
        acquired_at="2026-03-31T12:00:00+00:00",
    )
    assert intake["readiness"]["intake"] == "READY"
    assert intake["readiness"]["cloud"] == "READY"
    assert intake["readiness"]["crop"] == "BALKAN_1_SPECTRAL_HARMONISATION_REQUIRED"

    calibration_path = tmp_path / "parent.crop_calibration.json"
    _write_parent_calibration(calibration_path, source)
    calibration = load_calibration(calibration_path, source_path=output)
    assert calibration["_source_provenance"]["mode"] == (
        "VERIFIED_GEOMETRY_ONLY_ALIGNMENT_DERIVATION"
    )
    assert calibration["_source_provenance"]["alignment_report"] == (
        "aligned.alignment.json"
    )
    assert calibration["_verified_source_sha256"] == sha256_file(output)
    calibrated_intake = inspect_scene(
        output,
        sensor="balkan-1",
        acquired_at="2026-03-31T12:00:00+00:00",
        crop_calibration_path=calibration_path,
    )
    assert calibrated_intake["readiness"]["crop"] == "READY"
    assert calibrated_intake["source_sha256"] == sha256_file(output)
    assert (
        calibrated_intake["model_band_routes"]["crop_classification"]["spectral_adapter"][
            "source_provenance"
        ]["mode"]
        == "VERIFIED_GEOMETRY_ONLY_ALIGNMENT_DERIVATION"
    )
    analysis = materialize_balkan_analysis_grid(
        calibrated_intake,
        output_root=tmp_path / "pipeline-run",
    )
    calibrated_intake["analysis"] = analysis
    cloud_plan = build_cloud_stage_plan(calibrated_intake, reflectance_scale=1.0)
    assert cloud_plan["readiness"] == "READY"
    assert analysis["model_band_routes"]["crop_classification"][
        "source_band_indices"
    ] == [1, 2, 3, 4]
    report_path = output.with_suffix(".alignment.json")
    tampered_report = json.loads(report_path.read_text(encoding="utf-8"))
    tampered_report["registrations"]["BLUE"]["aligned"] = False
    report_path.write_text(json.dumps(tampered_report), encoding="utf-8")
    with pytest.raises(CalibrationError, match="does not match"):
        load_calibration(calibration_path, source_path=output)


def test_raw_band_start_rows_seed_shifts_beyond_search_radius(tmp_path: Path) -> None:
    source = tmp_path / "raw.tif"
    output = tmp_path / "raw-aligned.tif"
    metadata = tmp_path / "manifest.json"
    pan = _write_scene(
        source,
        ((-30.0, 0.0), (20.0, 0.0), (-15.0, 0.0), (25.0, 0.0), (0.0, 0.0)),
        georeferenced=False,
    )
    starts = [100.0] * 8
    starts[1] = 70.0
    starts[2] = 120.0
    starts[3] = 85.0
    starts[7] = 125.0
    metadata.write_text(
        json.dumps({"imager_configuration": {"BandStartRow": starts}}),
        encoding="utf-8",
    )

    loaded = load_band_start_rows(metadata)
    assert initial_row_shifts(loaded) == {
        "BLUE": -30.0,
        "GREEN": 20.0,
        "RED": -15.0,
        "NIR_BROAD": 25.0,
        "PANCHROMATIC": 0.0,
    }
    report = align_balkan_geotiff(
        source,
        output,
        metadata_path=metadata,
        config=_test_config(search_radius=6),
        require_georeferencing=False,
    )

    assert report["output"]["pipeline_ready"] is False
    assert report["warnings"]
    with rasterio.open(output) as aligned:
        assert aligned.crs is None
        values = aligned.read(masked=True)
    core = (slice(40, -40), slice(20, -20))
    for index in range(4):
        band = np.asarray(values[index][core])
        assert np.corrcoef(band.ravel(), pan[core].ravel())[0, 1] > 0.995


def test_raw_band_start_rows_can_seed_raster_columns(tmp_path: Path) -> None:
    source = tmp_path / "raw-column-oriented.tif"
    output = tmp_path / "raw-column-oriented-aligned.tif"
    metadata = tmp_path / "manifest.json"
    pan = _write_scene(
        source,
        ((0.0, -30.0), (0.0, 20.0), (0.0, -15.0), (0.0, 25.0), (0.0, 0.0)),
        georeferenced=False,
    )
    starts = [100.0] * 8
    starts[1] = 70.0
    starts[2] = 120.0
    starts[3] = 85.0
    starts[7] = 125.0
    metadata.write_text(
        json.dumps({"imager_configuration": {"BandStartRow": starts}}),
        encoding="utf-8",
    )

    report = align_balkan_geotiff(
        source,
        output,
        metadata_path=metadata,
        band_start_axis="column",
        config=_test_config(search_radius=6),
        require_georeferencing=False,
    )

    assert report["band_start_axis"] == "column"
    assert report["initial_row_shifts"] == {band: 0.0 for band in report["band_start_rows"]}
    assert report["initial_column_shifts"]["BLUE"] == -30.0
    with rasterio.open(output) as aligned:
        values = aligned.read(masked=True)
    core = (slice(20, -20), slice(40, -40))
    for index in range(4):
        band = np.asarray(values[index][core])
        assert np.corrcoef(band.ravel(), pan[core].ravel())[0, 1] > 0.995


def test_alignment_fails_closed_for_unregistrable_bands(tmp_path: Path) -> None:
    source = tmp_path / "flat.tif"
    output = tmp_path / "aligned.tif"
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        width=128,
        height=128,
        count=5,
        dtype="float32",
        crs="EPSG:32631",
        transform=from_origin(500_000.0, 4_500_000.0, 2.0, 2.0),
        nodata=0.0,
    ) as destination:
        destination.write(np.ones((5, 128, 128), dtype=np.float32))
        for index, description in enumerate(RAW_DESCRIPTIONS, start=1):
            destination.set_band_description(index, description)

    with pytest.raises(BalkanAlignmentError, match="failed closed"):
        align_balkan_geotiff(source, output, config=_test_config())
    assert not output.exists()


def test_pipeline_ready_alignment_rejects_raw_no_crs_input(tmp_path: Path) -> None:
    source = tmp_path / "raw.tif"
    _write_scene(
        source,
        ((0.0, 0.0),) * 5,
        georeferenced=False,
    )

    with pytest.raises(BalkanAlignmentError, match="requires orthorectification"):
        align_balkan_geotiff(source, tmp_path / "out.tif", config=_test_config())

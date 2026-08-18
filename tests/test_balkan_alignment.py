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
            from_origin(500_000.0, 4_500_000.0, 2.0, 2.0) if georeferenced else Affine.identity()
        ),
        nodata=0.0,
    ) as destination:
        destination.write(values)
        for index, description in enumerate(RAW_DESCRIPTIONS, start=1):
            destination.set_band_description(index, description)
    return pan


def _write_bridge_scene(path: Path) -> dict[str, float]:
    """Reproduce the corrected upstream repository's bridge test as a TIFF."""
    random = np.random.default_rng(0)
    size = 384

    def render_shapes(count: int) -> np.ndarray:
        canvas = np.zeros((size, size), dtype=np.float32)
        for _ in range(count):
            height, width = random.integers(6, 40, size=2)
            row = int(random.integers(0, size - height))
            column = int(random.integers(0, size - width))
            canvas[row : row + height, column : column + width] += np.float32(
                random.uniform(800.0, 3_500.0)
            )
        return canvas

    visible_background = random.normal(1_500.0, 60.0, (size, size)).astype(np.float32)
    nir_background = random.normal(1_500.0, 60.0, (size, size)).astype(np.float32)
    visible_structure = render_shapes(150)
    nir_structure = render_shapes(150)

    def noise() -> np.ndarray:
        return random.normal(0.0, 15.0, (size, size)).astype(np.float32)

    def translated(values: np.ndarray, dy: float, dx: float) -> np.ndarray:
        return shift(
            values,
            shift=(dy, dx),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )

    red = visible_background + visible_structure + noise()
    blue = translated(visible_background + 0.9 * visible_structure + noise(), -3.2, 2.1)
    green = translated(
        0.5 * visible_background
        + 0.5 * nir_background
        + 0.6 * visible_structure
        + 0.6 * nir_structure
        + noise(),
        1.5,
        -4.0,
    )
    nir = translated(nir_background + nir_structure + noise(), -6.0, 5.5)
    pan = translated(
        0.5 * visible_background
        + 0.5 * nir_background
        + 0.5 * visible_structure
        + 0.5 * nir_structure
        + noise(),
        -1.0,
        -1.5,
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=size,
        height=size,
        count=5,
        dtype="float32",
        crs="EPSG:32631",
        transform=from_origin(500_000.0, 4_500_000.0, 2.0, 2.0),
        nodata=0.0,
    ) as destination:
        destination.write(np.stack((blue, green, red, nir, pan)))
        for index, description in enumerate(RAW_DESCRIPTIONS, start=1):
            destination.set_band_description(index, description)
    return {
        "RED": 3_764.0,
        "BLUE": 4_616.0,
        "GREEN": 4_212.0,
        "NIR_BROAD": 3_044.0,
        "PANCHROMATIC": 3_396.0,
    }


def _test_config(*, search_radius: int = 12, device: str = "auto") -> AlignmentConfig:
    return AlignmentConfig(
        grid_rows=4,
        grid_cols=4,
        measurement_tile_size=64,
        local_search_radius_px=search_radius,
        global_search_radius_px=search_radius,
        minimum_confidence=0.15,
        write_block_size=64,
        device=device,
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
    assert report["algorithm"] == "balkan-pan-seeded-global-bridge-v1"
    assert report["source_repository"] == (
        "https://github.com/RalitsaZhekova/balkan1-band-alignment"
    )
    assert report["source_commit"] == "5ba3076f5d8a198248067512dbbff2728dc2e35b"
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
    assert calibration["_source_provenance"]["alignment_report"] == ("aligned.alignment.json")
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
    assert analysis["model_band_routes"]["crop_classification"]["source_band_indices"] == [
        1,
        2,
        3,
        4,
    ]
    report_path = output.with_suffix(".alignment.json")
    tampered_report = json.loads(report_path.read_text(encoding="utf-8"))
    tampered_report["registrations"]["BLUE"]["aligned"] = False
    report_path.write_text(json.dumps(tampered_report), encoding="utf-8")
    with pytest.raises(CalibrationError, match="does not match"):
        load_calibration(calibration_path, source_path=output)


def test_cuda_alignment_matches_cpu_output(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    source = tmp_path / "source.tif"
    cpu_output = tmp_path / "cpu.tif"
    cuda_output = tmp_path / "cuda.tif"
    _write_scene(
        source,
        ((5.0, -3.0), (-4.0, 2.0), (3.0, 4.0), (-6.0, -5.0), (0.0, 0.0)),
        georeferenced=True,
    )

    cpu_report = align_balkan_geotiff(
        source,
        cpu_output,
        config=_test_config(device="cpu"),
    )
    cuda_report = align_balkan_geotiff(
        source,
        cuda_output,
        config=_test_config(device="cuda"),
    )

    assert cpu_report["execution"]["resolved_device"] == "cpu"
    assert cuda_report["execution"]["resolved_device"] == "cuda"
    for band in ("BLUE", "GREEN", "RED", "NIR_BROAD"):
        cpu_registration = cpu_report["registrations"][band]
        cuda_registration = cuda_report["registrations"][band]
        assert cpu_registration["method"] == cuda_registration["method"]
        np.testing.assert_allclose(
            [cpu_registration["shift_dy"], cpu_registration["shift_dx"]],
            [cuda_registration["shift_dy"], cuda_registration["shift_dx"]],
            atol=0.02,
        )
    with rasterio.open(cpu_output) as cpu, rasterio.open(cuda_output) as cuda:
        cpu_values = cpu.read(masked=True)
        cuda_values = cuda.read(masked=True)
    cpu_mask = np.ma.getmaskarray(cpu_values)
    cuda_mask = np.ma.getmaskarray(cuda_values)
    assert np.mean(cpu_mask != cuda_mask) < 1e-4
    common_valid = ~cpu_mask & ~cuda_mask
    difference = np.abs(
        cpu_values.filled(0.0)[common_valid] - cuda_values.filled(0.0)[common_valid]
    )
    assert float(difference.mean()) < 0.05
    assert float(np.percentile(difference, 99)) < 0.5
    np.testing.assert_array_equal(cpu_values[4].filled(0.0), cuda_values[4].filled(0.0))


def test_low_confidence_direct_match_bridges_through_aligned_band(tmp_path: Path) -> None:
    source = tmp_path / "bridge-source.tif"
    output = tmp_path / "bridge-aligned.tif"
    band_start_rows = _write_bridge_scene(source)

    report = align_balkan_geotiff(
        source,
        output,
        band_start_rows=band_start_rows,
        band_start_row_scale=384.0 / 10_000.0,
        config=AlignmentConfig(
            measurement_tile_size=128,
            global_search_radius_px=48,
            minimum_confidence=0.30,
            write_block_size=64,
            device="cpu",
            build_overviews=False,
        ),
    )

    blue = report["registrations"]["BLUE"]
    assert blue["method"] == "via_GREEN"
    assert blue["confidence"] >= 0.30
    assert blue["shift_dy"] == pytest.approx(-2.0)
    assert blue["shift_dx"] == pytest.approx(3.0)
    assert all(item["aligned"] for item in report["registrations"].values())


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

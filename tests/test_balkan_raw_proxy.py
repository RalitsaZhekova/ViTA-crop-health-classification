from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import rasterio
from prithvi_payload.balkan_crop_calibration import sha256_file
from prithvi_payload.balkan_raw_proxy import build_raw_model_proxy, telemetry_grid
from pyproj import Transformer


def _telemetry(position: Path, attitude: Path) -> None:
    to_ecef = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)
    with position.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("px", "py", "pz"))
        writer.writeheader()
        for lon, lat, height in ((-113.0, 33.1, 500_000), (-113.1, 33.0, 500_000)):
            px, py, pz = to_ecef.transform(lon, lat, height)
            writer.writerow({"px": px, "py": py, "pz": pz})
    with attitude.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("estRpyRoll", "estRpyPitch", "estRpyYaw"),
        )
        writer.writeheader()
        writer.writerow({"estRpyRoll": 5.0, "estRpyPitch": 0.0, "estRpyYaw": 0.0})
        writer.writerow({"estRpyRoll": 5.0, "estRpyPitch": 0.0, "estRpyYaw": 0.0})


def test_telemetry_grid_is_rotated_and_projected(tmp_path: Path) -> None:
    position = tmp_path / "position.csv"
    attitude = tmp_path / "attitude.csv"
    _telemetry(position, attitude)
    grid = telemetry_grid(position, attitude, width=100, height=200, pixel_size_m=10)
    assert grid.crs.to_epsg() == 32612
    assert abs(grid.transform.b) > 0
    assert abs(grid.transform.d) > 0
    assert np.dot(grid.detector_column_unit, grid.left_cross_track_unit) < 0
    assert -114 < grid.center_lon < -112
    assert 32 < grid.center_lat < 34


def test_build_raw_model_proxy_marks_every_approximation(tmp_path: Path) -> None:
    source = tmp_path / "raw-aligned.tif"
    width = height = 64
    values = np.empty((5, height, width), dtype=np.uint16)
    for index in range(5):
        values[index] = np.uint16(600 + index * 100)
        values[index, :, 0] = np.uint16(100 + index)
        values[index, :, -1] = np.uint16(110 + index)
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=5,
        dtype="uint16",
    ) as destination:
        destination.write(values)
        for index, description in enumerate(
            ("BLUE", "GREEN", "RED", "NIR_BROAD", "PANCHROMATIC"),
            start=1,
        ):
            destination.set_band_description(index, description)

    source_hash = sha256_file(source)
    alignment = tmp_path / "raw-aligned.alignment.json"
    alignment.write_text(
        json.dumps(
            {
                "source": {"path": str(source), "sha256": source_hash},
                "output": {
                    "path": str(source),
                    "sha256": source_hash,
                    "radiometry_modified": False,
                },
                "registrations": {
                    band: {"aligned": True}
                    for band in (
                        "BLUE",
                        "GREEN",
                        "RED",
                        "NIR_BROAD",
                        "PANCHROMATIC",
                    )
                },
            }
        ),
        encoding="utf-8",
    )
    diagnostics = tmp_path / "diagnostics.json"
    diagnostics.write_text(
        json.dumps(
            {
                "radiometric_diagnostics": [
                    {
                        "band": band,
                        "diagnostic_affine_slope": 0.0001,
                        "diagnostic_affine_intercept": 0.02,
                    }
                    for band in ("BLUE", "GREEN", "RED", "NIR", "PAN")
                ]
            }
        ),
        encoding="utf-8",
    )
    parent_calibration = tmp_path / "parent.crop_calibration.json"
    parent_calibration.write_text(
        json.dumps({"acquired_at": "2026-06-16T12:00:00+00:00"}),
        encoding="utf-8",
    )
    position = tmp_path / "position.csv"
    attitude = tmp_path / "attitude.csv"
    _telemetry(position, attitude)
    output = tmp_path / "proxy.tif"

    report = build_raw_model_proxy(
        source,
        output,
        alignment_report_path=alignment,
        radiometric_diagnostics_path=diagnostics,
        position_path=position,
        attitude_path=attitude,
        parent_calibration_path=parent_calibration,
        raw_pixel_size_m=1.5,
        output_pixel_size_m=10.0,
        inactive_border_pixels=2,
        dark_reference_pixels=1,
    )

    assert report["status"] == "UNQUALIFIED_ENGINEERING_EXPERIMENT"
    assert report["radiometry"]["column_fixed_pattern_correction"] == "not applied"
    assert report["geolocation"]["terrain_model"] == "not applied"
    with rasterio.open(output) as result:
        assert result.crs.to_epsg() == 32612
        assert result.descriptions == (
            "BLUE",
            "GREEN",
            "RED",
            "NIR_BROAD",
            "PANCHROMATIC",
        )
        assert result.tags()["SCIENCE_QUALIFIED"] == "FALSE"
        assert result.tags()["DETECTOR_ORIENTATION"] == ("COLUMNS_RIGHT_OF_GROUND_TRACK")
        assert np.all(result.read(1) > 0)
    calibration = json.loads(
        output.with_name("proxy.crop_calibration.json").read_text(encoding="utf-8")
    )
    assert calibration["experimental_raw_proxy"]["status"] == ("UNQUALIFIED_ENGINEERING_EXPERIMENT")
    assert calibration["experimental_raw_proxy"]["crop_probability_threshold"] == 0.55
    assert calibration["experimental_raw_proxy"]["health_analysis_crop_threshold"] == 0.65

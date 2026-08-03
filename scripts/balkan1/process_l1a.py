"""Build a real, minimum Balkan-1 L1A product from delivered raw DN imagery."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import tempfile
import uuid
import warnings
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning
from rasterio.windows import Window

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "data" / "balkan1"
RAW_BAND_IDS = ("1", "2", "3", "7", "0")
BAND_NAMES = ("BLUE", "GREEN", "RED", "NIR", "PAN")
REFERENCE_BAND_INDEX = 2


def _csv_records(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream))
    return max(0, len(rows) - 1)


def _processing_evidence(log_path: Path) -> dict[str, object]:
    if not log_path.is_file():
        return {"log": str(log_path), "available": False}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    gipp = re.search(r"GIPP parsed: NPIX=(\d+) BPM=(\d+) BPC=(\d+)", text)
    offsets = re.search(r"band time offsets \[rows\]:\s*([^\)\r\n]+)", text)
    evidence: dict[str, object] = {"log": str(log_path), "available": True}
    if gipp:
        evidence.update(
            {
                "valid_detector_width": int(gipp.group(1)),
                "border_pixels": int(gipp.group(2)),
                "calibration_border_pixels": int(gipp.group(3)),
            }
        )
    if offsets:
        evidence["band_time_offsets_rows"] = offsets.group(1).strip()
    return evidence


def _load_scene_metadata(path: Path) -> dict[str, object]:
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read delivered scene metadata {path}: {error}") from error
    scenes = metadata.get("Scenes")
    if not isinstance(scenes, list) or len(scenes) != 1:
        raise ValueError(f"Expected exactly one scene in {path}")
    return metadata


def _extract_integer(text: str, pattern: str, label: str) -> int:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Extraction log is missing {label}")
    return int(match.group(1).replace(",", ""))


def _extract_integer_list(text: str, label: str) -> list[int]:
    match = re.search(rf"{label}\s*=\s*(\[[^\]]+\])", text)
    if not match:
        raise ValueError(f"Extraction log is missing {label}")
    values = ast.literal_eval(match.group(1))
    if not isinstance(values, list) or not all(isinstance(value, int) for value in values):
        raise ValueError(f"Extraction log has an invalid {label}")
    return values


def _metadata_from_extraction_log(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    width = _extract_integer(text, r"SceneWidth\s*=\s*([\d,]+)", "SceneWidth")
    height = _extract_integer(text, r"SceneHeight\s*=\s*([\d,]+)", "SceneHeight")
    line_tables: dict[str, list[list[int | None]]] = {band_id: [] for band_id in RAW_BAND_IDS}
    for match in re.finditer(
        r"LineData\s+SpectralBand\s*=\s*(\d+)\s+LineNumber\s*=\s*([\d,]+)", text
    ):
        band_id = match.group(1)
        if band_id in line_tables:
            line_tables[band_id].append(
                [int(match.group(2).replace(",", "")), None, None]
            )
    missing_tables = [band_id for band_id, lines in line_tables.items() if not lines]
    if missing_tables:
        raise ValueError(f"Extraction log has no line records for bands {missing_tables}")
    scene: dict[str, object] = {"Width": width, "Height": height}
    scene.update(line_tables)
    metadata: dict[str, object] = {
        "PlatformID": _extract_integer(text, r"PlatformID\s*=\s*(\d+)", "PlatformID"),
        "InstrumentID": _extract_integer(
            text, r"InstrumentID\s*=\s*(\d+)", "InstrumentID"
        ),
        "PacketVersion": _extract_integer_list(text, "PacketVersion"),
        "SessionClosed": bool(re.search(r"Closed\s*=\s*True|SessionEnd", text)),
        "ImagerConfiguration": {
            "LinePeriod": _extract_integer(
                text, r"LinePeriod\s*=\s*([\d,]+)", "LinePeriod"
            ),
            "SpectralBands": _extract_integer(
                text, r"SpectralBands\s*=\s*([\d,]+)", "SpectralBands"
            ),
            "ExposureTime": _extract_integer(
                text, r"ExposureTime\s*=\s*([\d,]+)", "ExposureTime"
            ),
            "BandSetup": _extract_integer_list(text, "BandSetup"),
            "BandStartRow": _extract_integer_list(text, "BandStartRow"),
            "BandCWL": _extract_integer_list(text, "BandCWL"),
        },
        "SensorConfiguration": {
            "PGAGain": _extract_integer(text, r"PGAGain\s*=\s*(-?\d+)", "PGAGain"),
            "ADCGain": _extract_integer(text, r"ADCGain\s*=\s*(-?\d+)", "ADCGain"),
            "DarkOffset": _extract_integer(
                text, r"DarkOffset\s*=\s*(-?\d+)", "DarkOffset"
            ),
        },
        "Scenes": [scene],
        "_metadata_source": "packet_extraction_log",
    }
    return metadata


def _validate_l0r(
    scene_id: str,
    scene_dir: Path,
    raw_path: Path,
    metadata_path: Path,
    processing_log: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    if metadata_path.suffix.lower() == ".json":
        metadata = _load_scene_metadata(metadata_path)
        metadata_source = "decoded_json"
    else:
        metadata = _metadata_from_extraction_log(metadata_path)
        metadata_source = "packet_extraction_log"
    scene = metadata["Scenes"][0]
    errors: list[str] = []
    expected_width = int(scene.get("Width", 0))
    expected_height = int(scene.get("Height", 0))
    line_diagnostics: dict[str, object] = {}
    total_missing_lines = 0
    for band_id in RAW_BAND_IDS:
        lines = scene.get(band_id)
        if not isinstance(lines, list):
            errors.append(f"Scene metadata has no line table for band {band_id}")
            continue
        line_numbers = {int(line[0]) for line in lines if isinstance(line, list) and line}
        missing = sorted(set(range(expected_height)) - line_numbers)
        total_missing_lines += len(missing)
        line_diagnostics[band_id] = {
            "records": len(lines),
            "missing_line_count": len(missing),
            "missing_line_sample": missing[:20],
            "first_imager_timestamp": lines[0][1] if lines else None,
            "last_imager_timestamp": lines[-1][1] if lines else None,
        }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(raw_path) as source:
            raw_tiff = {
                "path": str(raw_path),
                "bytes": raw_path.stat().st_size,
                "width": source.width,
                "height": source.height,
                "bands": source.count,
                "dtype": source.dtypes[0],
                "descriptions": list(source.descriptions),
                "crs": str(source.crs) if source.crs else None,
            }
            if source.width != expected_width or source.height != expected_height:
                errors.append(
                    "Raw TIFF dimensions do not match the delivered JSON: "
                    f"{source.width}x{source.height} != {expected_width}x{expected_height}"
                )
            if source.count != len(RAW_BAND_IDS):
                errors.append(f"Raw TIFF has {source.count} bands; expected {len(RAW_BAND_IDS)}")
            expected_descriptions = [f"Band_{band_id}" for band_id in RAW_BAND_IDS]
            if list(source.descriptions) != expected_descriptions:
                errors.append(
                    f"Raw TIFF descriptions {source.descriptions} do not match "
                    f"{expected_descriptions}"
                )

    binary_path = scene_dir / f"{scene_id}.bin"
    warnings_list: list[str] = []
    if total_missing_lines:
        warnings_list.append(
            f"Delivered reconstruction has {total_missing_lines} missing band lines in total; "
            "minimum L1A preserves affected pixels as nodata"
        )
    files = {
        "packet_stream": {
            "path": str(binary_path),
            "exists": binary_path.is_file(),
            "bytes": binary_path.stat().st_size if binary_path.is_file() else None,
        },
        "metadata": {"path": str(metadata_path), "bytes": metadata_path.stat().st_size},
        "raw_tiff": raw_tiff,
        "position": {
            "path": str(scene_dir / "position.csv"),
            "records": _csv_records(scene_dir / "position.csv"),
        },
        "attitude": {
            "path": str(scene_dir / "attitude.csv"),
            "records": _csv_records(scene_dir / "attitude.csv"),
        },
    }
    if not binary_path.is_file():
        warnings_list.append(
            "Original packet stream is absent; L0R validation starts from the delivered raw TIFF "
            "and packet-extraction log"
        )
    if metadata.get("SessionClosed") is not True:
        errors.append("Delivered metadata does not mark the session closed")

    if errors:
        status = "INVALID"
    elif binary_path.is_file() and metadata_source == "decoded_json":
        status = "VALIDATED"
    else:
        status = "VALIDATED_PARTIAL_PROVENANCE"
    l0r_manifest = {
        "schema_version": 1,
        "scene_id": scene_id,
        "processing_level": "L0R_DELIVERED_RECONSTRUCTED_DN",
        "status": status,
        "metadata_source": metadata_source,
        "files": files,
        "session": {
            "closed": metadata.get("SessionClosed"),
            "platform_id": metadata.get("PlatformID"),
            "instrument_id": metadata.get("InstrumentID"),
            "packet_version": metadata.get("PacketVersion"),
        },
        "imager_configuration": metadata.get("ImagerConfiguration"),
        "sensor_configuration": metadata.get("SensorConfiguration"),
        "line_diagnostics": line_diagnostics,
        "total_missing_band_lines": total_missing_lines,
        "production_log_evidence": _processing_evidence(processing_log),
        "errors": errors,
        "warnings": warnings_list,
        "boundary": (
            "The proprietary packet stream is preserved as L0 provenance. The delivered Raw TIFF "
            "is the reconstructed DN raster used as the processing input; this repository does not "
            "claim to re-decode the proprietary packet format without its ICD/SDK."
        ),
    }
    return l0r_manifest, metadata


def _display_u8(image: np.ndarray) -> np.ndarray:
    valid = np.isfinite(image) & (image > 0)
    if not np.any(valid):
        return np.zeros(image.shape, dtype=np.uint8)
    low, high = np.percentile(image[valid], (1, 99))
    if high <= low:
        high = low + 1
    scaled = np.clip((image - low) / (high - low), 0, 1)
    return np.round(scaled * 255).astype(np.uint8)


def _registration_preview(
    source: rasterio.DatasetReader,
    border: int,
    downsample: int,
) -> tuple[np.ndarray, float, float]:
    valid_width = source.width - 2 * border
    preview_height = max(512, source.height // downsample)
    preview_width = max(512, valid_width // downsample)
    preview = source.read(
        window=Window(border, 0, valid_width, source.height),
        out_shape=(source.count, preview_height, preview_width),
        out_dtype="float32",
        resampling=Resampling.average,
    )
    return preview, valid_width / preview_width, source.height / preview_height


def _estimate_registration(
    preview: np.ndarray,
    scale_x: float,
    scale_y: float,
) -> tuple[list[np.ndarray], list[dict[str, object]]]:
    prepared = [cv2.createCLAHE(2.0, (8, 8)).apply(_display_u8(band)) for band in preview]
    sift = cv2.SIFT_create(nfeatures=12_000, contrastThreshold=0.02)
    reference_keypoints, reference_descriptors = sift.detectAndCompute(
        prepared[REFERENCE_BAND_INDEX], None
    )
    if reference_descriptors is None or len(reference_keypoints) < 500:
        raise RuntimeError("Too few real features in the Red-band registration preview")
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matrices: list[np.ndarray] = []
    diagnostics: list[dict[str, object]] = []
    for band_index, (band_name, image) in enumerate(zip(BAND_NAMES, prepared, strict=True)):
        if band_index == REFERENCE_BAND_INDEX:
            matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
            matrices.append(matrix)
            diagnostics.append(
                {
                    "band": band_name,
                    "reference": True,
                    "keypoints": len(reference_keypoints),
                    "good_matches": len(reference_keypoints),
                    "inliers": len(reference_keypoints),
                    "inlier_ratio": 1.0,
                    "rmse_pixels": 0.0,
                    "matrix_source_to_red": matrix.tolist(),
                }
            )
            continue

        keypoints, descriptors = sift.detectAndCompute(image, None)
        if descriptors is None:
            raise RuntimeError(f"No descriptors found for real band {band_name}")
        pairs = matcher.knnMatch(descriptors, reference_descriptors, k=2)
        good = [first for first, second in pairs if first.distance < 0.72 * second.distance]
        if len(good) < 100:
            raise RuntimeError(f"Only {len(good)} cross-band matches found for {band_name}")
        source_points = np.float32([keypoints[match.queryIdx].pt for match in good])
        target_points = np.float32(
            [reference_keypoints[match.trainIdx].pt for match in good]
        )
        preview_matrix, inlier_mask = cv2.estimateAffinePartial2D(
            source_points,
            target_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=3,
            maxIters=10_000,
            confidence=0.999,
            refineIters=50,
        )
        if preview_matrix is None or inlier_mask is None:
            raise RuntimeError(f"RANSAC registration failed for {band_name}")
        inliers = inlier_mask.ravel().astype(bool)
        inlier_count = int(np.count_nonzero(inliers))
        inlier_ratio = inlier_count / len(good)
        if inlier_count < 100 or inlier_ratio < 0.5:
            raise RuntimeError(
                f"Registration rejected for {band_name}: {inlier_count} inliers, "
                f"ratio {inlier_ratio:.3f}"
            )

        scale = np.diag([scale_x, scale_y])
        linear = scale @ preview_matrix[:, :2] @ np.linalg.inv(scale)
        translation = scale @ preview_matrix[:, 2]
        matrix = np.column_stack([linear, translation]).astype(np.float64)
        predicted = cv2.transform(source_points[inliers, None, :], preview_matrix)[:, 0, :]
        residual = target_points[inliers] - predicted
        residual[:, 0] *= scale_x
        residual[:, 1] *= scale_y
        rmse = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
        matrices.append(matrix)
        diagnostics.append(
            {
                "band": band_name,
                "reference": False,
                "keypoints": len(keypoints),
                "good_matches": len(good),
                "inliers": inlier_count,
                "inlier_ratio": inlier_ratio,
                "rmse_pixels": rmse,
                "matrix_source_to_red": matrix.tolist(),
            }
        )
    return matrices, diagnostics


def _column_corrections(
    source: rasterio.DatasetReader,
    border: int,
    black_level: float,
    sample_rows: int = 512,
) -> tuple[list[np.ndarray], list[dict[str, float]]]:
    valid_width = source.width - 2 * border
    sampled = source.read(
        window=Window(border, 0, valid_width, source.height),
        out_shape=(source.count, sample_rows, valid_width),
        out_dtype="float32",
        resampling=Resampling.average,
    )
    corrections: list[np.ndarray] = []
    diagnostics: list[dict[str, float]] = []
    for band in sampled:
        corrected = np.maximum(band - black_level, 0)
        profile = np.median(corrected, axis=0)
        smooth = cv2.GaussianBlur(
            profile.reshape(1, -1), (0, 0), sigmaX=64, sigmaY=0
        ).reshape(-1)
        high_frequency = profile - smooth
        limit = float(np.percentile(np.abs(high_frequency), 99.5))
        high_frequency = np.clip(high_frequency, -limit, limit).astype(np.float32)
        corrections.append(high_frequency)
        diagnostics.append(
            {
                "median_abs_correction_dn": float(np.median(np.abs(high_frequency))),
                "max_abs_correction_dn": float(np.max(np.abs(high_frequency))),
            }
        )
    return corrections, diagnostics


def _remap_strip(
    source: rasterio.DatasetReader,
    band_number: int,
    inverse_matrix: np.ndarray,
    border: int,
    output_width: int,
    output_height: int,
    y_start: int,
    strip_height: int,
    black_level: float,
    column_correction: np.ndarray,
) -> np.ndarray:
    y_stop = min(output_height, y_start + strip_height)
    x_coordinates = np.arange(output_width, dtype=np.float32)[None, :]
    y_coordinates = np.arange(y_start, y_stop, dtype=np.float32)[:, None]
    map_x = (
        inverse_matrix[0, 0] * x_coordinates
        + inverse_matrix[0, 1] * y_coordinates
        + inverse_matrix[0, 2]
    )
    map_y = (
        inverse_matrix[1, 0] * x_coordinates
        + inverse_matrix[1, 1] * y_coordinates
        + inverse_matrix[1, 2]
    )
    x_min = max(0, int(np.floor(float(np.min(map_x)))) - 2)
    x_max = min(output_width, int(np.ceil(float(np.max(map_x)))) + 3)
    y_min = max(0, int(np.floor(float(np.min(map_y)))) - 2)
    y_max = min(output_height, int(np.ceil(float(np.max(map_y)))) + 3)
    if x_min >= x_max or y_min >= y_max:
        return np.zeros((y_stop - y_start, output_width), dtype=np.float32)
    source_band = source.read(
        band_number,
        window=Window(border + x_min, y_min, x_max - x_min, y_max - y_min),
        out_dtype="float32",
    )
    source_valid = (source_band > 0).astype(np.uint8)
    source_band -= black_level
    source_band -= column_correction[x_min:x_max][None, :]
    np.maximum(source_band, 0, out=source_band)
    remapped = cv2.remap(
        source_band,
        (map_x - x_min).astype(np.float32),
        (map_y - y_min).astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid = cv2.remap(
        source_valid,
        (map_x - x_min).astype(np.float32),
        (map_y - y_min).astype(np.float32),
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    remapped[valid == 0] = 0
    return remapped


def _write_l1a(
    source: rasterio.DatasetReader,
    output: Path,
    border: int,
    black_level: float,
    matrices: list[np.ndarray],
    column_corrections: list[np.ndarray],
    strip_height: int,
    overwrite: bool,
) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{uuid.uuid4().hex}.partial.tif")
    valid_width = source.width - 2 * border
    profile = {
        "driver": "GTiff",
        "width": valid_width,
        "height": source.height,
        "count": source.count,
        "dtype": "float32",
        "nodata": 0.0,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": "deflate",
        "predictor": 3,
        "BIGTIFF": "YES",
    }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(temporary, "w", **profile) as destination:
                destination.update_tags(
                    PROCESSING_LEVEL="L1A_MINIMUM_SENSOR_SPACE",
                    RADIOMETRIC_UNITS="scene-destriped relative DN",
                    RADIOMETRIC_STATUS="NOT_RADIANCE_NOT_REFLECTANCE",
                    GEOLOCATION_STATUS="UNREFERENCED_NOT_ORTHORECTIFIED",
                    REGISTRATION_METHOD="SIFT cross-band features + RANSAC partial affine",
                    REFERENCE_BAND="RED",
                    BLACK_LEVEL_DN=str(black_level),
                )
                for band_index, band_name in enumerate(BAND_NAMES):
                    inverse = cv2.invertAffineTransform(matrices[band_index])
                    for y_start in range(0, source.height, strip_height):
                        strip = _remap_strip(
                            source,
                            band_index + 1,
                            inverse,
                            border,
                            valid_width,
                            source.height,
                            y_start,
                            strip_height,
                            black_level,
                            column_corrections[band_index],
                        )
                        destination.write(
                            strip,
                            band_index + 1,
                            window=Window(0, y_start, valid_width, strip.shape[0]),
                        )
                    destination.set_band_description(band_index + 1, band_name)
                    destination.update_tags(
                        band_index + 1,
                        SOURCE_BAND=f"Band_{RAW_BAND_IDS[band_index]}",
                        REGISTRATION_MATRIX=json.dumps(matrices[band_index].tolist()),
                    )
                    print(f"L1A band complete: {band_name}", flush=True)
                overview_levels = [
                    level for level in (2, 4, 8, 16, 32) if valid_width // level >= 256
                ]
                destination.build_overviews(overview_levels, Resampling.average)
                destination.update_tags(ns="rio_overview", resampling="average")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def _stretch(image: np.ndarray, gamma: float) -> np.ndarray:
    channels = []
    for band in image:
        linear = _display_u8(band).astype(np.float32) / 255
        channels.append(linear**gamma)
    return np.stack(channels, axis=-1)


def _read_rgb_overview(path: Path, gamma: float, size: int = 1200) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as source:
            scale = min(size / source.width, size / source.height, 1)
            width = max(1, round(source.width * scale))
            height = max(1, round(source.height * scale))
            image = source.read(
                [3, 2, 1],
                out_shape=(3, height, width),
                out_dtype="float32",
                resampling=Resampling.average,
            )
    return _stretch(image, gamma)


def _write_preview(
    raw_path: Path,
    l1a_path: Path,
    reference_path: Path,
    output: Path,
    gamma: float,
) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "prithvi_payload_matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    panels = [
        (
            _read_rgb_overview(raw_path, gamma),
            "Delivered reconstructed raw DN\nunaligned RGB display",
        ),
        (
            _read_rgb_overview(l1a_path, gamma),
            "New minimum L1A\nreal feature-registered corrected DN",
        ),
    ]
    if reference_path.is_file():
        panels.append(
            (
                _read_rgb_overview(reference_path, gamma),
                "Supplied L1ORT reference\ncalibrated + orthorectified externally",
            )
        )
    figure, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 8))
    if len(panels) == 1:
        axes = [axes]
    for axis, (image, title) in zip(axes, panels, strict=True):
        axis.imshow(image)
        axis.set_title(title, fontsize=14)
        axis.axis("off")
    figure.suptitle(
        f"Balkan-1 real preprocessing comparison — 1–99% stretch, gamma {gamma:g}",
        fontsize=18,
        weight="bold",
    )
    figure.text(
        0.5,
        0.02,
        (
            "The new L1A is a real sensor-space product, not mock data. It is intentionally not "
            "labelled radiance, reflectance, georeferenced or orthorectified."
        ),
        ha="center",
        fontsize=11,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _sample_statistics(path: Path) -> list[dict[str, float]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as source:
            sampled = source.read(
                out_shape=(source.count, min(1000, source.height), min(1000, source.width)),
                out_dtype="float32",
                resampling=Resampling.average,
            )
    statistics = []
    for band in sampled:
        valid = band[np.isfinite(band) & (band > 0)]
        statistics.append(
            {
                "p02": float(np.percentile(valid, 2)),
                "p50": float(np.percentile(valid, 50)),
                "p98": float(np.percentile(valid, 98)),
            }
        )
    return statistics


def _post_write_registration(path: Path, downsample: int) -> dict[str, object]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as source:
            preview, scale_x, scale_y = _registration_preview(source, 0, downsample)
    _, diagnostics = _estimate_registration(preview, scale_x, scale_y)
    residual_translations = [
        max(
            abs(float(item["matrix_source_to_red"][0][2])),
            abs(float(item["matrix_source_to_red"][1][2])),
        )
        for item in diagnostics
        if not item["reference"]
    ]
    passed = bool(residual_translations) and max(residual_translations) <= 10
    return {
        "status": "PASS" if passed else "REVIEW_REQUIRED",
        "maximum_residual_translation_pixels": max(residual_translations),
        "acceptance_threshold_pixels": 10,
        "bands": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate delivered Balkan-1 L0 reconstruction and build a real minimum L1A "
            "sensor-space product."
        )
    )
    parser.add_argument("scene_id")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--registration-downsample", type=int, default=8)
    parser.add_argument("--strip-height", type=int, default=512)
    parser.add_argument(
        "--black-level-dn",
        type=float,
        help=(
            "Explicit calibrated black level to subtract. Omit unless a calibration source "
            "defines it; SensorConfiguration.DarkOffset is not assumed to be a black level."
        ),
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--preview-gamma", type=float, default=0.65)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.registration_downsample < 2:
        parser.error("--registration-downsample must be at least 2")
    if args.strip_height < 64:
        parser.error("--strip-height must be at least 64")
    if args.preview_gamma <= 0:
        parser.error("--preview-gamma must be greater than zero")
    if args.validate_only and args.preview_only:
        parser.error("--validate-only and --preview-only cannot be combined")

    data_root = args.data_root.resolve()
    scene_dir = data_root / "raw" / args.scene_id
    raw_path = scene_dir / f"{args.scene_id}_Raw.tif"
    json_metadata_path = scene_dir / f"{args.scene_id}.json"
    extraction_logs = sorted(scene_dir.glob("log_extract*.txt"))
    metadata_path = (
        json_metadata_path
        if json_metadata_path.is_file()
        else extraction_logs[0]
        if extraction_logs
        else json_metadata_path
    )
    processing_log = data_root / "preprocessed" / f"log_{args.scene_id}.txt"
    output = (
        args.output
        or data_root / "derived" / "l1a" / f"{args.scene_id}_L1A_MIN.tif"
    ).resolve()
    for required in (scene_dir, raw_path, metadata_path):
        if not required.exists():
            parser.error(f"required input does not exist: {required}")

    l0r_manifest, metadata = _validate_l0r(
        args.scene_id, scene_dir, raw_path, metadata_path, processing_log
    )
    manifest_dir = output.parent
    manifest_dir.mkdir(parents=True, exist_ok=True)
    l0r_manifest_path = manifest_dir / f"{args.scene_id}_L0R_manifest.json"
    l0r_manifest_path.write_text(json.dumps(l0r_manifest, indent=2), encoding="utf-8")
    if not str(l0r_manifest["status"]).startswith("VALIDATED"):
        raise RuntimeError(f"L0R validation failed; see {l0r_manifest_path}")
    if args.validate_only:
        print(json.dumps({"l0r_manifest": str(l0r_manifest_path)}, indent=2))
        return
    reference_path = data_root / "preprocessed" / f"{args.scene_id}_L1ORT.tif"
    preview_path = output.with_name(f"{args.scene_id}_L1A_comparison.png")
    if args.preview_only:
        if not output.is_file():
            parser.error(f"L1A product does not exist for preview: {output}")
        _write_preview(
            raw_path,
            output,
            reference_path,
            preview_path,
            args.preview_gamma,
        )
        print(json.dumps({"comparison_preview": str(preview_path)}, indent=2))
        return

    evidence = l0r_manifest["production_log_evidence"]
    border = evidence.get("border_pixels")
    valid_width = evidence.get("valid_detector_width")
    if not isinstance(border, int) or not isinstance(valid_width, int):
        raise RuntimeError(
            "Trusted log evidence for detector border/valid width is unavailable; refusing to guess"
        )
    if valid_width + 2 * border != int(metadata["Scenes"][0]["Width"]):
        raise RuntimeError("Trusted detector border evidence does not match the raw scene width")
    sensor_configuration = metadata.get("SensorConfiguration", {})
    dark_offset = float(sensor_configuration.get("DarkOffset", 0))
    black_level = args.black_level_dn if args.black_level_dn is not None else 0.0
    black_level_source = (
        "explicit --black-level-dn calibration input"
        if args.black_level_dn is not None
        else "not applied; delivered DarkOffset is retained as hardware configuration metadata"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(raw_path) as source:
            preview, scale_x, scale_y = _registration_preview(
                source, border, args.registration_downsample
            )
            matrices, registration = _estimate_registration(preview, scale_x, scale_y)
            corrections, correction_diagnostics = _column_corrections(
                source, border, black_level
            )
            _write_l1a(
                source,
                output,
                border,
                black_level,
                matrices,
                corrections,
                args.strip_height,
                args.overwrite,
            )

    _write_preview(
        raw_path,
        output,
        reference_path,
        preview_path,
        args.preview_gamma,
    )
    post_write_registration = _post_write_registration(
        output, args.registration_downsample
    )
    l1a_manifest = {
        "schema_version": 1,
        "scene_id": args.scene_id,
        "processing_level": "L1A_MINIMUM_SENSOR_SPACE",
        "source_l0r_manifest": str(l0r_manifest_path),
        "output": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "width": valid_width,
            "height": int(metadata["Scenes"][0]["Height"]),
            "bands": list(BAND_NAMES),
            "dtype": "float32",
            "units": "scene-destriped relative DN",
            "crs": None,
            "statistics": _sample_statistics(output),
        },
        "radiometric_processing": {
            "black_level_dn_applied": black_level,
            "black_level_source": black_level_source,
            "delivered_sensor_dark_offset_setting": dark_offset,
            "inactive_detector_border_pixels_removed_each_side": border,
            "column_fixed_pattern_correction": "scene-estimated high-frequency median profile",
            "column_correction_diagnostics": correction_diagnostics,
            "absolute_calibration_applied": False,
        },
        "registration": {
            "method": "SIFT cross-band features + RANSAC partial affine",
            "reference_band": "RED",
            "preview_downsample": args.registration_downsample,
            "bands": registration,
            "post_write_validation": post_write_registration,
        },
        "comparison_preview": str(preview_path),
        "comparison_preview_tone_mapping": {
            "percentiles": [1, 99],
            "gamma": args.preview_gamma,
            "display_only": True,
        },
        "scientific_boundaries": [
            "Output is corrected relative DN, not radiance or reflectance",
            "Output is sensor-space and has no CRS",
            "No synthetic orbit, mock calibration or random coefficients were used",
            "L1B/L1C/L1ORT requires the missing GIPP calibration, camera model, "
            "Orekit, DEM and registration-reference assets",
            "Do not use this minimum L1A as cloud/crop model input",
        ],
    }
    l1a_manifest_path = output.with_suffix(".json")
    l1a_manifest_path.write_text(json.dumps(l1a_manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "l0r_manifest": str(l0r_manifest_path),
                "l1a_product": str(output),
                "l1a_manifest": str(l1a_manifest_path),
                "comparison_preview": str(preview_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

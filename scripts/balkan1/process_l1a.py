"""Build a real, minimum Balkan-1 L1A product from delivered raw DN imagery."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import tempfile
import time
import uuid
import warnings
from dataclasses import dataclass
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


@dataclass(frozen=True)
class DarkReferenceModel:
    """Per-line detector bias measured from the delivered calibration columns."""

    left_dn: np.ndarray
    right_dn: np.ndarray
    left_detector_x: float
    right_detector_x: float
    source: str
    diagnostics: list[dict[str, float | str]]


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
    exposure_timestamp: int | None = None
    for line in text.splitlines():
        exposure = re.search(r"ExposureStart\s+Timestamp\s*=\s*([\d,]+)", line)
        if exposure:
            exposure_timestamp = int(exposure.group(1).replace(",", ""))
        line_data = re.search(
            r"LineData\s+SpectralBand\s*=\s*(\d+)\s+LineNumber\s*=\s*([\d,]+)",
            line,
        )
        if line_data and line_data.group(1) in line_tables:
            line_tables[line_data.group(1)].append(
                [
                    int(line_data.group(2).replace(",", "")),
                    exposure_timestamp,
                    exposure_timestamp,
                ]
            )
    missing_tables = [band_id for band_id, lines in line_tables.items() if not lines]
    if missing_tables:
        raise ValueError(f"Extraction log has no line records for bands {missing_tables}")
    scene: dict[str, object] = {"Width": width, "Height": height}
    scene.update(line_tables)
    time_sync = [
        {"ImagerTime": int(match.group(1)), "PPS": True}
        for match in re.finditer(r"\{'ImagerTime':\s*(\d+),\s*'PPS':\s*True\}", text)
    ]
    utc_anchors = [
        {
            "LastExposureTimestamp": int(match.group(1)),
            "Data": match.group(2),
        }
        for match in re.finditer(
            r"\{'ExposureTimestamp':\s*(\d+),\s*'Data':\s*b?'([0-9.]+)'\}",
            text,
        )
    ]
    metadata: dict[str, object] = {
        "PlatformID": _extract_integer(text, r"PlatformID\s*=\s*(\d+)", "PlatformID"),
        "InstrumentID": _extract_integer(text, r"InstrumentID\s*=\s*(\d+)", "InstrumentID"),
        "PacketVersion": _extract_integer_list(text, "PacketVersion"),
        "SessionClosed": bool(re.search(r"Closed\s*=\s*True|SessionEnd", text)),
        "ImagerConfiguration": {
            "LinePeriod": _extract_integer(text, r"LinePeriod\s*=\s*([\d,]+)", "LinePeriod"),
            "SpectralBands": _extract_integer(
                text, r"SpectralBands\s*=\s*([\d,]+)", "SpectralBands"
            ),
            "ExposureTime": _extract_integer(text, r"ExposureTime\s*=\s*([\d,]+)", "ExposureTime"),
            "BandSetup": _extract_integer_list(text, "BandSetup"),
            "BandStartRow": _extract_integer_list(text, "BandStartRow"),
            "BandCWL": _extract_integer_list(text, "BandCWL"),
        },
        "SensorConfiguration": {
            "PGAGain": _extract_integer(text, r"PGAGain\s*=\s*(-?\d+)", "PGAGain"),
            "ADCGain": _extract_integer(text, r"ADCGain\s*=\s*(-?\d+)", "ADCGain"),
            "DarkOffset": _extract_integer(text, r"DarkOffset\s*=\s*(-?\d+)", "DarkOffset"),
        },
        "Scenes": [scene],
        "Timesync": time_sync,
        "UserData": {"5": utc_anchors},
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
        "timing": {
            "pps_samples": len(metadata.get("Timesync", [])),
            "utc_anchor_samples": len(metadata.get("UserData", {}).get("5", [])),
            "source": metadata_source,
        },
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


def _phase_feature(image: np.ndarray) -> np.ndarray:
    return np.abs(cv2.Laplacian(image.astype(np.float32), cv2.CV_32F, ksize=3))


def _phase_shift(
    moving: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, float]:
    window = cv2.createHanningWindow((moving.shape[1], moving.shape[0]), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(
        _phase_feature(moving),
        _phase_feature(target),
        window,
    )
    return np.asarray(shift, dtype=np.float64), float(response)


def _full_resolution_matrix(
    preview_matrix: np.ndarray,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    scale = np.diag([scale_x, scale_y])
    linear = scale @ preview_matrix[:, :2] @ np.linalg.inv(scale)
    translation = scale @ preview_matrix[:, 2]
    return np.column_stack([linear, translation]).astype(np.float64)


def _phase_bridge_registration(
    image: np.ndarray,
    bridge_image: np.ndarray,
    bridge_to_reference: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    bridge_shift, bridge_response = _phase_shift(image, bridge_image)
    if bridge_response < 0.12 or np.max(np.abs(bridge_shift)) > 16:
        raise RuntimeError(
            "Phase-correlation bridge rejected: "
            f"response={bridge_response:.3f}, shift={bridge_shift.tolist()}"
        )
    source_to_bridge = np.array(
        [[1, 0, bridge_shift[0]], [0, 1, bridge_shift[1]], [0, 0, 1]],
        dtype=np.float64,
    )
    preview_matrix = (np.vstack([bridge_to_reference, [0, 0, 1]]) @ source_to_bridge)[:2]
    warped = cv2.warpAffine(
        image,
        preview_matrix,
        (reference.shape[1], reference.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    residual_shift, residual_response = _phase_shift(warped, reference)
    residual_applied = bool(residual_response >= 0.05 and np.max(np.abs(residual_shift)) <= 4)
    if residual_applied:
        preview_matrix[:, 2] += residual_shift
    return preview_matrix, {
        "phase_bridge_shift_preview_pixels": bridge_shift.tolist(),
        "phase_bridge_response": bridge_response,
        "phase_residual_shift_preview_pixels": residual_shift.tolist(),
        "phase_residual_response": residual_response,
        "phase_residual_applied": residual_applied,
    }


def _estimate_registration(
    preview: np.ndarray,
    scale_x: float,
    scale_y: float,
) -> tuple[list[np.ndarray], list[dict[str, object]]]:
    prepared = [cv2.createCLAHE(2.0, (8, 8)).apply(_display_u8(band)) for band in preview]
    sift = cv2.SIFT_create(nfeatures=12_000, contrastThreshold=0.02)
    reference = prepared[REFERENCE_BAND_INDEX]
    reference_keypoints, reference_descriptors = sift.detectAndCompute(reference, None)
    if reference_descriptors is None or len(reference_keypoints) < 500:
        raise RuntimeError("Too few real features in the Red-band registration preview")
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matrices: list[np.ndarray | None] = [None] * len(BAND_NAMES)
    preview_matrices: list[np.ndarray | None] = [None] * len(BAND_NAMES)
    diagnostics: list[dict[str, object] | None] = [None] * len(BAND_NAMES)
    identity = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
    matrices[REFERENCE_BAND_INDEX] = identity
    preview_matrices[REFERENCE_BAND_INDEX] = identity.copy()
    diagnostics[REFERENCE_BAND_INDEX] = {
        "band": BAND_NAMES[REFERENCE_BAND_INDEX],
        "reference": True,
        "registration_source": "reference",
        "keypoints": len(reference_keypoints),
        "good_matches": len(reference_keypoints),
        "inliers": len(reference_keypoints),
        "inlier_ratio": 1.0,
        "rmse_pixels": 0.0,
        "matrix_source_to_red": identity.tolist(),
    }

    # PAN is solved before NIR so it can provide a spectrally closer phase bridge
    # when vegetation or haze leaves NIR with too few direct Red-band features.
    for band_index in (0, 1, 4, 3):
        band_name = BAND_NAMES[band_index]
        image = prepared[band_index]
        keypoints, descriptors = sift.detectAndCompute(image, None)
        if descriptors is None:
            raise RuntimeError(f"No descriptors found for real band {band_name}")
        pairs = matcher.knnMatch(descriptors, reference_descriptors, k=2)
        good = [first for first, second in pairs if first.distance < 0.72 * second.distance]
        source_points = np.float32([keypoints[match.queryIdx].pt for match in good])
        target_points = np.float32([reference_keypoints[match.trainIdx].pt for match in good])

        if len(good) < 100:
            pan_preview_matrix = preview_matrices[4]
            if band_index != 3 or pan_preview_matrix is None:
                raise RuntimeError(f"Only {len(good)} cross-band matches found for {band_name}")
            preview_matrix, phase_bridge = _phase_bridge_registration(
                image,
                prepared[4],
                pan_preview_matrix,
                reference,
            )
            if len(good) < 4:
                raise RuntimeError(
                    f"Only {len(good)} independent features support the {band_name} phase bridge"
                )
            weak_matrix, weak_mask = cv2.estimateAffinePartial2D(
                source_points,
                target_points,
                method=cv2.RANSAC,
                ransacReprojThreshold=3,
                maxIters=20_000,
                confidence=0.999,
                refineIters=50,
            )
            phase_corroborated_sparse_fit = False
            maximum_model_disagreement = None
            if weak_matrix is not None and weak_mask is not None:
                weak_inliers = weak_mask.ravel().astype(bool)
                control_points = np.float32(
                    [
                        [0, 0],
                        [image.shape[1] - 1, 0],
                        [0, image.shape[0] - 1],
                        [image.shape[1] - 1, image.shape[0] - 1],
                        [(image.shape[1] - 1) / 2, (image.shape[0] - 1) / 2],
                    ]
                )
                weak_control = cv2.transform(control_points[:, None, :], weak_matrix)[:, 0, :]
                phase_control = cv2.transform(control_points[:, None, :], preview_matrix)[:, 0, :]
                maximum_model_disagreement = float(
                    np.max(np.linalg.norm(weak_control - phase_control, axis=1))
                )
                phase_corroborated_sparse_fit = bool(
                    np.count_nonzero(weak_inliers) >= 4
                    and np.count_nonzero(weak_inliers) / len(good) >= 0.5
                    and maximum_model_disagreement <= 8
                )
                if phase_corroborated_sparse_fit:
                    preview_matrix = weak_matrix
            predicted = cv2.transform(source_points[:, None, :], preview_matrix)[:, 0, :]
            preview_residual = target_points - predicted
            inliers = np.linalg.norm(preview_residual, axis=1) <= 5
            inlier_count = int(np.count_nonzero(inliers))
            if inlier_count < 4:
                raise RuntimeError(
                    f"Only {inlier_count} independent features validate the {band_name} "
                    "phase bridge"
                )
            matrix = _full_resolution_matrix(preview_matrix, scale_x, scale_y)
            residual = preview_residual[inliers].copy()
            residual[:, 0] *= scale_x
            residual[:, 1] *= scale_y
            rmse = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
            matrices[band_index] = matrix
            preview_matrices[band_index] = preview_matrix
            diagnostics[band_index] = {
                "band": band_name,
                "reference": False,
                "registration_source": (
                    "sparse_sift_ransac_corroborated_by_phase"
                    if phase_corroborated_sparse_fit
                    else "phase_correlation_via_pan"
                ),
                "keypoints": len(keypoints),
                "good_matches": len(good),
                "inliers": inlier_count,
                "inlier_ratio": inlier_count / len(good),
                "rmse_pixels": rmse,
                "phase_corroborated_sparse_fit": phase_corroborated_sparse_fit,
                "maximum_sparse_phase_model_disagreement_preview_pixels": (
                    maximum_model_disagreement
                ),
                **phase_bridge,
                "matrix_source_to_red": matrix.tolist(),
            }
            continue

        source_points = np.float32([keypoints[match.queryIdx].pt for match in good])
        target_points = np.float32([reference_keypoints[match.trainIdx].pt for match in good])
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

        warped = cv2.warpAffine(
            image,
            preview_matrix,
            (reference.shape[1], reference.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        phase_shift_array, phase_response = _phase_shift(warped, reference)
        predicted_before_phase = cv2.transform(source_points[inliers, None, :], preview_matrix)[
            :, 0, :
        ]
        residual_before_phase = target_points[inliers] - predicted_before_phase
        preview_rmse_before_phase = float(
            np.sqrt(np.mean(np.sum(residual_before_phase**2, axis=1)))
        )
        candidate_residual = residual_before_phase - phase_shift_array
        preview_rmse_after_phase = float(np.sqrt(np.mean(np.sum(candidate_residual**2, axis=1))))
        phase_applied = bool(
            phase_response >= 0.05
            and np.max(np.abs(phase_shift_array)) <= 8
            and preview_rmse_after_phase < preview_rmse_before_phase
        )
        if phase_applied:
            preview_matrix[:, 2] += phase_shift_array

        matrix = _full_resolution_matrix(preview_matrix, scale_x, scale_y)
        predicted = cv2.transform(source_points[inliers, None, :], preview_matrix)[:, 0, :]
        residual = target_points[inliers] - predicted
        residual[:, 0] *= scale_x
        residual[:, 1] *= scale_y
        rmse = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
        matrices[band_index] = matrix
        preview_matrices[band_index] = preview_matrix
        diagnostics[band_index] = {
            "band": band_name,
            "reference": False,
            "registration_source": "sift_ransac_with_phase_gate",
            "keypoints": len(keypoints),
            "good_matches": len(good),
            "inliers": inlier_count,
            "inlier_ratio": inlier_ratio,
            "rmse_pixels": rmse,
            "phase_correlation_shift_preview_pixels": phase_shift_array.tolist(),
            "phase_correlation_response": phase_response,
            "phase_correlation_applied": phase_applied,
            "phase_preview_rmse_before_pixels": preview_rmse_before_phase,
            "phase_preview_rmse_after_pixels": preview_rmse_after_phase,
            "matrix_source_to_red": matrix.tolist(),
        }

    if any(matrix is None for matrix in matrices) or any(
        diagnostic is None for diagnostic in diagnostics
    ):
        raise RuntimeError("Registration did not produce all five band transforms")
    return (
        [matrix for matrix in matrices if matrix is not None],
        [diagnostic for diagnostic in diagnostics if diagnostic is not None],
    )


def _fill_missing_rows(values: np.ndarray) -> np.ndarray:
    """Interpolate unavailable calibration rows while preserving real valid samples."""

    row_numbers = np.arange(values.size, dtype=np.float64)
    valid = np.isfinite(values)
    if not np.any(valid):
        raise RuntimeError("Dark-reference calibration pixels contain no valid samples")
    if np.all(valid):
        return values.astype(np.float32, copy=False)
    return np.interp(row_numbers, row_numbers[valid], values[valid]).astype(np.float32)


def _estimate_dark_reference(
    source: rasterio.DatasetReader,
    calibration_border_pixels: int,
    constant_override_dn: float | None,
) -> DarkReferenceModel:
    """Measure per-line dark bias from the real left/right calibration pixels."""

    if constant_override_dn is not None:
        values = np.full((source.count, source.height), constant_override_dn, dtype=np.float32)
        diagnostics: list[dict[str, float | str]] = [
            {
                "band": band_name,
                "left_median_dn": float(constant_override_dn),
                "right_median_dn": float(constant_override_dn),
                "median_cross_track_delta_dn": 0.0,
                "left_row_drift_p02_p98_dn": 0.0,
                "right_row_drift_p02_p98_dn": 0.0,
            }
            for band_name in BAND_NAMES[: source.count]
        ]
        return DarkReferenceModel(
            left_dn=values,
            right_dn=values.copy(),
            left_detector_x=0.0,
            right_detector_x=float(source.width - 1),
            source="explicit --black-level-dn override",
            diagnostics=diagnostics,
        )

    if calibration_border_pixels < 8:
        raise RuntimeError(
            "At least 8 trusted calibration-border pixels are required per detector side"
        )
    if calibration_border_pixels * 2 >= source.width:
        raise RuntimeError("Calibration-border evidence is incompatible with detector width")
    left = source.read(
        window=Window(0, 0, calibration_border_pixels, source.height),
        out_dtype="float32",
    )
    right = source.read(
        window=Window(
            source.width - calibration_border_pixels,
            0,
            calibration_border_pixels,
            source.height,
        ),
        out_dtype="float32",
    )
    left[left <= 0] = np.nan
    right[right <= 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        left_rows = np.nanmedian(left, axis=2)
        right_rows = np.nanmedian(right, axis=2)

    diagnostics = []
    for band_index in range(source.count):
        left_rows[band_index] = _fill_missing_rows(left_rows[band_index])
        right_rows[band_index] = _fill_missing_rows(right_rows[band_index])
        left_p02, left_p98 = np.percentile(left_rows[band_index], (2, 98))
        right_p02, right_p98 = np.percentile(right_rows[band_index], (2, 98))
        diagnostics.append(
            {
                "band": BAND_NAMES[band_index],
                "left_median_dn": float(np.median(left_rows[band_index])),
                "right_median_dn": float(np.median(right_rows[band_index])),
                "median_cross_track_delta_dn": float(
                    np.median(right_rows[band_index] - left_rows[band_index])
                ),
                "left_row_drift_p02_p98_dn": float(left_p98 - left_p02),
                "right_row_drift_p02_p98_dn": float(right_p98 - right_p02),
            }
        )
    left_detector_x = (calibration_border_pixels - 1) / 2
    right_detector_x = source.width - (calibration_border_pixels + 1) / 2
    return DarkReferenceModel(
        left_dn=left_rows.astype(np.float32),
        right_dn=right_rows.astype(np.float32),
        left_detector_x=left_detector_x,
        right_detector_x=right_detector_x,
        source=(
            f"per-line median of {calibration_border_pixels} delivered dark-reference "
            "pixels on each detector side; linear cross-track interpolation"
        ),
        diagnostics=diagnostics,
    )


def _dark_surface(
    model: DarkReferenceModel,
    band_index: int,
    detector_x: np.ndarray,
    row_y: np.ndarray,
) -> np.ndarray:
    """Evaluate the measured dark-reference plane at detector coordinates."""

    source_rows = np.arange(model.left_dn.shape[1], dtype=np.float32)
    left = np.interp(row_y, source_rows, model.left_dn[band_index]).astype(np.float32)
    right = np.interp(row_y, source_rows, model.right_dn[band_index]).astype(np.float32)
    denominator = model.right_detector_x - model.left_detector_x
    alpha = (detector_x - model.left_detector_x) / denominator
    return left[:, None] + (right - left)[:, None] * alpha[None, :]


def _column_corrections(
    source: rasterio.DatasetReader,
    border: int,
    dark_reference: DarkReferenceModel,
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
    detector_x = border + np.arange(valid_width, dtype=np.float32)
    sampled_y = (np.arange(sample_rows, dtype=np.float32) + 0.5) * (
        source.height / sample_rows
    ) - 0.5
    for band_index, band in enumerate(sampled):
        dark = _dark_surface(dark_reference, band_index, detector_x, sampled_y)
        corrected = np.maximum(band - dark, 0)
        profile = np.median(corrected, axis=0)
        smooth = cv2.GaussianBlur(profile.reshape(1, -1), (0, 0), sigmaX=64, sigmaY=0).reshape(-1)
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
    dark_reference: DarkReferenceModel,
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
    detector_x = border + np.arange(x_min, x_max, dtype=np.float32)
    source_y = np.arange(y_min, y_max, dtype=np.float32)
    source_band -= _dark_surface(
        dark_reference,
        band_number - 1,
        detector_x,
        source_y,
    )
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


def _remap_strip_cuda(
    source: rasterio.DatasetReader,
    band_number: int,
    inverse_matrix: np.ndarray,
    border: int,
    output_width: int,
    output_height: int,
    y_start: int,
    strip_height: int,
    dark_reference: DarkReferenceModel,
    column_correction: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Correct and resample one strip on CUDA while keeping raster I/O bounded."""

    import torch
    from torch.nn import functional as functional

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA preprocessing was requested but CUDA is unavailable")
    y_stop = min(output_height, y_start + strip_height)
    output_corners = np.asarray(
        [
            [0, y_start],
            [output_width - 1, y_start],
            [0, y_stop - 1],
            [output_width - 1, y_stop - 1],
        ],
        dtype=np.float64,
    )
    source_corners = cv2.transform(output_corners[:, None, :], inverse_matrix)[:, 0, :]
    x_min = max(0, int(np.floor(np.min(source_corners[:, 0]))) - 2)
    x_max = min(output_width, int(np.ceil(np.max(source_corners[:, 0]))) + 3)
    y_min = max(0, int(np.floor(np.min(source_corners[:, 1]))) - 2)
    y_max = min(output_height, int(np.ceil(np.max(source_corners[:, 1]))) + 3)
    if x_min >= x_max or y_min >= y_max:
        return np.zeros((y_stop - y_start, output_width), dtype=np.float32), 0.0

    source_band = source.read(
        band_number,
        window=Window(border + x_min, y_min, x_max - x_min, y_max - y_min),
        out_dtype="float32",
    )
    source_valid = source_band > 0
    device = torch.device("cuda")
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        values = torch.from_numpy(source_band).to(device=device).unsqueeze(0).unsqueeze(0)
        valid = (
            torch.from_numpy(source_valid.astype(np.float32))
            .to(device=device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        left = torch.from_numpy(dark_reference.left_dn[band_number - 1, y_min:y_max]).to(
            device=device
        )
        right = torch.from_numpy(dark_reference.right_dn[band_number - 1, y_min:y_max]).to(
            device=device
        )
        detector_x = border + torch.arange(x_min, x_max, device=device, dtype=torch.float32)
        alpha = (detector_x - dark_reference.left_detector_x) / (
            dark_reference.right_detector_x - dark_reference.left_detector_x
        )
        dark = left[:, None] + (right - left)[:, None] * alpha[None, :]
        correction = torch.from_numpy(column_correction[x_min:x_max]).to(device=device)
        values = torch.clamp(values - dark[None, None] - correction[None, None, None], min=0)

        output_x = torch.arange(output_width, device=device, dtype=torch.float32)[None, :]
        output_y = torch.arange(y_start, y_stop, device=device, dtype=torch.float32)[:, None]
        map_x = (
            inverse_matrix[0, 0] * output_x
            + inverse_matrix[0, 1] * output_y
            + inverse_matrix[0, 2]
            - x_min
        )
        map_y = (
            inverse_matrix[1, 0] * output_x
            + inverse_matrix[1, 1] * output_y
            + inverse_matrix[1, 2]
            - y_min
        )
        normalised_x = 2 * map_x / max(1, x_max - x_min - 1) - 1
        normalised_y = 2 * map_y / max(1, y_max - y_min - 1) - 1
        grid = torch.stack(
            (
                normalised_x.expand(y_stop - y_start, output_width),
                normalised_y.expand(y_stop - y_start, output_width),
            ),
            dim=-1,
        ).unsqueeze(0)
        remapped = functional.grid_sample(
            values,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        remapped_valid = functional.grid_sample(
            valid,
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        )
        remapped = remapped.masked_fill(remapped_valid < 0.5, 0)
        output = remapped[0, 0].cpu().numpy()
    torch.cuda.synchronize(device)
    cuda_wall_seconds = time.perf_counter() - started
    return output, cuda_wall_seconds


def _write_l1a(
    source: rasterio.DatasetReader,
    output: Path,
    border: int,
    dark_reference: DarkReferenceModel,
    matrices: list[np.ndarray],
    column_corrections: list[np.ndarray],
    strip_height: int,
    device: str,
    overwrite: bool,
) -> dict[str, object]:
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
    write_started = time.perf_counter()
    cuda_remap_seconds = 0.0
    strip_count = 0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(temporary, "w", **profile) as destination:
                destination.update_tags(
                    PROCESSING_LEVEL="L1A_MINIMUM_SENSOR_SPACE",
                    PROCESSING_DEVICE=device.upper(),
                    RADIOMETRIC_UNITS="scene-destriped relative DN",
                    RADIOMETRIC_STATUS="NOT_RADIANCE_NOT_REFLECTANCE",
                    GEOLOCATION_STATUS="UNREFERENCED_NOT_ORTHORECTIFIED",
                    REGISTRATION_METHOD=(
                        "SIFT + RANSAC partial affine + gated phase-correlation refinement"
                    ),
                    REFERENCE_BAND="RED",
                    DARK_REFERENCE_METHOD=dark_reference.source,
                )
                for band_index, band_name in enumerate(BAND_NAMES):
                    inverse = cv2.invertAffineTransform(matrices[band_index])
                    for y_start in range(0, source.height, strip_height):
                        if device == "cuda":
                            strip, cuda_seconds = _remap_strip_cuda(
                                source,
                                band_index + 1,
                                inverse,
                                border,
                                valid_width,
                                source.height,
                                y_start,
                                strip_height,
                                dark_reference,
                                column_corrections[band_index],
                            )
                            cuda_remap_seconds += cuda_seconds
                        else:
                            strip = _remap_strip(
                                source,
                                band_index + 1,
                                inverse,
                                border,
                                valid_width,
                                source.height,
                                y_start,
                                strip_height,
                                dark_reference,
                                column_corrections[band_index],
                            )
                        strip_count += 1
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
    runtime: dict[str, object] = {
        "device": device,
        "seconds": time.perf_counter() - write_started,
        "strip_count": strip_count,
        "strip_height": strip_height,
        "cuda_dense_correction_and_resampling_seconds": (
            cuda_remap_seconds if device == "cuda" else None
        ),
        "cuda_accelerated_operations": (
            [
                "dark-reference subtraction",
                "column fixed-pattern correction",
                "affine bilinear resampling",
                "validity-mask resampling",
            ]
            if device == "cuda"
            else []
        ),
    }
    if device == "cuda":
        import torch

        runtime.update(
            {
                "cuda_version": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(0),
            }
        )
    return runtime


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
            "rotated, or labelled radiance, reflectance, georeferenced or orthorectified."
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
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help=(
            "Dense correction/resampling device. CUDA is explicit and fails closed when "
            "unavailable; feature control and GeoTIFF I/O remain on CPU."
        ),
    )
    parser.add_argument(
        "--black-level-dn",
        type=float,
        help=(
            "Explicit constant black-level override. By default, the processor measures a "
            "per-line, per-band dark plane from the delivered calibration-border pixels. "
            "SensorConfiguration.DarkOffset is not assumed to be a black level."
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
        args.output or data_root / "derived" / "l1a" / f"{args.scene_id}_L1A_MIN.tif"
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
    preview_path = output.with_name(f"{args.scene_id}_L1A_comparison.png")
    if args.preview_only:
        if not output.is_file():
            parser.error(f"L1A product does not exist for preview: {output}")
        _write_preview(
            raw_path,
            output,
            preview_path,
            args.preview_gamma,
        )
        print(json.dumps({"comparison_preview": str(preview_path)}, indent=2))
        return

    evidence = l0r_manifest["production_log_evidence"]
    border = evidence.get("border_pixels")
    valid_width = evidence.get("valid_detector_width")
    calibration_border = evidence.get("calibration_border_pixels")
    if (
        not isinstance(border, int)
        or not isinstance(valid_width, int)
        or not isinstance(calibration_border, int)
    ):
        raise RuntimeError(
            "Trusted log evidence for detector border, calibration pixels, or valid width is "
            "unavailable; refusing to guess"
        )
    if valid_width + 2 * border != int(metadata["Scenes"][0]["Width"]):
        raise RuntimeError("Trusted detector border evidence does not match the raw scene width")
    if calibration_border > border:
        raise RuntimeError("Calibration-border evidence exceeds the detector margin")
    sensor_configuration = metadata.get("SensorConfiguration", {})
    dark_offset = float(sensor_configuration.get("DarkOffset", 0))

    processing_started = time.perf_counter()
    phase_seconds: dict[str, float] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(raw_path) as source:
            phase_started = time.perf_counter()
            dark_reference = _estimate_dark_reference(
                source,
                calibration_border,
                args.black_level_dn,
            )
            phase_seconds["dark_reference_estimation"] = time.perf_counter() - phase_started
            phase_started = time.perf_counter()
            preview, scale_x, scale_y = _registration_preview(
                source, border, args.registration_downsample
            )
            matrices, registration = _estimate_registration(preview, scale_x, scale_y)
            phase_seconds["registration_estimation"] = time.perf_counter() - phase_started
            phase_started = time.perf_counter()
            corrections, correction_diagnostics = _column_corrections(
                source, border, dark_reference
            )
            phase_seconds["column_profile_estimation"] = time.perf_counter() - phase_started
            if args.device == "cuda":
                import torch

                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "--device cuda was requested but torch.cuda.is_available() is false"
                    )
            write_runtime = _write_l1a(
                source,
                output,
                border,
                dark_reference,
                matrices,
                corrections,
                args.strip_height,
                args.device,
                args.overwrite,
            )
    core_processing_seconds = time.perf_counter() - processing_started

    qa_started = time.perf_counter()
    _write_preview(
        raw_path,
        output,
        preview_path,
        args.preview_gamma,
    )
    post_write_registration = _post_write_registration(output, args.registration_downsample)
    qa_seconds = time.perf_counter() - qa_started
    l1a_manifest = {
        "schema_version": 2,
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
            "dark_reference_method": dark_reference.source,
            "dark_reference_pixels_per_side": calibration_border,
            "dark_reference_diagnostics": dark_reference.diagnostics,
            "delivered_sensor_dark_offset_setting": dark_offset,
            "delivered_sensor_dark_offset_applied_as_black_level": False,
            "inactive_detector_border_pixels_removed_each_side": border,
            "column_fixed_pattern_correction": "scene-estimated high-frequency median profile",
            "column_correction_diagnostics": correction_diagnostics,
            "absolute_calibration_applied": False,
        },
        "registration": {
            "method": (
                "SIFT cross-band features + RANSAC partial affine + gated "
                "phase-correlation refinement"
            ),
            "reference_band": "RED",
            "preview_downsample": args.registration_downsample,
            "bands": registration,
            "post_write_validation": post_write_registration,
        },
        "runtime": {
            "core_processing_seconds": core_processing_seconds,
            "qa_preview_and_validation_seconds": qa_seconds,
            "phase_seconds": phase_seconds,
            "dense_product_write": write_runtime,
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

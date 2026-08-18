"""Fit scene-specific raw-DN to L1ORT engineering radiometric diagnostics."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

BANDS = ("BLUE", "GREEN", "RED", "NIR", "PAN")


def _display_byte(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values) & (values > 0)
    if not np.any(valid):
        raise ValueError("Registration band contains no positive pixels")
    low, high = np.percentile(values[valid], (1.0, 99.0))
    if high <= low:
        raise ValueError("Registration band has no usable dynamic range")
    return np.clip((values - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)


def _read_corrected_raw(
    aligned_path: Path,
    raw_path: Path,
    *,
    height: int,
    inactive_border: int,
    dark_pixels: int,
) -> np.ndarray:
    with rasterio.open(aligned_path) as aligned, rasterio.open(raw_path) as raw:
        active_width = aligned.width - 2 * inactive_border
        output_width = round(active_width * height / aligned.height)
        values = aligned.read(
            (1, 2, 3, 4, 5),
            window=Window(inactive_border, 0, active_width, aligned.height),
            out_shape=(5, height, output_width),
            out_dtype="float32",
            resampling=Resampling.average,
        )
        fraction = np.linspace(0.0, 1.0, output_width, dtype=np.float32)[None, :]
        for index in range(5):
            left = raw.read(
                index + 1,
                window=Window(0, 0, dark_pixels, raw.height),
                out_shape=(height, 1),
                out_dtype="float32",
                resampling=Resampling.average,
            )[:, 0]
            right = raw.read(
                index + 1,
                window=Window(raw.width - dark_pixels, 0, dark_pixels, raw.height),
                out_shape=(height, 1),
                out_dtype="float32",
                resampling=Resampling.average,
            )[:, 0]
            values[index] -= left[:, None] + (right - left)[:, None] * fraction
    return values[:, :, ::-1].copy()


def _read_reference(path: Path, *, height: int) -> np.ndarray:
    with rasterio.open(path) as source:
        width = round(source.width * height / source.height)
        return source.read(
            (1, 2, 3, 4, 5),
            out_shape=(5, height, width),
            out_dtype="float32",
            resampling=Resampling.average,
        )


def _fit_affine(source: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    source64 = source.astype(np.float64)
    target64 = target.astype(np.float64)
    source_centered = source64 - source64.mean()
    slope = float(np.dot(source_centered, target64 - target64.mean()))
    slope /= float(np.dot(source_centered, source_centered))
    intercept = float(target64.mean() - slope * source64.mean())
    return slope, intercept


def fit(
    aligned_path: Path,
    raw_path: Path,
    reference_path: Path,
    output_path: Path,
    *,
    scene_id: str,
    qa_height: int = 1500,
) -> dict[str, object]:
    corrected = _read_corrected_raw(
        aligned_path,
        raw_path,
        height=qa_height,
        inactive_border=88,
        dark_pixels=68,
    )
    reference = _read_reference(reference_path, height=qa_height)
    cv2.setRNGSeed(0)
    sift = cv2.SIFT_create(nfeatures=12_000, contrastThreshold=0.01)
    moving_keypoints, moving_descriptors = sift.detectAndCompute(
        _display_byte(corrected[4]), None
    )
    reference_keypoints, reference_descriptors = sift.detectAndCompute(
        _display_byte(reference[4]), None
    )
    if moving_descriptors is None or reference_descriptors is None:
        raise ValueError("SIFT could not describe the PAN registration images")
    candidates = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
        moving_descriptors,
        reference_descriptors,
        k=2,
    )
    good = [first for first, second in candidates if first.distance < 0.75 * second.distance]
    if len(good) < 20:
        raise ValueError(f"Too few PAN matches: {len(good)}")
    moving_points = np.float32(
        [moving_keypoints[match.queryIdx].pt for match in good]
    )
    reference_points = np.float32(
        [reference_keypoints[match.trainIdx].pt for match in good]
    )
    homography, inlier_mask = cv2.findHomography(
        moving_points,
        reference_points,
        cv2.RANSAC,
        4.0,
        maxIters=10_000,
        confidence=0.999,
    )
    if homography is None or inlier_mask is None:
        raise ValueError("PAN homography fitting failed")
    reference_valid = np.isfinite(reference).all(axis=0) & np.all(reference > 0, axis=0)
    diagnostics: list[dict[str, object]] = []
    for index, band in enumerate(BANDS):
        warped = cv2.warpPerspective(
            corrected[index],
            homography,
            (reference.shape[2], reference.shape[1]),
            flags=cv2.INTER_LINEAR,
        )
        valid = reference_valid & np.isfinite(warped) & (warped > 0)
        source_values = warped[valid]
        target_values = reference[index, valid]
        source_limits = np.percentile(source_values, (1.0, 99.0))
        target_limits = np.percentile(target_values, (1.0, 99.0))
        selected = (
            (source_values >= source_limits[0])
            & (source_values <= source_limits[1])
            & (target_values >= target_limits[0])
            & (target_values <= target_limits[1])
        )
        source_values = source_values[selected]
        target_values = target_values[selected]
        slope, intercept = _fit_affine(source_values, target_values)
        prediction = source_values * slope + intercept
        correlation = float(np.corrcoef(source_values, target_values)[0, 1])
        diagnostics.append(
            {
                "band": band,
                "correlation": correlation,
                "diagnostic_affine_slope": slope,
                "diagnostic_affine_intercept": intercept,
                "reference_p01": float(target_limits[0]),
                "reference_p99": float(target_limits[1]),
                "rmse": float(np.sqrt(np.mean(np.square(prediction - target_values)))),
                "samples": int(source_values.size),
                "status": "PASS" if correlation >= 0.6 else "WARN",
            }
        )
    minimum_correlation = min(float(item["correlation"]) for item in diagnostics)
    report: dict[str, object] = {
        "schema_version": 1,
        "scene_id": scene_id,
        "created_at": datetime.now(UTC).isoformat(),
        "purpose": "Scene-specific engineering radiometric QA required by raw proxy",
        "status": "PASS" if minimum_correlation >= 0.6 else "WARN",
        "quality_gate": {
            "status": "PASS" if minimum_correlation >= 0.6 else "WARN",
            "minimum_each_band_correlation": 0.6,
            "minimum_observed_correlation": minimum_correlation,
        },
        "preprocessing": {
            "dark_reference": "68 untouched raw-parent pixels per side, interpolated per line",
            "inactive_border_pixels_removed_each_side": 88,
            "orientation": "horizontal flip for offline reference matching only",
            "qa_height_pixels": qa_height,
            "pair_filter": "positive overlap followed by joint 1st-to-99th percentile trimming",
        },
        "registration": {
            "method": "SIFT plus RANSAC homography on PAN",
            "homography_moving_to_reference": homography.tolist(),
            "keypoints_moving": len(moving_keypoints),
            "keypoints_reference": len(reference_keypoints),
            "good_matches": len(good),
            "inliers": int(inlier_mask.sum()),
            "inlier_percentage": float(100.0 * inlier_mask.mean()),
        },
        "radiometric_diagnostics": diagnostics,
        "source": {
            "aligned_raw": str(aligned_path),
            "raw_pixels_used_for_pipeline": True,
            "reference": str(reference_path),
            "untouched_raw": str(raw_path),
        },
        "limitations": [
            "Coefficients are scene-specific QA fits, not absolute sensor calibration.",
            "The reference is used only offline; inference pixels remain raw-derived.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aligned", type=Path)
    parser.add_argument("raw", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scene-id", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            fit(
                args.aligned,
                args.raw,
                args.reference,
                args.output,
                scene_id=args.scene_id,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

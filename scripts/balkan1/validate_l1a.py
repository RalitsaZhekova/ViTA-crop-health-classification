"""Validate a real Balkan-1 L1A product against a supplied L1ORT reference."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import warnings
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "data" / "balkan1"
BAND_NAMES = ("BLUE", "GREEN", "RED", "NIR", "PAN")
RED_BAND_INDEX = 2


def _read_overview(path: Path, maximum_dimension: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as source:
            scale = min(maximum_dimension / source.width, maximum_dimension / source.height, 1)
            width = max(1, round(source.width * scale))
            height = max(1, round(source.height * scale))
            return source.read(
                out_shape=(source.count, height, width),
                out_dtype="float32",
                resampling=Resampling.average,
            )


def _display_u8(image: np.ndarray) -> np.ndarray:
    valid = np.isfinite(image) & (image > 0)
    if not np.any(valid):
        return np.zeros(image.shape, dtype=np.uint8)
    low, high = np.percentile(image[valid], (1, 99))
    if high <= low:
        high = low + 1
    return np.round(np.clip((image - low) / (high - low), 0, 1) * 255).astype(np.uint8)


def _rgb_display(image: np.ndarray, gamma: float) -> np.ndarray:
    channels = [
        (_display_u8(image[index]).astype(np.float32) / 255) ** gamma for index in (2, 1, 0)
    ]
    return np.stack(channels, axis=-1)


def _reference_homography(
    l1a: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    clahe = cv2.createCLAHE(2.0, (8, 8))
    moving = clahe.apply(_display_u8(l1a[RED_BAND_INDEX]))
    target = clahe.apply(_display_u8(reference[RED_BAND_INDEX]))
    sift = cv2.SIFT_create(nfeatures=30_000, contrastThreshold=0.01)
    moving_keypoints, moving_descriptors = sift.detectAndCompute(moving, None)
    target_keypoints, target_descriptors = sift.detectAndCompute(target, None)
    if moving_descriptors is None or target_descriptors is None:
        raise RuntimeError("Reference validation could not find real Red-band features")
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(moving_descriptors, target_descriptors, k=2)
    good = [first for first, second in pairs if first.distance < 0.75 * second.distance]
    if len(good) < 100:
        raise RuntimeError(f"Only {len(good)} L1A-to-L1ORT feature matches were found")
    source_points = np.float32([moving_keypoints[match.queryIdx].pt for match in good])
    target_points = np.float32([target_keypoints[match.trainIdx].pt for match in good])
    homography, inlier_mask = cv2.findHomography(
        source_points,
        target_points,
        cv2.USAC_MAGSAC,
        4,
        maxIters=20_000,
        confidence=0.999,
    )
    if homography is None or inlier_mask is None:
        raise RuntimeError("Robust L1A-to-L1ORT homography estimation failed")
    inliers = inlier_mask.ravel().astype(bool)
    inlier_count = int(np.count_nonzero(inliers))
    if inlier_count < 100 or inlier_count / len(good) < 0.7:
        raise RuntimeError(f"Reference registration rejected: {inlier_count}/{len(good)} inliers")
    predicted = cv2.perspectiveTransform(source_points[inliers, None, :], homography)[:, 0, :]
    residual = target_points[inliers] - predicted
    rmse = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    return homography, {
        "moving_keypoints": len(moving_keypoints),
        "reference_keypoints": len(target_keypoints),
        "good_matches": len(good),
        "inliers": inlier_count,
        "inlier_ratio": inlier_count / len(good),
        "overview_rmse_pixels": rmse,
    }


def _warp_product(
    product: np.ndarray,
    homography: np.ndarray,
    reference_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = reference_shape
    warped = np.stack(
        [
            cv2.warpPerspective(
                band,
                homography,
                (width, height),
                flags=cv2.INTER_AREA,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            for band in product
        ]
    )
    source_valid = np.all(product > 0, axis=0).astype(np.uint8)
    valid = cv2.warpPerspective(
        source_valid,
        homography,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    return warped, valid


def _radiometric_diagnostic(
    moving: np.ndarray,
    reference: np.ndarray,
    overlap: np.ndarray,
) -> dict[str, float]:
    valid = overlap & (moving > 0) & (reference > 0)
    x = moving[valid].astype(np.float64)
    y = reference[valid].astype(np.float64)
    if x.size < 10_000:
        raise RuntimeError("Insufficient real overlap for radiometric validation")
    x_low, x_high = np.percentile(x, (1, 99))
    y_low, y_high = np.percentile(y, (1, 99))
    keep = (x >= x_low) & (x <= x_high) & (y >= y_low) & (y <= y_high)
    x = x[keep]
    y = y[keep]
    design = np.column_stack([x, np.ones_like(x)])
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = slope * x + intercept
    residual = predicted - y
    rmse = float(np.sqrt(np.mean(residual**2)))
    target_range = float(y_high - y_low)
    correlation = float(np.corrcoef(x, y)[0, 1])
    return {
        "samples": int(x.size),
        "diagnostic_affine_slope": float(slope),
        "diagnostic_affine_intercept": float(intercept),
        "pearson_correlation": correlation,
        "rmse_after_diagnostic_affine": rmse,
        "rmse_fraction_of_reference_p01_p99_range": rmse / target_range,
    }


def _write_visualization(
    aligned: np.ndarray,
    reference: np.ndarray,
    overlap: np.ndarray,
    output: Path,
    scene_id: str,
    gamma: float,
) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "prithvi_payload_matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    moving_rgb = _rgb_display(aligned, gamma)
    reference_rgb = _rgb_display(reference, gamma)
    moving_rgb[~overlap] = 0
    reference_rgb[~overlap] = 0
    difference = np.clip(np.abs(moving_rgb - reference_rgb) * 2, 0, 1)
    figure, axes = plt.subplots(1, 3, figsize=(21, 8))
    panels = (
        (moving_rgb, "New L1A in sensor-row orientation\nindependent display stretch"),
        (
            reference_rgb,
            "Supplied L1ORT reprojected for QA\nno display rotation",
        ),
        (difference, "Display-space absolute difference\n2x amplification"),
    )
    for axis, (image, title) in zip(axes, panels, strict=True):
        axis.imshow(image)
        axis.set_title(title, fontsize=14)
        axis.axis("off")
    figure.suptitle(
        f"Balkan-1 scene {scene_id}: real L1A-to-L1ORT validation",
        fontsize=18,
        weight="bold",
    )
    figure.text(
        0.5,
        0.02,
        (
            "The supplied L1ORT is used only for validation here. It is not an input to the "
            "L1A preprocessing algorithm."
        ),
        ha="center",
        fontsize=11,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Geometrically align a real L1A overview to supplied L1ORT for QA only."
    )
    parser.add_argument("scene_id")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--l1a", type=Path)
    parser.add_argument("--maximum-dimension", type=int, default=1400)
    parser.add_argument("--gamma", type=float, default=0.65)
    args = parser.parse_args()
    if args.maximum_dimension < 512:
        parser.error("--maximum-dimension must be at least 512")
    if args.gamma <= 0:
        parser.error("--gamma must be greater than zero")

    data_root = args.data_root.resolve()
    l1a_path = (
        args.l1a or data_root / "derived" / "l1a" / f"{args.scene_id}_L1A_MIN.tif"
    ).resolve()
    reference_path = (data_root / "preprocessed" / f"{args.scene_id}_L1ORT.tif").resolve()
    for path in (l1a_path, reference_path):
        if not path.is_file():
            parser.error(f"required validation input does not exist: {path}")

    l1a = _read_overview(l1a_path, args.maximum_dimension)
    reference = _read_overview(reference_path, args.maximum_dimension)
    l1a = l1a[:, :, ::-1].copy()
    homography, registration = _reference_homography(l1a, reference)
    aligned, overlap = _warp_product(l1a, homography, reference.shape[1:])
    overlap &= np.all(reference > 0, axis=0)
    overlap_fraction = float(np.count_nonzero(overlap) / overlap.size)
    if overlap_fraction < 0.25:
        raise RuntimeError(f"Reference overlap is unexpectedly low: {overlap_fraction:.3f}")

    radiometric = [
        {
            "band": band_name,
            **_radiometric_diagnostic(aligned[index], reference[index], overlap),
        }
        for index, band_name in enumerate(BAND_NAMES)
    ]
    minimum_correlation = min(float(item["pearson_correlation"]) for item in radiometric)
    maximum_range_normalized_rmse = max(
        float(item["rmse_fraction_of_reference_p01_p99_range"]) for item in radiometric
    )
    acceptance = {
        "status": (
            "PASS"
            if minimum_correlation >= 0.8 and maximum_range_normalized_rmse <= 0.12
            else "REVIEW_REQUIRED"
        ),
        "minimum_band_correlation": minimum_correlation,
        "minimum_band_correlation_threshold": 0.8,
        "maximum_band_rmse_fraction_of_reference_range": (maximum_range_normalized_rmse),
        "maximum_band_rmse_fraction_threshold": 0.12,
        "scope": "overview similarity QA; not an absolute-calibration certificate",
    }
    output_dir = l1a_path.parent
    visualization_path = output_dir / f"{args.scene_id}_L1A_reference_validation.png"
    metrics_path = output_dir / f"{args.scene_id}_L1A_reference_validation.json"
    reference_in_sensor_orientation, reference_sensor_valid = _warp_product(
        reference,
        np.linalg.inv(homography),
        l1a.shape[1:],
    )
    sensor_overlap = (
        reference_sensor_valid
        & np.all(l1a > 0, axis=0)
        & np.all(reference_in_sensor_orientation > 0, axis=0)
    )
    _write_visualization(
        l1a,
        reference_in_sensor_orientation,
        sensor_overlap,
        visualization_path,
        args.scene_id,
        args.gamma,
    )
    metrics = {
        "schema_version": 1,
        "scene_id": args.scene_id,
        "purpose": "offline QA only",
        "l1a": str(l1a_path),
        "supplied_l1ort_reference": str(reference_path),
        "reference_used_by_l1a_processor": False,
        "overview": {
            "l1a_shape_after_flip_x": list(l1a.shape),
            "reference_shape": list(reference.shape),
            "overlap_fraction_of_reference_grid": overlap_fraction,
        },
        "registration": {
            **registration,
            "homography_l1a_overview_to_reference_overview": homography.tolist(),
        },
        "radiometric_diagnostics": radiometric,
        "acceptance": acceptance,
        "interpretation": (
            "The per-band affine fits are validation diagnostics, not a reusable physical "
            "calibration and are never applied by process_l1a.py."
        ),
        "visualization": str(visualization_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "metrics": str(metrics_path),
                "visualization": str(visualization_path),
                "registration": registration,
                "overlap_fraction": overlap_fraction,
                "acceptance": acceptance,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

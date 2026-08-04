"""Fit a validated Balkan-1 crop-input calibration from a co-registered Sentinel scene.

This is a ground-side preparation tool.  The resulting small JSON sidecar is the
only calibration artifact needed by payload inference; Sentinel imagery is never
required at runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
GDAL_WARP_THREADS = max(1, min(4, os.cpu_count() or 1))
sys.path.insert(0, str(REPOSITORY_ROOT / "payload" / "src"))

from prithvi_payload.balkan_crop_calibration import (  # noqa: E402
    ADAPTER_MODE,
    CALIBRATION_SCHEMA_VERSION,
    DEFAULT_ANALYSIS_RESOLUTION_METRES,
    MIN_BAND_CORRELATION,
    MIN_MEAN_BAND_CORRELATION,
    MIN_VALIDATION_PIXELS,
    MODEL_BAND_ORDER,
    SOURCE_BAND_ORDER,
    default_calibration_path,
    sha256_file,
)


def _isotonic(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Least-squares non-decreasing fit using the pool-adjacent-violators algorithm."""
    levels: list[float] = []
    masses: list[float] = []
    starts: list[int] = []
    ends: list[int] = []
    for index, (value, weight) in enumerate(zip(values, weights, strict=True)):
        levels.append(float(value))
        masses.append(float(weight))
        starts.append(index)
        ends.append(index + 1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            combined_mass = masses[-2] + masses[-1]
            combined_level = (levels[-2] * masses[-2] + levels[-1] * masses[-1]) / combined_mass
            levels[-2:] = [combined_level]
            masses[-2:] = [combined_mass]
            starts[-2:] = [starts[-2]]
            ends[-2:] = [ends[-1]]
    fitted = np.empty(values.size, dtype=np.float64)
    for level, start, end in zip(levels, starts, ends, strict=True):
        fitted[start:end] = level
    return fitted


def _fit_curve(source: np.ndarray, target: np.ndarray, *, bins: int, knots: int) -> dict:
    order = np.argsort(source)
    partitions = np.array_split(order, min(bins, order.size))
    source_medians = np.asarray([np.median(source[item]) for item in partitions])
    target_medians = np.asarray([np.median(target[item]) for item in partitions])
    weights = np.asarray([item.size for item in partitions], dtype=np.float64)

    unique_source, inverse = np.unique(source_medians, return_inverse=True)
    weighted_target = np.zeros(unique_source.size, dtype=np.float64)
    combined_weights = np.zeros(unique_source.size, dtype=np.float64)
    np.add.at(weighted_target, inverse, target_medians * weights)
    np.add.at(combined_weights, inverse, weights)
    weighted_target /= combined_weights
    fitted_target = _isotonic(weighted_target, combined_weights)
    if unique_source.size < 2:
        raise ValueError("A calibration band has insufficient radiometric variation")
    selected = np.unique(
        np.rint(np.linspace(0, unique_source.size - 1, min(knots, unique_source.size))).astype(int)
    )
    return {
        "source_knots": unique_source[selected].astype(float).tolist(),
        "target_values": fitted_target[selected].astype(float).tolist(),
    }


def _read_pair(
    source_path: Path,
    reference_path: Path,
    *,
    source_indices: list[int],
    reference_indices: list[int],
    source_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(reference_path) as reference:
        if reference.crs is None:
            raise ValueError("Sentinel reference has no CRS")
        reference_values = reference.read(reference_indices, out_dtype="float32")
        reference_valid = np.isfinite(reference_values).all(axis=0)
        if reference.nodata is not None:
            reference_valid &= ~np.any(reference_values == reference.nodata, axis=0)
        reference_valid &= np.all(reference_values > 0, axis=0)

        with rasterio.open(source_path) as source:
            if source.crs is None:
                raise ValueError("Balkan source has no CRS")
            if max(source_indices) > source.count:
                raise ValueError("Balkan source band indices exceed its band count")
            source_values = np.full(reference_values.shape, np.nan, dtype=np.float32)
            for destination_index, source_index in enumerate(source_indices):
                reproject(
                    source=rasterio.band(source, source_index),
                    destination=source_values[destination_index],
                    src_transform=source.transform,
                    src_crs=source.crs,
                    src_nodata=source.nodata,
                    dst_transform=reference.transform,
                    dst_crs=reference.crs,
                    dst_nodata=np.nan,
                    resampling=Resampling.average,
                    init_dest_nodata=True,
                    num_threads=GDAL_WARP_THREADS,
                )
    source_values *= np.float32(source_scale)
    valid = reference_valid & np.isfinite(source_values).all(axis=0)
    valid &= np.all(source_values > 0, axis=0)
    return source_values, np.where(valid[None, ...], reference_values, np.nan)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a per-scene monotonic Balkan-1 to Sentinel calibration for the "
            "existing Prithvi crop model. Inputs must overlap and be co-registered."
        )
    )
    parser.add_argument("balkan", type=Path)
    parser.add_argument("sentinel_reference", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--acquired-at", required=True)
    parser.add_argument("--source-band-indices", nargs=4, type=int, default=[1, 2, 3, 4])
    parser.add_argument("--reference-band-indices", nargs=4, type=int, default=[1, 2, 3, 4])
    parser.add_argument("--source-scale", type=float, default=10_000.0)
    parser.add_argument("--bins", type=int, default=128)
    parser.add_argument("--knots", type=int, default=65)
    parser.add_argument("--validation-block-size", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = args.balkan.resolve()
    reference = args.sentinel_reference.resolve()
    output = args.output.resolve() if args.output else default_calibration_path(source)
    if not source.is_file() or not reference.is_file():
        parser.error("Both Balkan and Sentinel GeoTIFF inputs must exist")
    if (
        not np.isfinite(args.source_scale)
        or args.source_scale <= 0
        or args.bins < 8
        or args.knots < 8
        or args.validation_block_size <= 0
    ):
        parser.error(
            "source-scale and validation-block-size must be positive; bins/knots must be at least 8"
        )
    if output in {source, reference} or output.suffix.lower() != ".json":
        parser.error("Calibration output must be a JSON file distinct from both GeoTIFF inputs")
    if output.is_relative_to((REPOSITORY_ROOT / "payload").resolve()):
        parser.error("Calibration sidecars must remain with local data, outside payload/")
    if output.exists() and not args.overwrite:
        parser.error(f"Calibration output already exists; pass --overwrite to replace it: {output}")
    if (
        len(set(args.source_band_indices)) != 4
        or min(args.source_band_indices) <= 0
        or len(set(args.reference_band_indices)) != 4
        or min(args.reference_band_indices) <= 0
    ):
        parser.error("Source and reference band indices must be four unique positive integers")
    acquired_at = datetime.fromisoformat(args.acquired_at.replace("Z", "+00:00"))
    if acquired_at.tzinfo is None:
        parser.error("--acquired-at must include a UTC offset or trailing Z")

    source_values, reference_values = _read_pair(
        source,
        reference,
        source_indices=args.source_band_indices,
        reference_indices=args.reference_band_indices,
        source_scale=args.source_scale,
    )
    valid = np.isfinite(reference_values).all(axis=0)
    valid_reference_values = reference_values[:, valid]
    if (
        valid_reference_values.size == 0
        or float(np.nanmin(valid_reference_values)) < 0
        or float(np.nanmax(valid_reference_values)) > 20_000
        or float(np.nanpercentile(valid_reference_values, 95)) <= 100
    ):
        raise SystemExit(
            "Sentinel reference must contain calibrated B02/B03/B04/B8A values "
            "in the model's approximately 0-10000 scale"
        )
    rows, columns = np.indices(valid.shape)
    held_out = valid & (
        ((rows // args.validation_block_size) + (columns // args.validation_block_size)) % 2 == 1
    )
    training = valid & ~held_out
    if np.count_nonzero(training) < MIN_VALIDATION_PIXELS:
        raise SystemExit("Calibration pair has too few valid training pixels")

    curves = []
    correlations = []
    median_errors = []
    for index, band in enumerate(SOURCE_BAND_ORDER):
        curve = _fit_curve(
            source_values[index][training],
            reference_values[index][training],
            bins=args.bins,
            knots=args.knots,
        )
        curve["band"] = band
        curves.append(curve)
        calibrated = np.interp(
            source_values[index][held_out],
            curve["source_knots"],
            curve["target_values"],
        )
        target = reference_values[index][held_out]
        correlations.append(_correlation(calibrated, target))
        median_errors.append(float(np.median(np.abs(calibrated - target))))

    held_out_count = int(np.count_nonzero(held_out))
    passed = (
        held_out_count >= MIN_VALIDATION_PIXELS
        and np.isfinite(correlations).all()
        and min(correlations) >= MIN_BAND_CORRELATION
        and float(np.mean(correlations)) >= MIN_MEAN_BAND_CORRELATION
    )
    calibration = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "adapter_mode": ADAPTER_MODE,
        "sensor": "balkan-1",
        "created_at": datetime.now(UTC).isoformat(),
        "acquired_at": acquired_at.isoformat(),
        "source_band_order": list(SOURCE_BAND_ORDER),
        "model_band_order": list(MODEL_BAND_ORDER),
        "source_band_indices_1_based": args.source_band_indices,
        "source_scale_to_model_units": args.source_scale,
        "analysis_resolution_metres": DEFAULT_ANALYSIS_RESOLUTION_METRES,
        "source": {
            "filename": source.name,
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        },
        "reference": {
            "filename": reference.name,
            "bytes": reference.stat().st_size,
            "sha256": sha256_file(reference),
            "sensor": "sentinel-2",
            "model_band_names": ["B02", "B03", "B04", "B8A"],
            "band_indices_1_based": args.reference_band_indices,
        },
        "fit": {
            "method": "equal_count_bin_medians_then_pava",
            "training_pixels": int(np.count_nonzero(training)),
            "bins": args.bins,
            "stored_knots_per_band_maximum": args.knots,
            "spatial_split": "alternating_checkerboard_blocks",
            "validation_block_size_pixels": args.validation_block_size,
            "pair_grid": "sentinel_reference_grid",
        },
        "curves": curves,
        "validation": {
            "status": "PASS" if passed else "FAIL",
            "held_out_valid_pixels": held_out_count,
            "held_out_band_correlations": correlations,
            "held_out_band_median_absolute_errors": median_errors,
            "quality_gate": {
                "minimum_pixels": MIN_VALIDATION_PIXELS,
                "minimum_each_band_correlation": MIN_BAND_CORRELATION,
                "minimum_mean_band_correlation": MIN_MEAN_BAND_CORRELATION,
            },
        },
    }
    if not passed:
        print(json.dumps(calibration["validation"], indent=2))
        raise SystemExit("Calibration rejected by held-out quality gate; no sidecar written")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "validation": calibration["validation"]}, indent=2))


if __name__ == "__main__":
    main()

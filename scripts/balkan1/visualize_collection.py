"""Build a bounded overview of delivered Balkan-1 L1ORT scenes for review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import rasterio
from rasterio.enums import Resampling

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPOSITORY_ROOT / "data" / "balkan1" / "preprocessed"
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT / "testing" / "runs" / "balkan1_collection_review" / "collection_overview.png"
)


def _stretch(channels: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = np.zeros_like(channels, dtype=np.float32)
    for index, channel in enumerate(channels):
        samples = channel[valid & np.isfinite(channel)]
        if samples.size == 0:
            continue
        lower, upper = np.percentile(samples, (2, 98))
        if upper > lower:
            output[index] = np.clip((channel - lower) / (upper - lower), 0, 1)
    return np.moveaxis(output, 0, -1)


def _overview(path: Path, size: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with rasterio.open(path) as source:
        if source.count < 4:
            raise ValueError(f"{path} has {source.count} bands; at least four are required")
        height = max(1, round(source.height * min(1.0, size / source.width)))
        width = max(1, round(source.width * min(1.0, size / source.height)))
        height = min(size, height)
        width = min(size, width)
        values = source.read(
            (1, 2, 3, 4),
            out_shape=(4, height, width),
            resampling=Resampling.average,
            out_dtype="float32",
        )
        valid = np.all(np.isfinite(values), axis=0) & np.any(values != 0, axis=0)
        denominator = values[3] + values[2]
        ndvi = np.divide(
            values[3] - values[2],
            denominator,
            out=np.zeros_like(denominator),
            where=valid & (np.abs(denominator) > 1e-8),
        )
        vegetation = valid & (ndvi >= 0.25)
        valid_pixels = int(np.count_nonzero(valid))
        metadata = {
            "scene_id": path.stem.removesuffix("_L1ORT"),
            "source": str(path.resolve()),
            "source_shape": [source.height, source.width],
            "overview_shape": [height, width],
            "crs": str(source.crs),
            "bounds": list(source.bounds),
            "valid_pixels": valid_pixels,
            "vegetation_percentage": (
                100 * int(np.count_nonzero(vegetation)) / valid_pixels if valid_pixels else 0.0
            ),
            "selection_note": (
                "NDVI >= 0.25 is only a vegetation review aid; it is not a crop label"
            ),
        }
    true_color = _stretch(values[[2, 1, 0]], valid)
    false_color = _stretch(values[[3, 2, 1]], valid)
    true_color[~valid] = 0
    false_color[~valid] = 0
    return true_color, false_color, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create true-color and vegetation false-color overviews of real Balkan-1 "
            "L1ORT scenes without materializing full rasters."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overview-size", type=int, default=512)
    args = parser.parse_args()
    if args.overview_size <= 0:
        parser.error("--overview-size must be positive")
    sources = sorted(args.source.resolve().glob("*_L1ORT.tif"))
    if not sources:
        parser.error(f"No *_L1ORT.tif files found below {args.source.resolve()}")

    records: list[dict[str, Any]] = []
    figure, axes = plt.subplots(len(sources), 2, figsize=(12, 5 * len(sources)))
    if len(sources) == 1:
        axes = np.asarray([axes])
    for row, source_path in enumerate(sources):
        true_color, false_color, record = _overview(source_path, args.overview_size)
        records.append(record)
        panels = (
            (true_color, f"{record['scene_id']} true color"),
            (
                false_color,
                f"NIR false color | vegetation aid {record['vegetation_percentage']:.1f}%",
            ),
        )
        for axis, (image, title) in zip(axes[row], panels, strict=True):
            axis.imshow(image)
            axis.set_title(title)
            axis.axis("off")
    figure.suptitle(
        "Balkan-1 delivered L1ORT collection review\n"
        "Independent display stretches; vegetation percentage is not a crop prediction",
        fontsize=16,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.985))

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    manifest = {
        "schema_version": 1,
        "purpose": "visual scene selection before Balkan-1 payload execution proof",
        "overview": str(output),
        "scenes": records,
    }
    manifest_path = output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

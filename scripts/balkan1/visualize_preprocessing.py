"""Visualize one Balkan-1 acquisition before and after L1ORT preprocessing."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import warnings
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "prithvi_payload_matplotlib")
)

import matplotlib
import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.windows import Window

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "data" / "balkan1"
DEFAULT_RUN_ROOT = REPOSITORY_ROOT / "testing" / "runs"
BAND_LABELS = (
    "B01 · Blue · 490 nm",
    "B02 · Green · 560 nm",
    "B03 · Red · 665 nm",
    "B07 · NIR · 842 nm",
    "PAN · broad · 625 nm",
)


def _center_window(width: int, height: int, size: int) -> Window:
    window_width = min(size, width)
    window_height = min(size, height)
    return Window(
        col_off=(width - window_width) // 2,
        row_off=(height - window_height) // 2,
        width=window_width,
        height=window_height,
    )


def _read_center(path: Path, size: int) -> tuple[np.ndarray, dict[str, object]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as dataset:
            if dataset.count != 5:
                raise ValueError(f"Expected five bands in {path}, found {dataset.count}")
            window = _center_window(dataset.width, dataset.height, size)
            image = dataset.read(window=window, out_dtype="float32")
            bounds = rasterio.windows.bounds(window, dataset.transform)
            metadata: dict[str, object] = {
                "path": str(path),
                "width": dataset.width,
                "height": dataset.height,
                "bands": dataset.count,
                "dtype": dataset.dtypes[0],
                "descriptions": list(dataset.descriptions),
                "crs": str(dataset.crs) if dataset.crs else None,
                "nodata": dataset.nodata,
                "window": {
                    "x": int(window.col_off),
                    "y": int(window.row_off),
                    "width": int(window.width),
                    "height": int(window.height),
                },
                "window_bounds": list(bounds) if dataset.crs else None,
            }
    return image, metadata


def _stretch_band(band: np.ndarray) -> np.ndarray:
    valid = np.isfinite(band) & (band != 0)
    if not np.any(valid):
        return np.zeros_like(band, dtype=np.float32)
    low, high = np.percentile(band[valid], (2, 98))
    if high <= low:
        high = low + 1
    stretched = np.clip((band - low) / (high - low), 0, 1)
    stretched[~valid] = 0
    return stretched.astype(np.float32)


def _composite(image: np.ndarray, indices: tuple[int, int, int]) -> np.ndarray:
    return np.stack([_stretch_band(image[index]) for index in indices], axis=-1)


def _edge_strength(band: np.ndarray) -> np.ndarray:
    stretched = _stretch_band(band)
    gradient_y, gradient_x = np.gradient(stretched)
    magnitude = np.hypot(gradient_x, gradient_y)
    nonzero = magnitude[magnitude > 0]
    if nonzero.size == 0:
        return magnitude
    scale = float(np.percentile(nonzero, 98.5))
    return np.clip(magnitude / max(scale, 1e-6), 0, 1) ** 0.65


def _alignment_overlay(image: np.ndarray) -> np.ndarray:
    red_edges = _edge_strength(image[2])
    blue_edges = _edge_strength(image[0])
    return np.stack([red_edges, blue_edges, blue_edges], axis=-1)


def _extract_log_metrics(log_path: Path) -> dict[str, object]:
    if not log_path.is_file():
        return {"log": str(log_path), "available": False}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    patterns = {
        "attitude_samples": r"Read (\d+) attitude samples",
        "position_samples": r"Read (\d+) position samples",
        "band_time_offsets_rows": r"band time offsets \[rows\]:\s*([^\)\r\n]+)",
        "gcp_rms_m": r"product assembled \(5 bands, \d+x\d+, GCP\s+rms ([\d.]+) m",
        "ce90_m": r"band=Overall[^\r\n]*ce90_m=([\d.]+)",
        "total_runtime": r"TOTAL \(top-level stages, GPU \+ host\)\s+([^\r\n]+)",
    }
    metrics: dict[str, object] = {"log": str(log_path), "available": True}
    for name, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip()
            if name.endswith("_samples"):
                metrics[name] = int(value)
            elif name.endswith("_m"):
                metrics[name] = float(value)
            else:
                metrics[name] = value
    return metrics


def _draw_flow(axis: plt.Axes, metrics: dict[str, object]) -> None:
    axis.axis("off")
    steps = (
        ("1 · RAW SENSOR", "5 stacked 12-bit DN bands\nno CRS / not map-ready"),
        ("2 · CORRECT", "destripe + deconvolve\nnormalize to reflectance"),
        (
            "3 · ALIGN",
            f"attitude: {metrics.get('attitude_samples', '?')} samples\n"
            f"position: {metrics.get('position_samples', '?')} samples",
        ),
        ("4 · ORTHORECTIFY", "per-band GPU registration\nreference image + terrain DEM"),
        ("5 · COMBINED L1ORT", "5 aligned reflectance bands\nEPSG:4326 GeoTIFF"),
    )
    centers = np.linspace(0.09, 0.91, len(steps))
    for index, ((title, body), center) in enumerate(zip(steps, centers, strict=True)):
        axis.text(
            center,
            0.5,
            f"{title}\n{body}",
            ha="center",
            va="center",
            fontsize=10,
            transform=axis.transAxes,
            bbox={
                "boxstyle": "round,pad=0.55",
                "facecolor": "#eef4fa" if index < 4 else "#e9f7ec",
                "edgecolor": "#52708f" if index < 4 else "#3d8650",
                "linewidth": 1.5,
            },
        )
        if index < len(steps) - 1:
            axis.annotate(
                "",
                xy=(centers[index + 1] - 0.08, 0.5),
                xytext=(center + 0.08, 0.5),
                xycoords=axis.transAxes,
                arrowprops={"arrowstyle": "->", "color": "#455a64", "lw": 2},
            )


def _render(
    scene_id: str,
    raw: np.ndarray,
    processed: np.ndarray,
    raw_metadata: dict[str, object],
    processed_metadata: dict[str, object],
    metrics: dict[str, object],
    output: Path,
) -> None:
    figure = plt.figure(figsize=(19, 15), constrained_layout=True)
    grid = figure.add_gridspec(4, 1, height_ratios=(0.65, 1.15, 1.45, 1.45))
    figure.suptitle(
        f"Balkan-1 preprocessing before payload inference — scene {scene_id}",
        fontsize=22,
        weight="bold",
    )

    _draw_flow(figure.add_subplot(grid[0]), metrics)

    band_grid = grid[1].subgridspec(1, 5)
    for index, label in enumerate(BAND_LABELS):
        axis = figure.add_subplot(band_grid[index])
        axis.imshow(_stretch_band(raw[index]), cmap="gray", vmin=0, vmax=1)
        axis.set_title(label, fontsize=12)
        axis.axis("off")
    comparison_grid = grid[2].subgridspec(1, 4)
    panels = (
        (
            _composite(raw, (2, 1, 0)),
            "BEFORE · raw RGB display\nBands are combined for viewing only",
        ),
        (
            _alignment_overlay(raw),
            "BEFORE · raw band edges\nred=Red band, cyan=Blue band",
        ),
        (
            _composite(processed, (2, 1, 0)),
            "AFTER · registered true color\nRed + Green + Blue",
        ),
        (
            _alignment_overlay(processed),
            "AFTER · registered band edges\nwhite features indicate overlap",
        ),
    )
    for index, (panel, title) in enumerate(panels):
        axis = figure.add_subplot(comparison_grid[index])
        axis.imshow(panel)
        axis.set_title(title, fontsize=12)
        axis.axis("off")

    result_grid = grid[3].subgridspec(1, 3, width_ratios=(1, 1, 1.7))
    false_color_axis = figure.add_subplot(result_grid[0])
    false_color_axis.imshow(_composite(processed, (3, 2, 1)))
    false_color_axis.set_title("L1ORT false color\nNIR + Red + Green", fontsize=12)
    false_color_axis.axis("off")

    pan_axis = figure.add_subplot(result_grid[1])
    pan_axis.imshow(_stretch_band(processed[4]), cmap="gray", vmin=0, vmax=1)
    pan_axis.set_title("L1ORT panchromatic\nbroadband detail", fontsize=12)
    pan_axis.axis("off")

    info_axis = figure.add_subplot(result_grid[2])
    info_axis.axis("off")
    raw_window = raw_metadata["window"]
    processed_window = processed_metadata["window"]
    info = (
        "WHAT CHANGED\n\n"
        f"Raw: {raw_metadata['width']} × {raw_metadata['height']} × 5, "
        f"{raw_metadata['dtype']}, no CRS\n"
        f"L1ORT: {processed_metadata['width']} × {processed_metadata['height']} × 5, "
        f"{processed_metadata['dtype']}, {processed_metadata['crs']}\n\n"
        f"Band timing offsets (rows): {metrics.get('band_time_offsets_rows', '?')}\n"
        f"Registration GCP RMS: {metrics.get('gcp_rms_m', '?')} m\n"
        f"Measured overall CE90: {metrics.get('ce90_m', '?')} m\n"
        f"Recorded preprocessing time: {metrics.get('total_runtime', '?')}\n\n"
        "DISPLAY NOTE\n"
        f"Raw center window: {raw_window['width']} × {raw_window['height']}\n"
        f"L1ORT center window: {processed_window['width']} × "
        f"{processed_window['height']}\n"
        "The windows are centered independently because the raw TIFF\n"
        "has no map coordinates. Percentile stretching changes display\n"
        "contrast only; it does not modify either source TIFF."
    )
    info_axis.text(
        0.02,
        0.98,
        info,
        ha="left",
        va="top",
        fontsize=12,
        linespacing=1.35,
        transform=info_axis.transAxes,
        bbox={"boxstyle": "round,pad=0.8", "facecolor": "#f7f7f5", "edgecolor": "#777"},
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render raw-band and combined-L1ORT preprocessing evidence without inference."
    )
    parser.add_argument("scene_id")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--processed", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--window-size", type=int, default=1536)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.window_size < 256:
        parser.error("--window-size must be at least 256 pixels")
    data_root = args.data_root.resolve()
    raw_path = (
        args.raw or data_root / "raw" / args.scene_id / f"{args.scene_id}_Raw.tif"
    ).resolve()
    processed_path = (
        args.processed or data_root / "preprocessed" / f"{args.scene_id}_L1ORT.tif"
    ).resolve()
    log_path = (args.log or data_root / "preprocessed" / f"log_{args.scene_id}.txt").resolve()
    output = (
        args.output
        or DEFAULT_RUN_ROOT
        / f"balkan1_preprocessing_{args.scene_id}"
        / f"{args.scene_id}_preprocessing_overview.png"
    ).resolve()
    if not raw_path.is_file():
        parser.error(f"raw TIFF does not exist: {raw_path}")
    if not processed_path.is_file():
        parser.error(f"processed TIFF does not exist: {processed_path}")
    if output.exists() and not args.overwrite:
        parser.error(f"output exists; pass --overwrite to replace it: {output}")

    raw, raw_metadata = _read_center(raw_path, args.window_size)
    processed, processed_metadata = _read_center(processed_path, args.window_size)
    metrics = _extract_log_metrics(log_path)
    _render(
        args.scene_id,
        raw,
        processed,
        raw_metadata,
        processed_metadata,
        metrics,
        output,
    )

    manifest = {
        "schema_version": 1,
        "purpose": "Balkan-1 preprocessing evidence before payload inference",
        "scene_id": args.scene_id,
        "raw": raw_metadata,
        "processed": processed_metadata,
        "processing_log": metrics,
        "visualization": str(output),
        "display_only": True,
    }
    manifest_path = output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"visualization": str(output), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()

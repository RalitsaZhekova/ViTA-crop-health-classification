"""Create one bounded Balkan-1 proof chip outside the payload source tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import rasterio
from rasterio.windows import Window, transform

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "prithvi_payload_matplotlib")
)
matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = REPOSITORY_ROOT / "testing" / "inputs" / "balkan1"
DEFAULT_BAND_ORDER = ("BLUE", "GREEN", "RED", "NIR", "PAN")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _window(
    width: int,
    height: int,
    requested: list[int] | None,
    size: int,
) -> Window:
    if requested is None:
        window_width = min(size, width)
        window_height = min(size, height)
        x_offset = (width - window_width) // 2
        y_offset = (height - window_height) // 2
    else:
        x_offset, y_offset, window_width, window_height = requested
    if min(x_offset, y_offset) < 0 or min(window_width, window_height) <= 0:
        raise ValueError("Window offsets must be non-negative and dimensions positive")
    if x_offset + window_width > width or y_offset + window_height > height:
        raise ValueError("Requested window extends beyond the source raster")
    return Window(x_offset, y_offset, window_width, window_height)


def _profile(source: rasterio.DatasetReader, window: Window) -> dict[str, Any]:
    profile = source.profile.copy()
    width = int(window.width)
    height = int(window.height)
    profile.update(
        width=width,
        height=height,
        transform=transform(window, source.transform),
        compress="deflate",
        BIGTIFF="IF_SAFER",
    )
    if width >= 16 and height >= 16:
        profile.update(
            tiled=True,
            blockxsize=max(16, min(256, width) // 16 * 16),
            blockysize=max(16, min(256, height) // 16 * 16),
        )
    else:
        profile.update(tiled=False)
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
    return profile


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


def _write_preview(values: np.ndarray, output: Path, scene_name: str) -> dict[str, Any]:
    valid = np.all(np.isfinite(values[:4]), axis=0) & np.any(values[:4] != 0, axis=0)
    rgb = _stretch(values[[2, 1, 0]].astype(np.float32), valid)
    false_color = _stretch(values[[3, 2, 1]].astype(np.float32), valid)
    denominator = values[3].astype(np.float32) + values[2]
    ndvi = np.divide(
        values[3] - values[2],
        denominator,
        out=np.zeros(values.shape[1:], dtype=np.float32),
        where=valid & (np.abs(denominator) > 1e-8),
    )
    rgb[~valid] = 0
    false_color[~valid] = 0
    ndvi_display = np.ma.masked_where(~valid, ndvi)
    figure, axes = plt.subplots(1, 3, figsize=(18, 6))
    panels = (
        (rgb, "True color · Red/Green/Blue"),
        (false_color, "Vegetation false color · NIR/Red/Green"),
    )
    for axis, (image, title) in zip(axes[:2], panels, strict=True):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    ndvi_image = axes[2].imshow(ndvi_display, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
    axes[2].set_title("NDVI review layer")
    axes[2].axis("off")
    figure.colorbar(ndvi_image, ax=axes[2], fraction=0.046, pad=0.04)
    figure.suptitle(
        f"Balkan-1 staged proof input · {scene_name}\n"
        "Independent display stretch; this is not model output",
        fontsize=16,
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, facecolor="white")
    plt.close(figure)
    samples = ndvi[valid]
    return {
        "path": str(output),
        "sha256": _sha256(output),
        "ndvi_valid_pixel_median": float(np.median(samples)) if samples.size else None,
        "ndvi_valid_pixel_p90": float(np.percentile(samples, 90)) if samples.size else None,
        "interpretation": "visual selection aid only; not a crop-model prediction",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a bounded, georeferenced Balkan-1 chip for local or target execution "
            "testing. The full source is never copied into payload/."
        )
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument(
        "--window",
        nargs=4,
        type=int,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
    )
    parser.add_argument("--band-order", nargs="+", default=list(DEFAULT_BAND_ORDER))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_path = args.source.resolve()
    if not source_path.is_file():
        parser.error(f"Source GeoTIFF does not exist: {source_path}")
    if args.size <= 0:
        parser.error("--size must be positive")
    output = (
        args.output.resolve()
        if args.output
        else (DEFAULT_INPUT_ROOT / f"{source_path.stem}_sample.tif").resolve()
    )
    if output == source_path:
        parser.error("Output must differ from the source")
    if output.exists() and not args.overwrite:
        parser.error(f"Output already exists; pass --overwrite to replace it: {output}")
    if output.is_relative_to((REPOSITORY_ROOT / "payload").resolve()):
        parser.error("Proof inputs must not be written below payload/")

    with rasterio.open(source_path) as source:
        if source.driver != "GTiff":
            parser.error(f"Expected a GeoTIFF, found {source.driver}")
        if source.crs is None:
            parser.error("Source is not georeferenced: CRS is missing")
        if len(args.band_order) != source.count:
            parser.error(
                "--band-order must contain exactly one label per source band: "
                f"expected {source.count}, received {len(args.band_order)}"
            )
        try:
            selected = _window(source.width, source.height, args.window, args.size)
        except ValueError as error:
            parser.error(str(error))
        values = source.read(window=selected)
        output.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output, "w", **_profile(source, selected)) as destination:
            destination.write(values)
            destination.update_tags(**source.tags())
            for index, description in enumerate(args.band_order, start=1):
                destination.set_band_description(index, description)
                destination.update_tags(index, **source.tags(index))

        source_record = {
            "path": str(source_path),
            "bytes": source_path.stat().st_size,
            "width": source.width,
            "height": source.height,
            "band_count": source.count,
        }

    preview_path = output.with_suffix(".preview.png")
    preview = _write_preview(values, preview_path, output.stem)
    manifest = {
        "schema_version": 1,
        "purpose": "Balkan-1 execution and acceleration proof input",
        "scientific_status": "unvalidated_sensor_transfer",
        "source": source_record,
        "window": {
            "x": int(selected.col_off),
            "y": int(selected.row_off),
            "width": int(selected.width),
            "height": int(selected.height),
        },
        "band_order": args.band_order,
        "output": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": _sha256(output),
            "preview": preview,
        },
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

"""Create one bounded Balkan-1 proof chip outside the payload source tree."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import rasterio
from rasterio.windows import Window, transform

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
        },
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

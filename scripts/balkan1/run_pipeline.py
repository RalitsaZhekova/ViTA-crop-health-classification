"""Hand a preprocessed Balkan-1 GeoTIFF to the payload science pipeline."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BAND_ORDER = ("BLUE", "GREEN", "RED", "NIR", "PAN")
STAGE_ORDER = ("intake", "cloud", "crop", "condition", "downlink")


def _default_scene_id(path: Path) -> str:
    return re.sub(r"(?i)_L1ORT(?:_sample)?$", "", path.stem)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the stage-gated payload pipeline on a preprocessed Balkan-1 product. "
            "Inputs remain outside payload/."
        )
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--acquired-at")
    parser.add_argument("--scene-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--band-order", nargs="+", default=list(DEFAULT_BAND_ORDER))
    parser.add_argument("--reflectance-scale", type=float)
    parser.add_argument("--stop-after", choices=STAGE_ORDER, default="cloud")
    parser.add_argument("--region-id")
    parser.add_argument("--max-crop-cloud-percentage", type=float, default=60.0)
    parser.add_argument("--condition-tile-size", type=int, default=512)
    parser.add_argument(
        "--allow-provisional-crop",
        action="store_true",
        help=(
            "Explicitly transfer Balkan NIR into the narrow-NIR crop input for software "
            "execution testing; this is not accuracy evidence"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_path = args.input.resolve()
    if not input_path.is_file():
        parser.error(f"Input GeoTIFF does not exist: {input_path}")
    if len(args.band_order) != 5:
        parser.error("Balkan-1 --band-order must contain exactly five labels")
    requested_crop = STAGE_ORDER.index(args.stop_after) >= STAGE_ORDER.index("crop")
    if requested_crop and not args.allow_provisional_crop:
        parser.error(
            "Balkan-1 crop transfer is unvalidated. Pass --allow-provisional-crop only "
            "for an explicitly labelled software execution test."
        )
    if requested_crop and not args.acquired_at:
        parser.error("--acquired-at is required for crop inference")
    if args.reflectance_scale is not None and args.reflectance_scale <= 0:
        parser.error("--reflectance-scale must be positive")

    scene_id = args.scene_id or _default_scene_id(input_path)
    output = (
        args.output.resolve()
        if args.output
        else (REPOSITORY_ROOT / "testing" / "runs" / f"balkan1_{scene_id}").resolve()
    )
    if output.is_relative_to((REPOSITORY_ROOT / "payload").resolve()):
        parser.error("Runtime products must not be written below payload/")
    result_path = output / "result.json"
    if output.is_file():
        parser.error(f"Output path is a file: {output}")
    if output.is_dir() and any(output.iterdir()) and not args.overwrite:
        parser.error(f"Run directory is not empty; pass --overwrite to reuse it: {output}")

    command = [
        sys.executable,
        "-m",
        "prithvi_payload.pipeline",
        str(input_path),
        "--sensor",
        "balkan-1",
        "--output",
        str(output),
        "--scene-id",
        scene_id,
        "--stop-after",
        args.stop_after,
        "--band-order",
        *args.band_order,
        "--max-crop-cloud-percentage",
        str(args.max_crop_cloud_percentage),
        "--condition-tile-size",
        str(args.condition_tile_size),
    ]
    if args.acquired_at:
        command.extend(("--acquired-at", args.acquired_at))
    if args.region_id:
        command.extend(("--region-id", args.region_id))
    if args.reflectance_scale is not None:
        command.extend(("--reflectance-scale", str(args.reflectance_scale)))
    if args.allow_provisional_crop:
        command.append("--allow-provisional-balkan-crop")
    if args.overwrite:
        command.append("--overwrite")

    environment = os.environ.copy()
    source_roots = [REPOSITORY_ROOT / "payload" / "src", REPOSITORY_ROOT / "shared" / "src"]
    existing_pythonpath = environment.get("PYTHONPATH")
    python_paths = [str(path) for path in source_roots]
    if existing_pythonpath:
        python_paths.append(existing_pythonpath)
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    environment["NO_ALBUMENTATIONS_UPDATE"] = "1"
    matplotlib_cache = REPOSITORY_ROOT / "outputs" / "cache" / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    environment["MPLCONFIGDIR"] = str(matplotlib_cache)

    print(subprocess.list2cmdline(command))
    completed = subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    result = json.loads(result_path.read_text(encoding="utf-8"))
    evidence = {
        "result": str(result_path),
        "status": result["status"],
        "scientific_status": (
            "unvalidated_balkan_sensor_transfer"
            if args.allow_provisional_crop
            else "provisional_cloud_transfer"
        ),
        "cloud": result.get("summary", {}).get("cloud"),
        "crop": result.get("summary", {}).get("crop"),
    }
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()

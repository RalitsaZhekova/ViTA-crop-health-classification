"""Run the real minimum L1A processor sequentially over selected Balkan-1 scenes."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "data" / "balkan1"
PROCESSOR = Path(__file__).with_name("process_l1a.py")


def _available_scenes(raw_root: Path) -> list[str]:
    return sorted(
        path.name
        for path in raw_root.iterdir()
        if path.is_dir()
        and (path / f"{path.name}_Raw.tif").is_file()
        and ((path / f"{path.name}.json").is_file() or any(path.glob("log_extract*.txt")))
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build real minimum L1A products sequentially; no mock calibration is used."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--scene-id", action="append", dest="scene_ids")
    selection.add_argument("--all", action="store_true", dest="all_scenes")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    raw_root = data_root / "raw"
    if not raw_root.is_dir():
        parser.error(f"raw root does not exist: {raw_root}")
    available = _available_scenes(raw_root)
    scene_ids = available if args.all_scenes else args.scene_ids
    assert scene_ids is not None
    unknown = sorted(set(scene_ids) - set(available))
    if unknown:
        parser.error(f"scenes are missing a raw TIFF or metadata JSON: {unknown}")
    output_root = (args.output_root or data_root / "derived" / "l1a").resolve()

    completed: list[str] = []
    skipped: list[str] = []
    for scene_id in scene_ids:
        output = output_root / f"{scene_id}_L1A_MIN.tif"
        if output.is_file() and not args.overwrite and not args.validate_only:
            print(f"SKIP existing L1A: {output}", flush=True)
            skipped.append(scene_id)
            continue
        command = [
            str(args.python.resolve()),
            str(PROCESSOR.resolve()),
            scene_id,
            "--data-root",
            str(data_root),
            "--output",
            str(output),
        ]
        if args.validate_only:
            command.append("--validate-only")
        if args.overwrite:
            command.append("--overwrite")
        print(subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
        completed.append(scene_id)

    print(
        f"COLLECTION COMPLETE: processed={len(completed)} skipped={len(skipped)} "
        f"selected={len(scene_ids)}"
    )


if __name__ == "__main__":
    main()

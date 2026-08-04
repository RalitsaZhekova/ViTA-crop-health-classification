"""Run a user-supplied Balkan-1 preprocessor over explicit scene folders."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "data" / "balkan1"
DEFAULT_ARGUMENTS = ("--input", "{scene_dir}", "--output", "{output}")


def _scene_ids(raw_root: Path, requested: list[str] | None, all_scenes: bool) -> list[str]:
    if requested:
        scene_ids = requested
    elif all_scenes:
        scene_ids = sorted(
            path.name
            for path in raw_root.iterdir()
            if path.is_dir() and (path / f"{path.name}_Raw.tif").is_file()
        )
    else:
        raise ValueError("Select at least one --scene-id or pass --all")
    if not scene_ids:
        raise ValueError(f"No acquisition folders were found below {raw_root}")
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("Scene identifiers must be unique")
    return scene_ids


def _output_path(root: Path, template: str, scene_id: str) -> Path:
    try:
        relative = Path(template.format(scene_id=scene_id))
    except (KeyError, ValueError) as error:
        raise ValueError(f"Invalid output template: {error}") from error
    if relative.is_absolute():
        raise ValueError("The output template must be relative to the preprocessed root")
    output = (root / relative).resolve()
    if not output.is_relative_to(root.resolve()):
        raise ValueError("The output template escapes the preprocessed root")
    return output


def _render_arguments(arguments: list[str], values: dict[str, str]) -> list[str]:
    rendered: list[str] = []
    for argument in arguments:
        try:
            rendered.append(argument.format_map(values))
        except KeyError as error:
            raise ValueError(f"Unknown preprocessor placeholder: {error.args[0]}") from error
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a checked-in Python preprocessor for selected Balkan-1 acquisition folders. "
            "Raw inputs remain in place."
        )
    )
    parser.add_argument("script", type=Path, help="Python preprocessing script to execute")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--preprocessed-root", type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--scene-id", action="append", dest="scene_ids")
    selection.add_argument("--all", action="store_true", dest="all_scenes")
    parser.add_argument(
        "--output-template",
        default="{scene_id}_L1ORT.tif",
        help="Relative output name; {scene_id} is available",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    command_line = sys.argv[1:]
    if "--" in command_line:
        delimiter = command_line.index("--")
        launcher_arguments = command_line[:delimiter]
        forwarded = command_line[delimiter + 1 :]
    else:
        launcher_arguments = command_line
        forwarded = []
    args = parser.parse_args(launcher_arguments)

    script = args.script.resolve()
    if not script.is_file() or script.suffix.lower() != ".py":
        parser.error(f"Preprocessor must be an existing .py file: {script}")
    python = args.python.resolve()
    if not python.is_file():
        parser.error(f"Python executable does not exist: {python}")

    data_root = args.data_root.resolve()
    raw_root = (args.raw_root or data_root / "raw").resolve()
    preprocessed_root = (args.preprocessed_root or data_root / "preprocessed").resolve()
    if not raw_root.is_dir():
        parser.error(f"Raw root does not exist: {raw_root}")
    preprocessed_root.mkdir(parents=True, exist_ok=True)

    try:
        scene_ids = _scene_ids(raw_root, args.scene_ids, args.all_scenes)
    except ValueError as error:
        parser.error(str(error))
    if not forwarded:
        forwarded = list(DEFAULT_ARGUMENTS)

    completed: list[Path] = []
    for scene_id in scene_ids:
        scene_dir = (raw_root / scene_id).resolve()
        if not scene_dir.is_dir():
            parser.error(f"Scene folder does not exist: {scene_dir}")
        try:
            output = _output_path(preprocessed_root, args.output_template, scene_id)
        except ValueError as error:
            parser.error(str(error))
        if output.exists() and not args.overwrite:
            print(f"SKIP existing output: {output}")
            completed.append(output)
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        values = {
            "scene_id": scene_id,
            "scene_dir": str(scene_dir),
            "raw_root": str(raw_root),
            "preprocessed_root": str(preprocessed_root),
            "output": str(output),
        }
        try:
            preprocessor_arguments = _render_arguments(forwarded, values)
        except ValueError as error:
            parser.error(str(error))
        command = [str(python), str(script), *preprocessor_arguments]
        print(subprocess.list2cmdline(command))
        if args.dry_run:
            continue
        subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
        if not output.is_file():
            raise RuntimeError(f"Preprocessor exited successfully but did not create {output}")
        completed.append(output)

    if args.dry_run:
        print(f"DRY RUN: {len(scene_ids)} scene(s) selected")
    else:
        print(f"PREPROCESSING COMPLETE: {len(completed)} product(s) available")


if __name__ == "__main__":
    main()

"""Validated, automatically resumable TerraTorch training launcher."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from prithvi_crop.preflight import run_preflight
from prithvi_crop.runtime import load_config


def _configured_last_checkpoint(checkpoint_dir: Path) -> Path | None:
    checkpoint = checkpoint_dir / "last.ckpt"
    return checkpoint if checkpoint.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run preflight and launch resumable TerraTorch training."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/prithvi_4band_head_only.yaml"),
    )
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=Path("outputs/dataset_validation.json"),
    )
    parser.add_argument(
        "--resume",
        default="auto",
        help="'auto', 'none', or an explicit checkpoint path.",
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    args, passthrough = parser.parse_known_args()

    run_preflight(args.config, args.validation_report)
    config = load_config(args.config)
    checkpoint_callbacks = [
        callback
        for callback in config["trainer"]["callbacks"]
        if callback["class_path"].endswith("ModelCheckpoint")
    ]
    checkpoint_dir = Path(checkpoint_callbacks[0]["init_args"]["dirpath"])
    command = [
        sys.executable,
        "-m",
        "terratorch",
        "fit",
        "--config",
        str(args.config),
    ]
    if args.batch_size is not None:
        command.append(f"--data.init_args.batch_size={args.batch_size}")
    if args.num_workers is not None:
        command.append(f"--data.init_args.num_workers={args.num_workers}")

    checkpoint: Path | None
    if args.resume == "auto":
        checkpoint = _configured_last_checkpoint(checkpoint_dir)
    elif args.resume.lower() == "none":
        checkpoint = None
    else:
        checkpoint = Path(args.resume)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
    if checkpoint is not None:
        print(f"Resuming from checkpoint: {checkpoint.resolve()}")
        command.extend(["--ckpt_path", str(checkpoint)])
    else:
        print("Starting a new training run (no last.ckpt found).")
    command.extend(passthrough)
    print("Launching:", subprocess.list2cmdline(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

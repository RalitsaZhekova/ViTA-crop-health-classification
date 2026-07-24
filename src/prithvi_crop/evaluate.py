"""Held-out evaluation launcher requiring a real checkpoint."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from prithvi_crop.check_config import check_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TerraTorch test on the held-out official validation split."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/prithvi_4band_head_only.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    args, passthrough = parser.parse_known_args()
    check_config(args.config)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    command = [
        sys.executable,
        "-m",
        "terratorch",
        "test",
        "--config",
        str(args.config),
        "--ckpt_path",
        str(args.checkpoint),
        f"--data.init_args.num_workers={args.num_workers}",
        *passthrough,
    ]
    print("Launching held-out evaluation:", subprocess.list2cmdline(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()


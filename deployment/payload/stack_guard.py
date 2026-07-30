"""Fail a payload build if pip replaces the base image's GPU stack."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PROTECTED_DISTRIBUTIONS = (
    "torch",
    "torchvision",
    "opencv-python",
    "opencv-python-headless",
    "cupy",
    "cupy-cuda12x",
)


def snapshot() -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for distribution in PROTECTED_DISTRIBUTIONS:
        try:
            values[distribution] = version(distribution)
        except PackageNotFoundError:
            values[distribution] = None
    if values["torch"] is None:
        raise RuntimeError("The selected payload base image must provide PyTorch")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("snapshot", "compare"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    current = snapshot()
    if args.action == "snapshot":
        args.path.write_text(json.dumps(current, sort_keys=True), encoding="utf-8")
        return
    before = json.loads(args.path.read_text(encoding="utf-8"))
    if current != before:
        changed = sorted(name for name in current if current[name] != before.get(name))
        raise RuntimeError(
            "Payload dependency installation changed protected GPU packages: "
            + ", ".join(changed)
        )


if __name__ == "__main__":
    main()

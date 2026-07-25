"""Download and verify the pinned prototype cloud-classifier checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
from cloudsen12_models import cloudsen12

MODEL_NAME = "dtacs4bands"
EXPECTED_SHA256 = "37205adce72fbbb65a3cfa8f47676c84ebf9b1555a27a3838d584072c954b22d"
DEFAULT_DIRECTORY = Path(__file__).resolve().parents[1] / "models" / "cloudsen12"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=Path,
        default=DEFAULT_DIRECTORY,
        help="Destination directory for dtacs4bands.pt.",
    )
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    cloudsen12.load_model_by_name(
        name=MODEL_NAME,
        weights_folder=str(args.directory),
        device=torch.device("cpu"),
    )
    model_path = args.directory / f"{MODEL_NAME}.pt"
    actual = sha256(model_path)
    if actual != EXPECTED_SHA256:
        raise SystemExit(
            f"Checkpoint hash mismatch: expected {EXPECTED_SHA256}, found {actual}. "
            "The file was preserved for inspection."
        )
    print(f"Verified {model_path} ({actual}).")


if __name__ == "__main__":
    main()

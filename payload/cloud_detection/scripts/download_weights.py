from __future__ import annotations

import hashlib
from pathlib import Path

from cloudsen12_models import cloudsen12

MODEL_NAME = "dtacs4bands"
WEIGHTS_FOLDER = Path(__file__).resolve().parents[2] / "models" / "cloudsen12"
EXPECTED_SHA256 = "37205adce72fbbb65a3cfa8f47676c84ebf9b1555a27a3838d584072c954b22d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    cloudsen12.load_model_by_name(
        name=MODEL_NAME,
        weights_folder=str(WEIGHTS_FOLDER),
    )
    model_path = WEIGHTS_FOLDER / "dtacs4bands.pt"
    actual = sha256(model_path)
    if actual != EXPECTED_SHA256:
        raise SystemExit(
            f"Checkpoint hash mismatch: expected {EXPECTED_SHA256}, found {actual}."
        )
    print(f"Verified {model_path} ({actual}).")

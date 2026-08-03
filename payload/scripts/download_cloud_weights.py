"""Download and verify the pinned OmniCloudMask V4 ensemble."""

from __future__ import annotations

import argparse
from pathlib import Path

from cloud_detection.backend import (
    OMNICLOUDMASK_ENSEMBLE_SHA256,
    OMNICLOUDMASK_MODEL_VERSION,
    ensemble_sha256,
)
from omnicloudmask.download_models import get_models

DEFAULT_DIRECTORY = Path(__file__).resolve().parents[1] / "models" / "omnicloudmask"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=Path,
        default=DEFAULT_DIRECTORY,
        help="Destination directory for the two OmniCloudMask V4 checkpoints.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download both checkpoints again before verification.",
    )
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    get_models(
        force_download=args.force,
        model_dir=args.directory,
        source="hugging_face",
        model_version=OMNICLOUDMASK_MODEL_VERSION,
    )
    actual = ensemble_sha256(args.directory)
    if actual != OMNICLOUDMASK_ENSEMBLE_SHA256:
        raise SystemExit(
            "Ensemble hash mismatch: "
            f"expected {OMNICLOUDMASK_ENSEMBLE_SHA256}, found {actual}. "
            "The files were preserved for inspection."
        )
    print(f"Verified OmniCloudMask V4 in {args.directory} ({actual}).")


if __name__ == "__main__":
    main()

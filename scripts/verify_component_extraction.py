"""Verify that the non-destructive component extraction remains intact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

SOURCE_CHECKPOINT_SHA256 = (
    "49383194174c56b66f3b7fa67254a42135db6ad64172e873c43d65d1afae6a65"
)
PAYLOAD_WEIGHTS_SHA256 = (
    "732e9b32f2a8fa7cbd1285882148d3103302fac90c96236b438849469ac28098"
)
SNAPSHOT_MODULES = (
    "__init__.py",
    "binary.py",
    "calibration.py",
    "check_config.py",
    "constants.py",
    "data.py",
    "download_data.py",
    "europe.py",
    "evaluate.py",
    "evaluate_binary.py",
    "pastis_validation.py",
    "preflight.py",
    "runtime.py",
    "smoke.py",
    "task.py",
    "train.py",
    "transforms.py",
    "validate_data.py",
    "visualize_examples.py",
)
COPIED_CONFIGS = (
    "crop_binary_calibration.yaml",
    "prithvi_4band_augmented_refine.yaml",
    "prithvi_4band_europe_replay.yaml",
    "prithvi_4band_head_only.yaml",
)
COPIED_SCRIPTS = (
    "prepare_pastis.py",
    "run_augmented_training.ps1",
    "run_europe_replay_pipeline.ps1",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_equal(source: Path, copy: Path) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(source)
    if not copy.is_file():
        raise FileNotFoundError(copy)
    source_digest = sha256(source)
    copy_digest = sha256(copy)
    if source_digest != copy_digest:
        raise RuntimeError(f"Extraction copy differs from source: {copy}")
    return {
        "source": str(source),
        "copy": str(copy),
        "bytes": source.stat().st_size,
        "sha256": source_digest,
    }


def verify(workspace: Path) -> dict[str, Any]:
    workspace = workspace.resolve()
    compared: list[dict[str, Any]] = []
    for name in SNAPSHOT_MODULES:
        compared.append(
            _verify_equal(
                workspace / "src" / "prithvi_crop" / name,
                workspace
                / "training"
                / "source_snapshot"
                / "prithvi_crop"
                / name,
            )
        )
    for name in COPIED_CONFIGS:
        compared.append(
            _verify_equal(
                workspace / "configs" / name,
                workspace / "training" / "configs" / name,
            )
        )
    for name in COPIED_SCRIPTS:
        compared.append(
            _verify_equal(
                workspace / "scripts" / name,
                workspace / "training" / "scripts" / name,
            )
        )

    source_model = (
        workspace
        / "outputs"
        / "prithvi_4band_europe_replay"
        / "checkpoints"
        / "epoch=02-macro_f1=0.5062.ckpt"
    )
    provenance_model = (
        workspace
        / "payload"
        / "models"
        / "prithvi_crop_classifier_epoch02_f1_0.5062.ckpt"
    )
    provenance_comparison = _verify_equal(source_model, provenance_model)
    if provenance_comparison["sha256"] != SOURCE_CHECKPOINT_SHA256:
        raise RuntimeError("Both model copies differ from the selected model digest")
    payload_model = (
        workspace
        / "payload"
        / "models"
        / "prithvi_crop_classifier_v1_weights.pt"
    )
    payload_digest = sha256(payload_model)
    if payload_digest != PAYLOAD_WEIGHTS_SHA256:
        raise RuntimeError("Payload weights differ from the selected artifact digest")

    manifest = yaml.safe_load(
        (workspace / "payload" / "models" / "selected_model.yaml").read_text(
            encoding="utf-8"
        )
    )
    if manifest["sha256"] != PAYLOAD_WEIGHTS_SHA256:
        raise RuntimeError("Payload model manifest has the wrong digest")
    if manifest["bytes"] != payload_model.stat().st_size:
        raise RuntimeError("Payload model manifest has the wrong byte count")

    forbidden_payload_files = [
        str(path.relative_to(workspace))
        for path in (workspace / "payload").rglob("*")
        if path.is_file()
        and (
            path.suffix.lower() in {".npy", ".tif", ".tgz", ".zip"}
            or path.name in {"train.py", "events.out.tfevents"}
        )
    ]
    if forbidden_payload_files:
        raise RuntimeError(
            f"Training/data artifacts leaked into payload: {forbidden_payload_files}"
        )

    dataset_manifest = yaml.safe_load(
        (workspace / "training" / "datasets" / "manifest.yaml").read_text(
            encoding="utf-8"
        )
    )
    for dataset in dataset_manifest["datasets"].values():
        dataset_path = (
            workspace / "training" / "datasets" / dataset["path"]
        ).resolve()
        if not dataset_path.is_dir():
            raise FileNotFoundError(dataset_path)

    return {
        "status": "ok",
        "mode": "non_destructive_copy",
        "compared_small_files": len(compared),
        "source_checkpoint": provenance_comparison,
        "payload_weights": {
            "path": str(payload_model),
            "bytes": payload_model.stat().st_size,
            "sha256": payload_digest,
        },
        "forbidden_payload_files": forbidden_payload_files,
        "dataset_copies_created": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.workspace)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from prithvi_crop.constants import CLASS_NAMES, MODEL_BANDS, NUM_CLASSES


class ConfigurationError(RuntimeError):
    """Raised when a safety-critical training setting is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigurationError(message)


def _all_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [
            item
            for nested in value.values()
            for item in _all_strings(nested)
        ]
    if isinstance(value, list):
        return [item for nested in value for item in _all_strings(nested)]
    return []


def _must_be_under_outputs(path_value: str, field: str) -> None:
    path = Path(path_value)
    _require(not path.is_absolute(), f"{field} must be repository-relative")
    _require(
        path.parts and path.parts[0] == "outputs",
        f"{field} must be under outputs/: {path}",
    )


def check_config(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _require(isinstance(config, dict), "Configuration root must be a mapping")
    try:
        trainer = config["trainer"]
        data = config["data"]
        data_args = data["init_args"]
        model = config["model"]
        model_args = model["init_args"]
        backbone_args = model_args["model_args"]
    except (KeyError, TypeError) as error:
        raise ConfigurationError(f"Missing required configuration key: {error}") from error

    expected_bands = list(MODEL_BANDS)
    _require(
        data["class_path"] == "prithvi_crop.data.CropTypeDataModule",
        "The leakage-safe CropTypeDataModule must be used",
    )
    _require(data_args.get("bands") == expected_bands, "Unexpected data band order")
    _require(
        backbone_args.get("backbone_bands") == expected_bands,
        "Data/model band order mismatch",
    )
    _require(backbone_args.get("backbone_num_frames") == 3, "Exactly 3 dates are required")
    _require(
        data_args.get("expand_temporal_dimension") is True,
        "Temporal dimension must be explicit",
    )
    _require(data_args.get("use_metadata") is True, "Dates and locations must be loaded")
    _require(
        backbone_args.get("backbone_coords_encoding") == ["time", "location"],
        "Prithvi TL time and location encodings must be enabled",
    )
    _require(model_args.get("freeze_backbone") is True, "Prithvi backbone must be frozen")
    initial_checkpoint = model_args.get("initial_checkpoint")
    if initial_checkpoint is not None:
        _require(
            isinstance(initial_checkpoint, str)
            and initial_checkpoint.endswith(".ckpt"),
            "initial_checkpoint must reference a .ckpt file",
        )
        _must_be_under_outputs(
            initial_checkpoint,
            "model.init_args.initial_checkpoint",
        )
        transforms = data_args.get("train_transform", [])
        mandatory_dihedral = [
            transform
            for transform in transforms
            if transform.get("class_path")
            == "prithvi_crop.transforms.RandomNonIdentityDihedral"
        ]
        _require(
            len(mandatory_dihedral) == 1
            and mandatory_dihedral[0].get("init_args", {}).get("p", 1.0) == 1.0,
            "Checkpoint refinement requires mandatory non-identity augmentation",
        )
    _require(
        model_args.get("freeze_decoder") is False,
        "The randomly initialized downstream decoder must remain trainable",
    )
    _require(
        model_args.get("freeze_head", False) is False,
        "The pixel-classification projection must remain trainable",
    )
    _require(
        backbone_args.get("backbone") == "prithvi_eo_v2_100_tl",
        "Unexpected Prithvi backbone",
    )
    _require(backbone_args.get("backbone_pretrained") is True, "Pretrained weights are required")
    _require(backbone_args.get("num_classes") == NUM_CLASSES, "Expected 13 output classes")
    _require(
        model_args.get("class_names") == list(CLASS_NAMES),
        "Class names/order must match the dataset",
    )
    weights = model_args.get("class_weights")
    _require(
        isinstance(weights, list)
        and len(weights) == NUM_CLASSES
        and all(isinstance(value, (int, float)) and value > 0 for value in weights),
        "class_weights must contain 13 positive values",
    )
    _require(model_args.get("ignore_index") == -1, "Reduced no-data labels must use -1")
    _require(model_args.get("loss") == "ce", "Expected weighted cross-entropy loss")

    _require(trainer.get("accelerator") == "gpu", "Training must explicitly require a GPU")
    _require(trainer.get("devices") == 1, "This single-GPU workflow must use one device")
    _require(
        trainer.get("precision") in {"16-mixed", "bf16-mixed", "32-true"},
        "Unsupported precision mode",
    )
    _require(trainer.get("enable_checkpointing") is True, "Checkpointing must be enabled")
    _must_be_under_outputs(trainer["default_root_dir"], "trainer.default_root_dir")
    _must_be_under_outputs(
        trainer["logger"]["init_args"]["save_dir"],
        "trainer.logger.init_args.save_dir",
    )
    _must_be_under_outputs(
        model_args["evaluation_output_dir"],
        "model.init_args.evaluation_output_dir",
    )

    callbacks = trainer.get("callbacks", [])
    checkpoints = [
        callback
        for callback in callbacks
        if callback.get("class_path", "").endswith("ModelCheckpoint")
    ]
    _require(len(checkpoints) == 1, "Exactly one ModelCheckpoint callback is required")
    checkpoint_args = checkpoints[0].get("init_args", {})
    checkpoint_dir = checkpoint_args.get("dirpath")
    _require(
        isinstance(checkpoint_dir, str),
        "ModelCheckpoint.dirpath must be explicitly configured",
    )
    _must_be_under_outputs(
        checkpoint_dir,
        "ModelCheckpoint.dirpath",
    )
    _require(checkpoint_args.get("save_last") is True, "A resumable last.ckpt is required")
    _require(checkpoint_args.get("save_top_k", 0) > 0, "At least one best checkpoint is required")
    _require(
        checkpoint_args.get("monitor") == "val/Macro_F1"
        and checkpoint_args.get("mode") == "max",
        "Checkpoint selection must maximize validation macro F1",
    )

    all_text = " ".join(_all_strings(config)).lower()
    forbidden = ("synthetic_swir", "fake_swir", "generated_swir", "imputed_swir")
    _require(
        not any(term in all_text for term in forbidden),
        "Configuration references a forbidden synthetic/imputed SWIR channel",
    )
    _require(
        "swir_1" not in [band.lower() for band in data_args["bands"]]
        and "swir_2" not in [band.lower() for band in data_args["bands"]],
        "SWIR bands are not permitted as model inputs",
    )

    validation_fraction = data_args.get("validation_fraction")
    _require(
        validation_fraction == 0.1,
        "The validated deterministic internal validation fraction must be 0.1",
    )
    _require(
        data_args.get("split_seed") == config.get("seed_everything"),
        "Data split seed and global seed must match",
    )
    european_data_root = data_args.get("european_data_root")
    if european_data_root is not None:
        path = Path(european_data_root)
        _require(
            not path.is_absolute() and path.parts and path.parts[0] == "data",
            "european_data_root must be repository-relative under data/",
        )
        european_fraction = data_args.get("european_fraction")
        _require(
            isinstance(european_fraction, (int, float))
            and 0 < european_fraction <= 0.2,
            "European replay must be no more than 20% of training samples",
        )
        folds = data_args.get("european_folds")
        _require(
            isinstance(folds, list)
            and folds
            and set(folds).issubset({1, 2, 3, 4}),
            "European training must reserve PASTIS fold 5",
        )
        _require(
            0 < model_args.get("crop_binary_loss_weight", 0) <= 0.5,
            "European replay requires a conservative binary crop loss",
        )

    summary = {
        "bands": expected_bands,
        "frames": 3,
        "classes": NUM_CLASSES,
        "backbone_frozen": True,
        "initial_checkpoint": initial_checkpoint,
        "european_data_root": european_data_root,
        "downstream_decoder_and_classifier_trainable": True,
        "accelerator": trainer["accelerator"],
        "precision": trainer["precision"],
        "logical_test_policy": "official validation split reserved for test",
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check safety-critical four-band, split, GPU, and freezing settings."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/prithvi_4band_head_only.yaml"),
    )
    args = parser.parse_args()
    summary = check_config(args.config)
    print("Configuration is valid:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

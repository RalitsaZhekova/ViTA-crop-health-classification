from pathlib import Path

import yaml

from prithvi_crop.check_config import check_config
from prithvi_crop.constants import CLASS_NAMES, MODEL_BANDS

CONFIG_PATH = Path("configs/prithvi_4band_head_only.yaml")
REFINE_CONFIG_PATH = Path("configs/prithvi_4band_augmented_refine.yaml")
EUROPE_CONFIG_PATH = Path("configs/prithvi_4band_europe_replay.yaml")


def test_head_only_config_contract() -> None:
    summary = check_config(CONFIG_PATH)
    assert summary["bands"] == list(MODEL_BANDS)
    assert summary["backbone_frozen"] is True
    assert summary["accelerator"] == "gpu"


def test_config_has_no_swir_model_input_or_synthetic_swir() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    data_bands = config["data"]["init_args"]["bands"]
    model = config["model"]["init_args"]
    model_bands = model["model_args"]["backbone_bands"]
    assert data_bands == list(MODEL_BANDS)
    assert model_bands == list(MODEL_BANDS)
    assert not any("SWIR" in band for band in data_bands + model_bands)
    assert "synthetic_swir" not in CONFIG_PATH.read_text(encoding="utf-8").lower()
    assert model["class_names"] == list(CLASS_NAMES)


def test_official_validation_is_reserved_for_test() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["data"]["class_path"] == "prithvi_crop.data.CropTypeDataModule"
    assert config["data"]["init_args"]["validation_fraction"] == 0.1
    assert config["model"]["init_args"]["freeze_backbone"] is True
    checkpoint = next(
        callback
        for callback in config["trainer"]["callbacks"]
        if callback["class_path"].endswith("ModelCheckpoint")
    )
    assert checkpoint["init_args"]["monitor"] == "val/Macro_F1"
    assert checkpoint["init_args"]["save_last"] is True


def test_augmented_refinement_reuses_best_checkpoint_safely() -> None:
    summary = check_config(REFINE_CONFIG_PATH)
    config = yaml.safe_load(REFINE_CONFIG_PATH.read_text(encoding="utf-8"))
    model = config["model"]["init_args"]
    transforms = config["data"]["init_args"]["train_transform"]

    assert summary["backbone_frozen"] is True
    assert model["initial_checkpoint"].endswith(
        "epoch=41-macro_f1=0.5046.ckpt"
    )
    assert model["freeze_decoder"] is False
    assert any(
        transform["class_path"]
        == "prithvi_crop.transforms.RandomNonIdentityDihedral"
        and transform["init_args"]["p"] == 1.0
        for transform in transforms
    )
    assert config["optimizer"]["init_args"]["lr"] < 0.0003


def test_europe_replay_keeps_taxonomy_and_original_checkpoint() -> None:
    summary = check_config(EUROPE_CONFIG_PATH)
    config = yaml.safe_load(EUROPE_CONFIG_PATH.read_text(encoding="utf-8"))
    data = config["data"]["init_args"]
    model = config["model"]["init_args"]

    assert summary["classes"] == len(CLASS_NAMES)
    assert data["european_fraction"] == 0.2
    assert 5 not in data["european_folds"]
    assert model["class_names"] == list(CLASS_NAMES)
    assert model["initial_checkpoint"].endswith(
        "epoch=06-macro_f1=0.5051.ckpt"
    )
    assert model["freeze_backbone"] is True

from pathlib import Path

import yaml

from prithvi_crop.check_config import check_config
from prithvi_crop.constants import CLASS_NAMES, MODEL_BANDS

CONFIG_PATH = Path("configs/prithvi_4band_head_only.yaml")


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


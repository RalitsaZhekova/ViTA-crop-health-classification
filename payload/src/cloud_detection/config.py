from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when the YAML configuration violates the model contract."""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigurationError(f"Configuration not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigurationError("Configuration root must be a mapping.")

    required = {
        "model",
        "input",
        "tiling",
        "classes",
        "postprocessing",
        "decision",
        "output",
    }
    missing = required - set(raw)
    if missing:
        raise ConfigurationError(f"Missing sections: {sorted(missing)}")

    expected_order = ["B08", "B04", "B03", "B02"]
    if raw["model"].get("name") != "dtacs4bands":
        raise ConfigurationError("Only the reviewed dtacs4bands model is supported.")
    if raw["input"].get("band_names") != expected_order:
        raise ConfigurationError(
            f"dtacs4bands requires the exact band order {expected_order}."
        )
    if raw["input"].get("processing_level") != "L1C":
        raise ConfigurationError("dtacs4bands is trained for Sentinel-2 L1C inputs.")
    if float(raw["input"].get("reflectance_scale", 0)) <= 0:
        raise ConfigurationError("Reflectance scale must be positive.")

    expected_classes = {
        "clear": 0,
        "thick_cloud": 1,
        "thin_cloud": 2,
        "cloud_shadow": 3,
    }
    if raw["classes"] != expected_classes:
        raise ConfigurationError(
            f"Class mapping must be exactly {expected_classes}."
        )

    size = int(raw["tiling"]["size"])
    overlap = int(raw["tiling"]["overlap"])
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ConfigurationError("Invalid tile size or overlap.")

    if int(raw["postprocessing"]["dilation_pixels"]) < 0:
        raise ConfigurationError("Dilation pixels cannot be negative.")
    if int(raw["postprocessing"]["minimum_region_pixels"]) < 1:
        raise ConfigurationError("Minimum region pixels must be at least one.")

    low = float(raw["decision"]["process_max_unusable_percentage"])
    high = float(raw["decision"]["reject_min_unusable_percentage"])
    if not 0 <= low < high <= 100:
        raise ConfigurationError(
            "Decision thresholds must satisfy 0 <= process < reject <= 100."
        )

    return raw

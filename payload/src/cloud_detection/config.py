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
    if raw["model"].get("name") != "omnicloudmask_v4":
        raise ConfigurationError("Only the reviewed OmniCloudMask V4 model is supported.")
    if raw["model"].get("backend") != "omnicloudmask":
        raise ConfigurationError("The cloud backend must be omnicloudmask.")
    if str(raw["model"].get("package_version")) != "1.7.1":
        raise ConfigurationError("OmniCloudMask package version must be pinned to 1.7.1.")
    if float(raw["model"].get("model_version", 0)) != 4.0:
        raise ConfigurationError("OmniCloudMask model version must be V4.")
    if raw["model"].get("inference_dtype") != "fp32":
        raise ConfigurationError("The reviewed OmniCloudMask baseline requires FP32.")
    if (
        int(raw["model"].get("patch_size", 0)) != 1000
        or int(raw["model"].get("patch_overlap", -1)) != 300
    ):
        raise ConfigurationError(
            "The reviewed OmniCloudMask baseline requires 1000/300 patch settings."
        )
    if raw["input"].get("band_names") != expected_order:
        raise ConfigurationError(
            f"The standalone four-band adapter requires the exact order {expected_order}."
        )
    if raw["model"].get("model_input_order") != ["RED", "GREEN", "NIR"]:
        raise ConfigurationError("OmniCloudMask requires Red, Green, NIR model input.")
    if raw["input"].get("processing_level") != "MULTISENSOR_REFLECTANCE":
        raise ConfigurationError(
            "OmniCloudMask input must use the reviewed multisensor reflectance contract."
        )
    if float(raw["input"].get("reflectance_scale", 0)) <= 0:
        raise ConfigurationError("Reflectance scale must be positive.")
    if float(raw["input"].get("balkan_target_resolution_m", 0)) != 10.0:
        raise ConfigurationError("Balkan-1 cloud inference must use the validated 10 m grid.")
    if int(raw["input"].get("balkan_max_analysis_pixels", 0)) <= 0:
        raise ConfigurationError("Balkan-1 analysis-grid memory bound must be positive.")

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

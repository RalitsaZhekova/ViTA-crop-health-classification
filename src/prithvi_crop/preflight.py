"""Training preflight that prints and verifies the full run contract."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from typing import Any

import torch

from prithvi_crop.check_config import check_config
from prithvi_crop.constants import CLASS_NAMES
from prithvi_crop.runtime import build_task, load_config


def _hugging_face_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def run_preflight(
    config_path: Path,
    validation_report: Path = Path("outputs/dataset_validation.json"),
    *,
    instantiate_model: bool = True,
) -> dict[str, Any]:
    check_config(config_path)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. This configuration intentionally does not fall back to CPU."
        )
    if not validation_report.is_file():
        raise FileNotFoundError(
            f"Missing exhaustive dataset validation report: {validation_report}. "
            "Run prithvi-validate first."
        )
    report = json.loads(validation_report.read_text(encoding="utf-8"))
    if report.get("complete") is not True:
        raise RuntimeError("The dataset validation report is not marked complete")

    config = load_config(config_path)
    data_args = config["data"]["init_args"]
    trainer = config["trainer"]
    device = torch.cuda.get_device_properties(0)
    split_report = report["logical_model_splits"]
    summary: dict[str, Any] = {
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_gib": round(device.total_memory / 1024**3, 2),
        "compute_capability": ".".join(
            str(value) for value in torch.cuda.get_device_capability(0)
        ),
        "batch_size": data_args["batch_size"],
        "num_workers": data_args["num_workers"],
        "precision": trainer["precision"],
        "gradient_accumulation": trainer.get("accumulate_grad_batches", 1),
        "logical_split_sizes": {
            name: split_report[name]
            for name in ("training", "validation", "test")
        },
        "logical_split_policy": split_report["policy"],
        "cross_split_chip_id_overlap": split_report["cross_split_chip_id_overlap"],
        "cross_split_spatial_overlap": split_report["cross_split_spatial_overlap"],
        "number_of_classes": len(CLASS_NAMES),
        "class_distribution": split_report["class_pixel_counts"]["training"],
        "class_distribution_scope": "logical training split only",
        "output_directory": trainer["default_root_dir"],
        "pretrained_source": (
            "https://huggingface.co/ibm-nasa-geospatial/"
            "Prithvi-EO-2.0-100M-TL"
        ),
        "pretrained_cache": str(_hugging_face_cache()),
    }

    if instantiate_model:
        task = build_task(config)
        trainable = sum(
            parameter.numel() for parameter in task.parameters() if parameter.requires_grad
        )
        frozen = sum(
            parameter.numel() for parameter in task.parameters() if not parameter.requires_grad
        )
        backbone_parameters = list(task.model.encoder.parameters())
        if not backbone_parameters or any(
            parameter.requires_grad for parameter in backbone_parameters
        ):
            raise RuntimeError("The Prithvi backbone is not completely frozen")
        summary["trainable_parameters"] = trainable
        summary["frozen_parameters"] = frozen
        del task

    print("Training preflight passed:")
    for key, value in summary.items():
        if key == "class_distribution":
            print("  class_distribution:")
            for class_name, count in value.items():
                print(f"    {class_name}: {count}")
        else:
            print(f"  {key}: {value}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and print the training run contract.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/prithvi_4band_head_only.yaml"),
    )
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=Path("outputs/dataset_validation.json"),
    )
    parser.add_argument(
        "--without-model",
        action="store_true",
        help="Do not instantiate/download the pretrained model.",
    )
    args = parser.parse_args()
    run_preflight(
        args.config,
        args.validation_report,
        instantiate_model=not args.without_model,
    )


if __name__ == "__main__":
    main()

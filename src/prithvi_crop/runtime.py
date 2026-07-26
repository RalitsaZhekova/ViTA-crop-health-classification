"""Helpers for constructing the configured data module and task outside LightningCLI."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import albumentations as A
import yaml
from albumentations.pytorch import ToTensorV2
from terratorch.datasets.transforms import (
    FlattenTemporalIntoChannels,
    UnflattenTemporalFromChannels,
)

from prithvi_crop.data import CropTypeDataModule
from prithvi_crop.task import CropSegmentationTask
from prithvi_crop.transforms import RandomNonIdentityDihedral


def load_config(config_path: Path) -> dict[str, Any]:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def _build_transform(items: list[dict[str, Any]]) -> A.Compose:
    classes = {
        "FlattenTemporalIntoChannels": FlattenTemporalIntoChannels,
        "terratorch.datasets.transforms.FlattenTemporalIntoChannels": (
            FlattenTemporalIntoChannels
        ),
        "UnflattenTemporalFromChannels": UnflattenTemporalFromChannels,
        "terratorch.datasets.transforms.UnflattenTemporalFromChannels": (
            UnflattenTemporalFromChannels
        ),
        "ToTensorV2": ToTensorV2,
        "albumentations.pytorch.ToTensorV2": ToTensorV2,
        "albumentations.HorizontalFlip": A.HorizontalFlip,
        "albumentations.VerticalFlip": A.VerticalFlip,
        "albumentations.RandomRotate90": A.RandomRotate90,
        "albumentations.Affine": A.Affine,
        "prithvi_crop.transforms.RandomNonIdentityDihedral": (
            RandomNonIdentityDihedral
        ),
    }
    transforms = []
    for item in items:
        class_path = item["class_path"]
        try:
            transform_class = classes[class_path]
        except KeyError as error:
            raise ValueError(f"Unsupported runtime transform: {class_path}") from error
        transforms.append(transform_class(**item.get("init_args", {})))
    return A.Compose(transforms, is_check_shapes=False)


def build_data_module(
    config: dict[str, Any],
    *,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> CropTypeDataModule:
    args = deepcopy(config["data"]["init_args"])
    for key in ("train_transform", "val_transform", "test_transform"):
        args[key] = _build_transform(args[key])
    if batch_size is not None:
        args["batch_size"] = batch_size
    if num_workers is not None:
        args["num_workers"] = num_workers
    return CropTypeDataModule(**args)


def build_task(
    config: dict[str, Any],
    *,
    load_initial_weights: bool = True,
) -> CropSegmentationTask:
    args = deepcopy(config["model"]["init_args"])
    if not load_initial_weights:
        # A complete Lightning checkpoint supplies every model tensor. Avoid a
        # redundant Hub lookup and a redundant warm-start checkpoint load when
        # constructing the architecture for evaluation.
        args["model_args"]["backbone_pretrained"] = False
        args["initial_checkpoint"] = None
        args["initial_checkpoint_adapter"] = None
    return CropSegmentationTask(**args)

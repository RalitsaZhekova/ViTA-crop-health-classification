"""Leakage-safe logical train/validation/test splits for the crop dataset."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import torch
from einops import rearrange
from terratorch.datamodules import MultiTemporalCropClassificationDataModule
from terratorch.datasets import MultiTemporalCropClassification
from torch.utils.data import ConcatDataset, DataLoader, Subset

from prithvi_crop.europe import (
    OriginalReplayDataset,
    PastisReplayDataset,
    european_sample_count,
)


def chip_id_from_image(path: str | Path) -> str:
    """Return the chip identifier from a TerraTorch image path."""
    return Path(path).name.removesuffix("_merged.tif")


def deterministic_partition(
    chip_ids: Sequence[str],
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Split indices reproducibly without separating a chip's temporal observations."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    if len(set(chip_ids)) != len(chip_ids):
        raise ValueError("chip_ids must be unique")
    if len(chip_ids) < 2:
        raise ValueError("At least two chips are required")

    ranked = sorted(
        range(len(chip_ids)),
        key=lambda index: hashlib.sha256(f"{seed}:{chip_ids[index]}".encode()).digest(),
    )
    validation_count = max(1, min(len(chip_ids) - 1, round(len(chip_ids) * validation_fraction)))
    validation_indices = sorted(ranked[:validation_count])
    validation_set = set(validation_indices)
    training_indices = [index for index in range(len(chip_ids)) if index not in validation_set]
    return training_indices, validation_indices


def select_temporal_bands(
    image: np.ndarray,
    band_indices: np.ndarray,
    all_band_count: int,
    expand_temporal_dimension: bool,
) -> np.ndarray:
    """Select requested bands independently from every temporal observation."""
    if image.ndim != 3 or image.shape[0] % all_band_count:
        raise ValueError("Expected a [time*bands,height,width] raster with a complete band set")
    temporal = rearrange(
        image,
        "(time channels) height width -> time height width channels",
        channels=all_band_count,
    )
    selected = temporal[..., band_indices]
    if expand_temporal_dimension:
        return selected
    return rearrange(selected, "time height width channels -> height width (time channels)")


class CropTypeDataset(MultiTemporalCropClassification):
    """Fix four-band temporal selection in the TerraTorch 1.2.10 dataset."""

    source_frame_count = 3

    def __init__(self, *args: Any, single_frame: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.single_frame = single_frame

    def __len__(self) -> int:
        base_length = super().__len__()
        return base_length * self.source_frame_count if self.single_frame else base_length

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.single_frame:
            base_index, frame_index = divmod(index, self.source_frame_count)
        else:
            base_index, frame_index = index, None
        image_data = self._load_file(
            self.image_files[base_index],
            nan_replace=self.no_data_replace,
        )

        location_coords, temporal_coords = None, None
        if self.use_metadata:
            location_coords = self._get_coords(image_data)
            metadata_index = self.image_to_metadata_index.get(base_index)
            if metadata_index is not None:
                temporal_coords = self._get_date(self.metadata.iloc[metadata_index])

        image = select_temporal_bands(
            image_data.to_numpy(),
            self.band_indices,
            len(self.all_band_names),
            self.expand_temporal_dimension,
        )
        if self.single_frame:
            if not self.expand_temporal_dimension:
                raise ValueError("Single-frame samples require an explicit temporal dimension")
            if image.shape[0] != self.source_frame_count:
                raise ValueError(
                    f"Expected {self.source_frame_count} source frames, got {image.shape[0]}"
                )
            image = image[frame_index : frame_index + 1]
            if temporal_coords is not None:
                temporal_coords = temporal_coords[frame_index : frame_index + 1]
        output = {
            "image": image.astype(np.float32),
            "mask": self._load_file(
                self.segmentation_mask_files[base_index],
                nan_replace=self.no_label_replace,
            ).to_numpy()[0],
        }
        if self.reduce_zero_label:
            output["mask"] -= 1
        if self.transform:
            output = self.transform(**output)
        output["mask"] = output["mask"].long()

        if self.use_metadata:
            output["location_coords"] = location_coords
            output["temporal_coords"] = temporal_coords
        return output


class CropTypeDataModule(MultiTemporalCropClassificationDataModule):
    """Use official training chips for train/val and official validation chips for test.

    The upstream TerraTorch data module aliases the official validation split as both
    validation and test. That leaks model-selection data into final evaluation. This
    subclass creates a deterministic internal validation subset from the official
    training chips and reserves every official validation chip for testing.
    """

    def __init__(
        self,
        data_root: str,
        batch_size: int = 4,
        num_workers: int = 0,
        bands: Sequence[str] = (
            "BLUE",
            "GREEN",
            "RED",
            "NIR_NARROW",
        ),
        train_transform: list[Any] | None = None,
        val_transform: list[Any] | None = None,
        test_transform: list[Any] | None = None,
        predict_transform: A.Compose | None | list = None,
        drop_last: bool = True,
        no_data_replace: float | None = 0,
        no_label_replace: int | None = -1,
        expand_temporal_dimension: bool = True,
        reduce_zero_label: bool = True,
        use_metadata: bool = False,
        metadata_file_name: str = "chips_df.csv",
        validation_fraction: float = 0.1,
        split_seed: int = 42,
        european_data_root: str | None = None,
        european_fraction: float = 0.0,
        european_folds: Sequence[int] = (1, 2, 3, 4),
        single_frame: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            bands=bands,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            drop_last=drop_last,
            no_data_replace=no_data_replace,
            no_label_replace=no_label_replace,
            expand_temporal_dimension=expand_temporal_dimension,
            reduce_zero_label=reduce_zero_label,
            use_metadata=use_metadata,
            metadata_file_name=metadata_file_name,
            **kwargs,
        )
        self.dataset_class = CropTypeDataset
        if not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be strictly between 0 and 1")
        self.validation_fraction = validation_fraction
        self.split_seed = split_seed
        if european_data_root is None and european_fraction != 0:
            raise ValueError("european_fraction requires european_data_root")
        if european_data_root is not None and not 0 < european_fraction < 1:
            raise ValueError("european_fraction must be strictly between 0 and 1")
        self.european_data_root = european_data_root
        self.european_fraction = european_fraction
        self.european_folds = tuple(european_folds)
        self.single_frame = single_frame

    def _make_dataset(self, split: str, transform: A.Compose | None):
        return self.dataset_class(
            split=split,
            data_root=self.data_root,
            transform=transform,
            bands=self.bands,
            no_data_replace=self.no_data_replace,
            no_label_replace=self.no_label_replace,
            expand_temporal_dimension=self.expand_temporal_dimension,
            reduce_zero_label=self.reduce_zero_label,
            use_metadata=self.use_metadata,
            metadata_file_name=self.metadata_file_name,
            single_frame=self.single_frame,
        )

    def _expand_base_indices(self, indices: Sequence[int]) -> list[int]:
        if not self.single_frame:
            return list(indices)
        return [
            index * self.dataset_class.source_frame_count + frame_index
            for index in indices
            for frame_index in range(self.dataset_class.source_frame_count)
        ]

    def _training_partition(self) -> tuple[list[int], list[int]]:
        dataset = self._make_dataset("train", self.val_transform)
        chip_ids = [chip_id_from_image(path) for path in dataset.image_files]
        train_indices, validation_indices = deterministic_partition(
            chip_ids,
            self.validation_fraction,
            self.split_seed,
        )
        return (
            self._expand_base_indices(train_indices),
            self._expand_base_indices(validation_indices),
        )

    def _dataloader_factory(self, split: str) -> DataLoader:
        """Keep Windows workers alive so the GPU is not starved every epoch."""
        dataset = self._valid_attribute(f"{split}_dataset", "dataset")
        batch_size = self._valid_attribute(f"{split}_batch_size", "batch_size")
        worker_options: dict[str, Any] = {}
        if self.num_workers > 0:
            worker_options.update(
                persistent_workers=True,
                prefetch_factor=2,
            )
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            drop_last=split == "train" and self.drop_last,
            pin_memory=torch.cuda.is_available(),
            **worker_options,
        )

    def setup(self, stage: str) -> None:
        if stage == "fit":
            train_base = self._make_dataset("train", self.train_transform)
            val_base = self._make_dataset("train", self.val_transform)
            chip_ids = [chip_id_from_image(path) for path in train_base.image_files]
            train_base_indices, val_base_indices = deterministic_partition(
                chip_ids,
                self.validation_fraction,
                self.split_seed,
            )
            train_indices = self._expand_base_indices(train_base_indices)
            val_indices = self._expand_base_indices(val_base_indices)
            original_train = Subset(train_base, train_indices)
            if self.european_data_root is None:
                self.train_dataset = original_train
            else:
                europe_count = european_sample_count(
                    len(original_train),
                    self.european_fraction,
                )
                source_frame_count = self.dataset_class.source_frame_count
                european_patch_count = (
                    math.ceil(europe_count / source_frame_count)
                    if self.single_frame
                    else europe_count
                )
                european_base = PastisReplayDataset(
                    self.european_data_root,
                    folds=self.european_folds,
                    max_samples=european_patch_count,
                    seed=self.split_seed,
                    augment=True,
                    single_frame=self.single_frame,
                )
                european_train = (
                    Subset(european_base, range(europe_count))
                    if self.single_frame
                    else european_base
                )
                self.train_dataset = ConcatDataset(
                    [
                        OriginalReplayDataset(original_train),
                        european_train,
                    ]
                )
            self.val_dataset = Subset(val_base, val_indices)
        elif stage == "validate":
            val_base = self._make_dataset("train", self.val_transform)
            chip_ids = [chip_id_from_image(path) for path in val_base.image_files]
            _, val_base_indices = deterministic_partition(
                chip_ids,
                self.validation_fraction,
                self.split_seed,
            )
            val_indices = self._expand_base_indices(val_base_indices)
            self.val_dataset = Subset(val_base, val_indices)
        elif stage == "test":
            self.test_dataset = self._make_dataset("val", self.test_transform)
        elif stage == "predict":
            self.predict_dataset = self._make_dataset("val", self.predict_transform)
        else:
            raise ValueError(f"Unsupported setup stage: {stage}")

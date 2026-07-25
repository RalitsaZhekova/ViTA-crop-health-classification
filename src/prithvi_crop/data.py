"""Leakage-safe logical train/validation/test splits for the crop dataset."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
from einops import rearrange
from terratorch.datamodules import MultiTemporalCropClassificationDataModule
from terratorch.datasets import MultiTemporalCropClassification
from torch.utils.data import ConcatDataset, Subset

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
        key=lambda index: hashlib.sha256(
            f"{seed}:{chip_ids[index]}".encode()
        ).digest(),
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
        raise ValueError(
            "Expected a [time*bands,height,width] raster with a complete band set"
        )
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

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_data = self._load_file(
            self.image_files[index],
            nan_replace=self.no_data_replace,
        )

        location_coords, temporal_coords = None, None
        if self.use_metadata:
            location_coords = self._get_coords(image_data)
            metadata_index = self.image_to_metadata_index.get(index)
            if metadata_index is not None:
                temporal_coords = self._get_date(self.metadata.iloc[metadata_index])

        image = select_temporal_bands(
            image_data.to_numpy(),
            self.band_indices,
            len(self.all_band_names),
            self.expand_temporal_dimension,
        )
        output = {
            "image": image.astype(np.float32),
            "mask": self._load_file(
                self.segmentation_mask_files[index],
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
            raise ValueError(
                "european_fraction must be strictly between 0 and 1"
            )
        self.european_data_root = european_data_root
        self.european_fraction = european_fraction
        self.european_folds = tuple(european_folds)

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
        )

    def _training_partition(self) -> tuple[list[int], list[int]]:
        dataset = self._make_dataset("train", self.val_transform)
        chip_ids = [chip_id_from_image(path) for path in dataset.image_files]
        return deterministic_partition(chip_ids, self.validation_fraction, self.split_seed)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            train_base = self._make_dataset("train", self.train_transform)
            val_base = self._make_dataset("train", self.val_transform)
            chip_ids = [chip_id_from_image(path) for path in train_base.image_files]
            train_indices, val_indices = deterministic_partition(
                chip_ids,
                self.validation_fraction,
                self.split_seed,
            )
            original_train = Subset(train_base, train_indices)
            if self.european_data_root is None:
                self.train_dataset = original_train
            else:
                europe_count = european_sample_count(
                    len(original_train),
                    self.european_fraction,
                )
                european_train = PastisReplayDataset(
                    self.european_data_root,
                    folds=self.european_folds,
                    max_samples=europe_count,
                    seed=self.split_seed,
                    augment=True,
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
            _, val_indices = deterministic_partition(
                chip_ids,
                self.validation_fraction,
                self.split_seed,
            )
            self.val_dataset = Subset(val_base, val_indices)
        elif stage == "test":
            self.test_dataset = self._make_dataset("val", self.test_transform)
        elif stage == "predict":
            self.predict_dataset = self._make_dataset("val", self.predict_transform)
        else:
            raise ValueError(f"Unsupported setup stage: {stage}")

"""PASTIS-to-Prithvi adapter with conservative label harmonization."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

# Only unambiguous mappings receive fine-grained labels. Other agricultural
# labels remain useful through binary crop/non-crop supervision.
PASTIS_FINE_CLASS_MAP = {
    2: 7,   # soft winter wheat -> Winter Wheat
    3: 2,   # corn -> Corn
    11: 7,  # winter durum wheat -> Winter Wheat
    15: 3,  # soybeans -> Soybeans
    18: 11, # sorghum -> Sorghum
}

PROJECT_NON_CROP_CLASSES = (0, 1, 4, 5, 6)
PROJECT_CROP_CLASSES = (2, 3, 7, 8, 9, 10, 11, 12)
PROJECT_BINARY_IGNORE_CLASSES = (12,)
PASTIS_BAND_INDICES = (0, 1, 2, 7)  # B02, B03, B04, B8A


def resolve_pastis_root(root: str | Path) -> Path:
    """Resolve the extracted optical PASTIS directory."""
    root = Path(root)
    for candidate in (root, root / "PASTIS"):
        if (
            (candidate / "metadata.geojson").is_file()
            and (candidate / "DATA_S2").is_dir()
            and (candidate / "ANNOTATIONS").is_dir()
        ):
            return candidate
    raise FileNotFoundError(
        f"PASTIS optical data not found under {root}; run the preparation pipeline"
    )


def european_sample_count(original_count: int, fraction: float) -> int:
    """Return the replay count needed for the requested total-data fraction."""
    if original_count <= 0:
        raise ValueError("original_count must be positive")
    if not 0 < fraction < 1:
        raise ValueError("fraction must be strictly between 0 and 1")
    return round(original_count * fraction / (1 - fraction))


def map_pastis_masks(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map PASTIS labels to fine-grained and binary project targets."""
    if mask.ndim != 2:
        raise ValueError("PASTIS mask must be two-dimensional")
    if np.any((mask < 0) | (mask > 19)):
        raise ValueError("PASTIS mask contains an unknown class")

    fine_mask = np.full(mask.shape, -1, dtype=np.int64)
    for source, destination in PASTIS_FINE_CLASS_MAP.items():
        fine_mask[mask == source] = destination

    crop_mask = np.full(mask.shape, -1, dtype=np.int64)
    crop_mask[mask == 0] = 0
    crop_mask[(mask >= 1) & (mask <= 18)] = 1
    return fine_mask, crop_mask


def select_seasonal_dates(
    raw_dates: Sequence[int | str],
    target_year: int = 2019,
    target_doys: Sequence[int] = (64, 199, 269),
) -> tuple[list[int], Tensor]:
    """Select observations nearest the original dataset's seasonal dates."""
    if len(raw_dates) < len(target_doys):
        raise ValueError("Not enough PASTIS observations for three seasonal dates")
    parsed = [
        date(int(str(value)[:4]), int(str(value)[4:6]), int(str(value)[6:8]))
        for value in raw_dates
    ]
    targets = [
        date(target_year, 1, 1) + timedelta(days=int(day_of_year) - 1)
        for day_of_year in target_doys
    ]
    indices = [
        min(
            range(len(parsed)),
            key=lambda index: abs((parsed[index] - target).days),
        )
        for target in targets
    ]
    selected = [parsed[index] for index in indices]
    temporal_coords = torch.tensor(
        [[value.year, value.timetuple().tm_yday] for value in selected],
        dtype=torch.float32,
    )
    return indices, temporal_coords


def _apply_dihedral(
    image: Tensor,
    fine_mask: Tensor,
    crop_mask: Tensor,
    transform_id: int,
) -> tuple[Tensor, Tensor, Tensor]:
    if transform_id not in range(1, 8):
        raise ValueError("transform_id must be from 1 through 7")

    def operation(value: Tensor) -> Tensor:
        if transform_id <= 3:
            return torch.rot90(value, transform_id, (-2, -1))
        if transform_id == 4:
            return torch.flip(value, (-1,))
        if transform_id == 5:
            return torch.flip(value, (-2,))
        transposed = value.transpose(-2, -1)
        if transform_id == 6:
            return transposed
        return torch.rot90(transposed, 2, (-2, -1))

    return operation(image), operation(fine_mask), operation(crop_mask)


class OriginalReplayDataset(Dataset):
    """Add binary crop supervision to unchanged original samples."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.dataset[index])
        mask = sample["mask"]
        crop_mask = torch.full_like(mask, -1)
        for class_index in PROJECT_NON_CROP_CLASSES:
            crop_mask[mask == class_index] = 0
        for class_index in PROJECT_CROP_CLASSES:
            if class_index not in PROJECT_BINARY_IGNORE_CLASSES:
                crop_mask[mask == class_index] = 1
        sample["crop_mask"] = crop_mask
        sample["dataset_source"] = torch.tensor(0, dtype=torch.long)
        return sample


class PastisReplayDataset(Dataset):
    """Return conservative PASTIS supervision in the current model's format."""

    def __init__(
        self,
        root: str | Path,
        folds: Sequence[int] = (1, 2, 3, 4),
        max_samples: int | None = None,
        seed: int = 42,
        output_size: int = 224,
        augment: bool = True,
    ) -> None:
        self.root = resolve_pastis_root(root)
        self.output_size = output_size
        self.augment = augment
        metadata = gpd.read_file(self.root / "metadata.geojson")
        metadata["Fold"] = metadata["Fold"].astype(int)
        metadata = metadata[metadata["Fold"].isin(folds)].copy()
        metadata["_rank"] = metadata["ID_PATCH"].map(
            lambda value: hashlib.sha256(
                f"{seed}:{int(value)}".encode()
            ).digest()
        )
        metadata.sort_values("_rank", inplace=True)
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_samples must be positive")
            metadata = metadata.head(max_samples)
        locations = metadata.to_crs(4326).geometry.representative_point()
        metadata["_latitude"] = locations.y
        metadata["_longitude"] = locations.x
        self.metadata = metadata.reset_index(drop=True)
        if self.metadata.empty:
            raise ValueError("The selected PASTIS folds contain no samples")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        row = self.metadata.iloc[index]
        patch_id = int(row["ID_PATCH"])
        image_array = np.load(
            self.root / "DATA_S2" / f"S2_{patch_id}.npy",
            mmap_mode="r",
        )
        raw_dates = row["dates-S2"]
        if isinstance(raw_dates, str):
            raw_dates = json.loads(raw_dates)
        ordered_dates = [
            value
            for _, value in sorted(
                raw_dates.items(),
                key=lambda item: int(item[0]),
            )
        ]
        date_indices, temporal_coords = select_seasonal_dates(ordered_dates)
        image = torch.from_numpy(
            np.asarray(
                image_array[np.asarray(date_indices)][:, PASTIS_BAND_INDICES],
                dtype=np.float32,
            ).copy()
        ).permute(1, 0, 2, 3)

        raw_mask = np.load(
            self.root / "ANNOTATIONS" / f"TARGET_{patch_id}.npy",
            mmap_mode="r",
        )[0]
        fine_array, crop_array = map_pastis_masks(np.asarray(raw_mask))
        fine_mask = torch.from_numpy(fine_array)
        crop_mask = torch.from_numpy(crop_array)

        height, width = fine_mask.shape
        if height > self.output_size or width > self.output_size:
            raise ValueError("PASTIS patch is larger than configured output size")
        vertical = self.output_size - height
        horizontal = self.output_size - width
        padding = (
            horizontal // 2,
            horizontal - horizontal // 2,
            vertical // 2,
            vertical - vertical // 2,
        )
        image = F.pad(image, padding, value=0)
        fine_mask = F.pad(fine_mask, padding, value=-1)
        crop_mask = F.pad(crop_mask, padding, value=-1)

        if self.augment:
            transform_id = int(torch.randint(1, 8, ()).item())
            image, fine_mask, crop_mask = _apply_dihedral(
                image,
                fine_mask,
                crop_mask,
                transform_id,
            )

        return {
            "image": image,
            "mask": fine_mask,
            "crop_mask": crop_mask,
            "location_coords": torch.tensor(
                [row["_latitude"], row["_longitude"]],
                dtype=torch.float32,
            ),
            "temporal_coords": temporal_coords,
            "dataset_source": torch.tensor(1, dtype=torch.long),
        }

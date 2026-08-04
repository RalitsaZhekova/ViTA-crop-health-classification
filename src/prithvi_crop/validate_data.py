from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as transform_coordinates

from prithvi_crop.constants import (
    CLASS_NAMES,
    EXPECTED_CRS,
    EXPECTED_IMAGE_SIZE,
    EXPECTED_TOTAL_CHIPS,
    EXPECTED_TRAINING_CHIPS,
    EXPECTED_VALIDATION_CHIPS,
    MODEL_BANDS,
    NUM_CLASSES,
    RASTER_BAND_ORDER,
)
from prithvi_crop.data import deterministic_partition


@dataclass
class SplitResult:
    split: str
    chips: int
    listed_ids: int
    masked_image_values: int
    invalid_image_values: int
    label_counts: list[int]
    image_minimum: float
    image_maximum: float
    band_descriptions_present: int


def _image_id(path: Path) -> str:
    return path.name.removesuffix("_merged.tif")


def _mask_id(path: Path) -> str:
    return path.name.removesuffix(".mask.tif")


def _read_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    duplicates = [item for item, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise RuntimeError(f"{path} contains duplicate IDs; examples: {sorted(duplicates)[:5]}")
    return ids


def _normalize_band_description(description: str) -> str:
    normalized = description.upper().strip().replace("-", "_").replace(" ", "_")
    aliases = {
        "B02": "BLUE",
        "B2": "BLUE",
        "B03": "GREEN",
        "B3": "GREEN",
        "B04": "RED",
        "B4": "RED",
        "B8A": "NIR_NARROW",
        "NIR": "NIR_NARROW",
        "NARROW_NIR": "NIR_NARROW",
        "B11": "SWIR_1",
        "SWIR1": "SWIR_1",
        "B12": "SWIR_2",
        "SWIR2": "SWIR_2",
    }
    return aliases.get(normalized, normalized)


def _check_band_descriptions(
    descriptions: tuple[str | None, ...],
    path: Path,
) -> bool:
    present = [description for description in descriptions if description]
    if not present:
        return False
    if len(present) != len(RASTER_BAND_ORDER):
        raise RuntimeError(f"Only some raster bands are described: {path}")
    normalized = tuple(_normalize_band_description(item) for item in present)
    if normalized != RASTER_BAND_ORDER:
        raise RuntimeError(f"Unexpected band descriptions/order in {path}: {normalized}")
    return True


def _geometry_record(
    split: str,
    chip_id: str,
    source: rasterio.io.DatasetReader,
) -> tuple[str, str, float, float, float, float]:
    bounds = source.bounds
    center_x = (bounds.left + bounds.right) / 2
    center_y = (bounds.bottom + bounds.top) / 2
    longitude, latitude = transform_coordinates(
        source.crs,
        "EPSG:4326",
        [center_x],
        [center_y],
    )
    if not (
        math.isfinite(longitude[0])
        and math.isfinite(latitude[0])
        and -180 <= longitude[0] <= 180
        and -90 <= latitude[0] <= 90
    ):
        raise RuntimeError(f"Invalid geographic center for {chip_id}")
    return (
        split,
        chip_id,
        bounds.left,
        bounds.bottom,
        bounds.right,
        bounds.top,
    )


def _compute_array_stats(
    image: np.ndarray,
    label: np.ndarray,
    device: str,
) -> tuple[int, float, float, np.ndarray]:
    import torch

    image_tensor = torch.from_numpy(np.ascontiguousarray(image)).to(device)
    finite = torch.isfinite(image_tensor)
    invalid = int((~finite).sum().item())
    if finite.any():
        valid_values = image_tensor[finite]
        minimum = float(valid_values.min().item())
        maximum = float(valid_values.max().item())
    else:
        minimum = math.nan
        maximum = math.nan

    label_tensor = torch.from_numpy(np.ascontiguousarray(label.astype(np.int64))).to(device)
    counts = torch.bincount(label_tensor.flatten(), minlength=NUM_CLASSES + 1).cpu().numpy()
    return invalid, minimum, maximum, counts


def _validate_split(
    root: Path,
    split: str,
    device: str,
) -> tuple[
    SplitResult,
    list[tuple[str, str, float, float, float, float]],
    set[str],
    dict[str, np.ndarray],
]:
    directory = root / f"{split}_chips"
    split_file = root / f"{split}_data.txt"
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing directory: {directory}")
    if not split_file.is_file():
        raise FileNotFoundError(f"Missing split file: {split_file}")

    images = sorted(directory.glob("*_merged.tif"))
    masks = sorted(directory.glob("*.mask.tif"))
    ids = _read_ids(split_file)
    expected_ids = set(ids)
    image_paths = {_image_id(path): path for path in images}
    mask_paths = {_mask_id(path): path for path in masks}
    if set(image_paths) != expected_ids:
        missing = expected_ids - set(image_paths)
        extra = set(image_paths) - expected_ids
        raise RuntimeError(
            f"{split} image IDs do not match the manifest: "
            f"{len(missing)} missing, {len(extra)} extra"
        )
    if set(mask_paths) != expected_ids:
        missing = expected_ids - set(mask_paths)
        extra = set(mask_paths) - expected_ids
        raise RuntimeError(
            f"{split} mask IDs do not match the manifest: "
            f"{len(missing)} missing, {len(extra)} extra"
        )

    geometries: list[tuple[str, str, float, float, float, float]] = []
    label_counts = np.zeros(NUM_CLASSES + 1, dtype=np.int64)
    masked_image_values = 0
    invalid_image_values = 0
    image_minimum = math.inf
    image_maximum = -math.inf
    descriptions_present = 0
    chip_label_counts: dict[str, np.ndarray] = {}

    for index, chip_id in enumerate(ids, start=1):
        image_path = image_paths[chip_id]
        mask_path = mask_paths[chip_id]
        with rasterio.open(image_path) as image_source:
            if image_source.count != len(RASTER_BAND_ORDER):
                raise RuntimeError(f"Expected 18 bands, got {image_source.count}: {image_path}")
            if (image_source.height, image_source.width) != EXPECTED_IMAGE_SIZE:
                raise RuntimeError(
                    f"Expected 224x224, got {image_source.height}x"
                    f"{image_source.width}: {image_path}"
                )
            if image_source.crs is None:
                raise RuntimeError(f"Image has no CRS: {image_path}")
            if image_source.crs.to_string().upper() != EXPECTED_CRS:
                raise RuntimeError(f"Expected {EXPECTED_CRS}, got {image_source.crs}: {image_path}")
            descriptions_present += int(
                _check_band_descriptions(image_source.descriptions, image_path)
            )
            geometries.append(_geometry_record(split, chip_id, image_source))
            masked_image = image_source.read(masked=True, out_dtype="float32")
            masked_image_values += int(np.ma.getmaskarray(masked_image).sum())
            image = masked_image.filled(np.nan)
            image_profile = (
                image_source.crs,
                image_source.transform,
                image_source.width,
                image_source.height,
            )

        with rasterio.open(mask_path) as mask_source:
            if mask_source.count != 1:
                raise RuntimeError(f"Expected one mask band, got {mask_source.count}: {mask_path}")
            if (mask_source.height, mask_source.width) != EXPECTED_IMAGE_SIZE:
                raise RuntimeError(f"Unexpected mask dimensions: {mask_path}")
            mask_profile = (
                mask_source.crs,
                mask_source.transform,
                mask_source.width,
                mask_source.height,
            )
            if mask_profile != image_profile:
                raise RuntimeError(f"Image/mask geospatial mismatch for {chip_id}")
            masked_label = mask_source.read(1, masked=True)
            if np.ma.getmaskarray(masked_label).any():
                raise RuntimeError(f"Mask contains masked/nodata pixels: {mask_path}")
            label = np.asarray(masked_label)

        if label.min() < 0 or label.max() > NUM_CLASSES:
            raise RuntimeError(f"Unexpected label range {label.min()}..{label.max()}: {mask_path}")
        invalid, minimum, maximum, counts = _compute_array_stats(
            image,
            label,
            device,
        )
        invalid_image_values += invalid
        image_minimum = min(image_minimum, minimum)
        image_maximum = max(image_maximum, maximum)
        label_counts += counts
        chip_label_counts[chip_id] = counts
        if index % 250 == 0 or index == len(ids):
            print(f"  {split}: validated {index}/{len(ids)} chips")

    if masked_image_values or invalid_image_values:
        raise RuntimeError(
            f"{split} contains {masked_image_values} masked and "
            f"{invalid_image_values} non-finite image values"
        )

    result = SplitResult(
        split=split,
        chips=len(images),
        listed_ids=len(ids),
        masked_image_values=masked_image_values,
        invalid_image_values=invalid_image_values,
        label_counts=label_counts.tolist(),
        image_minimum=image_minimum,
        image_maximum=image_maximum,
        band_descriptions_present=descriptions_present,
    )
    return result, geometries, expected_ids, chip_label_counts


def _validate_metadata(
    root: Path,
    all_ids: set[str],
) -> pd.DataFrame:
    metadata_path = root / "chips_df.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    frame = pd.read_csv(metadata_path)
    date_columns = ["first_img_date", "middle_img_date", "last_img_date"]
    required = {"chip_id", *date_columns}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"chips_df.csv is missing columns: {sorted(missing)}")
    if frame["chip_id"].isna().any() or frame["chip_id"].duplicated().any():
        raise RuntimeError("chips_df.csv contains missing or duplicate chip_id values")
    metadata_ids = set(frame["chip_id"])
    if metadata_ids != all_ids:
        raise RuntimeError(
            "Metadata IDs do not match image manifests: "
            f"{len(all_ids - metadata_ids)} missing and "
            f"{len(metadata_ids - all_ids)} extra"
        )

    parsed_dates = frame[date_columns].apply(
        lambda column: pd.to_datetime(column, errors="coerce", utc=True)
    )
    if parsed_dates.isna().any().any():
        raise RuntimeError("chips_df.csv contains missing or invalid acquisition dates")
    if not (
        (parsed_dates[date_columns[0]] < parsed_dates[date_columns[1]])
        & (parsed_dates[date_columns[1]] < parsed_dates[date_columns[2]])
    ).all():
        raise RuntimeError("Acquisition dates are not strictly chronological")
    years = {int(year) for column in date_columns for year in parsed_dates[column].dt.year.unique()}
    if years != {2022}:
        raise RuntimeError(f"Expected only 2022 acquisition dates, got {sorted(years)}")
    return frame


def _find_cross_split_overlaps(
    geometries: list[tuple[str, str, float, float, float, float]],
) -> list[tuple[str, str]]:
    ordered = sorted(geometries, key=lambda item: item[2])
    active: list[tuple[str, str, float, float, float, float]] = []
    overlaps: list[tuple[str, str]] = []
    for current in ordered:
        active = [candidate for candidate in active if candidate[4] > current[2]]
        for candidate in active:
            if candidate[0] == current[0]:
                continue
            horizontal = min(candidate[4], current[4]) - max(candidate[2], current[2])
            vertical = min(candidate[5], current[5]) - max(candidate[3], current[3])
            if horizontal > 1e-6 and vertical > 1e-6:
                overlaps.append((candidate[1], current[1]))
                if len(overlaps) >= 20:
                    return overlaps
        active.append(current)
    return overlaps


def _named_label_counts(counts: np.ndarray) -> dict[str, int]:
    return {
        "No Data": int(counts[0]),
        **{name: int(counts[index + 1]) for index, name in enumerate(CLASS_NAMES)},
    }


def _sum_chip_counts(
    chip_ids: set[str],
    chip_counts: dict[str, np.ndarray],
) -> np.ndarray:
    total = np.zeros(NUM_CLASSES + 1, dtype=np.int64)
    for chip_id in chip_ids:
        total += chip_counts[chip_id]
    return total


def _validate_terratorch_batches(root: Path, batch_size: int) -> dict[str, Any]:
    import albumentations as A
    import torch
    from albumentations.pytorch import ToTensorV2
    from terratorch.datasets.transforms import (
        FlattenTemporalIntoChannels,
        UnflattenTemporalFromChannels,
    )

    from prithvi_crop.data import CropTypeDataModule

    def make_transform() -> A.Compose:
        return A.Compose(
            [
                FlattenTemporalIntoChannels(),
                ToTensorV2(),
                UnflattenTemporalFromChannels(n_timesteps=3),
            ],
            is_check_shapes=False,
        )

    module = CropTypeDataModule(
        data_root=str(root),
        batch_size=batch_size,
        num_workers=0,
        bands=list(MODEL_BANDS),
        train_transform=make_transform(),
        val_transform=make_transform(),
        test_transform=make_transform(),
        expand_temporal_dimension=True,
        reduce_zero_label=True,
        use_metadata=True,
        drop_last=False,
        validation_fraction=0.1,
        split_seed=42,
    )
    module.setup("fit")
    train_batch = next(iter(module.train_dataloader()))
    module.setup("test")
    test_batch = next(iter(module.test_dataloader()))

    for split_name, batch in (("train", train_batch), ("test", test_batch)):
        image = batch["image"]
        mask = batch["mask"]
        expected = (batch_size, 4, 3, 224, 224)
        if tuple(image.shape) != expected:
            raise RuntimeError(
                f"Expected {split_name} image shape {expected}, got {tuple(image.shape)}"
            )
        if tuple(mask.shape) != (batch_size, 224, 224):
            raise RuntimeError(f"Unexpected {split_name} mask shape: {tuple(mask.shape)}")
        if tuple(batch["temporal_coords"].shape) != (batch_size, 3, 2):
            raise RuntimeError(
                f"Unexpected temporal metadata shape: {batch['temporal_coords'].shape}"
            )
        if tuple(batch["location_coords"].shape) != (batch_size, 2):
            raise RuntimeError(
                f"Unexpected location metadata shape: {batch['location_coords'].shape}"
            )
        if not torch.isfinite(image).all():
            raise RuntimeError(f"{split_name} loader produced non-finite image values")

    test_size = len(module.test_dataset)
    _, validation_indices = deterministic_partition(
        _read_ids(root / "training_data.txt"),
        0.1,
        42,
    )
    return {
        "tensor_shape": list(train_batch["image"].shape),
        "train_size": EXPECTED_TRAINING_CHIPS - len(validation_indices),
        "validation_size": len(validation_indices),
        "test_size": test_size,
        "bands": list(MODEL_BANDS),
        "temporal_coords_shape": list(train_batch["temporal_coords"].shape),
        "location_coords_shape": list(train_batch["location_coords"].shape),
    }


def validate_dataset(
    root: Path,
    batch_size: int = 2,
    device: str = "auto",
    report_path: Path | None = Path("outputs/dataset_validation.json"),
) -> dict[str, Any]:
    import torch

    root = root.resolve()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA validation was requested but CUDA is unavailable")
    print(f"Validation compute device: {device}")
    print(
        "Official raster band contract:",
        ", ".join(RASTER_BAND_ORDER[:6]),
        "repeated for 3 dates",
    )
    print("Model-selected bands:", ", ".join(MODEL_BANDS))

    training, training_geometries, training_ids, training_chip_counts = _validate_split(
        root,
        "training",
        device,
    )
    validation, validation_geometries, validation_ids, validation_chip_counts = _validate_split(
        root,
        "validation",
        device,
    )
    if training_ids & validation_ids:
        raise RuntimeError("Official training and validation manifests share chip IDs")
    if training.chips != EXPECTED_TRAINING_CHIPS:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAINING_CHIPS} training chips, got {training.chips}"
        )
    if validation.chips != EXPECTED_VALIDATION_CHIPS:
        raise RuntimeError(
            f"Expected {EXPECTED_VALIDATION_CHIPS} validation chips, got {validation.chips}"
        )
    if training.chips + validation.chips != EXPECTED_TOTAL_CHIPS:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_CHIPS} total chips, got {training.chips + validation.chips}"
        )

    overlaps = _find_cross_split_overlaps(training_geometries + validation_geometries)
    if overlaps:
        raise RuntimeError(
            f"Spatial footprints overlap across official splits; examples: {overlaps[:5]}"
        )

    training_manifest = _read_ids(root / "training_data.txt")
    train_indices, internal_validation_indices = deterministic_partition(
        training_manifest,
        0.1,
        42,
    )
    logical_ids = {
        "training": {training_manifest[index] for index in train_indices},
        "validation": {training_manifest[index] for index in internal_validation_indices},
        "test": validation_ids,
    }
    if (
        logical_ids["training"] & logical_ids["validation"]
        or logical_ids["training"] & logical_ids["test"]
        or logical_ids["validation"] & logical_ids["test"]
    ):
        raise RuntimeError("Logical train/validation/test split IDs overlap")

    logical_geometries = []
    internal_validation_ids = logical_ids["validation"]
    for geometry in training_geometries:
        logical_split = "validation" if geometry[1] in internal_validation_ids else "training"
        logical_geometries.append((logical_split, *geometry[1:]))
    logical_geometries.extend(("test", *geometry[1:]) for geometry in validation_geometries)
    logical_overlaps = _find_cross_split_overlaps(logical_geometries)
    if logical_overlaps:
        raise RuntimeError(
            "Spatial footprints overlap across logical train/validation/test splits; "
            f"examples: {logical_overlaps[:5]}"
        )

    logical_class_counts = {
        "training": _named_label_counts(
            _sum_chip_counts(logical_ids["training"], training_chip_counts)
        ),
        "validation": _named_label_counts(
            _sum_chip_counts(logical_ids["validation"], training_chip_counts)
        ),
        "test": _named_label_counts(_sum_chip_counts(logical_ids["test"], validation_chip_counts)),
    }
    metadata = _validate_metadata(root, training_ids | validation_ids)
    total_labels = np.asarray(training.label_counts) + np.asarray(validation.label_counts)
    missing_classes = [
        CLASS_NAMES[index] for index, count in enumerate(total_labels[1:]) if count == 0
    ]
    if missing_classes:
        raise RuntimeError(f"Classes with no labelled pixels: {missing_classes}")

    batch_contract = _validate_terratorch_batches(root, batch_size)
    report = {
        "root": str(root),
        "validation_device": device,
        "complete": True,
        "official_source_split": {
            "training": asdict(training),
            "validation_reserved_as_test": asdict(validation),
        },
        "logical_model_splits": {
            "training": batch_contract["train_size"],
            "validation": batch_contract["validation_size"],
            "test": batch_contract["test_size"],
            "policy": (
                "deterministic 90/10 partition of official training chips; "
                "official validation chips reserved for held-out test"
            ),
            "cross_split_chip_id_overlap": 0,
            "cross_split_spatial_overlap": 0,
            "class_pixel_counts": logical_class_counts,
        },
        "metadata_rows": len(metadata),
        "band_order": list(RASTER_BAND_ORDER),
        "selected_bands": list(MODEL_BANDS),
        "time_steps": 3,
        "image_size": list(EXPECTED_IMAGE_SIZE),
        "crs": EXPECTED_CRS,
        "cross_split_chip_id_overlap": 0,
        "cross_split_spatial_overlap": 0,
        "class_names": list(CLASS_NAMES),
        "class_pixel_counts": _named_label_counts(total_labels),
        "batch_contract": batch_contract,
    }
    if report_path is not None:
        report_path = report_path.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"Validation report: {report_path}")
    print("All dataset, metadata, split, and loader checks passed.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exhaustively validate the crop dataset and loader contract."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/multi_temporal_crop"),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Use CUDA for array checks when available.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/dataset_validation.json"),
    )
    args = parser.parse_args()
    validate_dataset(
        root=args.root,
        batch_size=args.batch_size,
        device=args.device,
        report_path=args.report,
    )


if __name__ == "__main__":
    main()

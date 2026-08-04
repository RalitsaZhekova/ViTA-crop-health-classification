"""Data-only validation for the extracted optical PASTIS dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np

from prithvi_crop.europe import resolve_pastis_root

EXPECTED_METADATA_PATCHES = 2_433
EXPECTED_FOLDS = {1, 2, 3, 4, 5}
EXPECTED_BANDS = 10
EXPECTED_SIZE = (128, 128)


def _dates_count(value: Any) -> int:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("dates-S2 must be a dictionary or encoded dictionary")
    return len(value)


def validate_pastis(
    root: str | Path,
    *,
    inspect_arrays: bool = True,
) -> dict[str, Any]:
    """Validate metadata-linked image/target pairs without loading a model."""
    dataset_root = resolve_pastis_root(root)
    metadata = gpd.read_file(dataset_root / "metadata.geojson")
    required_columns = {"ID_PATCH", "Fold", "dates-S2", "geometry"}
    missing_columns = required_columns - set(metadata.columns)
    if missing_columns:
        raise ValueError(f"PASTIS metadata is missing columns: {sorted(missing_columns)}")
    if len(metadata) != EXPECTED_METADATA_PATCHES:
        raise ValueError(
            f"Expected {EXPECTED_METADATA_PATCHES} metadata patches, found {len(metadata)}"
        )

    patch_ids = metadata["ID_PATCH"].astype(int)
    if patch_ids.duplicated().any():
        raise ValueError("PASTIS metadata contains duplicate patch IDs")
    folds = set(metadata["Fold"].astype(int))
    if folds != EXPECTED_FOLDS:
        raise ValueError(f"Expected PASTIS folds {sorted(EXPECTED_FOLDS)}, found {sorted(folds)}")
    if metadata.crs is None:
        raise ValueError("PASTIS metadata has no coordinate reference system")
    if metadata.geometry.isna().any() or metadata.geometry.is_empty.any():
        raise ValueError("PASTIS metadata contains missing or empty geometries")

    image_paths = {
        int(path.stem.removeprefix("S2_")): path
        for path in (dataset_root / "DATA_S2").glob("S2_*.npy")
    }
    target_paths = {
        int(path.stem.removeprefix("TARGET_")): path
        for path in (dataset_root / "ANNOTATIONS").glob("TARGET_*.npy")
    }
    required_ids = set(patch_ids)
    missing_images = required_ids - set(image_paths)
    missing_targets = required_ids - set(target_paths)
    if missing_images or missing_targets:
        raise ValueError(
            "PASTIS is incomplete: "
            f"{len(missing_images)} missing images and {len(missing_targets)} missing targets"
        )

    if inspect_arrays:
        dates_by_id = {
            int(patch_id): _dates_count(dates)
            for patch_id, dates in metadata[["ID_PATCH", "dates-S2"]].itertuples(
                index=False, name=None
            )
        }
        for patch_id in sorted(required_ids):
            image = np.load(image_paths[patch_id], mmap_mode="r")
            if (
                image.ndim != 4
                or image.shape[1] != EXPECTED_BANDS
                or image.shape[2:] != EXPECTED_SIZE
            ):
                raise ValueError(f"Unexpected image shape for patch {patch_id}: {image.shape}")
            if image.shape[0] != dates_by_id[patch_id]:
                raise ValueError(
                    f"Image/date count mismatch for patch {patch_id}: "
                    f"{image.shape[0]} versus {dates_by_id[patch_id]}"
                )
            if not np.issubdtype(image.dtype, np.integer):
                raise ValueError(f"Unexpected image dtype for patch {patch_id}: {image.dtype}")

            target = np.load(target_paths[patch_id], mmap_mode="r")
            if target.shape != (3, *EXPECTED_SIZE):
                raise ValueError(f"Unexpected target shape for patch {patch_id}: {target.shape}")
            semantic = np.asarray(target[0])
            if semantic.min() < 0 or semantic.max() > 19:
                raise ValueError(f"Unknown semantic label in patch {patch_id}")

    report = {
        "dataset_root": str(dataset_root.resolve()),
        "metadata_patches": len(required_ids),
        "training_folds": [1, 2, 3, 4],
        "reserved_fold": 5,
        "fold_counts": {
            str(fold): int(count)
            for fold, count in metadata["Fold"].astype(int).value_counts().sort_index().items()
        },
        "required_images": len(required_ids),
        "required_targets": len(required_ids),
        "extra_unlabelled_images_ignored": len(set(image_paths) - required_ids),
        "extra_targets_ignored": len(set(target_paths) - required_ids),
        "array_headers_inspected": inspect_arrays,
        "status": "ready",
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate extracted PASTIS arrays without loading the crop model."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/europe/pastis"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/prithvi_4band_europe_replay/data_readiness.json"),
    )
    parser.add_argument(
        "--skip-array-inspection",
        action="store_true",
        help="Check metadata/file pairing without opening every NumPy array header.",
    )
    args = parser.parse_args()
    report = validate_pastis(
        args.root,
        inspect_arrays=not args.skip_array_inspection,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

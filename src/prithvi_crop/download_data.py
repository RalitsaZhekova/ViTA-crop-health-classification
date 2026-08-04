from __future__ import annotations

import argparse
import os
import shutil
import tarfile
import uuid
from pathlib import Path

from huggingface_hub import hf_hub_download

from prithvi_crop.constants import (
    ARCHIVES,
    DATASET_ID,
    DATASET_REVISION,
    METADATA_FILES,
)


def _image_id(path: Path) -> str:
    return path.name.removesuffix("_merged.tif")


def _mask_id(path: Path) -> str:
    return path.name.removesuffix(".mask.tif")


def _is_macos_metadata(member_name: str) -> bool:
    """Return whether a tar member is Finder metadata, not dataset content."""
    return any(part == "__MACOSX" or part.startswith("._") for part in Path(member_name).parts)


def _safe_extract(archive: Path, destination: Path) -> None:
    """Extract a regular-file/directory tar archive while blocking traversal."""
    destination = destination.resolve()
    with tarfile.open(archive, mode="r:gz") as tar:
        members = [member for member in tar.getmembers() if not _is_macos_metadata(member.name)]
        for member in members:
            target = (destination / member.name).resolve()
            if destination != target and destination not in target.parents:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"Links are not allowed in dataset archive: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise RuntimeError(f"Unsupported special file in dataset archive: {member.name}")
        tar.extractall(destination, members=members, filter="data")


def _download(filename: str, download_dir: Path, revision: str) -> Path:
    """Download through huggingface_hub, which resumes interrupted transfers."""
    path = hf_hub_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        revision=revision,
        filename=filename,
        local_dir=download_dir,
    )
    return Path(path)


def _copy_metadata(downloaded: Path, root: Path) -> None:
    destination = root / downloaded.name
    if downloaded.resolve() != destination.resolve():
        temporary = destination.with_suffix(f"{destination.suffix}.tmp")
        shutil.copy2(downloaded, temporary)
        os.replace(temporary, destination)


def _read_split_ids(root: Path, split: str) -> set[str]:
    split_file = root / f"{split}_data.txt"
    if not split_file.is_file():
        return set()
    return {
        line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()
    }


def _split_complete(root: Path, split: str) -> bool:
    expected_ids = _read_split_ids(root, split)
    directory = root / f"{split}_chips"
    if not expected_ids or not directory.is_dir():
        return False
    image_ids = {_image_id(path) for path in directory.glob("*_merged.tif")}
    mask_ids = {_mask_id(path) for path in directory.glob("*.mask.tif")}
    return image_ids == expected_ids and mask_ids == expected_ids


def _install_extracted_split(
    staging_root: Path,
    root: Path,
    split: str,
) -> None:
    extracted = staging_root / f"{split}_chips"
    destination = root / f"{split}_chips"
    if not extracted.is_dir():
        raise RuntimeError(f"Archive extracted but expected directory was not found: {extracted}")

    expected_ids = _read_split_ids(root, split)
    image_ids = {_image_id(path) for path in extracted.glob("*_merged.tif")}
    mask_ids = {_mask_id(path) for path in extracted.glob("*.mask.tif")}
    if image_ids != expected_ids or mask_ids != expected_ids:
        raise RuntimeError(
            f"Extracted {split} archive is incomplete: expected {len(expected_ids)} "
            f"chips, found {len(image_ids)} images and {len(mask_ids)} masks"
        )

    backup = root / f".{split}_chips.incomplete-{uuid.uuid4().hex}"
    moved_old = False
    try:
        if destination.exists():
            os.replace(destination, backup)
            moved_old = True
        os.replace(extracted, destination)
    except Exception:
        if moved_old and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)


def _free_space_gib(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1024**3


def prepare_dataset(
    root: Path,
    validation_only: bool,
    delete_archives: bool,
    force_extract: bool,
    revision: str = DATASET_REVISION,
    minimum_free_gib: float = 30.0,
) -> None:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    free_gib = _free_space_gib(root)
    if free_gib < minimum_free_gib:
        raise RuntimeError(
            f"Only {free_gib:.2f} GiB free at {root}; "
            f"{minimum_free_gib:.2f} GiB is required before download"
        )

    download_dir = root / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)

    print(f"Official dataset: https://huggingface.co/datasets/{DATASET_ID}")
    print(f"Dataset revision: {revision}")
    print(f"Dataset root: {root}")
    print(f"Free disk space: {free_gib:.2f} GiB")
    print("Downloading metadata...")
    for filename in METADATA_FILES:
        downloaded = _download(filename, download_dir, revision)
        _copy_metadata(downloaded, root)

    selected = ["validation"] if validation_only else ["training", "validation"]
    for split in selected:
        archive_name = ARCHIVES[split]
        local_archive = download_dir / archive_name
        if _split_complete(root, split) and not force_extract:
            print(f"Verified complete; skipping download/extraction: {split}_chips")
            if delete_archives:
                local_archive.unlink(missing_ok=True)
            continue

        print(f"Downloading/resuming {archive_name}...")
        archive = _download(archive_name, download_dir, revision)
        staging_root = root / f".extract-{split}-{uuid.uuid4().hex}"
        staging_root.mkdir()
        try:
            print(f"Safely extracting {archive_name} to a staging directory...")
            _safe_extract(archive, staging_root)
            _install_extracted_split(staging_root, root, split)
            if not _split_complete(root, split):
                raise RuntimeError(f"{split} split failed the post-install completeness check")
        finally:
            if staging_root.exists():
                shutil.rmtree(staging_root)

        if delete_archives:
            archive.unlink(missing_ok=True)
            print(f"Deleted verified local archive: {archive}")

    print("Dataset preparation complete.")
    print(f"Run: prithvi-validate --root {root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and safely extract the official IBM-NASA crop dataset."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/multi_temporal_crop"),
        help="Destination dataset directory.",
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Download only the 1.18 GB official validation archive.",
    )
    parser.add_argument(
        "--delete-archives",
        action="store_true",
        help="Delete local .tgz files only after verified extraction.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Atomically replace an already complete extracted split.",
    )
    parser.add_argument(
        "--revision",
        default=DATASET_REVISION,
        help="Hugging Face dataset revision (branch, tag, or commit).",
    )
    parser.add_argument(
        "--minimum-free-gib",
        type=float,
        default=30.0,
        help="Required free space before downloading (default: 30 GiB).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepare_dataset(
        root=args.root,
        validation_only=args.validation_only,
        delete_archives=args.delete_archives,
        force_extract=args.force_extract,
        revision=args.revision,
        minimum_free_gib=args.minimum_free_gib,
    )


if __name__ == "__main__":
    main()

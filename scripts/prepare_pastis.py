"""Verify and selectively extract the optical PASTIS training data."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath

from prithvi_crop.pastis_validation import validate_pastis

PASTIS_MD5 = "cfc441bf18137ff0bbf4fad58828fb98"


def file_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_destination(member_name: str, destination: Path) -> Path | None:
    """Return a safe destination for required optical files only."""
    path = PurePosixPath(member_name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe archive member: {member_name}")
    if not path.parts or member_name.endswith("/"):
        return None

    filename = path.name
    if filename == "metadata.geojson":
        relative = Path("metadata.geojson")
    elif "DATA_S2" in path.parts and filename.startswith("S2_"):
        relative = Path("DATA_S2") / filename
    elif "ANNOTATIONS" in path.parts and filename.startswith("TARGET_"):
        relative = Path("ANNOTATIONS") / filename
    else:
        return None
    return destination / "PASTIS" / relative


def prepare(archive: Path, destination: Path) -> dict[str, int | str]:
    if not archive.is_file():
        raise FileNotFoundError(archive)
    checksum = file_md5(archive)
    if checksum != PASTIS_MD5:
        raise RuntimeError(f"PASTIS checksum mismatch: expected {PASTIS_MD5}, got {checksum}")

    extracted = 0
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            target = selected_destination(member.filename, destination)
            if target is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as input_file, target.open("wb") as output_file:
                shutil.copyfileobj(input_file, output_file, length=8 * 1024 * 1024)
            extracted += 1

    validation = validate_pastis(destination, inspect_arrays=True)
    root = Path(validation["dataset_root"])

    report: dict[str, int | str] = {
        "archive": str(archive.resolve()),
        "md5": checksum,
        "extracted_members": extracted,
        "image_patches": int(validation["required_images"]),
        "mask_patches": int(validation["required_targets"]),
        "extra_unlabelled_images_ignored": int(validation["extra_unlabelled_images_ignored"]),
        "output": str(root.resolve()),
    }
    (destination / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/downloads/PASTIS.zip"),
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/europe/pastis"),
    )
    args = parser.parse_args()
    report = prepare(args.archive, args.destination)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

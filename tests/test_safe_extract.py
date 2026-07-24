import io
import tarfile
from pathlib import Path

import pytest

from prithvi_crop.download_data import (
    _is_macos_metadata,
    _safe_extract,
    _split_complete,
)


def _write_archive(path: Path, member_name: str) -> None:
    payload = b"test"
    with tarfile.open(path, "w:gz") as tar:
        info = tarfile.TarInfo(member_name)
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))


def test_safe_extract_accepts_normal_file(tmp_path: Path) -> None:
    archive = tmp_path / "ok.tgz"
    _write_archive(archive, "folder/file.txt")
    output = tmp_path / "output"
    output.mkdir()
    _safe_extract(archive, output)
    assert (output / "folder/file.txt").read_bytes() == b"test"


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.tgz"
    _write_archive(archive, "../escape.txt")
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(RuntimeError, match="Unsafe archive member"):
        _safe_extract(archive, output)


def test_safe_extract_rejects_links(tmp_path: Path) -> None:
    archive = tmp_path / "link.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "../escape.txt"
        tar.addfile(info)
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(RuntimeError, match="Links are not allowed"):
        _safe_extract(archive, output)


def test_macos_metadata_is_ignored() -> None:
    assert _is_macos_metadata("._training_chips")
    assert _is_macos_metadata("training_chips/._chip_1_2_merged.tif")
    assert _is_macos_metadata("__MACOSX/training_chips/file")
    assert not _is_macos_metadata("training_chips/chip_1_2_merged.tif")


def test_split_complete_requires_every_image_and_mask(tmp_path: Path) -> None:
    (tmp_path / "training_data.txt").write_text("chip_1\nchip_2\n", encoding="utf-8")
    directory = tmp_path / "training_chips"
    directory.mkdir()
    for chip_id in ("chip_1", "chip_2"):
        (directory / f"{chip_id}_merged.tif").touch()
        (directory / f"{chip_id}.mask.tif").touch()
    assert _split_complete(tmp_path, "training")
    (directory / "chip_2.mask.tif").unlink()
    assert not _split_complete(tmp_path, "training")

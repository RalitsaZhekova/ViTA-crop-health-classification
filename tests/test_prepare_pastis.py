from pathlib import Path

import pytest

from scripts.prepare_pastis import selected_destination


def test_selective_extract_keeps_only_required_optical_files() -> None:
    destination = Path("data/europe/pastis")
    assert selected_destination(
        "PASTIS/DATA_S2/S2_42.npy",
        destination,
    ) == destination / "PASTIS/DATA_S2/S2_42.npy"
    assert selected_destination(
        "PASTIS/ANNOTATIONS/TARGET_42.npy",
        destination,
    ) == destination / "PASTIS/ANNOTATIONS/TARGET_42.npy"
    assert selected_destination(
        "PASTIS/INSTANCE_ANNOTATIONS/INSTANCES_42.npy",
        destination,
    ) is None


def test_selective_extract_rejects_path_traversal() -> None:
    with pytest.raises(ValueError, match="Unsafe"):
        selected_destination("../outside.npy", Path("data/europe/pastis"))

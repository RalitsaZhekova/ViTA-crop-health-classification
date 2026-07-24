import numpy as np
import pytest

from prithvi_crop.data import deterministic_partition, select_temporal_bands


def test_deterministic_partition_is_stable_complete_and_disjoint() -> None:
    chip_ids = [f"chip_{index:04d}" for index in range(100)]
    train_a, validation_a = deterministic_partition(chip_ids, 0.1, 42)
    train_b, validation_b = deterministic_partition(chip_ids, 0.1, 42)
    assert (train_a, validation_a) == (train_b, validation_b)
    assert len(train_a) == 90
    assert len(validation_a) == 10
    assert set(train_a).isdisjoint(validation_a)
    assert sorted(train_a + validation_a) == list(range(100))


def test_deterministic_partition_rejects_duplicate_chip_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        deterministic_partition(["chip_1", "chip_1"], 0.1, 42)


def test_temporal_band_selection_uses_first_four_bands_of_every_date() -> None:
    image = np.arange(18, dtype=np.float32).reshape(18, 1, 1)
    selected = select_temporal_bands(
        image,
        band_indices=np.asarray([0, 1, 2, 3]),
        all_band_count=6,
        expand_temporal_dimension=True,
    )
    assert selected.shape == (3, 1, 1, 4)
    assert selected[:, 0, 0].tolist() == [
        [0.0, 1.0, 2.0, 3.0],
        [6.0, 7.0, 8.0, 9.0],
        [12.0, 13.0, 14.0, 15.0],
    ]

import numpy as np
import pytest

from prithvi_crop.transforms import apply_dihedral


def test_all_dihedral_transforms_preserve_shape_values_and_alignment() -> None:
    image = np.arange(3 * 3 * 2).reshape(3, 3, 2)
    mask = image[..., 0]
    outputs = []

    for transform_id in range(1, 8):
        transformed_image = apply_dihedral(image, transform_id)
        transformed_mask = apply_dihedral(mask, transform_id)
        assert transformed_image.shape == image.shape
        assert transformed_image.flags.c_contiguous
        assert np.array_equal(transformed_image[..., 0], transformed_mask)
        outputs.append(transformed_mask)

    assert len({output.tobytes() for output in outputs}) == 7
    assert all(not np.array_equal(output, mask) for output in outputs)


@pytest.mark.parametrize("transform_id", [0, 8])
def test_dihedral_rejects_identity_and_unknown_transform(transform_id: int) -> None:
    with pytest.raises(ValueError, match="1 through 7"):
        apply_dihedral(np.zeros((2, 2)), transform_id)

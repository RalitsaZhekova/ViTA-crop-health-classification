"""Training-only augmentations for multitemporal crop chips."""

from __future__ import annotations

from typing import Any

import albumentations as A
import numpy as np


def apply_dihedral(array: np.ndarray, transform_id: int) -> np.ndarray:
    """Apply one of the seven non-identity square symmetries."""
    if transform_id == 1:
        transformed = np.rot90(array, 1, axes=(0, 1))
    elif transform_id == 2:
        transformed = np.rot90(array, 2, axes=(0, 1))
    elif transform_id == 3:
        transformed = np.rot90(array, 3, axes=(0, 1))
    elif transform_id == 4:
        transformed = np.flip(array, axis=1)
    elif transform_id == 5:
        transformed = np.flip(array, axis=0)
    elif transform_id == 6:
        transformed = np.swapaxes(array, 0, 1)
    elif transform_id == 7:
        transformed = np.rot90(np.swapaxes(array, 0, 1), 2, axes=(0, 1))
    else:
        raise ValueError("transform_id must be an integer from 1 through 7")
    return np.ascontiguousarray(transformed)


class RandomNonIdentityDihedral(A.DualTransform):
    """Randomly rotate/flip an image and its mask without choosing identity."""

    def __init__(self, p: float = 1.0) -> None:
        super().__init__(p=p)

    def get_params(self) -> dict[str, int]:
        return {"transform_id": self.py_random.randint(1, 7)}

    def apply(
        self,
        img: np.ndarray,
        transform_id: int,
        **params: Any,
    ) -> np.ndarray:
        return apply_dihedral(img, transform_id)

    def apply_to_mask(
        self,
        mask: np.ndarray,
        transform_id: int,
        **params: Any,
    ) -> np.ndarray:
        return apply_dihedral(mask, transform_id)

    def get_transform_init_args_names(self) -> tuple[()]:
        return ()

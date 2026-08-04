from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage


def postprocess(
    semantic: np.ndarray,
    classes: dict[str, int],
    config: dict[str, Any],
    invalid: np.ndarray,
) -> np.ndarray:
    if semantic.ndim != 2 or invalid.shape != semantic.shape:
        raise ValueError("Semantic and invalid masks must have matching two-dimensional shapes.")

    unusable = (semantic == int(classes["thick_cloud"])) | (semantic == int(classes["thin_cloud"]))
    if bool(config["include_shadow_as_unusable"]):
        unusable |= semantic == int(classes["cloud_shadow"])

    minimum_pixels = int(config["minimum_region_pixels"])
    labels, region_count = ndimage.label(unusable)
    if region_count and minimum_pixels > 1:
        sizes = np.bincount(labels.ravel())
        keep = sizes >= minimum_pixels
        keep[0] = False
        unusable = keep[labels]

    dilation_pixels = int(config["dilation_pixels"])
    if dilation_pixels > 0:
        unusable = ndimage.binary_dilation(unusable, iterations=dilation_pixels)

    unusable |= invalid
    return unusable.astype(np.uint8)

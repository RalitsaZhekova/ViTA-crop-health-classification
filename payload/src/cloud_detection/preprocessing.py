from __future__ import annotations

import numpy as np


def strict_valid_mask(image_rgn: np.ndarray) -> np.ndarray:
    """Return the supplied detector's strict-valid Red/Green/NIR footprint."""
    if image_rgn.ndim != 3 or image_rgn.shape[0] != 3:
        raise ValueError("image_rgn must have shape (3,height,width) in Red/Green/NIR order")
    tiny = np.finfo(np.float32).tiny
    finite = np.all(np.isfinite(image_rgn), axis=0)
    return finite & np.all(image_rgn > tiny, axis=0)


def normalize_reflectance(
    array: np.ndarray,
    scale: float,
    clip_min: float | None = None,
    clip_max: float | None = None,
    nodata_value: int | float | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert scaled sensor values to float32 reflectance.

    Scaling remains required by downstream science. OmniCloudMask dynamically
    normalizes each patch again, so this conversion does not define its model
    distribution. Clipping is disabled by default.
    """
    if array.ndim != 3:
        raise ValueError(f"Expected C x H x W input, found {array.shape}.")
    if scale <= 0:
        raise ValueError("Reflectance scale must be positive.")

    values = array.astype(np.float32, copy=False)
    invalid = ~np.isfinite(values).all(axis=0)
    if nodata_value is not None:
        invalid |= np.all(values == nodata_value, axis=0)

    values = values / np.float32(scale)
    if clip_min is not None or clip_max is not None:
        lower = -np.inf if clip_min is None else float(clip_min)
        upper = np.inf if clip_max is None else float(clip_max)
        values = np.clip(values, lower, upper)

    values[:, invalid] = 0.0
    return values, invalid

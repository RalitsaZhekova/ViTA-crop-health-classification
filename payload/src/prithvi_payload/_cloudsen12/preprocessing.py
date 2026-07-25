"""Reflectance normalization adapted from ViTA revision a1ce4d3."""

from __future__ import annotations

import numpy as np


def normalize_reflectance(
    array: np.ndarray,
    scale: float,
    clip_min: float | None = None,
    clip_max: float | None = None,
    nodata_value: int | float | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert digital numbers to float32 top-of-atmosphere reflectance."""
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

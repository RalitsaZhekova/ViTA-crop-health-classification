from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def save_preview(
    path: str | Path,
    image: np.ndarray,
    semantic: np.ndarray,
    unusable: np.ndarray,
) -> None:
    rgb = image[[1, 2, 3]].transpose(1, 2, 0)
    finite_values = rgb[np.isfinite(rgb)]
    if finite_values.size:
        low, high = np.percentile(finite_values, [2, 98])
        rgb = np.clip((rgb - low) / max(high - low, 1e-6), 0, 1)
    else:
        rgb = np.zeros_like(rgb)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 RGB")
    axes[1].imshow(
        semantic,
        cmap=ListedColormap(["black", "white", "cyan", "dimgray"]),
        vmin=0,
        vmax=3,
    )
    axes[1].set_title("CloudSEN12 semantic mask")
    axes[2].imshow(unusable, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Unusable-pixel mask")
    for axis in axes:
        axis.axis("off")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

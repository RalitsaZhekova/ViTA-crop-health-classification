from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "vita_cloud_matplotlib"),
)

from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


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

    figure = Figure(figsize=(15, 5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
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

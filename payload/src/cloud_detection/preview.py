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
from matplotlib.patches import Patch  # noqa: E402


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

    figure = Figure(figsize=(16, 5.5), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    axes[0].imshow(rgb)
    axes[0].set_title("RGB composite")
    semantic_colors = ["#1b1b1b", "#ffffff", "#33d6ff", "#666666"]
    semantic_map = ListedColormap(semantic_colors)
    semantic_map.set_bad("#d81b60")
    axes[1].imshow(
        np.ma.masked_equal(semantic, 255),
        cmap=semantic_map,
        vmin=0,
        vmax=3,
    )
    axes[1].set_title("CloudSEN12 semantic mask")
    axes[1].legend(
        handles=[
            Patch(facecolor=color, edgecolor="#777777", label=label)
            for color, label in zip(
                semantic_colors,
                ("Clear", "Thick cloud", "Thin cloud", "Shadow"),
                strict=True,
            )
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
    )
    axes[2].imshow(unusable, cmap="gray", vmin=0, vmax=1)
    unusable_percentage = 100.0 * float(np.count_nonzero(unusable == 1)) / unusable.size
    axes[2].set_title(f"Unusable-pixel mask ({unusable_percentage:.1f}%)")
    for axis in axes:
        axis.axis("off")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)

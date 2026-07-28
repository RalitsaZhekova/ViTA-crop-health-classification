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
from matplotlib.colors import ListedColormap, to_rgba  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

CLOUD_CLASS_SPECS = (
    (0, "Clear", "#20252b", 0.0),
    (1, "Thick cloud", "#ff4d4f", 0.62),
    (2, "Thin cloud", "#00b8d9", 0.62),
    (3, "Cloud shadow", "#7e57c2", 0.62),
    (255, "Invalid / nodata", "#ffc107", 0.78),
)


def _semantic_display(semantic: np.ndarray) -> np.ndarray:
    """Map model classes and any invalid code to consecutive display values."""
    values = np.asarray(semantic)
    display = np.full(values.shape, 4, dtype=np.uint8)
    for display_code, (class_value, _label, _color, _alpha) in enumerate(CLOUD_CLASS_SPECS[:-1]):
        display[values == class_value] = display_code
    return display


def _semantic_overlay(semantic: np.ndarray) -> np.ndarray:
    """Return a display-only RGBA overlay; this never changes the saved mask."""
    values = np.asarray(semantic)
    overlay = np.zeros((*values.shape, 4), dtype=np.float32)
    known = np.zeros(values.shape, dtype=bool)
    for class_value, _label, color, alpha in CLOUD_CLASS_SPECS[:-1]:
        mask = values == class_value
        known |= mask
        overlay[mask] = (*to_rgba(color)[:3], alpha)
    invalid = ~known
    _value, _label, color, alpha = CLOUD_CLASS_SPECS[-1]
    overlay[invalid] = (*to_rgba(color)[:3], alpha)
    return overlay


def _semantic_legend(semantic: np.ndarray) -> list[Patch]:
    values = np.asarray(semantic)
    known = np.isin(values, [item[0] for item in CLOUD_CLASS_SPECS[:-1]])
    total = max(values.size, 1)
    handles = []
    for class_value, label, color, _alpha in CLOUD_CLASS_SPECS:
        count = (
            np.count_nonzero(values == class_value)
            if class_value != 255
            else np.count_nonzero(~known)
        )
        handles.append(
            Patch(
                facecolor=color,
                edgecolor="#444444",
                label=f"{label}: {100.0 * count / total:.1f}%",
            )
        )
    return handles


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

    figure = Figure(figsize=(14, 11), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2).ravel()
    axes[0].imshow(rgb)
    axes[0].set_title("RGB composite")
    semantic_colors = [item[2] for item in CLOUD_CLASS_SPECS]
    semantic_map = ListedColormap(semantic_colors)
    axes[1].imshow(
        _semantic_display(semantic),
        cmap=semantic_map,
        vmin=-0.5,
        vmax=4.5,
        interpolation="nearest",
    )
    axes[1].set_title("Exact semantic classes")
    axes[2].imshow(rgb)
    axes[2].imshow(_semantic_overlay(semantic), interpolation="bilinear")
    axes[2].set_title("Cloud-class overlay on RGB")
    axes[3].imshow(
        unusable,
        cmap=ListedColormap(["#20252b", "#f4f6f8"]),
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    unusable_percentage = 100.0 * float(np.count_nonzero(unusable == 1)) / unusable.size
    axes[3].set_title(f"Operational unusable mask: {unusable_percentage:.1f}%")
    for axis in axes:
        axis.axis("off")
    figure.legend(
        handles=_semantic_legend(semantic),
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        "Cloud detail preview — exact masks use nearest-neighbor display; "
        "overlay feathering is visual only",
        fontsize=13,
    )

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)

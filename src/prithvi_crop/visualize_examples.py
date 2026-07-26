"""Render business-ready examples from real internal-validation scenes."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from prithvi_crop.binary import (
    binary_predictions_from_fine_logits,
    binary_targets_from_fine_targets,
    crop_probability_from_fine_logits,
)
from prithvi_crop.calibration import CROP_CLASSIFICATION_THRESHOLD
from prithvi_crop.constants import CLASS_NAMES
from prithvi_crop.runtime import build_data_module, build_task, load_config

MODEL_CHECKPOINT_NAME = "epoch=02-macro_f1=0.5062.ckpt"
MODEL_CHECKPOINT_SHA256 = (
    "49383194174c56b66f3b7fa67254a42135db6ad64172e873c43d65d1afae6a65"
)
SELECTIONS = (
    ("Lower-range example", 0.20),
    ("Median example", 0.50),
    ("Upper-range example", 0.80),
)
CLASS_COLORS = (
    "#78A85A",
    "#245B3A",
    "#F3C64E",
    "#63B7A6",
    "#68B8C4",
    "#8C8F96",
    "#3478C5",
    "#D9A441",
    "#A8CC65",
    "#D7C29E",
    "#DFA8B8",
    "#B8653B",
    "#596273",
)
NON_CROP_COLOR = "#CDD5DF"
CROP_COLOR = "#2C9C69"
IGNORED_COLOR = "#737C89"
BACKGROUND_COLOR = "#F3F6F9"
TEXT_COLOR = "#162536"
SUBTLE_TEXT_COLOR = "#526171"


@dataclass(frozen=True)
class ValidationExample:
    selection: str
    percentile: float
    validation_index: int
    chip_id: str
    full_accuracy: float
    binary_accuracy: float
    rgb_dates: tuple[np.ndarray, ...]
    fine_target: np.ndarray
    fine_prediction: np.ndarray
    binary_target: np.ndarray
    binary_prediction: np.ndarray
    crop_probability: np.ndarray


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _rgb(image: torch.Tensor, time_index: int) -> np.ndarray:
    array = image[[2, 1, 0], time_index].permute(1, 2, 0).numpy()
    low = np.nanpercentile(array, 2, axis=(0, 1), keepdims=True)
    high = np.nanpercentile(array, 98, axis=(0, 1), keepdims=True)
    stretched = np.clip((array - low) / np.maximum(high - low, 1e-6), 0, 1)
    return np.nan_to_num(stretched, nan=0.0) ** 0.9


def _chip_id(data_module: Any, validation_index: int) -> str:
    subset = data_module.val_dataset
    source_index = subset.indices[validation_index]
    image_path = Path(subset.dataset.image_files[source_index])
    return image_path.name.removesuffix("_merged.tif")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _batch_accuracies(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    ignore_index: int,
) -> list[float]:
    valid = target != ignore_index
    correct = (prediction == target) & valid
    return (
        correct.flatten(1).sum(dim=1)
        / valid.flatten(1).sum(dim=1).clamp_min(1)
    ).cpu().tolist()


def _load_selected_examples(
    config_path: Path,
    checkpoint: Path,
    *,
    batch_size: int,
    num_workers: int,
) -> tuple[list[ValidationExample], str]:
    if not torch.cuda.is_available():
        raise RuntimeError("Validation visualization requires CUDA")
    if checkpoint.name != MODEL_CHECKPOINT_NAME:
        raise ValueError(
            f"Expected pinned checkpoint {MODEL_CHECKPOINT_NAME}, got {checkpoint.name}"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    actual_checksum = _sha256(checkpoint)
    if actual_checksum != MODEL_CHECKPOINT_SHA256:
        raise ValueError(
            f"Pinned checkpoint checksum mismatch: {actual_checksum}"
        )

    config = load_config(config_path)
    data_module = build_data_module(
        config,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    data_module.setup("validate")

    config["model"]["init_args"]["initial_checkpoint"] = None
    task = build_task(config, load_initial_weights=False)
    checkpoint_data = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    task.load_state_dict(checkpoint_data["state_dict"], strict=True)
    device = torch.device("cuda", 0)
    task = task.to(device).eval()

    full_scores: list[float] = []
    binary_scores: list[float] = []
    loader = data_module.val_dataloader()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            normalized = _to_device(data_module.aug(batch), device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = task(
                    normalized["image"],
                    temporal_coords=normalized["temporal_coords"],
                    location_coords=normalized["location_coords"],
                ).output
            full_scores.extend(
                _batch_accuracies(
                    logits.argmax(dim=1),
                    normalized["mask"],
                    ignore_index=-1,
                )
            )
            binary_scores.extend(
                _batch_accuracies(
                    binary_predictions_from_fine_logits(logits),
                    binary_targets_from_fine_targets(normalized["mask"]),
                    ignore_index=-1,
                )
            )
            if batch_index % 5 == 0 or batch_index == len(loader):
                print(
                    f"Scored validation batches: "
                    f"{batch_index}/{len(loader)}",
                    flush=True,
                )

    ordered = np.argsort(full_scores)
    selected = [
        (
            selection,
            percentile,
            int(ordered[round(percentile * (len(ordered) - 1))]),
        )
        for selection, percentile in SELECTIONS
    ]

    examples: list[ValidationExample] = []
    with torch.inference_mode():
        for selection, percentile, validation_index in selected:
            sample = data_module.val_dataset[validation_index]
            batch = data_module.collate_fn([sample])
            normalized = _to_device(data_module.aug(batch), device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = task(
                    normalized["image"],
                    temporal_coords=normalized["temporal_coords"],
                    location_coords=normalized["location_coords"],
                ).output
            examples.append(
                ValidationExample(
                    selection=selection,
                    percentile=percentile,
                    validation_index=validation_index,
                    chip_id=_chip_id(data_module, validation_index),
                    full_accuracy=full_scores[validation_index],
                    binary_accuracy=binary_scores[validation_index],
                    rgb_dates=tuple(_rgb(sample["image"], index) for index in range(3)),
                    fine_target=sample["mask"].numpy(),
                    fine_prediction=logits.argmax(dim=1)[0].cpu().numpy(),
                    binary_target=(
                        binary_targets_from_fine_targets(sample["mask"]).numpy()
                    ),
                    binary_prediction=(
                        binary_predictions_from_fine_logits(logits)[0].cpu().numpy()
                    ),
                    crop_probability=(
                        crop_probability_from_fine_logits(logits)[0]
                        .float()
                        .cpu()
                        .numpy()
                    ),
                )
            )
    return examples, torch.cuda.get_device_name(device)


def _prepare_axes(axes: np.ndarray) -> None:
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_facecolor("white")
        for spine in axis.spines.values():
            spine.set_edgecolor("#D9E0E7")
            spine.set_linewidth(0.8)


def _add_header(
    figure: plt.Figure,
    *,
    title: str,
    subtitle: str,
) -> None:
    figure.patch.set_facecolor(BACKGROUND_COLOR)
    figure.text(
        0.04,
        0.965,
        title,
        ha="left",
        va="top",
        fontsize=22,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    figure.text(
        0.04,
        0.925,
        subtitle,
        ha="left",
        va="top",
        fontsize=11,
        color=SUBTLE_TEXT_COLOR,
    )


def _add_row_label(axis: plt.Axes, example: ValidationExample, metric: str) -> None:
    value = example.full_accuracy if metric == "full" else example.binary_accuracy
    axis.text(
        -0.06,
        0.5,
        f"{example.selection}\n{example.chip_id}\nPixel accuracy  {value:.1%}",
        transform=axis.transAxes,
        ha="right",
        va="center",
        fontsize=10,
        fontweight="semibold",
        color=TEXT_COLOR,
        linespacing=1.5,
    )


def _masked_discrete(
    values: np.ndarray,
    colors: tuple[str, ...],
) -> tuple[np.ma.MaskedArray, ListedColormap]:
    masked = np.ma.masked_where(values < 0, values)
    color_map = ListedColormap(colors)
    color_map.set_bad(IGNORED_COLOR)
    return masked, color_map


def _render_full_classifier(
    examples: list[ValidationExample],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        len(examples),
        5,
        figsize=(18, 11),
        gridspec_kw={"wspace": 0.04, "hspace": 0.15},
    )
    _prepare_axes(axes)
    _add_header(
        figure,
        title="Multi-class crop & land-cover segmentation",
        subtitle=(
            "Real internal-validation scenes  |  Three seasonal RGB views  |  "
            "13-class Prithvi prediction"
        ),
    )
    headers = (
        "Seasonal view 1",
        "Seasonal view 2",
        "Seasonal view 3",
        "Reference labels",
        "Model prediction",
    )
    class_map = ListedColormap(CLASS_COLORS)
    class_map.set_bad(IGNORED_COLOR)

    for row, example in enumerate(examples):
        for date_index, rgb in enumerate(example.rgb_dates):
            axes[row, date_index].imshow(rgb)
        target = np.ma.masked_where(example.fine_target < 0, example.fine_target)
        axes[row, 3].imshow(target, cmap=class_map, vmin=0, vmax=len(CLASS_NAMES) - 1)
        axes[row, 4].imshow(
            example.fine_prediction,
            cmap=class_map,
            vmin=0,
            vmax=len(CLASS_NAMES) - 1,
        )
        _add_row_label(axes[row, 0], example, "full")
        if row == 0:
            for axis, header in zip(axes[row], headers, strict=True):
                axis.set_title(
                    header,
                    fontsize=11,
                    fontweight="semibold",
                    color=TEXT_COLOR,
                    pad=8,
                )

    legend = [
        Patch(facecolor=color, label=name)
        for color, name in zip(CLASS_COLORS, CLASS_NAMES, strict=True)
    ]
    figure.legend(
        handles=legend,
        loc="lower center",
        ncol=7,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.54, 0.042),
        columnspacing=1.4,
        handlelength=1.4,
    )
    figure.text(
        0.04,
        0.012,
        (
            "Examples selected automatically at the 20th, 50th and 80th "
            "percentiles of fine-class accuracy. Per-scene display stretch; "
            "not a held-out test."
        ),
        fontsize=8.5,
        color=SUBTLE_TEXT_COLOR,
    )
    figure.subplots_adjust(left=0.18, right=0.98, top=0.87, bottom=0.15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor=BACKGROUND_COLOR)
    plt.close(figure)


def _render_binary_classifier(
    examples: list[ValidationExample],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        len(examples),
        6,
        figsize=(20, 11),
        gridspec_kw={"wspace": 0.04, "hspace": 0.15},
    )
    _prepare_axes(axes)
    _add_header(
        figure,
        title="Operational crop / no-crop view",
        subtitle=(
            "The same internal-validation scenes  |  Combined crop probability  |  "
            f"Decision threshold {CROP_CLASSIFICATION_THRESHOLD:.3f}"
        ),
    )
    headers = (
        "Seasonal view 1",
        "Seasonal view 2",
        "Seasonal view 3",
        "Reference split",
        "Model split",
        "Crop probability",
    )
    probability_images = []

    for row, example in enumerate(examples):
        for date_index, rgb in enumerate(example.rgb_dates):
            axes[row, date_index].imshow(rgb)
        target, binary_map = _masked_discrete(
            example.binary_target,
            (NON_CROP_COLOR, CROP_COLOR),
        )
        axes[row, 3].imshow(target, cmap=binary_map, vmin=0, vmax=1)
        axes[row, 4].imshow(
            example.binary_prediction,
            cmap=binary_map,
            vmin=0,
            vmax=1,
        )
        probability_images.append(
            axes[row, 5].imshow(
                example.crop_probability,
                cmap="YlGn",
                vmin=0,
                vmax=1,
            )
        )
        _add_row_label(axes[row, 0], example, "binary")
        if row == 0:
            for axis, header in zip(axes[row], headers, strict=True):
                axis.set_title(
                    header,
                    fontsize=11,
                    fontweight="semibold",
                    color=TEXT_COLOR,
                    pad=8,
                )

    figure.subplots_adjust(left=0.17, right=0.95, top=0.87, bottom=0.14)
    colorbar = figure.colorbar(
        probability_images[0],
        ax=axes[:, 5],
        fraction=0.025,
        pad=0.02,
    )
    colorbar.set_label("Probability", color=SUBTLE_TEXT_COLOR, fontsize=9)
    colorbar.ax.tick_params(labelsize=8, colors=SUBTLE_TEXT_COLOR)
    legend = [
        Patch(facecolor=NON_CROP_COLOR, label="No crop"),
        Patch(facecolor=CROP_COLOR, label="Crop"),
        Patch(facecolor=IGNORED_COLOR, label="Ignored / ambiguous reference"),
    ]
    figure.legend(
        handles=legend,
        loc="lower center",
        ncol=3,
        fontsize=10,
        frameon=False,
        bbox_to_anchor=(0.55, 0.06),
    )
    figure.text(
        0.04,
        0.012,
        (
            "Binary accuracy excludes no-data and the ambiguous “Other” reference "
            f"class. Threshold {CROP_CLASSIFICATION_THRESHOLD:.3f} was calibrated "
            "on internal validation; not a held-out test."
        ),
        fontsize=8.5,
        color=SUBTLE_TEXT_COLOR,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor=BACKGROUND_COLOR)
    plt.close(figure)


def render_validation_examples(
    config_path: Path,
    checkpoint: Path,
    output_directory: Path,
    *,
    batch_size: int = 32,
    num_workers: int = 4,
) -> list[dict[str, Any]]:
    examples, gpu_name = _load_selected_examples(
        config_path,
        checkpoint,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    full_path = output_directory / "full_classifier_validation.png"
    binary_path = output_directory / "crop_binary_validation.png"
    _render_full_classifier(examples, full_path)
    _render_binary_classifier(examples, binary_path)

    records = [
        {
            "selection": example.selection,
            "percentile": example.percentile,
            "chip_id": example.chip_id,
            "validation_index": example.validation_index,
            "fine_class_pixel_accuracy": example.full_accuracy,
            "crop_binary_pixel_accuracy": example.binary_accuracy,
        }
        for example in examples
    ]
    metadata = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": MODEL_CHECKPOINT_SHA256,
        "config": str(config_path),
        "evaluation_scope": "internal validation; held-out test not consumed",
        "gpu": gpu_name,
        "crop_threshold": CROP_CLASSIFICATION_THRESHOLD,
        "selection_method": "fine-class accuracy percentiles",
        "outputs": {
            "full_classifier": str(full_path),
            "crop_binary": str(binary_path),
        },
        "examples": records,
    }
    metadata_path = output_directory / "validation_examples.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/prithvi_4band_europe_replay.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "outputs/prithvi_4band_europe_replay/checkpoints/"
            f"{MODEL_CHECKPOINT_NAME}"
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("outputs/model_examples"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    records = render_validation_examples(
        args.config,
        args.checkpoint,
        args.output_directory,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Saved business visuals to: {args.output_directory.resolve()}")
    for record in records:
        print(
            f"  {record['selection']}: {record['chip_id']} | "
            f"fine {record['fine_class_pixel_accuracy']:.1%} | "
            f"binary {record['crop_binary_pixel_accuracy']:.1%}"
        )


if __name__ == "__main__":
    main()

"""Render representative real-validation predictions from a trained checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from prithvi_crop.constants import CLASS_NAMES
from prithvi_crop.runtime import build_data_module, build_task, load_config


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _rgb(image: torch.Tensor, time_index: int) -> np.ndarray:
    array = image[[2, 1, 0], time_index].permute(1, 2, 0).numpy()
    low = np.percentile(array, 2, axis=(0, 1), keepdims=True)
    high = np.percentile(array, 98, axis=(0, 1), keepdims=True)
    return np.clip((array - low) / np.maximum(high - low, 1e-6), 0, 1)


def _chip_id(data_module, validation_index: int) -> str:
    subset = data_module.val_dataset
    source_index = subset.indices[validation_index]
    image_path = Path(subset.dataset.image_files[source_index])
    return image_path.name.removesuffix("_merged.tif")


def render_validation_examples(
    config_path: Path,
    checkpoint: Path,
    output_path: Path,
) -> list[dict]:
    if not torch.cuda.is_available():
        raise RuntimeError("Validation visualization requires CUDA")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    config = load_config(config_path)
    data_module = build_data_module(config, num_workers=0)
    data_module.setup("validate")

    task = build_task(config)
    checkpoint_data = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    task.load_state_dict(checkpoint_data["state_dict"], strict=True)
    device = torch.device("cuda", 0)
    task = task.to(device).eval()

    scores: list[float] = []
    with torch.inference_mode():
        for batch in data_module.val_dataloader():
            normalized = data_module.aug(batch)
            normalized = _to_device(normalized, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = task(
                    normalized["image"],
                    temporal_coords=normalized["temporal_coords"],
                    location_coords=normalized["location_coords"],
                )
            predictions = output.output.argmax(dim=1)
            targets = normalized["mask"]
            valid = targets != -1
            correct = (predictions == targets) & valid
            scores.extend(
                (
                    correct.flatten(1).sum(dim=1)
                    / valid.flatten(1).sum(dim=1).clamp_min(1)
                )
                .cpu()
                .tolist()
            )

    ordered = np.argsort(scores)
    selections = [
        ("Challenging (10th percentile)", int(ordered[round(0.1 * (len(ordered) - 1))])),
        ("Typical (50th percentile)", int(ordered[round(0.5 * (len(ordered) - 1))])),
        ("Strong (90th percentile)", int(ordered[round(0.9 * (len(ordered) - 1))])),
    ]

    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(CLASS_NAMES)))
    class_cmap = ListedColormap(colors)
    figure, axes = plt.subplots(len(selections), 6, figsize=(20, 12))
    records = []

    with torch.inference_mode():
        for row, (selection_name, validation_index) in enumerate(selections):
            sample = data_module.val_dataset[validation_index]
            raw_image = sample["image"].clone()
            batch = data_module.collate_fn([sample])
            normalized = _to_device(data_module.aug(batch), device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = task(
                    normalized["image"],
                    temporal_coords=normalized["temporal_coords"],
                    location_coords=normalized["location_coords"],
                )
            prediction = output.output.argmax(dim=1)[0].cpu().numpy()
            target = sample["mask"].numpy()
            error = prediction != target
            chip_id = _chip_id(data_module, validation_index)
            accuracy = scores[validation_index]

            for time_index in range(3):
                axes[row, time_index].imshow(_rgb(raw_image, time_index))
                axes[row, time_index].set_title(f"Date {time_index + 1}")
            axes[row, 3].imshow(target, cmap=class_cmap, vmin=0, vmax=12)
            axes[row, 3].set_title("Ground truth")
            axes[row, 4].imshow(prediction, cmap=class_cmap, vmin=0, vmax=12)
            axes[row, 4].set_title("Prediction")
            axes[row, 5].imshow(error, cmap=ListedColormap(["#202020", "#ef4444"]))
            axes[row, 5].set_title("Errors in red")

            axes[row, 0].set_ylabel(
                f"{selection_name}\n{chip_id}\nPixel accuracy: {accuracy:.1%}",
                fontsize=10,
            )
            for axis in axes[row]:
                axis.set_xticks([])
                axis.set_yticks([])

            records.append(
                {
                    "selection": selection_name,
                    "chip_id": chip_id,
                    "validation_index": validation_index,
                    "pixel_accuracy": accuracy,
                }
            )

    legend = [
        Patch(facecolor=colors[index], label=f"{index}: {name}")
        for index, name in enumerate(CLASS_NAMES)
    ]
    figure.legend(
        handles=legend,
        loc="lower center",
        ncol=5,
        fontsize=9,
        frameon=False,
    )
    figure.suptitle(
        "Prithvi four-band crop/land-cover predictions — internal validation",
        fontsize=16,
    )
    figure.tight_layout(rect=(0, 0.12, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/prithvi_4band_head_only.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/prithvi_4band_head_only/examples/validation_examples.png"
        ),
    )
    args = parser.parse_args()
    records = render_validation_examples(
        args.config,
        args.checkpoint,
        args.output,
    )
    print(f"Saved: {args.output.resolve()}")
    for record in records:
        print(
            f"  {record['selection']}: {record['chip_id']} "
            f"({record['pixel_accuracy']:.1%})"
        )


if __name__ == "__main__":
    main()

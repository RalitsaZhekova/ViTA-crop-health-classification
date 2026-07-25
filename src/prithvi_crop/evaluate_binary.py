"""Calibrate and evaluate crop/non-crop decisions on internal validation data."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from prithvi_crop.binary import (
    binary_targets_from_fine_targets,
    crop_probability_from_fine_logits,
)
from prithvi_crop.runtime import build_data_module, build_task, load_config


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _threshold_metrics(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    tp = int(np.count_nonzero(positive_scores >= threshold))
    fn = int(positive_scores.size - tp)
    fp = int(np.count_nonzero(negative_scores >= threshold))
    tn = int(negative_scores.size - fp)
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    return {
        "threshold": float(threshold),
        "accuracy": _ratio(tp + tn, tp + tn + fp + fn),
        "balanced_accuracy": (recall + specificity) / 2,
        "crop_precision": precision,
        "crop_recall": recall,
        "crop_f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "crop_iou": _ratio(tp, tp + fp + fn),
        "false_crop_rate": _ratio(fp, tn + fp),
        "false_non_crop_rate": _ratio(fn, tp + fn),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def _best(records: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    return max(
        records,
        key=lambda record: (record[metric], -abs(record["threshold"] - 0.5)),
    )


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def evaluate_binary_validation(
    config_path: Path,
    checkpoint_path: Path,
    *,
    output_path: Path,
    batch_size: int = 16,
    num_workers: int = 4,
    target_false_crop_rate: float = 0.1,
) -> dict[str, Any]:
    """Evaluate a checkpoint without consuming the held-out test split."""
    if not torch.cuda.is_available():
        raise RuntimeError("Binary validation requires CUDA")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not 0 < target_false_crop_rate < 1:
        raise ValueError("target_false_crop_rate must be strictly between 0 and 1")

    config = load_config(config_path)
    data = build_data_module(
        config,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    data.setup("validate")

    task = build_task(config)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    task.load_state_dict(checkpoint["state_dict"], strict=True)
    device = torch.device("cuda", 0)
    task = task.to(device).eval()
    torch.cuda.reset_peak_memory_stats(device)

    positive_scores: list[np.ndarray] = []
    negative_scores: list[np.ndarray] = []
    loader = data.val_dataloader()
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            normalized = _to_device(data.aug(batch), device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = task(
                    normalized["image"],
                    temporal_coords=normalized["temporal_coords"],
                    location_coords=normalized["location_coords"],
                )
            probabilities = crop_probability_from_fine_logits(output.output)
            targets = binary_targets_from_fine_targets(normalized["mask"])
            valid = targets != -1
            positive_scores.append(
                probabilities[valid & (targets == 1)].float().cpu().numpy()
            )
            negative_scores.append(
                probabilities[valid & (targets == 0)].float().cpu().numpy()
            )
            if batch_index % 5 == 0 or batch_index == len(loader):
                print(f"Evaluated batches: {batch_index}/{len(loader)}", flush=True)

    elapsed = time.perf_counter() - started
    positive = np.concatenate(positive_scores)
    negative = np.concatenate(negative_scores)
    thresholds = np.unique(
        np.concatenate((np.linspace(0.05, 0.95, 181), np.asarray([0.5])))
    )
    records = [
        _threshold_metrics(positive, negative, float(threshold))
        for threshold in thresholds
    ]
    constrained = [
        record
        for record in records
        if record["false_crop_rate"] <= target_false_crop_rate
    ]
    safe = (
        max(
            constrained,
            key=lambda record: (
                record["crop_recall"],
                record["accuracy"],
                -record["threshold"],
            ),
        )
        if constrained
        else _best(records, "balanced_accuracy")
    )
    report = {
        "evaluation_scope": "internal validation only; held-out test not consumed",
        "checkpoint": str(checkpoint_path.resolve()),
        "validation_chips": len(data.val_dataset),
        "evaluated_pixels": int(positive.size + negative.size),
        "crop_pixels": int(positive.size),
        "non_crop_pixels": int(negative.size),
        "target_false_crop_rate": target_false_crop_rate,
        "candidates": {
            "default_0_5": _threshold_metrics(positive, negative, 0.5),
            "best_accuracy": _best(records, "accuracy"),
            "best_balanced_accuracy": _best(records, "balanced_accuracy"),
            "best_crop_f1": _best(records, "crop_f1"),
            "recommended_for_health_mask": safe,
        },
        "runtime": {
            "seconds": elapsed,
            "chips_per_second": len(data.val_dataset) / elapsed,
            "gpu": torch.cuda.get_device_name(device),
            "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
            "batch_size": batch_size,
            "num_workers": num_workers,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate crop/non-crop probabilities on internal validation."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/prithvi_4band_europe_replay.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/prithvi_4band_europe_replay/binary_validation_metrics.json"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--target-false-crop-rate", type=float, default=0.1)
    args = parser.parse_args()
    evaluate_binary_validation(
        args.config,
        args.checkpoint,
        output_path=args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_false_crop_rate=args.target_false_crop_rate,
    )


if __name__ == "__main__":
    main()

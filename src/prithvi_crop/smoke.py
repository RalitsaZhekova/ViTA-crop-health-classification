"""One-real-batch CUDA forward/backward verification."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from prithvi_crop.check_config import check_config
from prithvi_crop.runtime import build_data_module, build_task, load_config


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def run_smoke(config_path: Path, batch_size: int | None = None) -> None:
    check_config(config_path)
    if not torch.cuda.is_available():
        raise RuntimeError("Smoke test requires CUDA and will not silently use CPU")

    config = load_config(config_path)
    configured_batch_size = int(config["data"]["init_args"]["batch_size"])
    smoke_batch_size = max(2, batch_size or configured_batch_size)
    module = build_data_module(config, batch_size=smoke_batch_size, num_workers=0)
    module.setup("fit")
    batch = next(iter(module.train_dataloader()))
    expected_frames = int(config["model"]["init_args"]["model_args"]["backbone_num_frames"])
    expected_classes = int(config["model"]["init_args"]["model_args"]["num_classes"])
    binary_only = bool(config["model"]["init_args"].get("binary_only", False))
    expected_shape = (smoke_batch_size, 4, expected_frames, 224, 224)
    if tuple(batch["image"].shape) != expected_shape:
        raise RuntimeError(
            f"Expected real-data batch shape {expected_shape}, got {tuple(batch['image'].shape)}"
        )

    device = torch.device("cuda", 0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    batch = _to_device(batch, device)
    batch = module.aug(batch)
    task = build_task(config).to(device)
    task.train()

    frozen = list(task.model.encoder.named_parameters())
    if not frozen or any(parameter.requires_grad for _, parameter in frozen):
        raise RuntimeError("Backbone has trainable parameters")
    frozen_versions = {name: parameter._version for name, parameter in frozen}
    frozen_samples = {
        name: float(parameter.detach().reshape(-1)[0].cpu())
        for name, parameter in frozen
        if parameter.numel()
    }
    trainable = [
        (name, parameter) for name, parameter in task.named_parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("No downstream parameters are trainable")
    trainable_versions = {name: parameter._version for name, parameter in trainable}

    optimizer_args = config["optimizer"]["init_args"]
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        **optimizer_args,
    )
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=config["trainer"]["precision"] == "16-mixed",
    ):
        output = task(
            batch["image"],
            temporal_coords=batch["temporal_coords"],
            location_coords=batch["location_coords"],
        )
        if tuple(output.output.shape) != (
            smoke_batch_size,
            expected_classes,
            224,
            224,
        ):
            raise RuntimeError(f"Unexpected logits shape: {tuple(output.output.shape)}")
        target = task._binary_batch(batch)["mask"] if binary_only else batch["mask"]
        loss = task.criterion(output.output, target)
    loss.backward()

    if any(parameter.grad is not None for _, parameter in frozen):
        raise RuntimeError("Frozen backbone received gradients")
    trainable_with_gradient = [
        name
        for name, parameter in trainable
        if parameter.grad is not None and torch.isfinite(parameter.grad).all()
    ]
    if not trainable_with_gradient:
        raise RuntimeError("No trainable downstream parameter received a finite gradient")
    optimizer.step()
    torch.cuda.synchronize(device)

    changed_trainable = [
        name for name, parameter in trainable if parameter._version != trainable_versions[name]
    ]
    if not changed_trainable:
        raise RuntimeError("Optimizer step did not update downstream parameters")
    for name, parameter in frozen:
        if parameter._version != frozen_versions[name]:
            raise RuntimeError(f"Frozen backbone parameter version changed: {name}")
        current_sample = (
            float(parameter.detach().reshape(-1)[0].cpu()) if parameter.numel() else None
        )
        if parameter.numel() and current_sample != frozen_samples[name]:
            raise RuntimeError(f"Frozen backbone parameter value changed: {name}")

    print("GPU smoke test passed:")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  input: {tuple(batch['image'].shape)} {batch['image'].dtype} {batch['image'].device}")
    print(f"  temporal_coords: {tuple(batch['temporal_coords'].shape)}")
    print(f"  location_coords: {tuple(batch['location_coords'].shape)}")
    print(f"  logits: {tuple(output.output.shape)}")
    print(f"  loss: {float(loss.detach()):.6f}")
    print(f"  frozen backbone tensors: {len(frozen)}")
    print(f"  trainable tensors with gradients: {len(trainable_with_gradient)}")
    print(f"  updated downstream tensors: {len(changed_trainable)}")
    print(f"  peak GPU memory MiB: {torch.cuda.max_memory_allocated() / 1024**2:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one real-data CUDA training batch.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/prithvi_4band_head_only.yaml"),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Optional batch-size probe; UPerNet training requires at least 2.",
    )
    args = parser.parse_args()
    run_smoke(args.config, batch_size=args.batch_size)


if __name__ == "__main__":
    main()

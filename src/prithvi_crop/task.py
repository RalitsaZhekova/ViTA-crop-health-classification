"""Crop segmentation task with complete, persisted multiclass evaluation metrics."""

from __future__ import annotations

import csv
import json
from functools import partial
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from terratorch.tasks.segmentation_tasks import (
    SemanticSegmentationTask,
    to_segmentation_prediction,
)
from torch import Tensor, nn
from torchmetrics import ClasswiseWrapper, MeanMetric, MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassJaccardIndex,
    MulticlassPrecision,
    MulticlassRecall,
)

from prithvi_crop.binary import (
    binary_logits_from_fine_logits,
    binary_predictions_from_fine_logits,
    binary_targets_from_fine_targets,
)
from prithvi_crop.constants import CLASS_NAMES
from prithvi_crop.europe import PROJECT_CROP_CLASSES, PROJECT_NON_CROP_CLASSES


def _cross_entropy_or_zero(
    logits: Tensor,
    target: Tensor,
    ignore_index: int,
) -> Tensor:
    """Return a differentiable zero when a target contains no supervised pixels."""
    if torch.any(target != ignore_index):
        return F.cross_entropy(logits, target, ignore_index=ignore_index)
    return logits.sum() * 0


def _safe_class_labels(class_names: list[str] | None, num_classes: int) -> list[str]:
    names = class_names or list(CLASS_NAMES)
    if len(names) != num_classes:
        raise ValueError(f"Expected {num_classes} class names, got {len(names)}")
    return [name.lower().replace(" / ", "_").replace(" ", "_").replace("/", "_") for name in names]


def _adapt_three_frame_state_dict(
    source: dict[str, Tensor],
    destination: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Adapt a trained three-frame decoder to the native one-frame backbone."""
    adapted: dict[str, Tensor] = {}
    temporal_convolutions = {
        "decoder.psp_modules.0.1.conv.weight",
        "decoder.psp_modules.1.1.conv.weight",
        "decoder.psp_modules.2.1.conv.weight",
        "decoder.psp_modules.3.1.conv.weight",
        "decoder.lateral_convs.0.conv.weight",
        "decoder.lateral_convs.1.conv.weight",
        "decoder.lateral_convs.2.conv.weight",
    }
    for name, target in destination.items():
        if name not in source:
            raise ValueError(f"Initial checkpoint is missing model tensor: {name}")
        value = source[name]
        if value.shape == target.shape:
            adapted[name] = value
            continue
        if name == "encoder.pos_embed":
            # Prithvi's fixed 3-D sinusoidal positions depend on frame count.
            # Keep the correctly generated native one-frame buffer.
            adapted[name] = target
            continue
        if name in temporal_convolutions:
            if (
                value.ndim != 4
                or target.ndim != 4
                or value.shape[0] != target.shape[0]
                or value.shape[1] != target.shape[1] * 3
                or value.shape[2:] != target.shape[2:]
            ):
                raise ValueError(f"Cannot adapt temporal convolution {name}")
            adapted[name] = value.reshape(
                value.shape[0],
                3,
                target.shape[1],
                *value.shape[2:],
            ).sum(dim=1)
            continue
        if name == "decoder.bottleneck.conv.weight":
            temporal_channels = 768
            shared_channels = target.shape[1] - temporal_channels
            temporal_source = value[:, shared_channels:]
            if (
                value.ndim != 4
                or target.ndim != 4
                or shared_channels < 0
                or value.shape[0] != target.shape[0]
                or value.shape[1] != shared_channels + 3 * temporal_channels
                or value.shape[2:] != target.shape[2:]
            ):
                raise ValueError("Cannot adapt decoder bottleneck")
            collapsed = temporal_source.reshape(
                value.shape[0],
                3,
                temporal_channels,
                *value.shape[2:],
            ).sum(dim=1)
            adapted[name] = torch.cat((value[:, :shared_channels], collapsed), dim=1)
            continue
        raise ValueError(
            f"Unsupported three-to-one-frame tensor change for {name}: "
            f"{tuple(value.shape)} -> {tuple(target.shape)}"
        )
    return adapted


def _adapt_fine_head_to_binary(
    source: dict[str, Tensor],
    destination: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Warm-start a two-class model from a trained single-frame fine model."""
    adapted: dict[str, Tensor] = {}
    head_tensors = {"head.head.2.weight", "head.head.2.bias"}
    groups = (PROJECT_NON_CROP_CLASSES, PROJECT_CROP_CLASSES)
    for name, target in destination.items():
        if name not in source:
            raise ValueError(f"Initial checkpoint is missing model tensor: {name}")
        value = source[name]
        if value.shape == target.shape:
            adapted[name] = value
            continue
        if name in head_tensors and value.shape[0] == 13 and target.shape[0] == 2:
            adapted[name] = torch.stack(
                [value[list(class_indices)].mean(dim=0) for class_indices in groups]
            )
            continue
        raise ValueError(
            f"Unsupported fine-to-binary tensor change for {name}: "
            f"{tuple(value.shape)} -> {tuple(target.shape)}"
        )
    return adapted


class CropSegmentationTask(SemanticSegmentationTask):
    """TerraTorch segmentation task with the metrics required by this project."""

    def __init__(
        self,
        model_args: dict | None = None,
        model_factory: str | None = None,
        model: torch.nn.Module | None = None,
        loss: str | list[str] | dict[str, float] | None = "ce",
        aux_heads: list[Any] | None = None,
        aux_loss: dict[str, float] | None = None,
        class_weights: list[float] | None = None,
        ignore_index: int | None = None,
        lr: float = 0.001,
        optimizer: str | None = None,
        optimizer_hparams: dict | None = None,
        scheduler: str | None = None,
        scheduler_hparams: dict | None = None,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        freeze_head: bool = False,
        plot_on_val: bool | int = False,
        class_names: list[str] | None = None,
        tiled_inference_parameters: dict | None = None,
        test_dataloaders_names: list[str] | None = None,
        lr_overrides: dict[str, float] | None = None,
        output_on_inference: str | list[str] = "prediction",
        output_most_probable: bool = True,
        path_to_record_metrics: str | None = None,
        tiled_inference_on_testing: bool = False,
        tiled_inference_on_validation: bool = False,
        evaluation_output_dir: str = "outputs/prithvi_4band_head_only/evaluation",
        initial_checkpoint: str | None = None,
        initial_checkpoint_adapter: str | None = None,
        crop_binary_loss_weight: float = 0.0,
        validation_crop_threshold: float = 0.5,
        selection_binary_weight: float = 0.7,
        binary_only: bool = False,
    ) -> None:
        self.evaluation_output_dir = Path(evaluation_output_dir)
        self.binary_only = binary_only
        if not 0 <= crop_binary_loss_weight <= 1:
            raise ValueError("crop_binary_loss_weight must be between 0 and 1")
        self.crop_binary_loss_weight = crop_binary_loss_weight
        if not 0 < validation_crop_threshold < 1:
            raise ValueError("validation_crop_threshold must be strictly between 0 and 1")
        if not 0 <= selection_binary_weight <= 1:
            raise ValueError("selection_binary_weight must be between 0 and 1")
        self.validation_crop_threshold = validation_crop_threshold
        self.selection_binary_weight = selection_binary_weight
        self.initial_checkpoint_adapter = initial_checkpoint_adapter
        super().__init__(
            model_args=model_args,
            model_factory=model_factory,
            model=model,
            loss=loss,
            aux_heads=aux_heads,
            aux_loss=aux_loss,
            class_weights=class_weights,
            ignore_index=ignore_index,
            lr=lr,
            optimizer=optimizer,
            optimizer_hparams=optimizer_hparams,
            scheduler=scheduler,
            scheduler_hparams=scheduler_hparams,
            freeze_backbone=freeze_backbone,
            freeze_decoder=freeze_decoder,
            freeze_head=freeze_head,
            plot_on_val=plot_on_val,
            class_names=class_names,
            tiled_inference_parameters=tiled_inference_parameters,
            test_dataloaders_names=test_dataloaders_names,
            lr_overrides=lr_overrides,
            output_on_inference=output_on_inference,
            output_most_probable=output_most_probable,
            path_to_record_metrics=path_to_record_metrics,
            tiled_inference_on_testing=tiled_inference_on_testing,
            tiled_inference_on_validation=tiled_inference_on_validation,
        )
        if initial_checkpoint is not None:
            self._load_model_weights(
                Path(initial_checkpoint),
                adapter=initial_checkpoint_adapter,
            )

    def _load_model_weights(
        self,
        checkpoint_path: Path,
        *,
        adapter: str | None = None,
    ) -> None:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Initial checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError(f"Invalid Lightning checkpoint: {checkpoint_path}")
        model_state = {
            name.removeprefix("model."): value
            for name, value in state_dict.items()
            if name.startswith("model.")
        }
        if adapter is None:
            adapted_state = model_state
        elif adapter == "three_to_one_frame":
            adapted_state = _adapt_three_frame_state_dict(
                model_state,
                self.model.state_dict(),
            )
        elif adapter == "fine_to_binary":
            adapted_state = _adapt_fine_head_to_binary(
                model_state,
                self.model.state_dict(),
            )
        else:
            raise ValueError(f"Unknown initial checkpoint adapter: {adapter}")
        self.model.load_state_dict(adapted_state, strict=True)
        suffix = f" with {adapter}" if adapter else ""
        print(f"Initialized model weights from: {checkpoint_path.resolve()}{suffix}")

    def training_step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Tensor:
        if self.binary_only:
            return SemanticSegmentationTask.training_step(
                self,
                self._binary_batch(batch),
                batch_idx,
                dataloader_idx,
            )
        if "crop_mask" not in batch or self.crop_binary_loss_weight == 0:
            return super().training_step(batch, batch_idx, dataloader_idx)

        x = batch["image"]
        fine_target = self.squeeze_ground_truth(batch["mask"])
        crop_target = self.squeeze_ground_truth(batch["crop_mask"])
        excluded = {
            "image",
            "mask",
            "crop_mask",
            "dataset_source",
            "filename",
        }
        model_output = self(
            x,
            **{key: batch[key] for key in batch.keys() - excluded},
        )
        logits = model_output.output

        if torch.any(fine_target != self.hparams["ignore_index"]):
            fine_loss = self.criterion(logits, fine_target)
        else:
            fine_loss = logits.sum() * 0
        binary_logits = binary_logits_from_fine_logits(logits)
        binary_loss = _cross_entropy_or_zero(
            binary_logits,
            crop_target,
            self.hparams["ignore_index"],
        )
        loss = fine_loss + self.crop_binary_loss_weight * binary_loss
        batch_size = fine_target.shape[0]
        self.log("train/loss", loss, batch_size=batch_size)
        self.log("train/fine_loss", fine_loss, batch_size=batch_size)
        self.log(
            "train/crop_binary_loss",
            binary_loss,
            batch_size=batch_size,
        )
        self.train_metrics.update(logits.argmax(dim=1), fine_target)
        return loss

    def _binary_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        target = batch.get("crop_mask")
        if target is None:
            target = binary_targets_from_fine_targets(
                batch["mask"],
                ignore_index=self.hparams["ignore_index"],
            )
        converted = {
            key: batch[key]
            for key in (
                "image",
                "temporal_coords",
                "location_coords",
                "filename",
            )
            if key in batch
        }
        converted["mask"] = target
        return converted

    def configure_metrics(self) -> None:
        num_classes: int = self.hparams["model_args"]["num_classes"]
        ignore_index: int = self.hparams["ignore_index"]
        labels = _safe_class_labels(self.hparams["class_names"], num_classes)
        metrics = MetricCollection(
            {
                "Overall_Accuracy": MulticlassAccuracy(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="micro",
                ),
                "Balanced_Accuracy": MulticlassRecall(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="macro",
                ),
                "Macro_Precision": MulticlassPrecision(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="macro",
                ),
                "Macro_Recall": MulticlassRecall(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="macro",
                ),
                "Macro_F1": MulticlassF1Score(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="macro",
                ),
                "mIoU": MulticlassJaccardIndex(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    average="macro",
                ),
                "Per_Class_Precision": ClasswiseWrapper(
                    MulticlassPrecision(
                        num_classes=num_classes,
                        ignore_index=ignore_index,
                        average=None,
                    ),
                    labels=labels,
                    prefix="Precision_",
                ),
                "Per_Class_Recall": ClasswiseWrapper(
                    MulticlassRecall(
                        num_classes=num_classes,
                        ignore_index=ignore_index,
                        average=None,
                    ),
                    labels=labels,
                    prefix="Recall_",
                ),
                "Per_Class_F1": ClasswiseWrapper(
                    MulticlassF1Score(
                        num_classes=num_classes,
                        ignore_index=ignore_index,
                        average=None,
                    ),
                    labels=labels,
                    prefix="F1_",
                ),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        names = self.hparams["test_dataloaders_names"]
        prefixes = [f"test/{name}/" for name in names] if names else ["test/"]
        self.test_metrics = nn.ModuleList([metrics.clone(prefix=prefix) for prefix in prefixes])
        self.test_confusion_matrices = nn.ModuleList(
            [
                MulticlassConfusionMatrix(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                )
                for _ in prefixes
            ]
        )
        self.test_loss_metrics = nn.ModuleList([MeanMetric() for _ in prefixes])
        self.val_binary_metrics = MetricCollection(
            {
                "Crop_Binary_Accuracy": MulticlassAccuracy(
                    num_classes=2,
                    ignore_index=ignore_index,
                    average="micro",
                ),
                "Crop_Binary_Balanced_Accuracy": MulticlassRecall(
                    num_classes=2,
                    ignore_index=ignore_index,
                    average="macro",
                ),
                "Crop_Binary_Macro_F1": MulticlassF1Score(
                    num_classes=2,
                    ignore_index=ignore_index,
                    average="macro",
                ),
            },
            prefix="val/",
        )

    def validation_step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.binary_only:
            return SemanticSegmentationTask.validation_step(
                self,
                self._binary_batch(batch),
                batch_idx,
                dataloader_idx,
            )
        x = batch["image"]
        target = self.squeeze_ground_truth(batch["mask"])
        excluded = {
            "image",
            "mask",
            "crop_mask",
            "dataset_source",
            "filename",
        }
        model_output = self.handle_full_or_tiled_inference(
            x,
            self.tiled_inference_on_validation,
            **{key: batch[key] for key in batch.keys() - excluded},
        )
        loss = self.val_loss_handler.compute_loss(
            model_output,
            target,
            self.criterion,
            self.aux_loss,
        )
        self.val_loss_handler.log_loss(
            self.log,
            loss_dict=loss,
            batch_size=target.shape[0],
        )
        prediction = to_segmentation_prediction(model_output)
        self.val_metrics.update(prediction, target)
        binary_target = binary_targets_from_fine_targets(
            target,
            ignore_index=self.hparams["ignore_index"],
        )
        binary_prediction = binary_predictions_from_fine_logits(
            model_output.output,
            crop_threshold=self.validation_crop_threshold,
        )
        self.val_binary_metrics.update(binary_prediction, binary_target)

    def on_validation_epoch_end(self) -> None:
        if self.binary_only:
            return SemanticSegmentationTask.on_validation_epoch_end(self)
        fine_metrics = self.val_metrics.compute()
        binary_metrics = self.val_binary_metrics.compute()
        fine_f1 = fine_metrics["val/Macro_F1"]
        binary_accuracy = binary_metrics["val/Crop_Binary_Accuracy"]
        deployment_score = (
            self.selection_binary_weight * binary_accuracy
            + (1 - self.selection_binary_weight) * fine_f1
        )
        self.log_dict(binary_metrics)
        self.log("val/Deployment_Score", deployment_score)
        self.val_binary_metrics.reset()
        super().on_validation_epoch_end()

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if self.binary_only:
            return SemanticSegmentationTask.test_step(
                self,
                self._binary_batch(batch),
                batch_idx,
                dataloader_idx,
            )
        x = batch["image"]
        y = self.squeeze_ground_truth(batch["mask"])
        other_keys = batch.keys() - {"image", "mask", "filename"}
        rest = {key: batch[key] for key in other_keys}
        model_output = self.handle_full_or_tiled_inference(
            x,
            self.tiled_inference_on_testing,
            **rest,
        )

        if dataloader_idx >= len(self.test_loss_handler):
            raise ValueError(
                "More test dataloaders were returned than test_dataloaders_names entries."
            )
        loss = self.test_loss_handler[dataloader_idx].compute_loss(
            model_output,
            y,
            self.criterion,
            self.aux_loss,
        )
        self.test_loss_handler[dataloader_idx].log_loss(
            partial(self.log, add_dataloader_idx=False),
            loss_dict=loss,
            batch_size=y.shape[0],
        )
        prediction = to_segmentation_prediction(model_output)
        self.test_metrics[dataloader_idx].update(prediction, y)
        self.test_confusion_matrices[dataloader_idx].update(prediction, y)
        self.test_loss_metrics[dataloader_idx].update(
            loss["loss"].detach(),
            weight=y.shape[0],
        )

    def on_test_epoch_end(self) -> None:
        self.evaluation_output_dir.mkdir(parents=True, exist_ok=True)
        names = self.hparams["test_dataloaders_names"] or ["test"]
        for index, name in enumerate(names):
            metrics = {
                key: float(value.detach().cpu())
                for key, value in self.test_metrics[index].compute().items()
            }
            metrics["test/loss"] = float(self.test_loss_metrics[index].compute().detach().cpu())
            confusion = self.test_confusion_matrices[index].compute().detach().cpu()
            support = confusion.sum(dim=1)
            for class_index, class_name in enumerate(CLASS_NAMES):
                metrics[f"support/{class_name}"] = int(support[class_index])
            self._write_evaluation_artifacts(name, metrics, confusion)
        super().on_test_epoch_end()
        for metric in self.test_confusion_matrices:
            metric.reset()
        for metric in self.test_loss_metrics:
            metric.reset()

    def _write_evaluation_artifacts(
        self,
        name: str,
        metrics: dict[str, float | int],
        confusion: Tensor,
    ) -> None:
        safe_name = name.replace("/", "_")
        metrics_path = self.evaluation_output_dir / f"{safe_name}_metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        csv_path = self.evaluation_output_dir / f"{safe_name}_confusion_matrix.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["actual/predicted", *CLASS_NAMES])
            for class_name, row in zip(CLASS_NAMES, confusion.tolist(), strict=True):
                writer.writerow([class_name, *row])

        figure, axis = plt.subplots(figsize=(12, 10))
        image = axis.imshow(confusion.numpy(), cmap="Blues")
        axis.set(
            xlabel="Predicted class",
            ylabel="Actual class",
            title="Held-out test confusion matrix",
            xticks=range(len(CLASS_NAMES)),
            yticks=range(len(CLASS_NAMES)),
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
        )
        axis.tick_params(axis="x", labelrotation=90)
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        figure.savefig(
            self.evaluation_output_dir / f"{safe_name}_confusion_matrix.png",
            dpi=160,
        )
        plt.close(figure)

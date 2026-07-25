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
    return [
        name.lower().replace(" / ", "_").replace(" ", "_").replace("/", "_")
        for name in names
    ]


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
        crop_binary_loss_weight: float = 0.0,
    ) -> None:
        self.evaluation_output_dir = Path(evaluation_output_dir)
        if not 0 <= crop_binary_loss_weight <= 1:
            raise ValueError("crop_binary_loss_weight must be between 0 and 1")
        self.crop_binary_loss_weight = crop_binary_loss_weight
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
            self._load_model_weights(Path(initial_checkpoint))

    def _load_model_weights(self, checkpoint_path: Path) -> None:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Initial checkpoint not found: {checkpoint_path}"
            )
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
        self.model.load_state_dict(model_state, strict=True)
        print(f"Initialized model weights from: {checkpoint_path.resolve()}")

    def training_step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> Tensor:
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
        binary_logits = torch.stack(
            [
                torch.logsumexp(
                    logits[:, PROJECT_NON_CROP_CLASSES],
                    dim=1,
                ),
                torch.logsumexp(
                    logits[:, PROJECT_CROP_CLASSES],
                    dim=1,
                ),
            ],
            dim=1,
        )
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

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
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
            metrics["test/loss"] = float(
                self.test_loss_metrics[index].compute().detach().cpu()
            )
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

"""Direct TensorRT execution for the OmniCloudMask ensemble."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import torch
from prithvi_payload.tensorrt_runtime import NativeTensorRTPlan, TensorRTArtifactError
from torch import Tensor, nn


def _remove_zero_channel_cat_noops(exported: torch.export.ExportedProgram) -> int:
    """Remove static zero-channel inputs from exported ``aten.cat`` nodes.

    The EdgeNeXt encoder used by the pinned OmniCloudMask V4 ensemble emits one
    ``[N, 0, H, W]`` feature so that its feature-pyramid interface has the same
    number of levels as other encoders. PyTorch correctly treats concatenating
    that tensor along the channel dimension as an identity operation. TensorRT
    rejects a zero-element constant, so the offline ONNX exporter removes only
    this exact mathematical no-op.

    Static profiles make the zero channel visible in FX metadata. Replacing only
    a two-input channel concatenation whose other input already has the complete
    output shape is mathematically exact and leaves every non-empty concatenation
    untouched.
    """

    graph_module = exported.graph_module
    graph = graph_module.graph
    removed = 0
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target != torch.ops.aten.cat.default:
            continue
        inputs = node.args[0] if node.args else None
        dimension = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            continue
        if not isinstance(dimension, int):
            continue

        output = node.meta.get("val")
        output_shape = tuple(output.shape) if isinstance(output, Tensor) else None
        if output_shape is None:
            continue
        normalized_dimension = dimension % len(output_shape)

        zero_inputs = []
        live_inputs = []
        for input_node in inputs:
            value = getattr(input_node, "meta", {}).get("val")
            shape = tuple(value.shape) if isinstance(value, Tensor) else None
            if (
                shape is not None
                and len(shape) == len(output_shape)
                and shape[normalized_dimension] == 0
            ):
                zero_inputs.append(input_node)
            else:
                live_inputs.append(input_node)

        if len(zero_inputs) != 1 or len(live_inputs) != 1:
            continue
        live_value = getattr(live_inputs[0], "meta", {}).get("val")
        live_shape = tuple(live_value.shape) if isinstance(live_value, Tensor) else None
        if live_shape != output_shape:
            continue

        node.replace_all_uses_with(live_inputs[0])
        graph.erase_node(node)
        removed += 1

    if removed:
        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()
    return removed


class _CloudEnsemble(nn.Module):
    """Expose the reviewed mean-logit ensemble as one exportable graph."""

    def __init__(self, models: list[nn.Module]) -> None:
        super().__init__()
        if len(models) != 2:
            raise RuntimeError(
                f"OmniCloudMask V4 requires exactly two component models, got {len(models)}"
            )
        self.models = nn.ModuleList(models)

    def forward(self, image: Tensor) -> Tensor:
        return (self.models[0](image) + self.models[1](image)) * 0.5


class CloudTensorRTRouter(nn.Module):
    """Route fixed cloud patch sizes to prebuilt, accepted TensorRT plans."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        manifest_path: Path,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if device.type != "cuda":
            raise RuntimeError("TensorRT cloud inference requires a CUDA device")
        self.device = device
        self.dtype = dtype
        expected_precision = "fp16" if dtype == torch.float16 else "fp32"
        self._plans: dict[tuple[int, int], NativeTensorRTPlan] = {}
        self._records: dict[tuple[int, int], dict[str, Any]] = {}
        self._used_profiles: set[tuple[int, int, int]] = set()
        self._lock = threading.Lock()
        for record in records:
            if not isinstance(record, dict):
                raise TensorRTArtifactError("Cloud TensorRT profile record is invalid")
            if record.get("precision") != expected_precision:
                raise TensorRTArtifactError(
                    "Cloud TensorRT plan precision does not match inference_dtype"
                )
            patch_size = record.get("patch_size")
            if not isinstance(patch_size, int) or patch_size < 32:
                raise TensorRTArtifactError("Cloud TensorRT patch size is invalid")
            key = (patch_size, patch_size)
            if key in self._plans:
                raise TensorRTArtifactError(
                    f"Duplicate cloud TensorRT plan for {patch_size}px patches"
                )
            self._plans[key] = NativeTensorRTPlan(
                record,
                manifest_path=manifest_path,
                device=device,
            )
            self._records[key] = record
        if not self._plans:
            raise TensorRTArtifactError("No accepted cloud TensorRT plans were supplied")

    @property
    def engine_count(self) -> int:
        return len(self._plans)

    @property
    def profiles(self) -> list[dict[str, Any]]:
        profiles: list[dict[str, Any]] = []
        for key in sorted(self._records):
            record = self._records[key]
            profile = record["inputs"][0].get("profile")
            static_batch_size = record["inputs"][0]["shape"][0]
            profiles.append(
                {
                    "patch_size": record["patch_size"],
                    "minimum_batch_size": (
                        profile["min"][0] if profile else static_batch_size
                    ),
                    "maximum_batch_size": (
                        profile["max"][0] if profile else static_batch_size
                    ),
                    "engine_count": 1,
                    "precision": record["precision"],
                    "plan_sha256": record["plan_sha256"],
                }
            )
        return profiles

    def forward(self, image: Tensor) -> Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise RuntimeError(
                f"Cloud TensorRT expects [batch,3,height,width], got {tuple(image.shape)}"
            )
        batch_size, height, width = (
            int(image.shape[0]),
            int(image.shape[2]),
            int(image.shape[3]),
        )
        plan = self._plans.get((height, width))
        if plan is None:
            raise RuntimeError(
                "Cloud TensorRT has no accepted plan for "
                f"batch={batch_size}, height={height}, width={width}"
            )
        with self._lock:
            self._used_profiles.add((batch_size, height, width))
        output = plan(image)
        if not isinstance(output, Tensor):
            raise RuntimeError("Cloud TensorRT plan returned multiple outputs")
        return output

    def freeze(self) -> None:
        if not self._plans:
            raise RuntimeError("Cloud TensorRT has no accepted plans")

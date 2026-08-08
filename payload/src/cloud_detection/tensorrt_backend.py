"""Strict, profile-cached TensorRT execution for the OmniCloudMask ensemble."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


def _environment_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of 1/0, true/false, yes/no or on/off")


def _engine_count(module: nn.Module) -> int:
    graph = getattr(module, "graph", None)
    graph_nodes = 0
    if graph is not None:
        graph_nodes = sum(
            "tensorrt" in str(node.target).casefold()
            or "run_on_acc" in str(node.target).casefold()
            for node in graph.nodes
        )
    module_nodes = sum(
        "tensorrt" in type(child).__module__.casefold()
        or "tensorrt" in type(child).__name__.casefold()
        for _, child in module.named_modules()
    )
    return max(graph_nodes, module_nodes)


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


class CloudTensorRTRouter:
    """Build one strict static TensorRT engine for every observed MVP profile.

    Static profiles are intentional on the payload: they are more predictable than
    a broad dynamic range on the pinned Torch-TensorRT 2.6 stack, while exact scene
    warmup discovers every batch/patch pair before readiness. After ``freeze`` an
    unseen profile fails instead of compiling inside a timed job.
    """

    def __init__(
        self,
        models: list[nn.Module],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if device.type != "cuda":
            raise RuntimeError("TensorRT cloud inference requires a CUDA device")
        if dtype != torch.float16:
            raise RuntimeError("TensorRT cloud inference currently requires FP16")
        self.device = device
        self.dtype = dtype
        self._source: _CloudEnsemble | None = _CloudEnsemble(models).to(
            device=device,
            dtype=dtype,
        )
        self._source.eval()
        self._compiled: dict[tuple[int, int, int], nn.Module] = {}
        self._engine_counts: dict[tuple[int, int, int], int] = {}
        self._parity: dict[tuple[int, int, int], dict[str, Any]] = {}
        self._frozen = False
        self._lock = threading.Lock()
        self.cache_root = Path(
            os.environ.get("VITA_TRT_CACHE_DIR", "/tmp/vita-torch-tensorrt")
        ).resolve()
        self.engine_cache = self.cache_root / "cloud"
        self.engine_cache.mkdir(parents=True, exist_ok=True)

    @property
    def engine_count(self) -> int:
        return sum(self._engine_counts.values())

    @property
    def profiles(self) -> list[dict[str, Any]]:
        return [
            {
                "batch_size": batch_size,
                "height": height,
                "width": width,
                "engine_count": self._engine_counts[key],
                **self._parity[key],
            }
            for key in sorted(self._compiled)
            for batch_size, height, width in (key,)
        ]

    def _compile(
        self, sample: Tensor
    ) -> tuple[nn.Module, Tensor, int, dict[str, float]]:
        source = self._source
        if source is None:
            raise RuntimeError("Cloud TensorRT profile compilation is frozen")
        try:
            import torch_tensorrt
        except ImportError as error:
            raise RuntimeError(
                "VITA_CLOUD_BACKEND=tensorrt requires the NVIDIA Torch-TensorRT package"
            ) from error

        with torch.inference_mode():
            reference = source(sample)
        exported = torch.export.export(source, (sample,), strict=False)
        compiled = torch_tensorrt.dynamo.compile(
            exported,
            arg_inputs=[sample],
            enabled_precisions={torch.float16},
            require_full_compilation=_environment_flag(
                "VITA_CLOUD_TRT_REQUIRE_FULL", True
            ),
            pass_through_build_failures=True,
            optimization_level=int(
                os.environ.get("VITA_CLOUD_TRT_OPTIMIZATION_LEVEL", "3")
            ),
            num_avg_timing_iters=int(
                os.environ.get("VITA_CLOUD_TRT_NUM_AVG_TIMING_ITERS", "3")
            ),
            workspace_size=int(
                os.environ.get("VITA_CLOUD_TRT_WORKSPACE_BYTES", str(2 * 1024**3))
            ),
            timing_cache_path=str(self.cache_root / "cloud-timing-cache.bin"),
            cache_built_engines=True,
            reuse_cached_engines=True,
            # Required by the Torch-TensorRT 2.6 persistent engine cache API.
            make_refittable=True,
            engine_cache_dir=str(self.engine_cache),
            engine_cache_size=int(
                os.environ.get("VITA_CLOUD_TRT_CACHE_BYTES", str(8 * 1024**3))
            ),
        )
        count = _engine_count(compiled)
        if count < 1:
            raise RuntimeError("Torch-TensorRT produced no cloud engine partitions")
        with torch.inference_mode():
            accelerated = compiled(sample)
        parity = self._validate_output(
            sample,
            accelerated,
            reference=reference,
        )
        return compiled, accelerated, count, parity

    def _validate_output(
        self,
        sample: Tensor,
        accelerated: Any,
        *,
        reference: Tensor | None = None,
    ) -> dict[str, float]:
        source = self._source
        if source is None:
            raise RuntimeError("Cloud TensorRT parity validation requires source models")
        if reference is None:
            with torch.inference_mode():
                reference = source(sample)
        if not isinstance(accelerated, Tensor) or accelerated.shape != reference.shape:
            raise RuntimeError("Cloud TensorRT output contract does not match PyTorch")
        if not bool(torch.isfinite(accelerated).all()):
            raise RuntimeError("Cloud TensorRT produced non-finite logits")

        mismatch = float(
            (accelerated.argmax(dim=1) != reference.argmax(dim=1))
            .float()
            .mean()
            .item()
        )
        maximum_mismatch = float(
            os.environ.get("VITA_CLOUD_TRT_MAX_CLASS_MISMATCH", "0.001")
        )
        if not 0.0 <= maximum_mismatch <= 1.0:
            raise RuntimeError("VITA_CLOUD_TRT_MAX_CLASS_MISMATCH must be within 0..1")
        if mismatch > maximum_mismatch:
            raise RuntimeError(
                "Cloud TensorRT parity failed: "
                f"class mismatch {mismatch:.8f} exceeds {maximum_mismatch:.8f}"
            )
        absolute_error = (accelerated.float() - reference.float()).abs()
        parity = {
            "class_mismatch_fraction": mismatch,
            "mean_absolute_logit_error": float(absolute_error.mean().item()),
            "maximum_absolute_logit_error": float(absolute_error.max().item()),
        }
        return parity

    def __call__(self, image: Tensor) -> Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise RuntimeError(
                f"Cloud TensorRT expects [batch,3,height,width], got {tuple(image.shape)}"
            )
        key = tuple(int(value) for value in (image.shape[0], image.shape[2], image.shape[3]))
        compiled = self._compiled.get(key)
        if compiled is not None:
            accelerated = compiled(image)
            if not self._frozen and _environment_flag(
                "VITA_CLOUD_TRT_VALIDATE_WARMUP_CALLS", True
            ):
                self._parity[key] = self._validate_output(image, accelerated)
            return accelerated
        with self._lock:
            compiled = self._compiled.get(key)
            if compiled is not None:
                accelerated = compiled(image)
                if not self._frozen and _environment_flag(
                    "VITA_CLOUD_TRT_VALIDATE_WARMUP_CALLS", True
                ):
                    self._parity[key] = self._validate_output(image, accelerated)
                return accelerated
            if self._frozen:
                raise RuntimeError(
                    "Cloud TensorRT received an unwarmed profile after readiness: "
                    f"batch={key[0]}, height={key[1]}, width={key[2]}"
                )
            compiled, output, engine_count, parity = self._compile(image)
            self._compiled[key] = compiled
            self._engine_counts[key] = engine_count
            self._parity[key] = parity
            return output

    def freeze(self) -> None:
        if not self._compiled:
            raise RuntimeError("Cloud TensorRT cannot freeze without a compiled profile")
        self._frozen = True
        self._source = None

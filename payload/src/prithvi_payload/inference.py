"""Load the pinned model and run training-free tensor inference."""

from __future__ import annotations

import hashlib
import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from prithvi_shared import (
    CROP_CLASSIFICATION_THRESHOLD,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    MODEL_BANDS,
    NORMALIZATION_MEANS,
    NORMALIZATION_STDS,
    SELECTED_CHECKPOINT_NAME,
    SELECTED_CHECKPOINT_SHA256,
    TIME_STEPS,
)
from torch import Tensor, nn

PAYLOAD_ROOT = Path(__file__).resolve().parents[2]
OPTIMIZED_BATCH_SIZE = int(os.environ.get("VITA_CROP_BATCH_SIZE", "4"))
if not 1 <= OPTIMIZED_BATCH_SIZE <= 16:
    raise RuntimeError("VITA_CROP_BATCH_SIZE must be within 1..16")

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


def default_model_directory() -> Path:
    """Resolve models from an explicit deployment setting or source checkout."""
    configured = os.environ.get("PRITHVI_MODEL_DIR")
    if configured:
        return Path(configured)
    source_checkout = PAYLOAD_ROOT / "models"
    if source_checkout.is_dir():
        return source_checkout
    return Path.cwd() / "models"


@dataclass(frozen=True)
class InferenceOutput:
    """Per-pixel crop/non-crop outputs for one batch of model tiles."""

    crop_probability: Tensor
    crop_binary: Tensor
    crop_confidence: Tensor


class _TensorLogitsModel(nn.Module):
    """Expose TerraTorch's tensor output as a portable PyTorch graph."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        image: Tensor,
        temporal_coords: Tensor,
        location_coords: Tensor,
    ) -> Tensor:
        return self.model(
            image,
            temporal_coords=temporal_coords,
            location_coords=location_coords,
        ).output


def checkpoint_sha256(path: Path) -> str:
    """Hash a checkpoint without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_cache_directory(model_directory: Path) -> Path:
    configured = os.environ.get("VITA_MODEL_CACHE_DIR")
    cache = Path(configured) if configured else model_directory
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _optimized_model_path(model_directory: Path, device: torch.device) -> Path:
    checkpoint_stem = Path(SELECTED_CHECKPOINT_NAME).stem
    version = SELECTED_CHECKPOINT_SHA256[:12]
    torch_version = torch.__version__.split("+", maxsplit=1)[0].replace(".", "_")
    return _model_cache_directory(model_directory) / (
        f"{checkpoint_stem}.{version}.torch_{torch_version}.{device.type}."
        f"batch_{OPTIMIZED_BATCH_SIZE}.export.pt2"
    )


def _example_inputs(device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    return (
        torch.zeros(
            (
                OPTIMIZED_BATCH_SIZE,
                len(MODEL_BANDS),
                TIME_STEPS,
                INPUT_HEIGHT,
                INPUT_WIDTH,
            ),
            device=device,
        ),
        torch.zeros((OPTIMIZED_BATCH_SIZE, TIME_STEPS, 2), device=device),
        torch.zeros((OPTIMIZED_BATCH_SIZE, 2), device=device),
    )


def _export_optimized_model(
    model: nn.Module,
    destination: Path,
    device: torch.device,
) -> None:
    """Atomically cache the fixed-shape inference graph used by both MVPs."""
    image, temporal_coords, location_coords = _example_inputs(device)
    model = model.to(device).eval()
    optimized = torch.export.export(
        model,
        (image, temporal_coords, location_coords),
        strict=False,
    )
    temporary = destination.with_name(f".{destination.stem}.{os.getpid()}.tmp.pt2")
    try:
        torch.export.save(optimized, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _functionalize_prithvi_export_for_tensorrt(exported_program: Any) -> int:
    """Replace the pinned Prithvi helper's safe, temporary in-place divisions.

    TerraTorch 1.1.1 builds four sinusoidal-coordinate vectors with ``div_``.
    PyTorch can execute that exported graph, but portable ONNX graphs must not
    mutate frozen temporary storage. Each temporary has exactly one user, so
    replacing ``div_`` with ``div`` is mathematically identical.
    """
    graph_module = exported_program.graph_module
    replacements = 0
    for node in graph_module.graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.div_.Tensor:
            continue
        source = node.args[0] if node.args else None
        if not isinstance(source, torch.fx.Node) or set(source.users) != {node}:
            raise RuntimeError(
                "Cannot safely functionalize an aliased in-place division in the crop export"
            )
        node.target = torch.ops.aten.div.Tensor
        replacements += 1

    if replacements not in {0, 4}:
        raise RuntimeError(
            "Unexpected Prithvi export mutation count: "
            f"expected zero or four in-place divisions, got {replacements}"
        )
    if replacements:
        graph_module.graph.lint()
        graph_module.recompile()
    return replacements


def _model_state(checkpoint: dict[str, Any]) -> dict[str, Tensor]:
    state = checkpoint.get("state_dict")
    if isinstance(state, dict):
        model_state = {
            name.removeprefix("model."): value
            for name, value in state.items()
            if name.startswith("model.")
        }
    elif checkpoint and all(
        isinstance(name, str) and isinstance(value, Tensor) for name, value in checkpoint.items()
    ):
        model_state = checkpoint
    else:
        raise ValueError("Artifact does not contain a model state dictionary")
    if not model_state:
        raise ValueError("Artifact does not contain model weights")
    return model_state


def load_crop_exported_program(
    model_directory: Path,
    *,
    device: torch.device,
) -> Any:
    """Load or create the source PyTorch export used by offline builders."""
    architecture_path = model_directory / "architecture.yaml"
    checkpoint_path = model_directory / SELECTED_CHECKPOINT_NAME
    optimized_path = _optimized_model_path(model_directory, device)
    if optimized_path.is_file():
        return torch.export.load(optimized_path)
    if not architecture_path.is_file():
        raise FileNotFoundError(architecture_path)

    from terratorch.registry import MODEL_FACTORY_REGISTRY

    architecture = yaml.safe_load(architecture_path.read_text(encoding="utf-8"))
    factory = MODEL_FACTORY_REGISTRY.build(architecture["model_factory"])
    model = factory.build_model(
        task=architecture["task"],
        **architecture["model_args"],
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    model.load_state_dict(_model_state(checkpoint), strict=True)
    _export_optimized_model(_TensorLogitsModel(model), optimized_path, device)
    return torch.export.load(optimized_path)


class PayloadCropModel:
    """Pinned single-image Prithvi model with thresholded binary outputs."""

    def __init__(
        self,
        model: nn.Module,
        *,
        device: torch.device,
        fixed_batch_size: int | None = None,
        exported: bool = False,
        backend: str = "pytorch",
        tensorrt_engine_count: int = 0,
        tensorrt_parity: dict[str, float] | None = None,
        tensorrt_precision: str | None = None,
        tensorrt_tf32: bool | None = None,
    ) -> None:
        # A direct TensorRT runner owns an initialized execution context and has
        # no parameters for nn.Module.to() to move.
        self.model = model if backend == "tensorrt" else model.to(device)
        if not exported:
            self.model.eval()
        self.device = device
        self.backend = backend
        self.tensorrt_engine_count = tensorrt_engine_count
        self.tensorrt_parity = tensorrt_parity or {}
        self.tensorrt_precision = tensorrt_precision
        self.tensorrt_tf32 = tensorrt_tf32
        self.fixed_batch_size = fixed_batch_size
        self._means = torch.tensor(
            NORMALIZATION_MEANS,
            dtype=torch.float32,
            device=device,
        ).view(1, len(MODEL_BANDS), 1, 1, 1)
        self._stds = torch.tensor(
            NORMALIZATION_STDS,
            dtype=torch.float32,
            device=device,
        ).view(1, len(MODEL_BANDS), 1, 1, 1)

    @classmethod
    def load(
        cls,
        model_directory: Path | None = None,
        *,
        device: str | torch.device = "cuda",
    ) -> PayloadCropModel:
        """Build the architecture and load only the selected checkpoint."""
        model_directory = model_directory or default_model_directory()
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")

        checkpoint_path = model_directory / SELECTED_CHECKPOINT_NAME
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        actual_digest = checkpoint_sha256(checkpoint_path)
        if actual_digest != SELECTED_CHECKPOINT_SHA256:
            raise RuntimeError(
                "Selected checkpoint checksum mismatch: "
                f"expected {SELECTED_CHECKPOINT_SHA256}, got {actual_digest}"
            )
        backend = os.environ.get("VITA_CROP_BACKEND", "pytorch").strip().casefold()
        if backend not in {"pytorch", "tensorrt"}:
            raise ValueError("VITA_CROP_BACKEND must be pytorch or tensorrt")
        if backend == "pytorch" and requested_device.type == "cuda":
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        if backend == "tensorrt":
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            from prithvi_payload.tensorrt_runtime import (
                NativeTensorRTPlan,
                TensorRTArtifactError,
                load_accepted_manifest,
            )

            manifest = load_accepted_manifest(required_models=("crop",))
            crop_record = manifest["models"]["crop"]
            if crop_record.get("source_sha256") != actual_digest:
                raise TensorRTArtifactError(
                    "Accepted crop TensorRT plan does not match the installed checkpoint"
                )
            optimized = NativeTensorRTPlan(
                crop_record,
                manifest_path=manifest["_manifest_path"],
                device=requested_device,
            )
            tensorrt_engine_count = 1
            tensorrt_parity = crop_record.get("parity", {})
            tensorrt_precision = crop_record.get("precision")
            tensorrt_tf32 = crop_record.get("tf32")
        else:
            exported_program = load_crop_exported_program(
                model_directory,
                device=requested_device,
            )
            tensorrt_engine_count = 0
            tensorrt_parity = {}
            tensorrt_precision = None
            tensorrt_tf32 = None
            optimized = exported_program.module()
        return cls(
            optimized,
            device=requested_device,
            fixed_batch_size=OPTIMIZED_BATCH_SIZE,
            exported=True,
            backend=backend,
            tensorrt_engine_count=tensorrt_engine_count,
            tensorrt_parity=tensorrt_parity,
            tensorrt_precision=tensorrt_precision,
            tensorrt_tf32=tensorrt_tf32,
        )

    def _validate_inputs(
        self,
        image: Tensor,
        temporal_coords: Tensor,
        location_coords: Tensor,
    ) -> None:
        if image.ndim != 5:
            raise ValueError("image must have shape [batch, bands, time, height, width]")
        if image.shape[1] != len(MODEL_BANDS) or image.shape[2] != TIME_STEPS:
            raise ValueError(
                f"Expected {len(MODEL_BANDS)} bands and {TIME_STEPS} dates, "
                f"got shape {tuple(image.shape)}"
            )
        batch_size = image.shape[0]
        if batch_size < 1:
            raise ValueError("image batch must contain at least one tile")
        if temporal_coords.shape != (batch_size, TIME_STEPS, 2):
            raise ValueError("temporal_coords must have shape [batch, time, 2] as year/day-of-year")
        if location_coords.shape != (batch_size, 2):
            raise ValueError("location_coords must have shape [batch, 2] as latitude/longitude")

    def predict(
        self,
        image: Tensor,
        *,
        temporal_coords: Tensor,
        location_coords: Tensor,
    ) -> InferenceOutput:
        """Predict one or more tiles supplied on the training numeric scale."""
        self._validate_inputs(image, temporal_coords, location_coords)
        output_batch_size = image.shape[0]
        if self.fixed_batch_size is not None:
            if output_batch_size > self.fixed_batch_size:
                raise ValueError(
                    f"Optimized model accepts at most {self.fixed_batch_size} tiles per batch"
                )
            padding = self.fixed_batch_size - output_batch_size
            if padding:
                image = torch.cat((image, image[-1:].expand(padding, -1, -1, -1, -1)))
                temporal_coords = torch.cat(
                    (temporal_coords, temporal_coords[-1:].expand(padding, -1, -1))
                )
                location_coords = torch.cat(
                    (location_coords, location_coords[-1:].expand(padding, -1))
                )
        image = image.to(self.device, dtype=torch.float32, non_blocking=True)
        temporal_coords = temporal_coords.to(
            self.device,
            dtype=torch.float32,
            non_blocking=True,
        )
        location_coords = location_coords.to(
            self.device,
            dtype=torch.float32,
            non_blocking=True,
        )
        normalized = (image - self._means) / self._stds
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.device.type == "cuda" and self.backend == "pytorch"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            logits = self.model(normalized, temporal_coords, location_coords)
            crop_probability = logits.softmax(dim=1)[:output_batch_size, 1]
            crop_binary = (crop_probability >= CROP_CLASSIFICATION_THRESHOLD).to(dtype=torch.uint8)
            crop_confidence = torch.maximum(crop_probability, 1 - crop_probability)
        return InferenceOutput(
            crop_probability=crop_probability,
            crop_binary=crop_binary,
            crop_confidence=crop_confidence,
        )

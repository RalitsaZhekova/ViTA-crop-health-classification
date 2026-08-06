"""Load the pinned model and run training-free tensor inference."""

from __future__ import annotations

import hashlib
import os
import warnings
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
OPTIMIZED_BATCH_SIZE = 4


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
        f"{checkpoint_stem}.{version}.torch_{torch_version}.{device.type}.export.pt2"
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


def _compile_tensorrt(
    exported_program: Any,
    *,
    device: torch.device,
) -> tuple[nn.Module, int]:
    """Compile and cache the fixed crop graph with the Jetson Torch-TensorRT stack."""
    if device.type != "cuda":
        raise RuntimeError("TensorRT crop inference requires a CUDA device")
    try:
        import torch_tensorrt
    except ImportError as error:
        raise RuntimeError(
            "VITA_CROP_BACKEND=tensorrt requires the NVIDIA Torch-TensorRT package"
        ) from error

    cache_root = Path(
        os.environ.get("VITA_TRT_CACHE_DIR", "/tmp/vita-torch-tensorrt")
    ).resolve()
    engine_cache = cache_root / "crop"
    engine_cache.mkdir(parents=True, exist_ok=True)
    inputs = list(_example_inputs(device))
    compiled = torch_tensorrt.dynamo.compile(
        exported_program,
        arg_inputs=inputs,
        enabled_precisions={torch.float16},
        require_full_compilation=_environment_flag("VITA_TRT_REQUIRE_FULL", False),
        pass_through_build_failures=True,
        optimization_level=int(os.environ.get("VITA_TRT_OPTIMIZATION_LEVEL", "3")),
        workspace_size=int(os.environ.get("VITA_TRT_WORKSPACE_BYTES", str(2 * 1024**3))),
        timing_cache_path=str(cache_root / "timing-cache.bin"),
        cache_built_engines=True,
        reuse_cached_engines=True,
        engine_cache_dir=str(engine_cache),
        engine_cache_size=int(os.environ.get("VITA_TRT_CACHE_BYTES", str(8 * 1024**3))),
    )
    engine_nodes = sum(
        "tensorrt" in str(node.target).casefold()
        or "run_on_acc" in str(node.target).casefold()
        for node in compiled.graph.nodes
    )
    engine_modules = sum(
        "tensorrt" in type(module).__module__.casefold()
        or "tensorrt" in type(module).__name__.casefold()
        for _, module in compiled.named_modules()
    )
    engine_count = max(engine_nodes, engine_modules)
    if engine_count < 1:
        raise RuntimeError("Torch-TensorRT produced no TensorRT engine partitions")
    return compiled, engine_count


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


class PayloadCropModel:
    """Pinned single-image Prithvi model with calibrated binary outputs."""

    def __init__(
        self,
        model: nn.Module,
        *,
        device: torch.device,
        fixed_batch_size: int | None = None,
        exported: bool = False,
        backend: str = "pytorch",
        tensorrt_engine_count: int = 0,
    ) -> None:
        # Torch-TensorRT returns an already placed graph containing initialized
        # engine modules; applying nn.Module.to() again can invalidate runtime state.
        self.model = model if backend == "tensorrt" else model.to(device)
        if not exported:
            self.model.eval()
        self.device = device
        self.backend = backend
        self.tensorrt_engine_count = tensorrt_engine_count
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

        architecture_path = model_directory / "architecture.yaml"
        checkpoint_path = model_directory / SELECTED_CHECKPOINT_NAME
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        actual_digest = checkpoint_sha256(checkpoint_path)
        if actual_digest != SELECTED_CHECKPOINT_SHA256:
            raise RuntimeError(
                "Selected checkpoint checksum mismatch: "
                f"expected {SELECTED_CHECKPOINT_SHA256}, got {actual_digest}"
            )
        optimized_path = _optimized_model_path(model_directory, requested_device)
        exported_program = None
        if optimized_path.is_file():
            exported_program = torch.export.load(optimized_path)
        if exported_program is None:
            if not architecture_path.is_file():
                raise FileNotFoundError(architecture_path)

            # Keep TerraTorch out of commands that never reach crop classification.
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
            tensor_model = _TensorLogitsModel(model)
            _export_optimized_model(tensor_model, optimized_path, requested_device)
            exported_program = torch.export.load(optimized_path)

        backend = os.environ.get("VITA_CROP_BACKEND", "pytorch").strip().casefold()
        if backend not in {"pytorch", "tensorrt"}:
            raise ValueError("VITA_CROP_BACKEND must be pytorch or tensorrt")
        if backend == "tensorrt":
            try:
                optimized, tensorrt_engine_count = _compile_tensorrt(
                    exported_program,
                    device=requested_device,
                )
            except Exception as error:
                if _environment_flag("VITA_TRT_STRICT", True):
                    raise RuntimeError("Crop model TensorRT compilation failed") from error
                warnings.warn(
                    f"TensorRT compilation failed; using exported PyTorch graph: {error}",
                    stacklevel=2,
                )
                backend = "pytorch"
                tensorrt_engine_count = 0
                optimized = exported_program.module()
        else:
            tensorrt_engine_count = 0
            optimized = exported_program.module()
        return cls(
            optimized,
            device=requested_device,
            fixed_batch_size=OPTIMIZED_BATCH_SIZE,
            exported=True,
            backend=backend,
            tensorrt_engine_count=tensorrt_engine_count,
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
            probabilities = logits.softmax(dim=1)
            crop_probability = probabilities[:output_batch_size, 1]
            crop_binary = (crop_probability >= CROP_CLASSIFICATION_THRESHOLD).to(dtype=torch.uint8)
            crop_confidence = torch.maximum(crop_probability, 1 - crop_probability)
        return InferenceOutput(
            crop_probability=crop_probability,
            crop_binary=crop_binary,
            crop_confidence=crop_confidence,
        )

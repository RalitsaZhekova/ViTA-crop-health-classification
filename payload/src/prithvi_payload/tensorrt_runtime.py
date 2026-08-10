"""Direct TensorRT plan loading and execution without Torch-TensorRT.

The service never builds engines.  It accepts only checksum-bound plans from the
offline payload builder and executes them through TensorRT's Python runtime API.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path, PurePosixPath
from typing import Any

import torch
from torch import Tensor, nn

MANIFEST_SCHEMA_VERSION = 1
DEFAULT_MANIFEST_PATH = Path(
    os.environ.get(
        "VITA_TRT_MANIFEST",
        "/engine-cache/tensorrt/direct/accepted.json",
    )
)


class TensorRTArtifactError(RuntimeError):
    """Raised when a direct TensorRT artifact is missing, stale, or invalid."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def seal_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with a deterministic whole-manifest integrity digest."""
    sealed = dict(manifest)
    sealed.pop("manifest_sha256", None)
    sealed["manifest_sha256"] = hashlib.sha256(_canonical_json(sealed)).hexdigest()
    return sealed


def write_manifest_atomic(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    sealed = seal_manifest(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(sealed, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return sealed


def current_target_signature() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise TensorRTArtifactError("Direct TensorRT requires a visible CUDA device")
    try:
        import tensorrt as trt
    except ImportError as error:
        raise TensorRTArtifactError(
            "Direct TensorRT requires the TensorRT Python package"
        ) from error
    major, minor = torch.cuda.get_device_capability(0)
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": f"{major}.{minor}",
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "tensorrt": trt.__version__,
    }


def _resolve_artifact(manifest_path: Path, relative_value: Any) -> Path:
    if not isinstance(relative_value, str) or not relative_value:
        raise TensorRTArtifactError("TensorRT plan path is missing from the manifest")
    relative = PurePosixPath(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise TensorRTArtifactError("TensorRT plan paths must stay below the manifest directory")
    root = manifest_path.parent.resolve()
    candidate = root.joinpath(*relative.parts).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise TensorRTArtifactError(
            "TensorRT plan escaped the manifest directory"
        ) from error
    return candidate


def load_accepted_manifest(
    path: Path | None = None,
    *,
    required_models: tuple[str, ...] = (),
) -> dict[str, Any]:
    manifest_path = (path or DEFAULT_MANIFEST_PATH).resolve()
    if not manifest_path.is_file():
        raise TensorRTArtifactError(
            f"Accepted direct TensorRT manifest is missing: {manifest_path}. "
            "Run python -m prithvi_payload.tensorrt_builder before starting the service."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TensorRTArtifactError(
            f"Could not read direct TensorRT manifest {manifest_path}"
        ) from error
    if not isinstance(manifest, dict):
        raise TensorRTArtifactError("Direct TensorRT manifest must be a JSON object")
    recorded_digest = manifest.get("manifest_sha256")
    if not isinstance(recorded_digest, str) or seal_manifest(manifest)[
        "manifest_sha256"
    ] != recorded_digest:
        raise TensorRTArtifactError("Direct TensorRT manifest integrity check failed")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise TensorRTArtifactError("Direct TensorRT manifest schema is unsupported")
    if manifest.get("status") != "accepted":
        raise TensorRTArtifactError("Direct TensorRT manifest has not passed acceptance")
    if manifest.get("target") != current_target_signature():
        raise TensorRTArtifactError(
            "Direct TensorRT plans were built for a different GPU or software stack"
        )
    models = manifest.get("models")
    if not isinstance(models, dict):
        raise TensorRTArtifactError("Direct TensorRT manifest has no model records")
    for model_name in required_models:
        if model_name not in models:
            raise TensorRTArtifactError(
                f"Direct TensorRT manifest has no accepted {model_name} model"
            )
    manifest["_manifest_path"] = manifest_path
    return manifest


_TORCH_DTYPES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "float32": torch.float32,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


def _shape(value: Any, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value or any(
        not isinstance(dimension, int) for dimension in value
    ):
        raise TensorRTArtifactError(f"Invalid TensorRT {name} shape")
    return tuple(value)


def _resolve_cuda_device(device: torch.device) -> torch.device:
    """Resolve an unindexed CUDA alias to the active device used by TensorRT."""

    if device.type != "cuda":
        raise TensorRTArtifactError("TensorRT plans can only execute on CUDA")
    current_index = torch.cuda.current_device()
    requested_index = current_index if device.index is None else device.index
    if requested_index != current_index:
        raise TensorRTArtifactError(
            "TensorRT plan device must match the active CUDA device: "
            f"requested cuda:{requested_index}, active cuda:{current_index}"
        )
    return torch.device("cuda", requested_index)


class NativeTensorRTPlan(nn.Module):
    """One checksum-verified TensorRT engine with a private execution context."""

    def __init__(
        self,
        record: dict[str, Any],
        *,
        manifest_path: Path,
        device: torch.device,
    ) -> None:
        super().__init__()
        resolved_device = _resolve_cuda_device(device)
        try:
            import tensorrt as trt
        except ImportError as error:
            raise TensorRTArtifactError(
                "Direct TensorRT requires the TensorRT Python package"
            ) from error

        plan_path = _resolve_artifact(manifest_path, record.get("plan"))
        if not plan_path.is_file():
            raise TensorRTArtifactError(f"TensorRT plan is missing: {plan_path}")
        expected_digest = record.get("plan_sha256")
        if not isinstance(expected_digest, str) or file_sha256(plan_path) != expected_digest:
            raise TensorRTArtifactError(f"TensorRT plan checksum mismatch: {plan_path}")

        inputs = record.get("inputs")
        outputs = record.get("outputs")
        if (
            not isinstance(inputs, list)
            or not inputs
            or not isinstance(outputs, list)
            or not outputs
        ):
            raise TensorRTArtifactError("TensorRT plan has an invalid I/O contract")
        self.input_specs = self._validate_specs(inputs, kind="input")
        self.output_specs = self._validate_specs(outputs, kind="output")
        self.device = resolved_device
        self.record = record
        self.plan_path = plan_path
        self._logger = trt.Logger(trt.Logger.ERROR)
        self._runtime = trt.Runtime(self._logger)
        self._engine = self._runtime.deserialize_cuda_engine(plan_path.read_bytes())
        if self._engine is None:
            raise TensorRTArtifactError(f"TensorRT could not deserialize {plan_path}")
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise TensorRTArtifactError(
                f"TensorRT could not create an execution context for {plan_path}"
            )
        self._lock = threading.Lock()
        self._trt = trt
        self._validate_engine_contract()

    @staticmethod
    def _validate_specs(specs: list[Any], *, kind: str) -> list[dict[str, Any]]:
        validated: list[dict[str, Any]] = []
        for spec in specs:
            if not isinstance(spec, dict):
                raise TensorRTArtifactError(f"TensorRT {kind} specification is invalid")
            name = spec.get("name")
            dtype_name = spec.get("dtype")
            if not isinstance(name, str) or not name or dtype_name not in _TORCH_DTYPES:
                raise TensorRTArtifactError(f"TensorRT {kind} specification is invalid")
            shape = _shape(spec.get("shape"), name=kind)
            if any(dimension == 0 or dimension < -1 for dimension in shape):
                raise TensorRTArtifactError(f"TensorRT {kind} contains an invalid dimension")
            normalized = dict(spec)
            normalized["shape"] = list(shape)
            profile = spec.get("profile")
            if profile is not None:
                if not isinstance(profile, dict):
                    raise TensorRTArtifactError(f"TensorRT {kind} profile is invalid")
                normalized_profile = {
                    key: list(_shape(profile.get(key), name=f"{kind} {key}"))
                    for key in ("min", "opt", "max")
                }
                profile_shapes = [
                    tuple(normalized_profile[key]) for key in ("min", "opt", "max")
                ]
                if any(len(value) != len(shape) for value in profile_shapes):
                    raise TensorRTArtifactError(
                        f"TensorRT {kind} profile rank does not match its shape"
                    )
                for minimum, optimum, maximum in zip(*profile_shapes, strict=True):
                    if minimum <= 0 or not minimum <= optimum <= maximum:
                        raise TensorRTArtifactError(
                            f"TensorRT {kind} profile bounds are invalid"
                        )
                for declared, minimum, maximum in zip(
                    shape,
                    profile_shapes[0],
                    profile_shapes[2],
                    strict=True,
                ):
                    if declared >= 0 and (minimum != declared or maximum != declared):
                        raise TensorRTArtifactError(
                            f"TensorRT {kind} profile changes a static dimension"
                        )
                normalized["profile"] = normalized_profile
            validated.append(normalized)
        return validated

    def _validate_engine_contract(self) -> None:
        trt = self._trt
        trt_dtype_names = {}
        for attribute, name in (
            ("float16", "float16"),
            ("float32", "float32"),
            ("int32", "int32"),
            ("int64", "int64"),
            ("uint8", "uint8"),
            ("bool", "bool"),
        ):
            value = getattr(trt, attribute, None)
            if value is not None:
                trt_dtype_names[value] = name
        actual: dict[str, tuple[Any, tuple[int, ...], str]] = {}
        for index in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(index)
            mode = self._engine.get_tensor_mode(name)
            kind = "input" if mode == trt.TensorIOMode.INPUT else "output"
            engine_dtype = self._engine.get_tensor_dtype(name)
            if engine_dtype not in trt_dtype_names:
                raise TensorRTArtifactError(
                    f"TensorRT engine {name} uses unsupported dtype {engine_dtype}"
                )
            dtype_name = trt_dtype_names[engine_dtype]
            actual[name] = (
                kind,
                tuple(int(value) for value in self._engine.get_tensor_shape(name)),
                dtype_name,
            )
        expected_names = {
            spec["name"] for spec in [*self.input_specs, *self.output_specs]
        }
        if set(actual) != expected_names:
            raise TensorRTArtifactError(
                f"TensorRT engine I/O names do not match the manifest for {self.plan_path}"
            )
        for kind, specs in (("input", self.input_specs), ("output", self.output_specs)):
            for spec in specs:
                actual_kind, actual_shape, actual_dtype = actual[spec["name"]]
                if actual_kind != kind or actual_dtype != spec["dtype"]:
                    raise TensorRTArtifactError(
                        f"TensorRT engine {spec['name']} type does not match the manifest"
                    )
                expected_shape = tuple(spec["shape"])
                if actual_shape != expected_shape:
                    raise TensorRTArtifactError(
                        f"TensorRT engine {spec['name']} shape {actual_shape} does not match "
                        f"the manifest {expected_shape}"
                    )

    @staticmethod
    def _validate_profile(spec: dict[str, Any], shape: tuple[int, ...]) -> None:
        expected = tuple(spec["shape"])
        if len(shape) != len(expected):
            raise TensorRTArtifactError(
                f"TensorRT input {spec['name']} rank does not match its plan"
            )
        if all(dimension >= 0 for dimension in expected):
            if shape != expected:
                raise TensorRTArtifactError(
                    f"TensorRT input {spec['name']} has shape {shape}, expected {expected}"
                )
            return
        profile = spec.get("profile")
        if not isinstance(profile, dict):
            raise TensorRTArtifactError(
                f"Dynamic TensorRT input {spec['name']} has no optimization profile"
            )
        minimum = tuple(profile["min"])
        maximum = tuple(profile["max"])
        if any(
            actual < low or actual > high
            for actual, low, high in zip(shape, minimum, maximum, strict=True)
        ):
            raise TensorRTArtifactError(
                f"TensorRT input {spec['name']} shape {shape} is outside "
                f"{minimum}..{maximum}"
            )

    def forward(self, *values: Tensor) -> Tensor | tuple[Tensor, ...]:
        if len(values) != len(self.input_specs):
            raise TensorRTArtifactError(
                f"TensorRT plan expects {len(self.input_specs)} inputs, got {len(values)}"
            )
        prepared: list[Tensor] = []
        for value, spec in zip(values, self.input_specs, strict=True):
            if not isinstance(value, Tensor):
                raise TensorRTArtifactError(f"TensorRT input {spec['name']} is not a tensor")
            if value.device != self.device:
                raise TensorRTArtifactError(
                    f"TensorRT input {spec['name']} must already be on {self.device}"
                )
            expected_dtype = _TORCH_DTYPES[spec["dtype"]]
            if value.dtype != expected_dtype:
                raise TensorRTArtifactError(
                    f"TensorRT input {spec['name']} has dtype {value.dtype}, "
                    f"expected {expected_dtype}"
                )
            contiguous = value.contiguous()
            self._validate_profile(spec, tuple(contiguous.shape))
            prepared.append(contiguous)

        with self._lock:
            for value, spec in zip(prepared, self.input_specs, strict=True):
                if -1 in spec["shape"] and not self._context.set_input_shape(
                    spec["name"], tuple(value.shape)
                ):
                    raise TensorRTArtifactError(
                        f"TensorRT rejected input shape for {spec['name']}"
                    )
                if not self._context.set_tensor_address(spec["name"], value.data_ptr()):
                    raise TensorRTArtifactError(
                        f"TensorRT rejected the input address for {spec['name']}"
                    )

            outputs: list[Tensor] = []
            for spec in self.output_specs:
                output_shape = tuple(
                    int(value) for value in self._context.get_tensor_shape(spec["name"])
                )
                if any(dimension <= 0 for dimension in output_shape):
                    raise TensorRTArtifactError(
                        f"TensorRT did not resolve output shape for {spec['name']}: "
                        f"{output_shape}"
                    )
                output = torch.empty(
                    output_shape,
                    dtype=_TORCH_DTYPES[spec["dtype"]],
                    device=self.device,
                )
                if not self._context.set_tensor_address(spec["name"], output.data_ptr()):
                    raise TensorRTArtifactError(
                        f"TensorRT rejected the output address for {spec['name']}"
                    )
                outputs.append(output)

            stream = torch.cuda.current_stream(self.device)
            if not self._context.execute_async_v3(stream_handle=stream.cuda_stream):
                raise TensorRTArtifactError(f"TensorRT execution failed for {self.plan_path}")
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

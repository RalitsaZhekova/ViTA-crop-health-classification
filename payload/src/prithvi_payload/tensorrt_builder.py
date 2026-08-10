"""Offline ONNX-to-TensorRT build and scientific acceptance on the payload GPU."""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import shutil
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cloud_detection.backend import OMNICLOUDMASK_ENSEMBLE_SHA256
from cloud_detection.tensorrt_backend import (
    CloudTensorRTRouter,
    _CloudEnsemble,
    _remove_zero_channel_cat_noops,
)
from prithvi_shared import SELECTED_CHECKPOINT_SHA256
from torch import Tensor, nn

from prithvi_payload.crop_parity import build_crop_parity_inputs
from prithvi_payload.inference import (
    OPTIMIZED_BATCH_SIZE,
    _functionalize_prithvi_export_for_tensorrt,
)
from prithvi_payload.tensorrt_runtime import (
    DEFAULT_MANIFEST_PATH,
    MANIFEST_SCHEMA_VERSION,
    NativeTensorRTPlan,
    current_target_signature,
    file_sha256,
    write_manifest_atomic,
)

CLOUD_MAX_CLASS_MISMATCH = 0.001
CROP_MAX_CLASS_MISMATCH = 0.002
CROP_MAX_MEAN_PROBABILITY_ERROR = 0.005
ONNX_OPSET = 18


class TensorRTBuildError(RuntimeError):
    """Raised when any prebuild, engine, or parity gate fails."""


@contextlib.contextmanager
def _temporary_environment(updates: dict[str, str]) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _precision_dtype(value: str, *, name: str) -> torch.dtype:
    normalized = value.strip().casefold()
    if normalized == "fp32":
        return torch.float32
    if normalized == "fp16":
        return torch.float16
    raise TensorRTBuildError(f"{name} must be fp32 or fp16")


def _load_source_runtime(cloud_precision: str) -> Any:
    source_environment = {
        "VITA_CROP_BACKEND": "pytorch",
        "VITA_CLOUD_BACKEND": "pytorch",
        "VITA_CLOUD_INFERENCE_DTYPE": cloud_precision,
        "VITA_WARMUP": "0",
        "VITA_TRT_CUDAGRAPHS": "0",
    }
    with _temporary_environment(source_environment):
        from prithvi_payload.service import PayloadRuntime

        return PayloadRuntime()


def _release_cuda_memory() -> None:
    """Make engine construction independent of source-model GPU residency."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _require_build_dependencies() -> dict[str, str]:
    if not torch.cuda.is_available():
        raise TensorRTBuildError("The direct TensorRT builder requires CUDA")
    try:
        import onnx
        import onnxscript
        import tensorrt
    except ImportError as error:
        raise TensorRTBuildError(
            "The payload image is missing onnx, onnxscript, or tensorrt"
        ) from error
    return {
        "onnx": onnx.__version__,
        "onnxscript": onnxscript.__version__,
        "tensorrt": tensorrt.__version__,
    }


def _find_trtexec() -> Path:
    configured = os.environ.get("VITA_TRTEXEC")
    candidates = [
        Path(configured) if configured else None,
        Path(value) if (value := shutil.which("trtexec")) else None,
        Path("/opt/tensorrt/bin/trtexec"),
        Path("/usr/src/tensorrt/bin/trtexec"),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise TensorRTBuildError(
        "TensorRT trtexec was not found. Set VITA_TRTEXEC to the executable path."
    )


def _validate_trtexec(executable: Path) -> str:
    result = subprocess.run(
        [str(executable), "--help"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
    )
    help_text = result.stdout
    required = (
        "--onnx",
        "--saveEngine",
        "--stronglyTyped",
        "--skipInference",
        "--memPoolSize",
        "--timingCacheFile",
        "--noTF32",
    )
    missing = [flag for flag in required if flag not in help_text]
    if result.returncode != 0 or missing:
        raise TensorRTBuildError(
            f"trtexec is incompatible with this builder; missing options: {missing}"
        )
    version_lines = [line.strip() for line in help_text.splitlines() if "TensorRT" in line]
    return version_lines[0] if version_lines else str(executable)


def _onnx_dtype_name(element_type: int) -> str:
    import onnx

    mapping = {
        onnx.TensorProto.FLOAT: "float32",
        onnx.TensorProto.FLOAT16: "float16",
        onnx.TensorProto.INT32: "int32",
        onnx.TensorProto.INT64: "int64",
        onnx.TensorProto.UINT8: "uint8",
        onnx.TensorProto.BOOL: "bool",
    }
    try:
        return mapping[element_type]
    except KeyError as error:
        raise TensorRTBuildError(f"Unsupported ONNX tensor type {element_type}") from error


def _onnx_value_spec(value: Any) -> dict[str, Any]:
    tensor_type = value.type.tensor_type
    shape = [
        int(dimension.dim_value) if dimension.HasField("dim_value") else -1
        for dimension in tensor_type.shape.dim
    ]
    if not shape or any(dimension == 0 or dimension < -1 for dimension in shape):
        raise TensorRTBuildError(f"ONNX value {value.name} has invalid shape {shape}")
    return {
        "name": value.name,
        "dtype": _onnx_dtype_name(tensor_type.elem_type),
        "shape": shape,
    }


def _validate_onnx(
    path: Path,
    *,
    expected_inputs: tuple[str, ...],
    expected_outputs: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import onnx

    try:
        model = onnx.load(str(path), load_external_data=True)
        onnx.checker.check_model(model, full_check=True)
    except Exception as error:
        raise TensorRTBuildError(f"ONNX validation failed for {path}: {error}") from error

    initializer_names = {initializer.name for initializer in model.graph.initializer}
    graph_inputs = [
        value for value in model.graph.input if value.name not in initializer_names
    ]
    inputs = [_onnx_value_spec(value) for value in graph_inputs]
    outputs = [_onnx_value_spec(value) for value in model.graph.output]
    if tuple(spec["name"] for spec in inputs) != expected_inputs:
        raise TensorRTBuildError(
            f"ONNX inputs for {path.name} are not {expected_inputs}: {inputs}"
        )
    if tuple(spec["name"] for spec in outputs) != expected_outputs:
        raise TensorRTBuildError(
            f"ONNX outputs for {path.name} are not {expected_outputs}: {outputs}"
        )

    for initializer in model.graph.initializer:
        if math.prod(initializer.dims) == 0:
            raise TensorRTBuildError(
                f"ONNX graph {path.name} contains zero-sized initializer {initializer.name}"
            )
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR and math.prod(attribute.t.dims) == 0:
                raise TensorRTBuildError(
                    f"ONNX graph {path.name} contains zero-sized Constant {node.name}"
                )
    return inputs, outputs


def _save_onnx_program(program: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        program.save(str(temporary), external_data=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _normalize_onnxscript_integer_attributes(program: Any) -> int:
    """Coerce ONNX INT attributes away from Python bool before protobuf save.

    PyTorch 2.6's exporter can represent flags such as ReduceMean ``keepdims``
    as ``Attr(INT, True)``. ONNX Script 0.1 accepts that IR value, but the pure
    Python protobuf implementation in the pinned Jetson image correctly rejects
    bool when serializing the int64 ``AttributeProto.i`` field. The ONNX value
    is exactly 1/0, so normalizing only INT-typed bools is lossless.
    """

    model = getattr(program, "model", None)
    root = getattr(model, "graph", None)
    if root is None:
        raise TensorRTBuildError("PyTorch ONNXProgram has no mutable IR graph")

    normalized = 0

    def visit(graph: Any) -> None:
        nonlocal normalized
        for node in graph:
            attributes = getattr(node, "attributes", None)
            if not isinstance(attributes, dict):
                continue
            for attribute in attributes.values():
                type_name = getattr(getattr(attribute, "type", None), "name", None)
                value = getattr(attribute, "value", None)
                if type_name == "INT" and isinstance(value, bool):
                    attribute.value = int(value)
                    normalized += 1
                elif type_name == "GRAPH":
                    visit(value)
                elif type_name == "GRAPHS":
                    for child in value:
                        visit(child)

    visit(root)
    return normalized


def _exception_summary(error: BaseException) -> str:
    parts: list[str] = []
    current: BaseException | None = error
    while current is not None:
        message = str(current).splitlines()[0].strip()
        parts.append(f"{type(current).__name__}: {message}"[:500])
        current = current.__cause__
    return " <- ".join(parts)


def _export_program_to_onnx(
    exported: Any,
    path: Path,
    *,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    print(f"[onnx] exporting {path.name}", flush=True)
    try:
        program = torch.onnx.export(
            exported,
            args=(),
            f=None,
            input_names=list(input_names),
            output_names=list(output_names),
            opset_version=ONNX_OPSET,
            dynamo=True,
            external_data=False,
        )
        if program is None or not hasattr(program, "save"):
            raise TensorRTBuildError("PyTorch did not return an ONNXProgram")
        normalized = _normalize_onnxscript_integer_attributes(program)
        if normalized:
            print(
                f"[onnx] normalized {normalized} integer flag attributes",
                flush=True,
            )
        _save_onnx_program(program, path)
    except Exception as error:
        raise TensorRTBuildError(
            f"PyTorch ONNX export failed for {path}: {_exception_summary(error)}"
        ) from error
    return _validate_onnx(
        path,
        expected_inputs=input_names,
        expected_outputs=output_names,
    )


def _export_crop_onnx(runtime: Any, path: Path) -> dict[str, Any]:
    exported = runtime.crop.model
    # PayloadRuntime exposes the loaded ExportedProgram module. Re-exporting the
    # module captures the same fixed batch contract without compiler partitions.
    from prithvi_payload.inference import _example_inputs

    inputs = _example_inputs(torch.device("cuda"))
    recaptured = torch.export.export(exported, inputs, strict=False)
    _functionalize_prithvi_export_for_tensorrt(recaptured)
    input_specs, output_specs = _export_program_to_onnx(
        recaptured,
        path,
        input_names=("image", "temporal_coords", "location_coords"),
        output_names=("logits",),
    )
    return {"path": path, "inputs": input_specs, "outputs": output_specs}


def _export_cloud_onnx(
    source: nn.Module,
    path: Path,
    *,
    batch_size: int,
    patch_size: int,
    dtype: torch.dtype,
    dynamic_batch: bool,
) -> dict[str, Any]:
    sample = torch.zeros(
        (batch_size, 3, patch_size, patch_size),
        device="cuda",
        dtype=dtype,
    )
    dynamic_shapes = None
    if dynamic_batch:
        batch = torch.export.Dim("batch", min=1, max=batch_size)
        dynamic_shapes = ({0: batch},)
    exported = torch.export.export(
        source,
        (sample,),
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )
    removed = _remove_zero_channel_cat_noops(exported)
    if removed != 1:
        raise TensorRTBuildError(
            "Expected exactly one zero-channel OmniCloudMask concatenation, "
            f"removed {removed} for {patch_size}px"
        )
    input_specs, output_specs = _export_program_to_onnx(
        exported,
        path,
        input_names=("image",),
        output_names=("logits",),
    )
    profile = None
    if dynamic_batch:
        profile = {
            "min": [1, 3, patch_size, patch_size],
            "opt": [batch_size, 3, patch_size, patch_size],
            "max": [batch_size, 3, patch_size, patch_size],
        }
        input_specs[0]["profile"] = profile
    del sample
    return {
        "path": path,
        "inputs": input_specs,
        "outputs": output_specs,
        "patch_size": patch_size,
        "profile": profile,
    }


def _parse_onnx_with_tensorrt(path: Path) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if hasattr(parser, "parse_from_file"):
        accepted = parser.parse_from_file(str(path))
    else:
        accepted = parser.parse(path.read_bytes())
    if not accepted:
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise TensorRTBuildError(
            f"TensorRT ONNX parser rejected {path.name}: " + " | ".join(errors)
        )


def _relative_plan_path(manifest_path: Path, plan_path: Path) -> str:
    return plan_path.resolve().relative_to(manifest_path.parent.resolve()).as_posix()


def _tail(path: Path, lines: int = 80) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return "<build log unavailable>"


def _build_plan(
    graph: dict[str, Any],
    *,
    name: str,
    precision: str,
    manifest_path: Path,
    executable: Path,
    target: dict[str, Any],
) -> dict[str, Any]:
    onnx_path = graph["path"]
    onnx_digest = file_sha256(onnx_path)
    target_digest = __import__("hashlib").sha256(
        json.dumps(target, sort_keys=True).encode("utf-8")
    ).hexdigest()
    plans_root = manifest_path.parent / "plans"
    logs_root = manifest_path.parent / "logs"
    plans_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    plan_path = plans_root / f"{name}.{onnx_digest[:12]}.{target_digest[:12]}.plan"
    build_record_path = plan_path.with_suffix(".build.json")
    timing_cache = manifest_path.parent / "timing" / f"{name}.cache"
    timing_cache.parent.mkdir(parents=True, exist_ok=True)
    workspace_mib = int(os.environ.get("VITA_TRT_WORKSPACE_MIB", "4096"))
    if workspace_mib < 256:
        raise TensorRTBuildError("VITA_TRT_WORKSPACE_MIB must be at least 256")
    arguments = [
        str(executable),
        f"--onnx={onnx_path}",
        "--stronglyTyped",
        "--noTF32",
        "--skipInference",
        f"--memPoolSize=workspace:{workspace_mib}",
        f"--timingCacheFile={timing_cache}",
    ]
    profile = graph.get("profile")
    if profile:
        for key in ("min", "opt", "max"):
            dimensions = "x".join(str(value) for value in profile[key])
            arguments.append(f"--{key}Shapes=image:{dimensions}")
    build_key = {
        "onnx_sha256": onnx_digest,
        "target": target,
        "precision": precision,
        "arguments": arguments[1:],
    }
    if plan_path.is_file() and build_record_path.is_file():
        try:
            cached = json.loads(build_record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = None
        if isinstance(cached, dict) and cached.get("build_key") == build_key:
            expected = cached.get("plan_sha256")
            if isinstance(expected, str) and file_sha256(plan_path) == expected:
                print(f"[tensorrt] reusing {plan_path.name}", flush=True)
                return {
                    **graph,
                    "plan_path": plan_path,
                    "plan_sha256": expected,
                    "onnx_sha256": onnx_digest,
                }

    temporary = plan_path.with_name(f".{plan_path.name}.{os.getpid()}.tmp")
    log_path = logs_root / f"{name}.trtexec.log"
    command = [*arguments, f"--saveEngine={temporary}"]
    print(f"[tensorrt] building {name}; full log: {log_path}", flush=True)
    started = time.perf_counter()
    try:
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                check=False,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if result.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
            raise TensorRTBuildError(
                f"trtexec failed for {name}. Last log lines:\n{_tail(log_path)}"
            )
        os.replace(temporary, plan_path)
    finally:
        temporary.unlink(missing_ok=True)
    plan_digest = file_sha256(plan_path)
    build_record = {
        "build_key": build_key,
        "plan_sha256": plan_digest,
        "build_seconds": time.perf_counter() - started,
        "log": log_path.name,
    }
    build_record_path.write_text(
        json.dumps(build_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        **graph,
        "plan_path": plan_path,
        "plan_sha256": plan_digest,
        "onnx_sha256": onnx_digest,
    }


def _plan_record(
    built: dict[str, Any],
    *,
    manifest_path: Path,
    precision: str,
) -> dict[str, Any]:
    return {
        "plan": _relative_plan_path(manifest_path, built["plan_path"]),
        "plan_sha256": built["plan_sha256"],
        "onnx_sha256": built["onnx_sha256"],
        "precision": precision,
        "inputs": built["inputs"],
        "outputs": built["outputs"],
    }


def _validate_crop_parity(
    source: nn.Module,
    record: dict[str, Any],
    *,
    manifest_path: Path,
    profiles: list[dict[str, Any]],
) -> dict[str, float]:
    batch = build_crop_parity_inputs(profiles, batch_size=OPTIMIZED_BATCH_SIZE)
    image, temporal, location, thresholds, valid_mask = [value.cuda() for value in batch]
    runner = NativeTensorRTPlan(
        record,
        manifest_path=manifest_path,
        device=torch.device("cuda"),
    )
    with torch.inference_mode():
        reference = source(image, temporal, location)
        accelerated = runner(image, temporal, location)
    if not isinstance(accelerated, Tensor) or accelerated.shape != reference.shape:
        raise TensorRTBuildError("Crop TensorRT output contract does not match PyTorch")
    if not bool(torch.isfinite(accelerated).all()):
        raise TensorRTBuildError("Crop TensorRT produced non-finite logits")
    reference_probability = reference.float().softmax(dim=1)[:, 1]
    accelerated_probability = accelerated.float().softmax(dim=1)[:, 1]
    threshold = thresholds[:, 0].view(-1, 1, 1)
    valid = valid_mask.bool()
    mismatch = float(
        (
            (reference_probability >= threshold)
            != (accelerated_probability >= threshold)
        )[valid]
        .float()
        .mean()
        .item()
    )
    absolute_error = (accelerated_probability - reference_probability).abs()[valid]
    mean_error = float(absolute_error.mean().item())
    parity = {
        "class_mismatch_fraction": mismatch,
        "mean_absolute_probability_error": mean_error,
        "maximum_absolute_probability_error": float(absolute_error.max().item()),
        "validation_pixel_count": float(valid.count_nonzero().item()),
    }
    if mismatch > CROP_MAX_CLASS_MISMATCH or mean_error > CROP_MAX_MEAN_PROBABILITY_ERROR:
        raise TensorRTBuildError(
            "Crop direct TensorRT parity failed: "
            f"decision mismatch {mismatch:.8f}/{CROP_MAX_CLASS_MISMATCH:.8f}, "
            f"mean probability error {mean_error:.8f}/"
            f"{CROP_MAX_MEAN_PROBABILITY_ERROR:.8f}"
        )
    return parity


def _load_cloud_scene(
    profile: dict[str, Any],
    *,
    input_config: dict[str, Any],
) -> np.ndarray:
    import rasterio
    from cloud_detection.preprocessing import normalize_reflectance, strict_valid_mask

    with rasterio.open(profile["source_path"]) as source:
        raw_image = source.read(profile["source_band_indices"])
    image, invalid = normalize_reflectance(
        raw_image,
        scale=profile["reflectance_scale"],
        clip_min=input_config.get("clip_min"),
        clip_max=input_config.get("clip_max"),
        nodata_value=profile["nodata_value"],
    )
    invalid |= ~strict_valid_mask(image[[1, 2, 0]])
    image[:, invalid] = 0.0
    return image.astype(np.float32, copy=False)


def _validate_cloud_parity(
    runtime: Any,
    records: list[dict[str, Any]],
    *,
    manifest_path: Path,
    dtype: torch.dtype,
) -> dict[str, Any]:
    backend = runtime.cloud.backend
    source_models = backend.models
    router = CloudTensorRTRouter(
        records,
        manifest_path=manifest_path,
        device=torch.device("cuda"),
        dtype=dtype,
    )
    scene_results: list[dict[str, Any]] = []
    total_pixels = 0
    total_mismatches = 0
    try:
        for profile in runtime._scene_cloud_warmups:
            image = _load_cloud_scene(
                profile,
                input_config=runtime.cloud.config["input"],
            )
            backend.models = source_models
            reference = backend.predict_semantic(image)
            backend.models = [router]
            accelerated = backend.predict_semantic(image)
            mismatches = int(np.count_nonzero(accelerated != reference))
            pixels = int(reference.size)
            fraction = mismatches / pixels
            if fraction > CLOUD_MAX_CLASS_MISMATCH:
                raise TensorRTBuildError(
                    "Cloud direct TensorRT parity failed for "
                    f"{profile['input']}: {fraction:.8f}/{CLOUD_MAX_CLASS_MISMATCH:.8f}"
                )
            total_pixels += pixels
            total_mismatches += mismatches
            scene_results.append(
                {
                    "input": profile["input"],
                    "class_mismatch_fraction": fraction,
                    "mismatch_count": mismatches,
                    "pixel_count": pixels,
                }
            )
    finally:
        backend.models = source_models
    aggregate = total_mismatches / total_pixels
    if aggregate > CLOUD_MAX_CLASS_MISMATCH:
        raise TensorRTBuildError(
            f"Cloud aggregate TensorRT mismatch {aggregate:.8f} exceeds "
            f"{CLOUD_MAX_CLASS_MISMATCH:.8f}"
        )
    return {
        "class_mismatch_fraction": aggregate,
        "mismatch_count": total_mismatches,
        "pixel_count": total_pixels,
        "scenes": scene_results,
    }


def build(manifest_path: Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    versions = _require_build_dependencies()
    target = current_target_signature()
    executable = _find_trtexec()
    trtexec_record = _validate_trtexec(executable)
    crop_precision = os.environ.get("VITA_CROP_TRT_PRECISION", "fp32").strip().casefold()
    cloud_precision = os.environ.get("VITA_CLOUD_TRT_PRECISION", "fp32").strip().casefold()
    crop_dtype = _precision_dtype(crop_precision, name="VITA_CROP_TRT_PRECISION")
    cloud_dtype = _precision_dtype(cloud_precision, name="VITA_CLOUD_TRT_PRECISION")
    if crop_dtype != torch.float32:
        raise TensorRTBuildError(
            "The first direct crop qualification is accuracy-first and requires FP32"
        )

    print("[preflight] direct TensorRT dependencies and trtexec passed", flush=True)
    runtime = _load_source_runtime(cloud_precision)

    onnx_root = manifest_path.parent / "onnx"
    onnx_root.mkdir(parents=True, exist_ok=True)
    crop_graph = _export_crop_onnx(
        runtime,
        onnx_root
        / f"crop.{SELECTED_CHECKPOINT_SHA256[:12]}.batch{OPTIMIZED_BATCH_SIZE}.fp32.onnx",
    )
    cloud_source = _CloudEnsemble(runtime.cloud.backend.models).to(
        device="cuda",
        dtype=cloud_dtype,
    )
    cloud_source.eval()
    cloud_graphs = [
        _export_cloud_onnx(
            cloud_source,
            onnx_root / f"cloud.{OMNICLOUDMASK_ENSEMBLE_SHA256[:12]}.b4.869.{cloud_precision}.onnx",
            batch_size=int(os.environ.get("VITA_CLOUD_BATCH_SIZE", "4")),
            patch_size=869,
            dtype=cloud_dtype,
            dynamic_batch=True,
        ),
        _export_cloud_onnx(
            cloud_source,
            onnx_root
            / (
                f"cloud.{OMNICLOUDMASK_ENSEMBLE_SHA256[:12]}."
                f"b1.1000.{cloud_precision}.onnx"
            ),
            batch_size=1,
            patch_size=1000,
            dtype=cloud_dtype,
            dynamic_batch=False,
        ),
    ]

    # trtexec is a separate process. Do not make its tactic selection compete
    # with two complete source ensembles left resident by ONNX export.
    del cloud_source
    del runtime
    _release_cuda_memory()

    print("[parse] validating every ONNX graph with TensorRT before builds", flush=True)
    for graph in [crop_graph, *cloud_graphs]:
        _parse_onnx_with_tensorrt(graph["path"])
    print("[parse] all ONNX graphs accepted by TensorRT", flush=True)

    built_crop = _build_plan(
        crop_graph,
        name="crop",
        precision=crop_precision,
        manifest_path=manifest_path,
        executable=executable,
        target=target,
    )
    built_cloud = [
        _build_plan(
            graph,
            name=f"cloud-{graph['patch_size']}",
            precision=cloud_precision,
            manifest_path=manifest_path,
            executable=executable,
            target=target,
        )
        for graph in cloud_graphs
    ]
    crop_record = {
        **_plan_record(
            built_crop,
            manifest_path=manifest_path,
            precision=crop_precision,
        ),
        "source_sha256": SELECTED_CHECKPOINT_SHA256,
        "tf32": False,
    }
    cloud_records = []
    for built in built_cloud:
        cloud_records.append(
            {
                **_plan_record(
                    built,
                    manifest_path=manifest_path,
                    precision=cloud_precision,
                ),
                "patch_size": built["patch_size"],
            }
        )

    # Reload the untouched PyTorch source only after all plans exist. The
    # accepted manifest is still absent, so no unvalidated plan can start.
    runtime = _load_source_runtime(cloud_precision)
    print("[parity] validating crop plan on balanced real-scene tiles", flush=True)
    crop_record["parity"] = _validate_crop_parity(
        runtime.crop.model,
        crop_record,
        manifest_path=manifest_path,
        profiles=runtime._crop_parity_profiles,
    )
    print("[parity] validating cloud plans on all four complete scenes", flush=True)
    cloud_parity = _validate_cloud_parity(
        runtime,
        cloud_records,
        manifest_path=manifest_path,
        dtype=cloud_dtype,
    )

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "accepted",
        "builder": {
            "kind": "onnx-trtexec",
            "onnx_opset": ONNX_OPSET,
            "packages": versions,
            "trtexec": trtexec_record,
        },
        "target": target,
        "models": {
            "crop": crop_record,
            "cloud": {
                "source_sha256": OMNICLOUDMASK_ENSEMBLE_SHA256,
                "precision": cloud_precision,
                "plans": cloud_records,
                "parity": cloud_parity,
            },
        },
    }
    sealed = write_manifest_atomic(manifest_path, manifest)
    del runtime
    _release_cuda_memory()
    print(
        json.dumps(
            {
                "status": "DIRECT_TENSORRT_ACCEPTED",
                "manifest": str(manifest_path),
                "manifest_sha256": sealed["manifest_sha256"],
                "crop_parity": crop_record["parity"],
                "cloud_parity": cloud_parity,
                "engine_count": 1 + len(cloud_records),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return sealed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and accept direct TensorRT plans on the target payload GPU"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    args = parser.parse_args()
    build(args.manifest)


if __name__ == "__main__":
    main()

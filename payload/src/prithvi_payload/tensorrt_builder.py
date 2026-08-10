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
    CLOUD_MAX_AGGREGATE_CLASS_MISMATCH,
    CLOUD_MAX_SCENE_CLASS_MISMATCH,
    REVIEWED_CLOUD_BASE_PATCH_SIZE,
    REVIEWED_CLOUD_SCENE_BATCH_SIZES,
    REVIEWED_CLOUD_SCENE_PATCH_SIZES,
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

CROP_MAX_CLASS_MISMATCH = 0.002
CROP_MAX_MEAN_PROBABILITY_ERROR = 0.005
ONNX_OPSET = 18
_EXPECTED_CROP_CANONICALIZATION = {
    "as_strided_reshapes": 1,
    "layer_norm_aux_outputs_removed": 50,
    "pow_base_casts": 4,
    "reduce_axes_initializers": 2,
    "sequence_gathers": 36,
    "split_sequences_removed": 12,
}
_EXPECTED_CLOUD_CANONICALIZATION = {
    "layer_norm_aux_outputs_removed": 50,
    "matmul_wrappers_removed": 6,
    "pow_base_casts": 1,
    "reduce_axes_initializers": 16,
    "sequence_gathers": 9,
    "sequence_slices": 9,
    "split_sequences_removed": 6,
    "vector_norm_wrappers_removed": 6,
}


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


def _crop_precision(value: str) -> str:
    normalized = value.strip().casefold()
    if normalized not in {"fp32", "mixed-fp16"}:
        raise TensorRTBuildError(
            "VITA_CROP_TRT_PRECISION must be fp32 or mixed-fp16"
        )
    return normalized


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
        "--fp16",
        "--builderOptimizationLevel",
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
    full_check: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import onnx

    try:
        model = onnx.load(str(path), load_external_data=True)
        onnx.checker.check_model(model, full_check=full_check)
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


def _onnx_constant_evaluator(model: Any) -> Any:
    """Return a strict evaluator for the small constant subgraphs we canonicalize."""

    import onnx
    from onnx import helper, numpy_helper

    initializers = {
        initializer.name: initializer for initializer in model.graph.initializer
    }
    producers = {
        output: node for node in model.graph.node for output in node.output if output
    }
    cache: dict[str, np.ndarray] = {}

    def evaluate(name: str) -> np.ndarray:
        if name in cache:
            return cache[name]
        if name in initializers:
            value = numpy_helper.to_array(initializers[name])
            cache[name] = value
            return value
        try:
            node = producers[name]
        except KeyError as error:
            raise TensorRTBuildError(
                f"ONNX value {name!r} is not a compile-time constant"
            ) from error

        attributes = {
            attribute.name: helper.get_attribute_value(attribute)
            for attribute in node.attribute
        }
        if node.op_type == "Constant":
            if "value" in attributes:
                value = numpy_helper.to_array(attributes["value"])
            elif "value_int" in attributes:
                value = np.asarray(attributes["value_int"], dtype=np.int64)
            elif "value_ints" in attributes:
                value = np.asarray(attributes["value_ints"], dtype=np.int64)
            else:
                raise TensorRTBuildError(
                    f"Unsupported ONNX Constant representation for {name!r}"
                )
        elif node.op_type == "Identity":
            value = evaluate(node.input[0])
        elif node.op_type == "Reshape":
            value = np.reshape(evaluate(node.input[0]), evaluate(node.input[1]))
        elif node.op_type == "Cast":
            value = evaluate(node.input[0]).astype(
                onnx.helper.tensor_dtype_to_np_dtype(attributes["to"])
            )
        else:
            raise TensorRTBuildError(
                f"Unsupported ONNX constant expression {node.op_type} for {name!r}"
            )
        cache[name] = value
        return value

    return evaluate


def _is_contiguous_index_view(
    size: np.ndarray,
    stride: np.ndarray,
    storage_offset: int,
) -> bool:
    """Whether as_strided enumerates the source storage in ordinary row-major order."""

    dimensions = [int(value) for value in np.asarray(size).reshape(-1)]
    strides = [int(value) for value in np.asarray(stride).reshape(-1)]
    if storage_offset != 0 or len(dimensions) != len(strides):
        return False
    if not dimensions or any(dimension <= 0 for dimension in dimensions):
        return False
    expected_stride = 1
    for dimension, actual_stride in zip(
        reversed(dimensions), reversed(strides), strict=True
    ):
        # A singleton axis never advances storage, so any non-negative stride is
        # value-equivalent. Non-singleton axes must be exactly contiguous.
        if actual_stride < 0 or (dimension > 1 and actual_stride != expected_stride):
            return False
        expected_stride *= dimension
    return True


def _remove_dead_onnx_nodes(model: Any) -> int:
    """Remove side-effect-free exporter helpers unreachable from graph outputs."""

    import onnx

    for node in model.graph.node:
        if any(
            attribute.type
            in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
            for attribute in node.attribute
        ):
            raise TensorRTBuildError(
                f"Crop ONNX graph still contains control flow at {node.name or node.op_type}"
            )

    needed = {output.name for output in model.graph.output}
    kept = []
    for node in reversed(model.graph.node):
        if any(output in needed for output in node.output):
            kept.append(node)
            needed.update(name for name in node.input if name)
    kept.reverse()
    removed = len(model.graph.node) - len(kept)
    del model.graph.node[:]
    model.graph.node.extend(kept)
    return removed


def _canonicalize_crop_onnx(
    path: Path,
    *,
    expected: dict[str, int] | None = None,
) -> dict[str, int]:
    """Lower pinned PyTorch 2.6-alpha ONNX encodings to TensorRT 10.8 ONNX.

    Every rewrite is value-preserving and guarded by structural assertions. The
    target parser check and real-scene parity gates remain authoritative.
    """

    import onnx
    from onnx import helper, inliner, numpy_helper

    model = onnx.load(str(path), load_external_data=True)
    evaluate = _onnx_constant_evaluator(model)
    stats = {
        "as_strided_reshapes": 0,
        "layer_norm_aux_outputs_removed": 0,
        "pow_base_casts": 0,
        "reduce_axes_initializers": 0,
        "sequence_gathers": 0,
        "split_sequences_removed": 0,
    }

    # The crop decoder's 1x1 adaptive-pool view is exported as a local
    # as_strided function containing ONNX sequence/control-flow operators.
    # Its fixed sizes and strides enumerate contiguous storage exactly, so a
    # Reshape is identical and directly supported by TensorRT.
    preinline_nodes = []
    for node in model.graph.node:
        if not (
            node.domain == "pkg.onnxscript.torch_lib"
            and node.op_type == "_aten_as_strided_onnx"
        ):
            preinline_nodes.append(node)
            continue
        if len(node.input) not in (3, 4):
            raise TensorRTBuildError(
                f"Unexpected crop as_strided signature at {node.name}: {node.input}"
            )
        size = np.asarray(evaluate(node.input[1]), dtype=np.int64)
        stride = np.asarray(evaluate(node.input[2]), dtype=np.int64)
        storage_offset = (
            int(np.asarray(evaluate(node.input[3])).item())
            if len(node.input) == 4
            else 0
        )
        if not _is_contiguous_index_view(size, stride, storage_offset):
            raise TensorRTBuildError(
                f"Crop as_strided at {node.name} is not a contiguous value-preserving view"
            )
        preinline_nodes.append(
            helper.make_node(
                "Reshape",
                [node.input[0], node.input[1]],
                list(node.output),
                name=f"{node.name}_contiguous_reshape",
            )
        )
        stats["as_strided_reshapes"] += 1
    del model.graph.node[:]
    model.graph.node.extend(preinline_nodes)

    model = inliner.inline_local_functions(model)
    custom_nodes = [
        f"{node.domain}::{node.op_type}" for node in model.graph.node if node.domain
    ]
    if model.functions or custom_nodes:
        raise TensorRTBuildError(
            "Crop ONNX local-function inlining was incomplete: "
            f"functions={len(model.functions)}, custom_nodes={sorted(set(custom_nodes))}"
        )

    users: dict[str, list[Any]] = {}
    for node in model.graph.node:
        for name in node.input:
            users.setdefault(name, []).append(node)
    graph_outputs = {output.name for output in model.graph.output}
    for node in model.graph.node:
        if node.op_type != "LayerNormalization" or len(node.output) <= 1:
            continue
        auxiliary = list(node.output[1:])
        if any(users.get(name) or name in graph_outputs for name in auxiliary):
            raise TensorRTBuildError(
                f"Crop LayerNormalization auxiliary output is used at {node.name}"
            )
        stats["layer_norm_aux_outputs_removed"] += len(auxiliary)
        del node.output[1:]

    evaluate = _onnx_constant_evaluator(model)
    split_sources: dict[str, tuple[str, int]] = {}
    for node in model.graph.node:
        if node.op_type != "SplitToSequence":
            continue
        attributes = {
            attribute.name: helper.get_attribute_value(attribute)
            for attribute in node.attribute
        }
        if attributes.get("keepdims") != 0 or len(node.input) != 2:
            raise TensorRTBuildError(
                f"Unexpected crop SplitToSequence encoding at {node.name}"
            )
        split = np.asarray(evaluate(node.input[1])).reshape(-1)
        if split.size != 1 or int(split.item()) != 1:
            raise TensorRTBuildError(
                f"Crop SplitToSequence at {node.name} is not an unbind-by-one"
            )
        sequence_name = node.output[0]
        if sequence_name in graph_outputs or any(
            user.op_type != "SequenceAt" for user in users.get(sequence_name, [])
        ):
            raise TensorRTBuildError(
                f"Crop sequence {sequence_name} has unsupported consumers"
            )
        split_sources[sequence_name] = (
            node.input[0],
            int(attributes.get("axis", 0)),
        )

    rewritten_nodes = []
    for index, node in enumerate(model.graph.node):
        if node.op_type == "SplitToSequence" and node.output[0] in split_sources:
            stats["split_sequences_removed"] += 1
            continue
        if node.op_type == "SequenceAt" and node.input[0] in split_sources:
            index_value = np.asarray(evaluate(node.input[1]))
            if index_value.size != 1:
                raise TensorRTBuildError(
                    f"Crop SequenceAt index is not scalar at {node.name}"
                )
            data, axis = split_sources[node.input[0]]
            rewritten_nodes.append(
                helper.make_node(
                    "Gather",
                    [data, node.input[1]],
                    list(node.output),
                    name=f"{node.name}_direct_gather",
                    axis=axis,
                )
            )
            stats["sequence_gathers"] += 1
            continue
        if node.op_type == "Pow":
            cast_output = f"{node.input[0]}__pow_castlike__{index}"
            rewritten_nodes.append(
                helper.make_node(
                    "CastLike",
                    [node.input[0], node.input[1]],
                    [cast_output],
                    name=f"{node.name}_base_castlike",
                )
            )
            node.input[0] = cast_output
            stats["pow_base_casts"] += 1
        rewritten_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)

    evaluate = _onnx_constant_evaluator(model)
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    for index, node in enumerate(model.graph.node):
        if (
            not node.op_type.startswith("Reduce")
            or len(node.input) < 2
            or not node.input[1]
            or node.input[1] in initializer_names
        ):
            continue
        try:
            axes = np.asarray(evaluate(node.input[1]), dtype=np.int64)
        except TensorRTBuildError:
            continue
        name = f"{node.input[1]}__trt_initializer_{index}"
        model.graph.initializer.append(numpy_helper.from_array(axes, name=name))
        initializer_names.add(name)
        node.input[1] = name
        stats["reduce_axes_initializers"] += 1

    _remove_dead_onnx_nodes(model)
    del model.graph.value_info[:]
    remaining_sequences = sorted(
        {node.op_type for node in model.graph.node if "Sequence" in node.op_type}
    )
    if remaining_sequences:
        raise TensorRTBuildError(
            f"Crop ONNX still contains sequence operators: {remaining_sequences}"
        )

    required = _EXPECTED_CROP_CANONICALIZATION if expected is None else expected
    if stats != required:
        raise TensorRTBuildError(
            "Crop ONNX export does not match the pinned canonicalization contract: "
            f"observed={stats}, expected={required}"
        )

    # Validate the in-memory candidate before atomically replacing the raw export.
    onnx.checker.check_model(model, full_check=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.canonical.tmp")
    try:
        onnx.save_model(model, str(temporary), save_as_external_data=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return stats


def _canonicalize_cloud_onnx(
    path: Path,
    *,
    expected: dict[str, int] | None = None,
) -> dict[str, int]:
    """Lower the pinned fixed-batch OmniCloudMask export to TensorRT 10.8 ONNX."""

    import onnx
    from onnx import helper, inliner, numpy_helper

    model = onnx.load(str(path), load_external_data=True)
    value_specs = {
        value.name: value
        for value in [
            *model.graph.input,
            *model.graph.value_info,
            *model.graph.output,
        ]
    }
    evaluate = _onnx_constant_evaluator(model)
    stats = {
        "layer_norm_aux_outputs_removed": 0,
        "matmul_wrappers_removed": 0,
        "pow_base_casts": 0,
        "reduce_axes_initializers": 0,
        "sequence_gathers": 0,
        "sequence_slices": 0,
        "split_sequences_removed": 0,
        "vector_norm_wrappers_removed": 0,
    }

    # ONNX Script's generic wrappers branch over scalar cases. Every pinned
    # attention operand has a recorded rank >= 2, and every vector norm is the
    # fixed ord=2/keepdim form, so the corresponding standard ONNX nodes are
    # exactly equivalent and avoid unsupported control-flow branches.
    preinline_nodes = []
    for index, node in enumerate(model.graph.node):
        if node.domain == "pkg.onnxscript.torch_lib" and node.op_type == "aten_matmul":
            ranks = []
            for name in node.input:
                value = value_specs.get(name)
                ranks.append(
                    len(value.type.tensor_type.shape.dim) if value is not None else -1
                )
            if len(ranks) != 2 or any(rank < 2 for rank in ranks):
                raise TensorRTBuildError(
                    f"Cloud matmul at {node.name} is not a recorded rank>=2 operation: {ranks}"
                )
            preinline_nodes.append(
                helper.make_node(
                    "MatMul",
                    list(node.input),
                    list(node.output),
                    name=f"{node.name}_ranked",
                )
            )
            stats["matmul_wrappers_removed"] += 1
            continue
        if (
            node.domain == "pkg.onnxscript.torch_lib"
            and node.op_type == "_aten_linalg_vector_norm_onnx"
        ):
            attributes = {
                attribute.name: helper.get_attribute_value(attribute)
                for attribute in node.attribute
            }
            value = value_specs.get(node.input[0])
            rank = len(value.type.tensor_type.shape.dim) if value is not None else -1
            axes = np.asarray(evaluate(node.input[1]), dtype=np.int64).reshape(-1)
            if (
                float(attributes.get("ord", math.nan)) != 2.0
                or int(attributes.get("keepdim", -1)) != 1
                or rank < 1
                or axes.size == 0
                or any(not -rank <= int(axis) < rank for axis in axes)
            ):
                raise TensorRTBuildError(
                    f"Cloud vector norm at {node.name} is not the pinned rank/ord/axes form"
                )
            axes_name = f"{node.name}_axes_{index}"
            model.graph.initializer.append(
                numpy_helper.from_array(axes, name=axes_name)
            )
            preinline_nodes.append(
                helper.make_node(
                    "ReduceL2",
                    [node.input[0], axes_name],
                    list(node.output),
                    name=f"{node.name}_reduce_l2",
                    keepdims=1,
                )
            )
            stats["vector_norm_wrappers_removed"] += 1
            continue
        preinline_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(preinline_nodes)

    model = inliner.inline_local_functions(model)
    custom_nodes = [
        f"{node.domain}::{node.op_type}" for node in model.graph.node if node.domain
    ]
    if model.functions or custom_nodes:
        raise TensorRTBuildError(
            "Cloud ONNX local-function inlining was incomplete: "
            f"functions={len(model.functions)}, custom_nodes={sorted(set(custom_nodes))}"
        )

    users: dict[str, list[Any]] = {}
    for node in model.graph.node:
        for name in node.input:
            users.setdefault(name, []).append(node)
    graph_outputs = {output.name for output in model.graph.output}
    for node in model.graph.node:
        if node.op_type != "LayerNormalization" or len(node.output) <= 1:
            continue
        auxiliary = list(node.output[1:])
        if any(users.get(name) or name in graph_outputs for name in auxiliary):
            raise TensorRTBuildError(
                f"Cloud LayerNormalization auxiliary output is used at {node.name}"
            )
        stats["layer_norm_aux_outputs_removed"] += len(auxiliary)
        del node.output[1:]

    evaluate = _onnx_constant_evaluator(model)
    split_sources: dict[str, tuple[str, int, int, int]] = {}
    for node in model.graph.node:
        if node.op_type != "SplitToSequence":
            continue
        attributes = {
            attribute.name: helper.get_attribute_value(attribute)
            for attribute in node.attribute
        }
        if len(node.input) != 2:
            raise TensorRTBuildError(
                f"Unexpected cloud SplitToSequence signature at {node.name}"
            )
        split = np.asarray(evaluate(node.input[1])).reshape(-1)
        keepdims = int(attributes.get("keepdims", 1))
        if split.size != 1 or int(split.item()) <= 0 or keepdims not in (0, 1):
            raise TensorRTBuildError(
                f"Unexpected cloud SplitToSequence contract at {node.name}"
            )
        sequence_name = node.output[0]
        if sequence_name in graph_outputs or any(
            user.op_type != "SequenceAt" for user in users.get(sequence_name, [])
        ):
            raise TensorRTBuildError(
                f"Cloud sequence {sequence_name} has unsupported consumers"
            )
        split_sources[sequence_name] = (
            node.input[0],
            int(attributes.get("axis", 0)),
            keepdims,
            int(split.item()),
        )

    rewritten_nodes = []
    for index, node in enumerate(model.graph.node):
        if node.op_type == "SplitToSequence" and node.output[0] in split_sources:
            stats["split_sequences_removed"] += 1
            continue
        if node.op_type == "SequenceAt" and node.input[0] in split_sources:
            data, axis, keepdims, split_size = split_sources[node.input[0]]
            item_array = np.asarray(evaluate(node.input[1]))
            if item_array.size != 1 or int(item_array.item()) < 0:
                raise TensorRTBuildError(
                    f"Cloud SequenceAt index is not a non-negative scalar at {node.name}"
                )
            item = int(item_array.item())
            if keepdims == 0:
                rewritten_nodes.append(
                    helper.make_node(
                        "Gather",
                        [data, node.input[1]],
                        list(node.output),
                        name=f"{node.name}_direct_gather",
                        axis=axis,
                    )
                )
                stats["sequence_gathers"] += 1
            else:
                values = {
                    "starts": [item * split_size],
                    "ends": [(item + 1) * split_size],
                    "axes": [axis],
                    "steps": [1],
                }
                inputs = [data]
                for kind, value in values.items():
                    name = f"{node.name}_{kind}_{index}"
                    model.graph.initializer.append(
                        numpy_helper.from_array(
                            np.asarray(value, dtype=np.int64),
                            name=name,
                        )
                    )
                    inputs.append(name)
                rewritten_nodes.append(
                    helper.make_node(
                        "Slice",
                        inputs,
                        list(node.output),
                        name=f"{node.name}_direct_slice",
                    )
                )
                stats["sequence_slices"] += 1
            continue
        if node.op_type == "Pow":
            cast_output = f"{node.input[0]}__pow_castlike__{index}"
            rewritten_nodes.append(
                helper.make_node(
                    "CastLike",
                    [node.input[0], node.input[1]],
                    [cast_output],
                    name=f"{node.name}_base_castlike",
                )
            )
            node.input[0] = cast_output
            stats["pow_base_casts"] += 1
        rewritten_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)

    evaluate = _onnx_constant_evaluator(model)
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    for index, node in enumerate(model.graph.node):
        if (
            not node.op_type.startswith("Reduce")
            or len(node.input) < 2
            or not node.input[1]
            or node.input[1] in initializer_names
        ):
            continue
        try:
            axes = np.asarray(evaluate(node.input[1]), dtype=np.int64)
        except TensorRTBuildError:
            continue
        name = f"{node.input[1]}__trt_initializer_{index}"
        model.graph.initializer.append(numpy_helper.from_array(axes, name=name))
        initializer_names.add(name)
        node.input[1] = name
        stats["reduce_axes_initializers"] += 1

    _remove_dead_onnx_nodes(model)
    del model.graph.value_info[:]
    remaining_sequences = sorted(
        {node.op_type for node in model.graph.node if "Sequence" in node.op_type}
    )
    if remaining_sequences:
        raise TensorRTBuildError(
            f"Cloud ONNX still contains sequence operators: {remaining_sequences}"
        )

    required = _EXPECTED_CLOUD_CANONICALIZATION if expected is None else expected
    if stats != required:
        raise TensorRTBuildError(
            "Cloud ONNX export does not match the pinned canonicalization contract: "
            f"observed={stats}, expected={required}"
        )

    onnx.checker.check_model(model, full_check=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.canonical.tmp")
    try:
        onnx.save_model(model, str(temporary), save_as_external_data=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return stats


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
    canonicalize_crop: bool = False,
    canonicalize_cloud: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if canonicalize_crop and canonicalize_cloud:
        raise TensorRTBuildError("An ONNX graph cannot use two canonicalization contracts")
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
        if canonicalize_crop:
            canonicalization = _canonicalize_crop_onnx(path)
            print(
                f"[onnx] canonicalized crop graph: {canonicalization}",
                flush=True,
            )
        if canonicalize_cloud:
            canonicalization = _canonicalize_cloud_onnx(path)
            print(
                f"[onnx] canonicalized cloud graph: {canonicalization}",
                flush=True,
            )
    except Exception as error:
        raise TensorRTBuildError(
            f"PyTorch ONNX export failed for {path}: {_exception_summary(error)}"
        ) from error
    return _validate_onnx(
        path,
        expected_inputs=input_names,
        expected_outputs=output_names,
        full_check=not (canonicalize_crop or canonicalize_cloud),
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
        canonicalize_crop=True,
    )
    return {"path": path, "inputs": input_specs, "outputs": output_specs}


def _export_cloud_onnx(
    source: nn.Module,
    path: Path,
    *,
    batch_size: int,
    patch_size: int,
    dtype: torch.dtype,
) -> dict[str, Any]:
    sample = torch.zeros(
        (batch_size, 3, patch_size, patch_size),
        device="cuda",
        dtype=dtype,
    )
    exported = torch.export.export(
        source,
        (sample,),
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
        canonicalize_cloud=True,
    )
    del sample
    return {
        "path": path,
        "inputs": input_specs,
        "outputs": output_specs,
        "batch_size": batch_size,
        "patch_size": patch_size,
        "profile": None,
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
    timing_cache = manifest_path.parent / "timing" / f"{name}.{precision}.cache"
    timing_cache.parent.mkdir(parents=True, exist_ok=True)
    workspace_mib = int(os.environ.get("VITA_TRT_WORKSPACE_MIB", "4096"))
    if workspace_mib < 256:
        raise TensorRTBuildError("VITA_TRT_WORKSPACE_MIB must be at least 256")
    builder_optimization_level = int(
        os.environ.get("VITA_TRT_BUILDER_OPTIMIZATION_LEVEL", "5")
    )
    if not 0 <= builder_optimization_level <= 5:
        raise TensorRTBuildError(
            "VITA_TRT_BUILDER_OPTIMIZATION_LEVEL must be between 0 and 5"
        )
    arguments = [
        str(executable),
        f"--onnx={onnx_path}",
        "--noTF32",
        "--skipInference",
        f"--memPoolSize=workspace:{workspace_mib}",
        f"--timingCacheFile={timing_cache}",
        f"--builderOptimizationLevel={builder_optimization_level}",
    ]
    if precision == "mixed-fp16":
        # TensorRT 10.8 weak typing retains the FP32 ONNX I/O contract while
        # selecting FP16 tactics internally where supported. This mirrors the
        # established PyTorch CUDA autocast execution used by the operational
        # baseline without converting numerically sensitive unsupported layers.
        arguments.append("--fp16")
    else:
        arguments.append("--stronglyTyped")
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
    precision: str,
) -> dict[str, Any]:
    batch = build_crop_parity_inputs(profiles, batch_size=OPTIMIZED_BATCH_SIZE)
    image, temporal, location, thresholds, valid_mask = [value.cuda() for value in batch]
    runner = NativeTensorRTPlan(
        record,
        manifest_path=manifest_path,
        device=torch.device("cuda"),
    )
    reference_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if precision == "mixed-fp16"
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), reference_context:
        reference = source(image, temporal, location)
    with torch.inference_mode():
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
        "reference": (
            "pytorch_cuda_autocast_fp16"
            if precision == "mixed-fp16"
            else "pytorch_cuda_fp32"
        ),
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


def _discover_cloud_scene_patch_sizes(runtime: Any) -> dict[str, int]:
    """Apply OmniCloudMask's pinned no-data rule to every accepted scene."""

    from omnicloudmask.cloud_mask import check_patch_size

    backend = runtime.cloud.backend
    if int(backend.patch_size) != REVIEWED_CLOUD_BASE_PATCH_SIZE:
        raise TensorRTBuildError(
            "Cloud base patch size does not match the reviewed TensorRT contract"
        )
    scene_patch_sizes: dict[str, int] = {}
    for profile in runtime._scene_cloud_warmups:
        scene_input = profile.get("input")
        if not isinstance(scene_input, str) or not scene_input:
            raise TensorRTBuildError("Cloud scene profile has no stable input identity")
        if scene_input in scene_patch_sizes:
            raise TensorRTBuildError(f"Duplicate cloud scene profile: {scene_input}")
        image = _load_cloud_scene(
            profile,
            input_config=runtime.cloud.config["input"],
        )
        prepared = backend._prepare_input(image)
        if prepared is None:
            raise TensorRTBuildError(
                f"Cloud acceptance scene contains no model-valid pixels: {scene_input}"
            )
        requested = min(int(backend.patch_size), *prepared.shape[1:])
        overlap = min(int(backend.patch_overlap), max(0, requested // 2))
        _, adjusted = check_patch_size(
            prepared,
            0.0,
            requested,
            overlap,
        )
        scene_patch_sizes[scene_input] = int(adjusted)
        del image, prepared

    observed = set(scene_patch_sizes.values())
    expected = set(REVIEWED_CLOUD_SCENE_PATCH_SIZES)
    if observed != expected or len(scene_patch_sizes) != 4:
        raise TensorRTBuildError(
            "Cloud scene patch sizes do not match the reviewed four-scene contract: "
            f"observed={scene_patch_sizes}, expected_sizes={sorted(expected)}"
        )
    print(
        f"[cloud] reviewed fixed-scene patch sizes: {scene_patch_sizes}",
        flush=True,
    )
    return scene_patch_sizes


def _validate_cloud_parity(
    runtime: Any,
    records: list[dict[str, Any]],
    *,
    manifest_path: Path,
    dtype: torch.dtype,
    scene_patch_sizes: dict[str, int],
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
            scene_input = profile["input"]
            patch_size = scene_patch_sizes.get(scene_input)
            if patch_size is None:
                raise TensorRTBuildError(
                    f"Cloud parity scene has no accepted patch size: {scene_input}"
                )
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
            total_pixels += pixels
            total_mismatches += mismatches
            scene_results.append(
                {
                    "input": scene_input,
                    "patch_size": patch_size,
                    "class_mismatch_fraction": fraction,
                    "mismatch_count": mismatches,
                    "pixel_count": pixels,
                }
            )
    finally:
        backend.models = source_models
    aggregate = total_mismatches / total_pixels
    worst_scene = max(
        scene_results,
        key=lambda result: float(result["class_mismatch_fraction"]),
    )
    parity = {
        "class_mismatch_fraction": aggregate,
        "maximum_scene_class_mismatch_fraction": worst_scene[
            "class_mismatch_fraction"
        ],
        "mismatch_count": total_mismatches,
        "pixel_count": total_pixels,
        "acceptance_limits": {
            "aggregate_class_mismatch_fraction": (
                CLOUD_MAX_AGGREGATE_CLASS_MISMATCH
            ),
            "scene_class_mismatch_fraction": CLOUD_MAX_SCENE_CLASS_MISMATCH,
        },
        "scenes": scene_results,
    }
    aggregate_failed = aggregate > CLOUD_MAX_AGGREGATE_CLASS_MISMATCH
    scene_failed = (
        float(worst_scene["class_mismatch_fraction"])
        > CLOUD_MAX_SCENE_CLASS_MISMATCH
    )
    if aggregate_failed or scene_failed:
        print(
            "[parity] rejected cloud report:\n"
            + json.dumps(parity, indent=2, sort_keys=True),
            flush=True,
        )
        raise TensorRTBuildError(
            "Cloud direct TensorRT parity failed after all four scenes: "
            f"aggregate={aggregate:.8f}/"
            f"{CLOUD_MAX_AGGREGATE_CLASS_MISMATCH:.8f}, "
            f"worst_scene={worst_scene['input']}="
            f"{float(worst_scene['class_mismatch_fraction']):.8f}/"
            f"{CLOUD_MAX_SCENE_CLASS_MISMATCH:.8f}"
        )
    return parity


def build(manifest_path: Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    versions = _require_build_dependencies()
    target = current_target_signature()
    executable = _find_trtexec()
    trtexec_record = _validate_trtexec(executable)
    crop_precision = _crop_precision(
        os.environ.get("VITA_CROP_TRT_PRECISION", "mixed-fp16")
    )
    cloud_precision = os.environ.get("VITA_CLOUD_TRT_PRECISION", "fp16").strip().casefold()
    cloud_dtype = _precision_dtype(cloud_precision, name="VITA_CLOUD_TRT_PRECISION")

    print("[preflight] direct TensorRT dependencies and trtexec passed", flush=True)
    runtime = _load_source_runtime(cloud_precision)
    scene_patch_sizes = _discover_cloud_scene_patch_sizes(runtime)

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
    cloud_batch_size = int(os.environ.get("VITA_CLOUD_BATCH_SIZE", "4"))
    if cloud_batch_size < 1:
        raise TensorRTBuildError("VITA_CLOUD_BATCH_SIZE must be positive")
    reviewed_batches = dict(REVIEWED_CLOUD_SCENE_BATCH_SIZES)
    if max(reviewed_batches.values()) != cloud_batch_size:
        raise TensorRTBuildError(
            "VITA_CLOUD_BATCH_SIZE does not match the reviewed Balkan batch contract"
        )
    cloud_profiles = [
        (patch_size, reviewed_batches[patch_size])
        for patch_size in sorted(set(scene_patch_sizes.values()))
    ]
    cloud_graphs = [
        _export_cloud_onnx(
            cloud_source,
            onnx_root
            / (
                f"cloud.{OMNICLOUDMASK_ENSEMBLE_SHA256[:12]}."
                f"b{batch_size}.{patch_size}.{cloud_precision}.onnx"
            ),
            batch_size=batch_size,
            patch_size=patch_size,
            dtype=cloud_dtype,
        )
        for patch_size, batch_size in cloud_profiles
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
        name=f"crop-{crop_precision}",
        precision=crop_precision,
        manifest_path=manifest_path,
        executable=executable,
        target=target,
    )
    crop_record = {
        **_plan_record(
            built_crop,
            manifest_path=manifest_path,
            precision=crop_precision,
        ),
        "source_sha256": SELECTED_CHECKPOINT_SHA256,
        "tf32": False,
    }
    # Qualify the crop candidate before spending time constructing every cloud
    # plan. Candidate files remain inert and the previous accepted manifest is
    # untouched if this first scientific gate fails.
    runtime = _load_source_runtime(cloud_precision)
    print("[parity] validating crop plan on balanced real-scene tiles", flush=True)
    crop_record["parity"] = _validate_crop_parity(
        runtime.crop.model,
        crop_record,
        manifest_path=manifest_path,
        profiles=runtime._crop_parity_profiles,
        precision=crop_precision,
    )
    del runtime
    _release_cuda_memory()

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
                "logical_min_batch_size": 1,
                "logical_max_batch_size": built["batch_size"],
            }
        )

    # Reload the untouched source for the complete-scene cloud comparison only
    # after every required shape-specific plan exists.
    runtime = _load_source_runtime(cloud_precision)
    print("[parity] validating cloud plans on all four complete scenes", flush=True)
    cloud_parity = _validate_cloud_parity(
        runtime,
        cloud_records,
        manifest_path=manifest_path,
        dtype=cloud_dtype,
        scene_patch_sizes=scene_patch_sizes,
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
                "scene_patch_sizes": scene_patch_sizes,
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

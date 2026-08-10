from __future__ import annotations

import json
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace

import pytest
from prithvi_payload import tensorrt_runtime
from prithvi_payload.tensorrt_builder import (
    _canonicalize_cloud_onnx,
    _canonicalize_crop_onnx,
    _normalize_onnxscript_integer_attributes,
)
from prithvi_payload.tensorrt_runtime import (
    MANIFEST_SCHEMA_VERSION,
    TensorRTArtifactError,
    load_accepted_manifest,
    write_manifest_atomic,
)


def _manifest() -> dict:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "accepted",
        "target": {"gpu": "test"},
        "models": {"crop": {}, "cloud": {}},
    }


def test_direct_tensorrt_manifest_is_atomic_and_integrity_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "direct" / "accepted.json"
    sealed = write_manifest_atomic(path, _manifest())
    monkeypatch.setattr(
        tensorrt_runtime,
        "current_target_signature",
        lambda: {"gpu": "test"},
    )

    loaded = load_accepted_manifest(path, required_models=("crop", "cloud"))

    assert loaded["manifest_sha256"] == sealed["manifest_sha256"]
    assert loaded["_manifest_path"] == path.resolve()
    assert not list(path.parent.glob("*.tmp"))


def test_direct_tensorrt_manifest_rejects_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "accepted.json"
    write_manifest_atomic(path, _manifest())
    record = json.loads(path.read_text(encoding="utf-8"))
    record["status"] = "rejected"
    path.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(
        tensorrt_runtime,
        "current_target_signature",
        lambda: {"gpu": "test"},
    )

    with pytest.raises(TensorRTArtifactError, match="integrity"):
        load_accepted_manifest(path)


def test_direct_tensorrt_manifest_rejects_another_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "accepted.json"
    write_manifest_atomic(path, _manifest())
    monkeypatch.setattr(
        tensorrt_runtime,
        "current_target_signature",
        lambda: {"gpu": "different"},
    )

    with pytest.raises(TensorRTArtifactError, match="different GPU"):
        load_accepted_manifest(path)


def test_onnxscript_boolean_integer_attributes_are_losslessly_normalized() -> None:
    class _Type(Enum):
        INT = auto()
        FLOAT = auto()
        GRAPH = auto()

    root_int = SimpleNamespace(type=_Type.INT, value=True)
    nested_int = SimpleNamespace(type=_Type.INT, value=False)
    untouched = SimpleNamespace(type=_Type.FLOAT, value=True)
    child = [SimpleNamespace(attributes={"nested": nested_int})]
    graph_attribute = SimpleNamespace(type=_Type.GRAPH, value=child)
    graph = [
        SimpleNamespace(
            attributes={
                "root": root_int,
                "child": graph_attribute,
                "float": untouched,
            }
        )
    ]
    program = SimpleNamespace(model=SimpleNamespace(graph=graph))

    count = _normalize_onnxscript_integer_attributes(program)

    assert count == 2
    assert root_int.value == 1 and type(root_int.value) is int
    assert nested_int.value == 0 and type(nested_int.value) is int
    assert untouched.value is True


def test_crop_onnx_canonicalization_lowers_pinned_exporter_patterns(
    tmp_path: Path,
) -> None:
    onnx = pytest.importorskip("onnx")
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    nodes = [
        helper.make_node("Constant", [], ["pow_base"], value_int=10000),
        helper.make_node("Pow", ["pow_base", "exponent"], ["pow_out"], name="pow"),
        helper.make_node(
            "LayerNormalization",
            ["ln_x", "ln_scale", "ln_bias"],
            ["ln_out", "ln_mean", "ln_invstd"],
            name="layer_norm",
            axis=-1,
        ),
        helper.make_node("Constant", [], ["split_size"], value_int=1),
        helper.make_node(
            "SplitToSequence",
            ["sequence_x", "split_size"],
            ["sequence"],
            name="unbind",
            axis=0,
            keepdims=0,
        ),
        helper.make_node("Constant", [], ["sequence_index"], value_int=1),
        helper.make_node(
            "SequenceAt",
            ["sequence", "sequence_index"],
            ["gathered"],
            name="getitem",
        ),
        helper.make_node(
            "Constant",
            [],
            ["view_size"],
            value=numpy_helper.from_array(np.asarray([2, 2], dtype=np.int64)),
        ),
        helper.make_node(
            "Constant",
            [],
            ["view_stride"],
            value=numpy_helper.from_array(np.asarray([2, 1], dtype=np.int64)),
        ),
        helper.make_node(
            "_aten_as_strided_onnx",
            ["view_x", "view_size", "view_stride"],
            ["viewed"],
            name="as_strided",
            domain="pkg.onnxscript.torch_lib",
        ),
        helper.make_node(
            "Constant",
            [],
            ["axes_value"],
            value=numpy_helper.from_array(np.asarray([-1], dtype=np.int64)),
        ),
        helper.make_node("Constant", [], ["axes_shape"], value_ints=[-1]),
        helper.make_node("Reshape", ["axes_value", "axes_shape"], ["axes"]),
        helper.make_node("ReduceMean", ["view_x", "axes"], ["reduced"], keepdims=1),
    ]
    graph = helper.make_graph(
        nodes,
        "canonicalization_test",
        [
            helper.make_tensor_value_info("exponent", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("ln_x", TensorProto.FLOAT, [1, 2]),
            helper.make_tensor_value_info("sequence_x", TensorProto.FLOAT, [3, 2]),
            helper.make_tensor_value_info("view_x", TensorProto.FLOAT, [2, 2]),
        ],
        [
            helper.make_tensor_value_info("pow_out", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("ln_out", TensorProto.FLOAT, [1, 2]),
            helper.make_tensor_value_info("gathered", TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info("viewed", TensorProto.FLOAT, [2, 2]),
            helper.make_tensor_value_info("reduced", TensorProto.FLOAT, [2, 1]),
        ],
        initializer=[
            numpy_helper.from_array(np.ones(2, dtype=np.float32), name="ln_scale"),
            numpy_helper.from_array(np.zeros(2, dtype=np.float32), name="ln_bias"),
        ],
    )
    model = helper.make_model(
        graph,
        ir_version=9,
        opset_imports=[
            helper.make_opsetid("", 18),
            helper.make_opsetid("pkg.onnxscript.torch_lib", 1),
        ],
    )
    path = tmp_path / "crop.onnx"
    onnx.save(model, path)

    stats = _canonicalize_crop_onnx(
        path,
        expected={
            "as_strided_reshapes": 1,
            "layer_norm_aux_outputs_removed": 2,
            "pow_base_casts": 1,
            "reduce_axes_initializers": 1,
            "sequence_gathers": 1,
            "split_sequences_removed": 1,
        },
    )

    canonical = onnx.load(path)
    onnx.checker.check_model(canonical, full_check=True)
    assert stats["sequence_gathers"] == 1
    assert not canonical.functions
    assert all(not node.domain for node in canonical.graph.node)
    assert all("Sequence" not in node.op_type for node in canonical.graph.node)
    layer_norm = next(
        node for node in canonical.graph.node if node.op_type == "LayerNormalization"
    )
    assert list(layer_norm.output) == ["ln_out"]
    initializer_names = {value.name for value in canonical.graph.initializer}
    reduction = next(node for node in canonical.graph.node if node.op_type == "ReduceMean")
    assert reduction.input[1] in initializer_names


def test_cloud_onnx_canonicalization_lowers_pinned_exporter_patterns(
    tmp_path: Path,
) -> None:
    onnx = pytest.importorskip("onnx")
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    nodes = [
        helper.make_node("Constant", [], ["pow_base"], value_int=10000),
        helper.make_node("Pow", ["pow_base", "exponent"], ["pow_out"], name="pow"),
        helper.make_node(
            "aten_matmul",
            ["matrix_a", "matrix_b"],
            ["matmul_out"],
            name="matmul_wrapper",
            domain="pkg.onnxscript.torch_lib",
        ),
        helper.make_node(
            "Constant",
            [],
            ["norm_axes"],
            value=numpy_helper.from_array(np.asarray([-1], dtype=np.int64)),
        ),
        helper.make_node(
            "_aten_linalg_vector_norm_onnx",
            ["norm_x", "norm_axes"],
            ["norm_out"],
            name="norm_wrapper",
            domain="pkg.onnxscript.torch_lib",
            ord=2.0,
            keepdim=1,
        ),
        helper.make_node(
            "LayerNormalization",
            ["ln_x", "ln_scale", "ln_bias"],
            ["ln_out", "ln_mean", "ln_invstd"],
            name="layer_norm",
            axis=-1,
        ),
        helper.make_node("Constant", [], ["split_two"], value_int=2),
        helper.make_node(
            "SplitToSequence",
            ["chunk_x", "split_two"],
            ["chunks"],
            name="split_chunks",
            axis=1,
        ),
        helper.make_node("Constant", [], ["chunk_index"], value_int=1),
        helper.make_node(
            "SequenceAt", ["chunks", "chunk_index"], ["chunked"], name="chunk"
        ),
        helper.make_node("Constant", [], ["split_one"], value_int=1),
        helper.make_node(
            "SplitToSequence",
            ["unbind_x", "split_one"],
            ["items"],
            name="unbind",
            axis=0,
            keepdims=0,
        ),
        helper.make_node("Constant", [], ["item_index"], value_int=0),
        helper.make_node(
            "SequenceAt", ["items", "item_index"], ["unbound"], name="getitem"
        ),
        helper.make_node(
            "Constant",
            [],
            ["reduce_axes_value"],
            value=numpy_helper.from_array(np.asarray([-1], dtype=np.int64)),
        ),
        helper.make_node("Constant", [], ["reduce_axes_shape"], value_ints=[-1]),
        helper.make_node(
            "Reshape", ["reduce_axes_value", "reduce_axes_shape"], ["reduce_axes"]
        ),
        helper.make_node(
            "ReduceMean", ["ln_x", "reduce_axes"], ["reduced"], keepdims=1
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "cloud_canonicalization_test",
        [
            helper.make_tensor_value_info("exponent", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("matrix_a", TensorProto.FLOAT, [1, 2, 3]),
            helper.make_tensor_value_info("matrix_b", TensorProto.FLOAT, [1, 3, 4]),
            helper.make_tensor_value_info("norm_x", TensorProto.FLOAT, [1, 2, 3]),
            helper.make_tensor_value_info("ln_x", TensorProto.FLOAT, [1, 2]),
            helper.make_tensor_value_info("chunk_x", TensorProto.FLOAT, [1, 4]),
            helper.make_tensor_value_info("unbind_x", TensorProto.FLOAT, [2, 3]),
        ],
        [
            helper.make_tensor_value_info("pow_out", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("matmul_out", TensorProto.FLOAT, [1, 2, 4]),
            helper.make_tensor_value_info("norm_out", TensorProto.FLOAT, [1, 2, 1]),
            helper.make_tensor_value_info("ln_out", TensorProto.FLOAT, [1, 2]),
            helper.make_tensor_value_info("chunked", TensorProto.FLOAT, [1, 2]),
            helper.make_tensor_value_info("unbound", TensorProto.FLOAT, [3]),
            helper.make_tensor_value_info("reduced", TensorProto.FLOAT, [1, 1]),
        ],
        initializer=[
            numpy_helper.from_array(np.ones(2, dtype=np.float32), name="ln_scale"),
            numpy_helper.from_array(np.zeros(2, dtype=np.float32), name="ln_bias"),
        ],
    )
    model = helper.make_model(
        graph,
        ir_version=9,
        opset_imports=[
            helper.make_opsetid("", 18),
            helper.make_opsetid("pkg.onnxscript.torch_lib", 1),
        ],
    )
    path = tmp_path / "cloud.onnx"
    onnx.save(model, path)

    stats = _canonicalize_cloud_onnx(
        path,
        expected={
            "layer_norm_aux_outputs_removed": 2,
            "matmul_wrappers_removed": 1,
            "pow_base_casts": 1,
            "reduce_axes_initializers": 1,
            "sequence_gathers": 1,
            "sequence_slices": 1,
            "split_sequences_removed": 2,
            "vector_norm_wrappers_removed": 1,
        },
    )

    canonical = onnx.load(path)
    onnx.checker.check_model(canonical, full_check=True)
    assert stats["matmul_wrappers_removed"] == 1
    assert not canonical.functions
    assert all(not node.domain for node in canonical.graph.node)
    assert all(
        node.op_type not in {"If", "Loop", "SequenceAt", "SplitToSequence"}
        for node in canonical.graph.node
    )
    operator_types = {node.op_type for node in canonical.graph.node}
    assert {"Gather", "MatMul", "ReduceL2", "Slice"} <= operator_types

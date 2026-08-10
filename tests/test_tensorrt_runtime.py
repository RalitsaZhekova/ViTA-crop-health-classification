from __future__ import annotations

import json
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from cloud_detection.backend import OMNICLOUDMASK_ENSEMBLE_SHA256
from prithvi_payload import tensorrt_runtime
from prithvi_payload.inference import OPTIMIZED_BATCH_SIZE
from prithvi_payload.tensorrt_builder import (
    _canonicalize_cloud_onnx,
    _canonicalize_crop_onnx,
    _CloudPrecisionReference,
    _discover_cloud_scene_patch_sizes,
    _load_reusable_cloud_candidates,
    _load_reusable_cloud_records,
    _load_reusable_crop_record,
    _normalize_onnxscript_integer_attributes,
    _prune_unaccepted_build_artifacts,
    _target_digest,
    _trtexec_build_arguments,
)
from prithvi_payload.tensorrt_runtime import (
    MANIFEST_SCHEMA_VERSION,
    TensorRTArtifactError,
    _resolve_cuda_device,
    file_sha256,
    load_accepted_manifest,
    write_manifest_atomic,
)
from prithvi_shared import SELECTED_CHECKPOINT_SHA256


def _manifest() -> dict:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "accepted",
        "target": {"gpu": "test"},
        "models": {"crop": {}, "cloud": {}},
    }


class _ReferenceDtype(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen: torch.dtype | None = None
        self.offset = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.seen = image.dtype
        return image + self.offset


def test_cloud_precision_reference_promotes_operational_input() -> None:
    from omnicloudmask.cloud_mask import collect_models

    first = _ReferenceDtype()
    second = _ReferenceDtype()
    reference = _CloudPrecisionReference(
        [first, second],
        dtype=torch.float32,
    )
    collected = collect_models(
        custom_models=[reference],
        inference_device=torch.device("cpu"),
        inference_dtype=torch.float16,
        source="hugging_face",
    )

    output = collected[0](torch.ones((1, 3, 8, 8), dtype=torch.float16))

    assert first.seen == torch.float32
    assert second.seen == torch.float32
    assert first.offset.dtype == torch.float32
    assert second.offset.dtype == torch.float32
    assert output.dtype == torch.float32


def test_direct_tensorrt_resolves_cuda_alias_to_active_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    assert _resolve_cuda_device(torch.device("cuda")) == torch.device("cuda:0")
    assert _resolve_cuda_device(torch.device("cuda:0")) == torch.device("cuda:0")

    with pytest.raises(TensorRTArtifactError, match="active CUDA device"):
        _resolve_cuda_device(torch.device("cuda:1"))


def test_direct_tensorrt_discovers_every_reviewed_scene_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np
    import omnicloudmask.cloud_mask

    sizes = {
        "sentinel-a": 1000,
        "sentinel-b": 1000,
        "balkan-3370": 891,
        "balkan-3408": 869,
    }
    profiles = [
        {"input": scene_input, "marker": marker}
        for marker, scene_input in enumerate(sizes, start=1)
    ]
    backend = SimpleNamespace(
        patch_size=1000,
        patch_overlap=300,
        _prepare_input=lambda image: image,
    )
    runtime = SimpleNamespace(
        cloud=SimpleNamespace(backend=backend, config={"input": {}}),
        _scene_cloud_warmups=profiles,
    )
    monkeypatch.setattr(
        "prithvi_payload.tensorrt_builder._load_cloud_scene",
        lambda profile, input_config: np.full(
            (3, 4, 4),
            profile["marker"],
            dtype=np.float32,
        ),
    )
    monkeypatch.setattr(
        omnicloudmask.cloud_mask,
        "check_patch_size",
        lambda image, no_data_value, patch_size, patch_overlap: (
            patch_overlap,
            sizes[profiles[int(image[0, 0, 0]) - 1]["input"]],
        ),
    )

    assert _discover_cloud_scene_patch_sizes(runtime) == sizes


def test_direct_tensorrt_reuses_only_checksum_bound_matching_cloud_profiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "direct" / "accepted.json"
    plan_path = manifest_path.parent / "plans" / "cloud-869.plan"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_bytes(b"accepted-plan")
    target = {"gpu": "test"}
    record = {
        "plan": "plans/cloud-869.plan",
        "plan_sha256": file_sha256(plan_path),
        "onnx_sha256": "b" * 64,
        "precision": "fp16",
        "inputs": [
            {
                "name": "image",
                "dtype": "float16",
                "shape": [4, 3, 869, 869],
            }
        ],
        "outputs": [],
        "patch_size": 869,
        "logical_min_batch_size": 1,
        "logical_max_batch_size": 4,
    }
    write_manifest_atomic(
        manifest_path,
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "status": "accepted",
            "target": target,
            "models": {
                "crop": {},
                "cloud": {
                    "source_sha256": OMNICLOUDMASK_ENSEMBLE_SHA256,
                    "precision": "fp16",
                    "plans": [record],
                },
            },
        },
    )
    monkeypatch.setattr(tensorrt_runtime, "current_target_signature", lambda: target)

    reusable = _load_reusable_cloud_records(
        manifest_path,
        target=target,
        source_precision="fp16",
        profiles=[(869, 4, "fp16"), (891, 4, "fp16"), (1000, 1, "fp32")],
    )

    assert reusable == {869: record}


def test_direct_tensorrt_reuses_checksum_bound_matching_crop_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "direct" / "accepted.json"
    plan_path = manifest_path.parent / "plans" / "crop.plan"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_bytes(b"accepted-crop-plan")
    target = {"gpu": "test"}
    record = {
        "plan": "plans/crop.plan",
        "plan_sha256": file_sha256(plan_path),
        "onnx_sha256": "c" * 64,
        "precision": "mixed-fp16",
        "source_sha256": SELECTED_CHECKPOINT_SHA256,
        "tf32": False,
        "inputs": [
            {
                "name": "image",
                "dtype": "float32",
                "shape": [OPTIMIZED_BATCH_SIZE, 4, 1, 224, 224],
            },
            {
                "name": "temporal_coords",
                "dtype": "float32",
                "shape": [OPTIMIZED_BATCH_SIZE, 1, 2],
            },
            {
                "name": "location_coords",
                "dtype": "float32",
                "shape": [OPTIMIZED_BATCH_SIZE, 2],
            },
        ],
        "outputs": [],
    }
    write_manifest_atomic(
        manifest_path,
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "status": "accepted",
            "target": target,
            "models": {"crop": record, "cloud": {}},
        },
    )
    monkeypatch.setattr(tensorrt_runtime, "current_target_signature", lambda: target)

    reusable = _load_reusable_crop_record(
        manifest_path,
        target=target,
        precision="mixed-fp16",
    )

    assert reusable == record


def test_direct_tensorrt_revalidates_matching_inert_cloud_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "direct" / "accepted.json"
    onnx_path = (
        manifest_path.parent
        / "onnx"
        / f"cloud.{OMNICLOUDMASK_ENSEMBLE_SHA256[:12]}.b1.1000.fp16.onnx"
    )
    onnx_path.parent.mkdir(parents=True)
    onnx_path.write_bytes(b"canonical-onnx")
    onnx_digest = file_sha256(onnx_path)
    target = {"gpu": "test"}
    plan_path = manifest_path.parent / "plans" / (
        f"cloud-1000.{onnx_digest[:12]}.{_target_digest(target)[:12]}.plan"
    )
    plan_path.parent.mkdir()
    plan_path.write_bytes(b"candidate-plan")
    graph = {
        "path": onnx_path,
        "batch_size": 1,
        "patch_size": 1000,
        "profile": None,
    }
    build_key = {
        "onnx_sha256": onnx_digest,
        "target": target,
        "precision": "fp16",
        "arguments": _trtexec_build_arguments(
            graph,
            name="cloud-1000",
            precision="fp16",
            manifest_path=manifest_path,
        ),
    }
    plan_path.with_suffix(".build.json").write_text(
        json.dumps(
            {"build_key": build_key, "plan_sha256": file_sha256(plan_path)}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "prithvi_payload.tensorrt_builder._validate_onnx",
        lambda *args, **kwargs: (
            [
                {
                    "name": "image",
                    "dtype": "float16",
                    "shape": [1, 3, 1000, 1000],
                }
            ],
            [{"name": "logits", "dtype": "float16", "shape": [1, 4, 1000, 1000]}],
        ),
    )

    recovered = _load_reusable_cloud_candidates(
        manifest_path,
        target=target,
        profiles=[(1000, 1, "fp16")],
    )

    assert recovered[1000]["plan_sha256"] == file_sha256(plan_path)
    assert recovered[1000]["inputs"][0]["shape"] == [1, 3, 1000, 1000]


def test_direct_tensorrt_prunes_only_unaccepted_build_artifacts(tmp_path: Path) -> None:
    manifest_path = tmp_path / "direct" / "accepted.json"
    plans_root = manifest_path.parent / "plans"
    onnx_root = manifest_path.parent / "onnx"
    plans_root.mkdir(parents=True)
    onnx_root.mkdir()
    accepted_crop = plans_root / "crop.plan"
    accepted_cloud = plans_root / "cloud-1000.plan"
    obsolete = plans_root / "cloud-700.plan"
    for path in (accepted_crop, accepted_cloud, obsolete):
        path.write_bytes(path.name.encode())
        path.with_suffix(".build.json").write_text("{}", encoding="utf-8")
    (onnx_root / "candidate.onnx").write_bytes(b"intermediate")
    manifest = {
        "models": {
            "crop": {"plan": "plans/crop.plan"},
            "cloud": {"plans": [{"plan": "plans/cloud-1000.plan"}]},
        }
    }

    _prune_unaccepted_build_artifacts(manifest_path, manifest)

    assert accepted_crop.is_file()
    assert accepted_cloud.is_file()
    assert accepted_crop.with_suffix(".build.json").is_file()
    assert accepted_cloud.with_suffix(".build.json").is_file()
    assert not obsolete.exists()
    assert not obsolete.with_suffix(".build.json").exists()
    assert not list(onnx_root.glob("*.onnx"))


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

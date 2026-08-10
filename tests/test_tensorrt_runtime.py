from __future__ import annotations

import json
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace

import pytest
from prithvi_payload import tensorrt_runtime
from prithvi_payload.tensorrt_builder import (
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

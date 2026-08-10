from __future__ import annotations

import json
from pathlib import Path

import pytest
from prithvi_payload import tensorrt_runtime
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

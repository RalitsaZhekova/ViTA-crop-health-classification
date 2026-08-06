from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from prithvi_payload.runtime_config import environment_flag
from prithvi_shared.files import sha256_file


def test_shared_file_digest_preserves_sha256_contract(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    content = b"ViTA\x00checksum\n" * 1000
    path.write_bytes(content)

    assert sha256_file(path, chunk_size=17) == hashlib.sha256(content).hexdigest()


def test_payload_environment_flag_is_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VITA_TEST_RUNTIME_FLAG", raising=False)
    assert environment_flag("VITA_TEST_RUNTIME_FLAG", True) is True
    monkeypatch.setenv("VITA_TEST_RUNTIME_FLAG", "off")
    assert environment_flag("VITA_TEST_RUNTIME_FLAG", True) is False
    monkeypatch.setenv("VITA_TEST_RUNTIME_FLAG", "maybe")
    with pytest.raises(RuntimeError, match="must be a boolean value"):
        environment_flag("VITA_TEST_RUNTIME_FLAG", True)

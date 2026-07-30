from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from vita_integration.ground_client import GroundClient, GroundClientError


class _Response:
    def __init__(self, status_code: int, body: Any) -> None:
        self.status_code = status_code
        self._body = body

    def json(self) -> Any:
        return self._body


class _Session:
    def __init__(self, response: _Response) -> None:
        self.response = response
        self.url: str | None = None
        self.uploads: dict[str, tuple[str, bytes, str]] = {}

    def post(self, url: str, *, files: dict[str, tuple[str, Any, str]], timeout: float):
        assert timeout == 30.0
        self.url = url
        self.uploads = {
            field: (filename, stream.read(), media_type)
            for field, (filename, stream, media_type) in files.items()
        }
        return self.response


def _bundle(root: Path) -> Path:
    root.mkdir()
    (root / "scene.json").write_bytes(b"manifest")
    (root / "scene.webp").write_bytes(b"preview")
    (root / "condition.png").write_bytes(b"condition")
    return root


def test_ground_client_uploads_exact_three_file_bundle(tmp_path: Path) -> None:
    session = _Session(_Response(201, {"created": True, "scene": {"scene_id": "scene-1"}}))
    bundle = _bundle(tmp_path / "bundle")
    result = GroundClient("http://127.0.0.1:8000/", session=session).ingest_bundle(bundle)

    assert result["created"] is True
    assert session.url == "http://127.0.0.1:8000/api/v1/scenes"
    assert session.uploads == {
        "scene_json": ("scene.json", b"manifest", "application/json"),
        "scene_webp": ("scene.webp", b"preview", "image/webp"),
        "condition_png": ("condition.png", b"condition", "image/png"),
    }


def test_ground_client_rejects_extra_file(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle")
    (bundle / "debug.tif").write_bytes(b"not routine downlink")
    client = GroundClient(
        "http://127.0.0.1:8000", session=_Session(_Response(201, {"scene": {}}))
    )
    with pytest.raises(GroundClientError, match="exactly"):
        client.ingest_bundle(bundle)


def test_ground_client_sanitizes_server_errors(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "bundle")
    client = GroundClient(
        "http://127.0.0.1:8000",
        session=_Session(_Response(422, {"detail": "provider secret"})),
    )
    with pytest.raises(GroundClientError, match="HTTP 422") as raised:
        client.ingest_bundle(bundle)
    assert "provider secret" not in str(raised.value)

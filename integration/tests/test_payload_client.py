from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
from vita_integration.payload_client import (
    ROUTINE_ARTIFACTS,
    PayloadClient,
    PayloadClientError,
)


class _Response:
    status_code = 200

    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def iter_content(self, chunk_size: int) -> list[bytes]:
        del chunk_size
        return [self.body]


class _Session:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.urls: list[str] = []

    def get(self, url: str, **_: Any) -> _Response:
        self.urls.append(url)
        return _Response(self.files[url.rsplit("/", 1)[-1]])


def _checksums(files: dict[str, bytes]) -> dict[str, str]:
    return {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}


def test_payload_client_downloads_exactly_three_and_verifies_each_checksum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = {filename: f"contents:{filename}".encode() for filename in ROUTINE_ARTIFACTS}
    session = _Session(files)
    monkeypatch.setattr("vita_integration.payload_client.validate_bundle", lambda _: None)
    client = PayloadClient("http://127.0.0.1:8081", session=session)  # type: ignore[arg-type]

    bundle = client.download_bundle(
        "safe-job",
        tmp_path / "bundle",
        expected_checksums=_checksums(files),
    )

    assert {path.name for path in bundle.iterdir()} == set(ROUTINE_ARTIFACTS)
    assert [url.rsplit("/", 1)[-1] for url in session.urls] == list(ROUTINE_ARTIFACTS)


def test_payload_client_rejects_status_checksum_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = {filename: f"contents:{filename}".encode() for filename in ROUTINE_ARTIFACTS}
    checksums = _checksums(files)
    checksums["condition.png"] = "0" * 64
    monkeypatch.setattr("vita_integration.payload_client.validate_bundle", lambda _: None)
    client = PayloadClient(
        "http://127.0.0.1:8081",
        session=_Session(files),  # type: ignore[arg-type]
    )

    with pytest.raises(PayloadClientError, match="Checksum mismatch for condition.png"):
        client.download_bundle(
            "safe-job",
            tmp_path / "bundle",
            expected_checksums=checksums,
        )


def test_payload_client_rejects_missing_or_malformed_status_checksums(tmp_path: Path) -> None:
    client = PayloadClient(
        "http://127.0.0.1:8081",
        session=_Session({}),  # type: ignore[arg-type]
    )
    with pytest.raises(PayloadClientError, match="missing"):
        client.download_bundle(
            "safe-job",
            tmp_path / "missing",
            expected_checksums={"scene.json": "a" * 64},
        )
    with pytest.raises(PayloadClientError, match="invalid"):
        client.download_bundle(
            "safe-job",
            tmp_path / "invalid",
            expected_checksums={filename: "not-a-checksum" for filename in ROUTINE_ARTIFACTS},
        )


@pytest.mark.parametrize(
    "url",
    [
        "file:///data/jobs",
        "http://user:secret@127.0.0.1:8081",
        "http://127.0.0.1:8081?access_token=secret",
    ],
)
def test_payload_client_rejects_unsafe_service_urls(url: str) -> None:
    with pytest.raises(ValueError):
        PayloadClient(url)

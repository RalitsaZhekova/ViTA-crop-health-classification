"""Ground-side client for the fixed payload job API."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests
from prithvi_ground.catalog import validate_bundle
from prithvi_shared import PayloadAcquisitionCommand

ROUTINE_ARTIFACTS = ("scene.json", "scene.webp", "condition.png")
MAX_ARTIFACT_BYTES = 25 * 1024 * 1024


class PayloadClientError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PayloadClient:
    def __init__(
        self,
        base_url: str,
        *,
        session: requests.Session | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("payload_url must be an HTTP(S) service URL")
        if parsed.query or parsed.fragment or parsed.username or parsed.password:
            raise ValueError("payload_url must not contain credentials, query or fragment")
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout_seconds = timeout_seconds

    def _json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self.session.request(
                method,
                f"{self.base_url}{path}",
                timeout=self.timeout_seconds,
                **kwargs,
            )
        except requests.RequestException as error:
            raise PayloadClientError("Payload service request failed") from error
        if response.status_code >= 400:
            raise PayloadClientError(f"Payload service returned HTTP {response.status_code}")
        try:
            value = response.json()
        except (ValueError, requests.RequestException) as error:
            raise PayloadClientError("Payload service returned invalid JSON") from error
        if not isinstance(value, dict):
            raise PayloadClientError("Payload service returned an invalid response object")
        return value

    def health(self) -> dict[str, Any]:
        return self._json("GET", "/health")

    def submit(self, command: PayloadAcquisitionCommand) -> dict[str, Any]:
        return self._json("POST", "/v1/jobs", json=command.model_dump(mode="json"))

    def status(self, job_id: str) -> dict[str, Any]:
        return self._json("GET", f"/v1/jobs/{job_id}")

    def download_bundle(
        self,
        job_id: str,
        destination: Path,
        *,
        expected_checksums: dict[str, str],
    ) -> Path:
        if set(expected_checksums) != set(ROUTINE_ARTIFACTS):
            raise PayloadClientError("Payload status is missing routine artifact checksums")
        if any(
            not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum)
            for checksum in expected_checksums.values()
        ):
            raise PayloadClientError("Payload status contains an invalid artifact checksum")
        destination.mkdir(parents=True, exist_ok=False)
        for filename in ROUTINE_ARTIFACTS:
            path = destination / filename
            received = 0
            try:
                with self.session.get(
                    f"{self.base_url}/v1/jobs/{job_id}/artifacts/{filename}",
                    stream=True,
                    timeout=self.timeout_seconds,
                ) as response:
                    if response.status_code != 200:
                        raise PayloadClientError(
                            f"Payload artifact download returned HTTP {response.status_code}"
                        )
                    with path.open("wb") as output:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if not chunk:
                                continue
                            received += len(chunk)
                            if received > MAX_ARTIFACT_BYTES:
                                raise PayloadClientError("Payload artifact exceeded the byte limit")
                            output.write(chunk)
            except requests.RequestException as error:
                raise PayloadClientError("Payload artifact download failed") from error
            if _sha256(path) != expected_checksums[filename]:
                raise PayloadClientError(f"Checksum mismatch for {filename}")
        if {path.name for path in destination.iterdir()} != set(ROUTINE_ARTIFACTS):
            raise PayloadClientError("Downloaded bundle contains unexpected files")
        validate_bundle(destination)
        return destination

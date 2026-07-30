"""Ground-side upload client for the canonical verified-bundle API."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

from vita_integration.payload_client import ROUTINE_ARTIFACTS


class GroundClientError(RuntimeError):
    """Raised when the verified bundle cannot be ingested by the ground API."""


class GroundClient:
    def __init__(
        self,
        base_url: str,
        *,
        session: requests.Session | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("dashboard_url must be an HTTP(S) service URL")
        if parsed.query or parsed.fragment or parsed.username or parsed.password:
            raise ValueError("dashboard_url must not contain credentials, query or fragment")
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout_seconds = timeout_seconds

    def ingest_bundle(self, bundle: Path) -> dict[str, Any]:
        entries = {path.name for path in bundle.iterdir()}
        if entries != set(ROUTINE_ARTIFACTS):
            raise GroundClientError("Ground upload requires exactly the routine artifacts")
        try:
            with ExitStack() as stack:
                files = {
                    "scene_json": (
                        "scene.json",
                        stack.enter_context((bundle / "scene.json").open("rb")),
                        "application/json",
                    ),
                    "scene_webp": (
                        "scene.webp",
                        stack.enter_context((bundle / "scene.webp").open("rb")),
                        "image/webp",
                    ),
                    "condition_png": (
                        "condition.png",
                        stack.enter_context((bundle / "condition.png").open("rb")),
                        "image/png",
                    ),
                }
                response = self.session.post(
                    f"{self.base_url}/api/v1/scenes",
                    files=files,
                    timeout=self.timeout_seconds,
                )
        except (OSError, requests.RequestException) as error:
            raise GroundClientError("Ground ingestion request failed") from error
        if response.status_code not in {200, 201}:
            raise GroundClientError(f"Ground service returned HTTP {response.status_code}")
        try:
            value = response.json()
        except (ValueError, requests.RequestException) as error:
            raise GroundClientError("Ground service returned invalid JSON") from error
        if not isinstance(value, dict) or not isinstance(value.get("scene"), dict):
            raise GroundClientError("Ground service returned an invalid ingestion response")
        return value

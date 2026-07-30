"""Safe acquisition failures suitable for persistent status records."""

from __future__ import annotations

from typing import Any


class AcquisitionError(RuntimeError):
    """An acquisition error that never embeds provider URLs or credentials."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message
        self.details = details or {}

    def safe_record(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.safe_message,
            "details": self.details,
        }

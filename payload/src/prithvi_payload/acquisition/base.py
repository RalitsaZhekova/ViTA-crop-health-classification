"""Provider protocol for payload acquisition."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from prithvi_shared import PayloadAcquisitionCommand

from prithvi_payload.acquisition.models import AcquiredScene


class AcquisitionProvider(Protocol):
    def acquire_candidates(
        self,
        command: PayloadAcquisitionCommand,
        destination_directory: Path,
    ) -> list[AcquiredScene]:
        """Query, order, download and validate bounded candidate scenes."""

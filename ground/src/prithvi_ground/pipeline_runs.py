"""Ground-side launcher for the existing accelerated ViTA pipeline entry point."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
SAFE_RELATIVE_PATH = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,511}$")
PAYLOAD_TOTAL = re.compile(r"PAYLOAD TOTAL\s+([0-9]+(?:[.,][0-9]+)?)\s*s", re.IGNORECASE)


class PipelineLaunchError(ValueError):
    """Raised when a requested web launch is unsafe or cannot be started."""


@dataclass(frozen=True)
class PipelineLaunch:
    sensor: str
    region_id: str
    input_path: str | None = None
    image: str | None = None

    @classmethod
    def from_payload(cls, payload: Any) -> PipelineLaunch:
        if not isinstance(payload, dict):
            raise PipelineLaunchError("The analysis request must be a JSON object.")
        if set(payload) - {"sensor", "region_id", "input_path", "image"}:
            raise PipelineLaunchError("The analysis request contains unsupported fields.")

        sensor = payload.get("sensor")
        if sensor not in {"sentinel-2", "balkan-1"}:
            raise PipelineLaunchError("Choose Sentinel-2 or Balkan-1.")
        region_id = payload.get("region_id")
        if not isinstance(region_id, str) or not SAFE_ID.fullmatch(region_id):
            raise PipelineLaunchError(
                "Region ID may contain letters, numbers, dots, underscores and dashes."
            )

        input_path = _optional_relative_path(payload.get("input_path"), name="Input path")
        image = _optional_relative_path(payload.get("image"), name="Image", filename_only=True)
        if sensor == "balkan-1" and image:
            raise PipelineLaunchError("A separate image file is only used with Sentinel-2 folders.")
        return cls(sensor=sensor, region_id=region_id, input_path=input_path, image=image)


def _optional_relative_path(
    value: Any,
    *,
    name: str,
    filename_only: bool = False,
) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str) or not SAFE_RELATIVE_PATH.fullmatch(value):
        raise PipelineLaunchError(
            f"{name} must be a safe path relative to the payload data folder."
        )
    parts = re.split(r"[/\\]", value)
    if any(part in {"", ".", ".."} for part in parts):
        raise PipelineLaunchError(f"{name} contains an unsupported path segment.")
    if filename_only and len(parts) != 1:
        raise PipelineLaunchError("Image must be a filename inside the selected Sentinel folder.")
    return value.replace("\\", "/")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PipelineRunManager:
    """Run one pipeline job at a time without changing its execution contract."""

    def __init__(self, repository_root: str | Path):
        self.repository_root = Path(repository_root).resolve()
        self.script_path = self.repository_root / "vita.ps1"
        self._lock = threading.Lock()
        self._current: dict[str, Any] | None = None

    def capability(self) -> dict[str, Any]:
        disabled = os.environ.get("VITA_WEB_PIPELINE_ENABLED", "1").strip().lower()
        if disabled in {"0", "false", "no", "off"}:
            return {
                "available": False,
                "reason": "Web launches are disabled for this ground service.",
            }
        if not self.script_path.is_file():
            return {
                "available": False,
                "reason": "Pipeline launch is unavailable in this dashboard environment.",
            }
        if self._powershell_executable() is None:
            return {
                "available": False,
                "reason": "PowerShell is unavailable in this dashboard environment.",
            }
        return {
            "available": True,
            "reason": None,
            "sensors": ["sentinel-2", "balkan-1"],
        }

    def current(self) -> dict[str, Any] | None:
        with self._lock:
            return None if self._current is None else dict(self._current)

    def start(self, launch: PipelineLaunch) -> dict[str, Any]:
        capability = self.capability()
        if not capability["available"]:
            raise PipelineLaunchError(str(capability["reason"]))

        with self._lock:
            if self._current and self._current["status"] in {"queued", "running"}:
                raise PipelineLaunchError("An analysis is already running. Please let it finish.")
            prefix = "sentinel" if launch.sensor == "sentinel-2" else "balkan"
            run_id = (
                f"web-{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
                f"{uuid.uuid4().hex[:8]}"
            )
            self._current = {
                "run_id": run_id,
                "scene_id": run_id,
                "sensor": launch.sensor,
                "region_id": launch.region_id,
                "status": "queued",
                "created_at": _utc_now(),
                "started_at": None,
                "completed_at": None,
                "elapsed_seconds": None,
                "payload_seconds": None,
                "message": "Preparing the analysis…",
            }
            thread = threading.Thread(
                target=self._execute,
                args=(run_id, launch),
                name=f"vita-web-{run_id}",
                daemon=True,
            )
            thread.start()
            return dict(self._current)

    def _powershell_executable(self) -> str | None:
        configured = os.environ.get("VITA_WEB_POWERSHELL")
        if configured:
            return configured
        return shutil.which("pwsh") or shutil.which("powershell.exe") or shutil.which("powershell")

    def _command(self, run_id: str, launch: PipelineLaunch) -> list[str]:
        executable = self._powershell_executable()
        if executable is None:
            raise PipelineLaunchError("PowerShell is unavailable in this dashboard environment.")
        command = "sentinel" if launch.sensor == "sentinel-2" else "balkan"
        arguments = [
            executable,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(self.script_path),
            command,
            "-RegionId",
            launch.region_id,
            "-JobId",
            run_id,
        ]
        if launch.input_path:
            arguments.extend(["-InputPath", launch.input_path])
        if launch.image:
            arguments.extend(["-Image", launch.image])
        return arguments

    def _execute(self, run_id: str, launch: PipelineLaunch) -> None:
        started = time.perf_counter()
        self._update(
            run_id,
            status="running",
            started_at=_utc_now(),
            message="The payload is analyzing the observation…",
        )
        try:
            completed = subprocess.run(
                self._command(run_id, launch),
                cwd=self.repository_root,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=2100,
                check=False,
            )
            elapsed = time.perf_counter() - started
            if completed.returncode != 0:
                detail = _last_output_line(completed.stderr) or _last_output_line(completed.stdout)
                message = "Analysis could not be completed."
                if detail:
                    message = f"{message} {detail}"
                self._update(
                    run_id,
                    status="failed",
                    completed_at=_utc_now(),
                    elapsed_seconds=round(elapsed, 3),
                    message=message[:320],
                )
                return

            match = PAYLOAD_TOTAL.search(completed.stdout)
            payload_seconds = None
            if match:
                payload_seconds = round(float(match.group(1).replace(",", ".")), 4)
            self._update(
                run_id,
                status="succeeded",
                completed_at=_utc_now(),
                elapsed_seconds=round(elapsed, 3),
                payload_seconds=payload_seconds,
                message="Analysis complete. The new observation is ready.",
            )
        except subprocess.TimeoutExpired:
            self._update(
                run_id,
                status="failed",
                completed_at=_utc_now(),
                elapsed_seconds=round(time.perf_counter() - started, 3),
                message="Analysis timed out before a result was returned.",
            )
        except Exception:
            self._update(
                run_id,
                status="failed",
                completed_at=_utc_now(),
                elapsed_seconds=round(time.perf_counter() - started, 3),
                message="Analysis could not be started in this dashboard environment.",
            )

    def _update(self, run_id: str, **changes: Any) -> None:
        with self._lock:
            if self._current and self._current["run_id"] == run_id:
                self._current.update(changes)


def _last_output_line(value: str | None) -> str | None:
    if not value:
        return None
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if not lines:
        return None
    line = lines[-1]
    line = re.sub(r"\s+", " ", line)
    return line[:220]

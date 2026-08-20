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
from datetime import date, datetime, timezone
from math import cos, isfinite, radians
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
    bbox_wgs84: tuple[float, float, float, float] | None = None
    start_date: date | None = None
    end_date: date | None = None

    @classmethod
    def from_payload(cls, payload: Any) -> PipelineLaunch:
        if not isinstance(payload, dict):
            raise PipelineLaunchError("The analysis request must be a JSON object.")
        if set(payload) - {
            "sensor",
            "region_id",
            "input_path",
            "image",
            "bbox_wgs84",
            "start_date",
            "end_date",
        }:
            raise PipelineLaunchError("The analysis request contains unsupported fields.")

        sensor = payload.get("sensor")
        if sensor not in {
            "sentinel-2",
            "sentinel-2-live",
            "balkan-1",
            "balkan-1-raw",
        }:
            raise PipelineLaunchError(
                "Choose stored Sentinel-2, live Sentinel-2, Balkan-1, or Balkan-1 raw."
            )
        region_id = payload.get("region_id")
        if not isinstance(region_id, str) or not SAFE_ID.fullmatch(region_id):
            raise PipelineLaunchError(
                "Region ID may contain letters, numbers, dots, underscores and dashes."
            )

        input_path = _optional_relative_path(payload.get("input_path"), name="Input path")
        image = _optional_relative_path(payload.get("image"), name="Image", filename_only=True)
        if sensor != "sentinel-2" and image:
            raise PipelineLaunchError("A separate image file is only used with Sentinel-2 folders.")
        if sensor == "balkan-1-raw" and input_path and not SAFE_ID.fullmatch(input_path):
            raise PipelineLaunchError("Balkan-1 raw input must be a scene ID such as 3408.")
        bbox = _optional_bbox(payload.get("bbox_wgs84"))
        start_date = _optional_date(payload.get("start_date"), name="Start date")
        end_date = _optional_date(payload.get("end_date"), name="End date")
        live_values = (bbox, start_date, end_date)
        if sensor == "sentinel-2-live":
            if input_path or image:
                raise PipelineLaunchError(
                    "Live Sentinel analysis uses the selected area, not a payload file path."
                )
            if any(value is None for value in live_values):
                raise PipelineLaunchError(
                    "Live Sentinel analysis requires an area and start/end dates."
                )
            assert start_date is not None and end_date is not None
            if start_date > end_date:
                raise PipelineLaunchError("Start date must not follow end date.")
            if (end_date - start_date).days + 1 > 92:
                raise PipelineLaunchError("The live Sentinel search may span at most 92 days.")
        elif any(value is not None for value in live_values):
            raise PipelineLaunchError(
                "Area and date fields are only used by live Sentinel analysis."
            )
        return cls(
            sensor=sensor,
            region_id=region_id,
            input_path=input_path,
            image=image,
            bbox_wgs84=bbox,
            start_date=start_date,
            end_date=end_date,
        )


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


def _optional_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
    ):
        raise PipelineLaunchError("Selected area must contain west, south, east and north.")
    west, south, east, north = (float(item) for item in value)
    if not all(isfinite(item) for item in (west, south, east, north)):
        raise PipelineLaunchError("Selected area coordinates must be finite.")
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise PipelineLaunchError("Selected area is outside valid longitude/latitude bounds.")
    latitude_km = (north - south) * 110.574
    longitude_km = (east - west) * 111.320 * cos(radians((south + north) / 2.0))
    if min(latitude_km, longitude_km) < 0.6:
        raise PipelineLaunchError("Select an area at least 0.6 km wide and high.")
    if max(latitude_km, longitude_km) > 10.2:
        raise PipelineLaunchError("Select an area no more than 10 km wide and high.")
    return west, south, east, north


def _optional_date(value: Any, *, name: str) -> date | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise PipelineLaunchError(f"{name} must use YYYY-MM-DD.")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise PipelineLaunchError(f"{name} must use YYYY-MM-DD.") from error


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
        sensors = ["sentinel-2", "balkan-1"]
        raw_target = (
            os.environ.get("VITA_JETSON_SSH_TARGET", "").strip()
            or os.environ.get("VITA_RAW_SSH_TARGET", "").strip()
        )
        raw_available = bool(
            re.fullmatch(r"[A-Za-z0-9._-]+@[A-Za-z0-9._-]+", raw_target)
        )
        if raw_available:
            sensors.extend(("balkan-1-raw", "sentinel-2-live"))
        return {
            "available": True,
            "reason": None,
            "sensors": sensors,
            "raw": {
                "available": raw_available,
                "reason": (
                    None
                    if raw_available
                    else "Set VITA_JETSON_SSH_TARGET to enable warm Jetson analysis."
                ),
            },
            "earth_engine": {
                "available": raw_available,
                "reason": (
                    None
                    if raw_available
                    else "Set VITA_JETSON_SSH_TARGET to enable live Sentinel analysis."
                ),
                "maximum_search_days": 92,
                "maximum_area_km": 10,
            },
        }

    def current(self) -> dict[str, Any] | None:
        with self._lock:
            return None if self._current is None else dict(self._current)

    def start(self, launch: PipelineLaunch) -> dict[str, Any]:
        capability = self.capability()
        if not capability["available"]:
            raise PipelineLaunchError(str(capability["reason"]))
        if launch.sensor not in capability.get("sensors", []):
            raw = capability.get("raw", {})
            raise PipelineLaunchError(
                str(raw.get("reason") or "The selected sensor is unavailable.")
            )

        with self._lock:
            if self._current and self._current["status"] in {"queued", "running"}:
                raise PipelineLaunchError("An analysis is already running. Please let it finish.")
            prefix = {
                "sentinel-2": "sentinel",
                "sentinel-2-live": "live-sentinel",
                "balkan-1": "balkan",
                "balkan-1-raw": "raw",
            }[launch.sensor]
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
        command = {
            "sentinel-2": "sentinel",
            "sentinel-2-live": "earth-engine",
            "balkan-1": "balkan",
            "balkan-1-raw": "raw",
        }[launch.sensor]
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
        if launch.bbox_wgs84 is not None:
            arguments.extend(
                [
                    "-BboxWgs84",
                    ",".join(f"{value:.8f}" for value in launch.bbox_wgs84),
                ]
            )
        if launch.start_date is not None:
            arguments.extend(["-StartDate", launch.start_date.isoformat()])
        if launch.end_date is not None:
            arguments.extend(["-EndDate", launch.end_date.isoformat()])
        return arguments

    def _execute(self, run_id: str, launch: PipelineLaunch) -> None:
        started = time.perf_counter()
        self._update(
            run_id,
            status="running",
            started_at=_utc_now(),
            message=(
                "The payload is finding a clear Sentinel-2 scene for the selected area…"
                if launch.sensor == "sentinel-2-live"
                else "The payload is analyzing the observation…"
            ),
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

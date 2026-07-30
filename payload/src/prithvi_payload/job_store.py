"""Atomic, restart-safe payload job state storage."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prithvi_shared import PayloadAcquisitionCommand

JOB_STATES = frozenset(
    {
        "queued",
        "searching_candidates",
        "acquiring",
        "evaluating_cloud",
        "validating_input",
        "running_crop",
        "running_condition",
        "packaging",
        "completed",
        "rejected",
        "failed",
    }
)
TERMINAL_STATES = frozenset({"completed", "rejected", "failed"})
HISTORY_SCHEMA_VERSION = "1.0"
_EVENT_FIELDS = (
    "current_candidate_number",
    "maximum_candidate_attempts",
    "safe_candidate_scene_id",
    "safe_metadata_cloud_percentage",
    "safe_payload_measured_cloud_percentage",
    "selected_scene",
    "metadata_cloud_percentage",
    "payload_measured_cloud_percentage",
    "payload_measured_shadow_percentage",
    "payload_measured_unusable_percentage",
    "condition",
    "payload_status",
)


class JobExistsError(ValueError):
    pass


class JobNotFoundError(FileNotFoundError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class JobStore:
    """One directory per validated job with atomic JSON state transitions."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._lock:
            self._refresh_history_unlocked()

    def job_directory(self, job_id: str) -> Path:
        path = (self.root / job_id).resolve()
        if path.parent != self.root:
            raise ValueError("Unsafe job identifier")
        return path

    def _status_path(self, job_id: str) -> Path:
        return self.job_directory(job_id) / "status.json"

    def _command_path(self, job_id: str) -> Path:
        return self.job_directory(job_id) / "command.json"

    def _history_path(self) -> Path:
        return self.root / "history.json"

    @staticmethod
    def _event(status: dict[str, Any]) -> dict[str, Any]:
        event: dict[str, Any] = {
            "timestamp": status["updated_at"],
            "state": status["state"],
        }
        for field in _EVENT_FIELDS:
            if field in status:
                event[field] = status[field]
        attempts = status.get("candidate_attempts")
        if isinstance(attempts, list):
            event["candidate_attempt_count"] = len(attempts)
        error = status.get("error")
        if isinstance(error, dict) and isinstance(error.get("code"), str):
            event["error_code"] = error["code"]
        return event

    def _refresh_history_unlocked(self) -> dict[str, Any]:
        jobs: list[dict[str, Any]] = []
        for status_path in self.root.glob("*/status.json"):
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
                if not isinstance(status, dict):
                    continue
                if "request" not in status:
                    command_path = status_path.parent / "command.json"
                    command = PayloadAcquisitionCommand.model_validate_json(
                        command_path.read_text(encoding="utf-8")
                    )
                    status["request"] = command.source.model_dump(mode="json")
                jobs.append(status)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        jobs.sort(
            key=lambda job: (str(job.get("created_at", "")), str(job.get("job_id", ""))),
            reverse=True,
        )
        history = {
            "schema_version": HISTORY_SCHEMA_VERSION,
            "updated_at": _utc_now(),
            "job_count": len(jobs),
            "jobs": jobs,
        }
        _write_json_atomic(self._history_path(), history)
        return history

    def create(self, command: PayloadAcquisitionCommand) -> dict[str, Any]:
        with self._lock:
            directory = self.job_directory(command.job_id)
            if directory.exists():
                raise JobExistsError(command.job_id)
            directory.mkdir(parents=True)
            now = _utc_now()
            status = {
                "schema_version": "1.0",
                "job_id": command.job_id,
                "region_id": command.region_id,
                "state": "queued",
                "created_at": now,
                "updated_at": now,
                "current_candidate_number": 0,
                "maximum_candidate_attempts": 5,
                "safe_candidate_scene_id": None,
                "safe_metadata_cloud_percentage": None,
                "safe_payload_measured_cloud_percentage": None,
                "candidate_attempts": [],
                "request": command.source.model_dump(mode="json"),
                "error": None,
            }
            status["events"] = [self._event(status)]
            _write_json_atomic(
                self._command_path(command.job_id),
                command.model_dump(mode="json"),
            )
            _write_json_atomic(self._status_path(command.job_id), status)
            self._refresh_history_unlocked()
            return status

    def get(self, job_id: str) -> dict[str, Any]:
        path = self._status_path(job_id)
        if not path.is_file():
            raise JobNotFoundError(job_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def command(self, job_id: str) -> PayloadAcquisitionCommand:
        path = self._command_path(job_id)
        if not path.is_file():
            raise JobNotFoundError(job_id)
        return PayloadAcquisitionCommand.model_validate_json(path.read_text(encoding="utf-8"))

    def update(self, job_id: str, state: str, **fields: Any) -> dict[str, Any]:
        if state not in JOB_STATES:
            raise ValueError(f"Unsupported payload job state: {state}")
        with self._lock:
            status = self.get(job_id)
            status.update(fields)
            status["state"] = state
            status["updated_at"] = _utc_now()
            events = status.setdefault("events", [])
            if not isinstance(events, list):
                events = []
                status["events"] = events
            events.append(self._event(status))
            _write_json_atomic(self._status_path(job_id), status)
            self._refresh_history_unlocked()
            return status

    def history(self) -> dict[str, Any]:
        with self._lock:
            return self._refresh_history_unlocked()

    def write_result(self, job_id: str, value: dict[str, Any]) -> None:
        with self._lock:
            _write_json_atomic(self.job_directory(job_id) / "job_result.json", value)

    def result(self, job_id: str) -> dict[str, Any]:
        path = self.job_directory(job_id) / "job_result.json"
        if not path.is_file():
            raise JobNotFoundError(job_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def artifact_path(self, job_id: str, filename: str) -> Path:
        if filename not in {"scene.json", "scene.webp", "condition.png"}:
            raise ValueError("Unsupported payload artifact")
        status = self.get(job_id)
        if status.get("state") != "completed":
            raise JobNotFoundError(job_id)
        relative_directory = self.result(job_id).get("artifact_directory")
        if not isinstance(relative_directory, str):
            raise JobNotFoundError(job_id)
        directory = (self.job_directory(job_id) / relative_directory).resolve()
        if self.job_directory(job_id) not in directory.parents:
            raise RuntimeError("Stored artifact directory is unsafe")
        path = directory / filename
        if not path.is_file():
            raise JobNotFoundError(filename)
        return path

    def recover_pending(self) -> list[str]:
        """Return interrupted jobs to queued state while retaining completed jobs."""
        recovered: list[str] = []
        with self._lock:
            for status_path in sorted(self.root.glob("*/status.json")):
                try:
                    status = json.loads(status_path.read_text(encoding="utf-8"))
                    job_id = status["job_id"]
                    if status.get("state") not in TERMINAL_STATES:
                        self.update(
                            job_id,
                            "queued",
                            error=None,
                            recovery="requeued_after_payload_restart",
                        )
                        recovered.append(job_id)
                except (KeyError, OSError, ValueError, json.JSONDecodeError):
                    continue
        return recovered

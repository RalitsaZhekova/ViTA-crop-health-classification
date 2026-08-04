"""Persistent HTTP service for coordinate-driven payload jobs."""

from __future__ import annotations

import argparse
import os
import queue
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from prithvi_shared import PayloadAcquisitionCommand

from prithvi_payload.acquisition.errors import AcquisitionError
from prithvi_payload.job_store import (
    JobExistsError,
    JobNotFoundError,
    JobStore,
)
from prithvi_payload.runtime import PayloadRuntime

DEFAULT_JOB_ROOT = Path("/data/jobs")
DEFAULT_QUEUE_SIZE = 8


class PayloadJobService:
    def __init__(
        self,
        runtime: PayloadRuntime,
        store: JobStore,
        *,
        queue_size: int = DEFAULT_QUEUE_SIZE,
    ) -> None:
        self.runtime = runtime
        self.store = store
        self.queue: queue.Queue[str | None] = queue.Queue(maxsize=queue_size)
        self.worker: threading.Thread | None = None
        self.started = False

    def start(self) -> None:
        if self.started:
            return
        self.runtime.initialize()
        self.worker = threading.Thread(target=self._worker, name="payload-gpu-worker", daemon=True)
        self.worker.start()
        self.started = True
        for job_id in self.store.recover_pending():
            try:
                self.queue.put_nowait(job_id)
            except queue.Full:
                self.store.update(
                    job_id,
                    "failed",
                    error={"code": "PAYLOAD_QUEUE_FULL", "message": "Payload queue is full"},
                )

    def stop(self) -> None:
        if not self.started:
            return
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            return
        if self.worker is not None:
            self.worker.join(timeout=30)
        self.started = False

    def submit(self, command: PayloadAcquisitionCommand) -> dict[str, Any]:
        if self.queue.full():
            raise queue.Full
        status_record = self.store.create(command)
        try:
            self.queue.put_nowait(command.job_id)
        except queue.Full:
            self.store.update(
                command.job_id,
                "failed",
                error={"code": "PAYLOAD_QUEUE_FULL", "message": "Payload queue is full"},
            )
            raise
        return status_record

    def _worker(self) -> None:
        while True:
            job_id = self.queue.get()
            if job_id is None:
                self.queue.task_done()
                return
            try:
                command = self.store.command(job_id)

                def update(
                    state: str,
                    fields: dict[str, Any],
                    current_job_id: str = job_id,
                ) -> None:
                    self.store.update(current_job_id, state, **fields)

                result = self.runtime.process(
                    command,
                    self.store.job_directory(job_id),
                    update,
                )
                self.store.write_result(job_id, result)
                self.store.update(
                    job_id,
                    "completed",
                    selected_scene=result["selected_scene"],
                    metadata_cloud_percentage=result["metadata_cloud_percentage"],
                    payload_measured_cloud_percentage=result["payload_cloud_percentage"],
                    payload_measured_shadow_percentage=result["payload_shadow_percentage"],
                    payload_measured_unusable_percentage=result["payload_unusable_percentage"],
                    condition=result["condition"],
                    artifacts=result["artifacts"],
                    artifact_checksums=result["artifact_checksums"],
                    payload_status=result["payload_status"],
                    error=None,
                )
            except AcquisitionError as error:
                state = (
                    "rejected"
                    if error.code
                    in {
                        "EARTH_ENGINE_NO_SCENE",
                        "EARTH_ENGINE_NO_TARGET_CLOUD_SCENE",
                        "REGION_TOO_LARGE",
                        "PAYLOAD_NO_SCENE_IN_TARGET_CLOUD_RANGE",
                    }
                    else "failed"
                )
                fields: dict[str, Any] = {"error": error.safe_record()}
                if isinstance(error.details.get("candidate_attempts"), list):
                    fields["candidate_attempts"] = error.details["candidate_attempts"]
                self.store.update(job_id, state, **fields)
            except Exception:
                self.store.update(
                    job_id,
                    "failed",
                    error={
                        "code": "PAYLOAD_JOB_FAILED",
                        "message": "Payload job failed; inspect protected payload logs",
                    },
                )
            finally:
                self.queue.task_done()


def create_app(
    *,
    runtime: PayloadRuntime | None = None,
    store: JobStore | None = None,
    start_service: bool = True,
) -> FastAPI:
    runtime = runtime or PayloadRuntime()
    jobs_directory = os.environ.get(
        "VITA_JOBS_DIR",
        os.environ.get("VITA_JOB_ROOT", str(DEFAULT_JOB_ROOT)),
    )
    store = store or JobStore(jobs_directory)
    service = PayloadJobService(
        runtime,
        store,
        queue_size=int(os.environ.get("PAYLOAD_QUEUE_SIZE", DEFAULT_QUEUE_SIZE)),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if start_service:
            service.start()
        yield
        if start_service:
            service.stop()

    app = FastAPI(title="ViTA Payload Acquisition API", version="1.0", lifespan=lifespan)
    app.state.payload_service = service

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            **runtime.health(),
            "queue_depth": service.queue.qsize(),
            "queue_capacity": service.queue.maxsize,
        }

    @app.post("/v1/jobs", status_code=status.HTTP_202_ACCEPTED)
    def submit_job(command: PayloadAcquisitionCommand) -> dict[str, Any]:
        try:
            return service.submit(command)
        except JobExistsError as error:
            raise HTTPException(status_code=409, detail="job_id already exists") from error
        except queue.Full as error:
            raise HTTPException(status_code=503, detail="payload queue is full") from error

    @app.get("/v1/jobs")
    def list_jobs() -> dict[str, Any]:
        return store.history()

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        try:
            return store.get(job_id)
        except (JobNotFoundError, ValueError) as error:
            raise HTTPException(status_code=404, detail="job not found") from error

    def artifact(job_id: str, filename: str, media_type: str) -> FileResponse:
        try:
            path = store.artifact_path(job_id, filename)
        except (JobNotFoundError, ValueError) as error:
            raise HTTPException(status_code=404, detail="artifact not found") from error
        return FileResponse(path, media_type=media_type, filename=filename)

    @app.get("/v1/jobs/{job_id}/artifacts/scene.json")
    def scene_json(job_id: str) -> FileResponse:
        return artifact(job_id, "scene.json", "application/json")

    @app.get("/v1/jobs/{job_id}/artifacts/scene.webp")
    def scene_webp(job_id: str) -> FileResponse:
        return artifact(job_id, "scene.webp", "image/webp")

    @app.get("/v1/jobs/{job_id}/artifacts/condition.png")
    def condition_png(job_id: str) -> FileResponse:
        return artifact(job_id, "condition.png", "image/png")

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the persistent ViTA payload API.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

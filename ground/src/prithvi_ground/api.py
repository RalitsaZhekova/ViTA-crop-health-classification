"""Versioned HTTP API for verified ViTA crop-condition downlinks."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlsplit

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .catalog import SceneCatalog
from .pipeline_runs import PipelineLaunch, PipelineLaunchError, PipelineRunManager

API_VERSION = "1.1.0"


def _links(scene_id: str, region_id: str) -> dict[str, str]:
    scene_root = f"/api/v1/scenes/{scene_id}"
    region_root = f"/api/v1/regions/{region_id}"
    return {
        "self": scene_root,
        "manifest": f"{scene_root}/manifest",
        "preview": f"{scene_root}/preview",
        "condition_overlay": f"{scene_root}/condition-overlay",
        "region_latest": f"{region_root}/latest",
        "region_history": f"{region_root}/history",
    }


def _present_scene(scene: dict[str, Any]) -> dict[str, Any]:
    return {**scene, "links": _links(scene["scene_id"], scene["region_id"])}


def _not_found(kind: str, identifier: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"{kind} not found: {identifier}")


def _asset_response(
    request: Request,
    path: Path,
    *,
    sha256: str,
    media_type: str,
) -> Response:
    etag = f'"{sha256}"'
    headers = {
        "Cache-Control": "public, max-age=31536000, immutable",
        "ETag": etag,
        "X-Content-Type-Options": "nosniff",
    }
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    return FileResponse(path, media_type=media_type, headers=headers)


def create_app(
    store_root: str | Path | None = None,
    *,
    pipeline_manager: PipelineRunManager | None = None,
) -> FastAPI:
    """Create an API bound to one verified scene catalog."""
    resolved_store = Path(
        store_root or os.environ.get("VITA_GROUND_STORE", "ground/runtime")
    ).resolve()
    catalog = SceneCatalog(resolved_store)
    app = FastAPI(
        title="ViTA Crop Intelligence API",
        summary="Verified crop-condition observations from compact satellite downlinks",
        description=(
            "Serves exact payload-computed spectral measurements, spatial grid summaries, "
            "and visualization assets. Condition labels are screening priorities, not "
            "agronomic diagnoses."
        ),
        version=API_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.catalog = catalog
    app.state.store_root = resolved_store
    app.state.pipeline_runs = pipeline_manager or PipelineRunManager(
        Path(__file__).resolve().parents[3]
    )
    web_root = Path(__file__).with_name("web")
    app.mount("/static", StaticFiles(directory=web_root), name="static")

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'self'",
        )
        return response

    @app.get("/api/v1/health", tags=["system"])
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "vita-crop-ground",
            "api_version": API_VERSION,
            "catalog_schema_version": 1,
        }

    @app.get("/", include_in_schema=False)
    def web_application() -> FileResponse:
        return FileResponse(
            web_root / "index.html",
            media_type="text/html",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/api/v1/scenes", tags=["scenes"])
    def list_scenes(
        region_id: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> dict[str, Any]:
        items = [
            _present_scene(scene)
            for scene in catalog.list_scenes(region_id=region_id, limit=limit, offset=offset)
        ]
        return {"count": len(items), "limit": limit, "offset": offset, "items": items}

    @app.get("/api/v1/scenes/{scene_id}", tags=["scenes"])
    def get_scene(scene_id: str) -> dict[str, Any]:
        scene = catalog.get_scene(scene_id)
        if scene is None:
            raise _not_found("Scene", scene_id)
        return _present_scene(scene)

    @app.get("/api/v1/scenes/{scene_id}/manifest", tags=["scenes"])
    def get_manifest(scene_id: str) -> dict[str, Any]:
        try:
            manifest = catalog.load_manifest(scene_id)
        except FileNotFoundError as error:
            raise _not_found("Scene", scene_id) from error
        return {"scene": manifest, "links": _links(scene_id, manifest["region_id"])}

    @app.get("/api/v1/scenes/{scene_id}/preview", tags=["assets"])
    def get_preview(scene_id: str, request: Request) -> Response:
        try:
            manifest = catalog.load_manifest(scene_id)
            path = catalog.asset_path(scene_id, "scene.webp")
        except FileNotFoundError as error:
            raise _not_found("Scene", scene_id) from error
        asset = manifest["assets"]["rgb_preview"]
        return _asset_response(
            request,
            path,
            sha256=asset["sha256"],
            media_type="image/webp",
        )

    @app.get("/api/v1/scenes/{scene_id}/condition-overlay", tags=["assets"])
    def get_condition_overlay(scene_id: str, request: Request) -> Response:
        try:
            manifest = catalog.load_manifest(scene_id)
            path = catalog.asset_path(scene_id, "condition.png")
        except FileNotFoundError as error:
            raise _not_found("Scene", scene_id) from error
        asset = manifest["assets"]["condition_overlay"]
        return _asset_response(
            request,
            path,
            sha256=asset["sha256"],
            media_type="image/png",
        )

    @app.get("/api/v1/scenes/{scene_id}/cells/{cell_id}", tags=["scenes"])
    def get_cell(scene_id: str, cell_id: str) -> dict[str, Any]:
        try:
            manifest = catalog.load_manifest(scene_id)
        except FileNotFoundError as error:
            raise _not_found("Scene", scene_id) from error
        for cell in manifest["interaction_grid"]["cells"]:
            if cell.get("id") == cell_id:
                return {"scene_id": scene_id, "cell": cell}
        raise _not_found("Cell", cell_id)

    @app.get("/api/v1/regions", tags=["regions"])
    def list_regions() -> dict[str, Any]:
        items = catalog.list_regions()
        return {"count": len(items), "items": items}

    @app.get("/api/v1/regions/{region_id}/latest", tags=["regions"])
    def latest_for_region(region_id: str) -> dict[str, Any]:
        scene = catalog.latest_for_region(region_id)
        if scene is None:
            raise _not_found("Region", region_id)
        return _present_scene(scene)

    @app.get("/api/v1/regions/{region_id}/history", tags=["regions"])
    def history_for_region(
        region_id: str,
        limit: Annotated[int, Query(ge=1, le=200)] = 200,
    ) -> dict[str, Any]:
        items = [
            _present_scene(scene) for scene in catalog.history_for_region(region_id, limit=limit)
        ]
        if not items:
            raise _not_found("Region", region_id)
        return {"region_id": region_id, "count": len(items), "items": items}

    @app.get("/api/v1/pipeline-runs/capability", tags=["analysis"])
    def pipeline_capability(response: Response) -> dict[str, Any]:
        response.headers["Cache-Control"] = "no-store"
        return app.state.pipeline_runs.capability()

    @app.get("/api/v1/pipeline-runs/current", tags=["analysis"])
    def current_pipeline_run(response: Response) -> dict[str, Any]:
        response.headers["Cache-Control"] = "no-store"
        return {"run": app.state.pipeline_runs.current()}

    @app.post("/api/v1/pipeline-runs", tags=["analysis"], status_code=202)
    async def start_pipeline_run(request: Request, response: Response) -> dict[str, Any]:
        response.headers["Cache-Control"] = "no-store"
        origin = request.headers.get("origin")
        host = request.headers.get("host")
        if origin and host and urlsplit(origin).netloc.lower() != host.lower():
            raise HTTPException(
                status_code=403,
                detail="Cross-origin analysis launches are blocked.",
            )
        if "application/json" not in request.headers.get("content-type", "").lower():
            raise HTTPException(status_code=415, detail="Analysis launches require JSON.")
        try:
            payload = await request.json()
        except ValueError as error:
            raise HTTPException(
                status_code=400,
                detail="Analysis request is not valid JSON.",
            ) from error
        try:
            launch = PipelineLaunch.from_payload(payload)
        except PipelineLaunchError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        try:
            run = app.state.pipeline_runs.start(launch)
        except PipelineLaunchError as error:
            capability = app.state.pipeline_runs.capability()
            status_code = 409 if capability.get("available") else 503
            raise HTTPException(status_code=status_code, detail=str(error)) from error
        return {"run": run}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the ViTA ground API and web client.")
    parser.add_argument("--store", type=Path, default=Path("ground/runtime"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(args.store), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

"""Versioned HTTP API for verified ViTA crop-condition downlinks."""

from __future__ import annotations

import argparse
import hmac
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Any

import uvicorn
from fastapi import FastAPI, File, Header, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from .catalog import BundleValidationError, SceneCatalog, SceneConflictError

API_VERSION = "1.0.0"
DEFAULT_MAX_BUNDLE_BYTES = 25 * 1024 * 1024
UPLOAD_FILENAMES = {
    "scene_json": "scene.json",
    "scene_webp": "scene.webp",
    "condition_png": "condition.png",
}


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


def _require_upload_key(configured_key: str | None, provided_key: str | None) -> None:
    if configured_key is None:
        return
    if provided_key is None or not hmac.compare_digest(configured_key, provided_key):
        raise HTTPException(
            status_code=401,
            detail="A valid upload API key is required",
            headers={"WWW-Authenticate": "ApiKey"},
        )


def _save_uploads(
    destination: Path,
    uploads: dict[str, UploadFile],
    *,
    max_bundle_bytes: int,
) -> None:
    total_bytes = 0
    for field_name, expected_filename in UPLOAD_FILENAMES.items():
        upload = uploads[field_name]
        if upload.filename != expected_filename:
            raise HTTPException(
                status_code=422,
                detail=f"{field_name} must be uploaded as {expected_filename}",
            )
        output_path = destination / expected_filename
        with output_path.open("wb") as output:
            while chunk := upload.file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > max_bundle_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Bundle exceeds the {max_bundle_bytes}-byte upload limit",
                    )
                output.write(chunk)


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
    upload_api_key: str | None = None,
    max_bundle_bytes: int = DEFAULT_MAX_BUNDLE_BYTES,
) -> FastAPI:
    """Create an API bound to one verified scene catalog."""
    if max_bundle_bytes <= 0:
        raise ValueError("max_bundle_bytes must be positive")
    resolved_store = Path(
        store_root or os.environ.get("VITA_GROUND_STORE", "ground/runtime")
    ).resolve()
    configured_key = (
        upload_api_key if upload_api_key is not None else os.environ.get("VITA_UPLOAD_API_KEY")
    )
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

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "DENY")
        return response

    @app.exception_handler(BundleValidationError)
    async def invalid_bundle_handler(_request: Request, error: BundleValidationError):
        return JSONResponse(status_code=422, content={"detail": str(error)})

    @app.exception_handler(SceneConflictError)
    async def scene_conflict_handler(_request: Request, error: SceneConflictError):
        return JSONResponse(status_code=409, content={"detail": str(error)})

    @app.get("/api/v1/health", tags=["system"])
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "vita-crop-ground",
            "api_version": API_VERSION,
            "catalog_schema_version": 1,
        }

    @app.post("/api/v1/scenes", tags=["scenes"], status_code=201)
    def ingest_scene(
        response: Response,
        scene_json: Annotated[UploadFile, File(description="Verified scene.json manifest")],
        scene_webp: Annotated[UploadFile, File(description="RGB scene.webp preview")],
        condition_png: Annotated[
            UploadFile, File(description="RGBA condition.png visualization overlay")
        ],
        x_api_key: Annotated[str | None, Header()] = None,
    ) -> dict[str, Any]:
        _require_upload_key(configured_key, x_api_key)
        uploads = {
            "scene_json": scene_json,
            "scene_webp": scene_webp,
            "condition_png": condition_png,
        }
        with TemporaryDirectory(prefix="vita-ground-upload-") as temporary:
            _save_uploads(Path(temporary), uploads, max_bundle_bytes=max_bundle_bytes)
            scene, created = catalog.ingest(temporary)
        response.status_code = 201 if created else 200
        return {"created": created, "scene": _present_scene(scene)}

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

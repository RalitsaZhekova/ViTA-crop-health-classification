# ViTA Crop Intelligence API

The implemented FastAPI service exposes versioned routes on top of the verified
scene catalog:

```text
GET  /api/v1/health
POST /api/v1/scenes
GET  /api/v1/scenes
GET  /api/v1/scenes/{scene_id}
GET  /api/v1/scenes/{scene_id}/manifest
GET  /api/v1/scenes/{scene_id}/preview
GET  /api/v1/scenes/{scene_id}/condition-overlay
GET  /api/v1/scenes/{scene_id}/cells/{cell_id}
GET  /api/v1/regions
GET  /api/v1/regions/{region_id}/latest
GET  /api/v1/regions/{region_id}/history
```

Start the service from the repository root:

```powershell
$env:PYTHONPATH = "shared/src;ground/src"
.venv\Scripts\python.exe -m prithvi_ground.api --store ground/runtime
```

Open `http://127.0.0.1:8000/docs` for the generated OpenAPI interface. Set
`VITA_UPLOAD_API_KEY` to require the `X-API-Key` header on uploads. The upload is
limited to 25 MiB by default, accepts exactly `scene.json`, `scene.webp` and
`condition.png`, and revalidates the entire bundle before atomic ingestion.

Scene responses follow `shared/schemas/downlink_bundle.schema.json`. Exact
client measurements come from the JSON record and its interaction grid; WebP
and PNG assets are presentation layers and are never decoded as scientific
values. The PNG contains only the crop-condition heat map; cloud and quality
facts remain in JSON. Immutable asset responses include checksums as ETags.

Optional API-key protection is appropriate for the single-operator MVP. Full
authentication, authorization, rate limiting and multi-tenant isolation remain
production hardening work rather than demo claims.

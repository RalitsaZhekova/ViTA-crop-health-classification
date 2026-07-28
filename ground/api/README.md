# API boundary

The API layer will expose the following versioned routes on top of the verified
scene catalog:

```text
POST /api/v1/scenes
GET  /api/v1/scenes/{scene_id}
GET  /api/v1/scenes/{scene_id}/preview
GET  /api/v1/scenes/{scene_id}/condition-overlay
GET  /api/v1/regions/{region_id}/latest
GET  /api/v1/regions/{region_id}/history
```

Scene responses use `shared/schemas/downlink_bundle.schema.json`. Exact client
measurements come from the JSON record and its interaction grid; WebP and PNG
assets are presentation layers and are never decoded as scientific values.

Optional API-key protection will be available for scene uploads in the MVP.
Full authentication, authorization and multi-tenant isolation remain production
hardening work rather than demo claims.

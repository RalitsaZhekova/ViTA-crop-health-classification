# API boundary

The minimal future API should expose:

```text
POST /scenes
GET  /scenes/{scene_id}
GET  /scenes/{scene_id}/preview
GET  /scenes/{scene_id}/condition-overlay
GET  /regions/{region_id}/latest
GET  /regions/{region_id}/history
```

Scene responses use `shared/schemas/downlink_bundle.schema.json`. Exact client
measurements come from the JSON record and its interaction grid; WebP and PNG
assets are presentation layers and are never decoded as scientific values.

No authentication, tenancy or upload implementation exists yet. Those choices
must be made before this becomes a B2B service.

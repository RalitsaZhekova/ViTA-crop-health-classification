# API boundary

The minimal future API should expose:

```text
POST /scenes
GET  /regions/{region_id}/latest
GET  /regions/{region_id}/history
GET  /scenes/{scene_id}/layers/{layer_name}
```

Responses must use the schemas under `shared/schemas/`. Large rasters remain
COGs or tiles; they are not embedded in JSON.

No authentication, tenancy or upload implementation exists yet. Those choices
must be made before this becomes a B2B service.

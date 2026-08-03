# Ground component

`ground/` is the receiving side of the mission. Crop-condition calculations run
on the payload; the ground component verifies and stores compact downlink
products, builds historical series, exposes the API and serves the professional
web client.

The authoritative scientific implementation remains available from the shared
package so payload and ground validation use exactly the same formulas. Ground
processing must not silently recalculate a different headline score.

## Ground responsibilities

- validate the downlink manifest and asset checksums;
- index scenes by region, footprint and acquisition time;
- serve the RGB preview and crop-condition heat map;
- expose exact measurements, explanations and evidence quality from JSON;
- compare compatible observations through time;
- produce alerts and client-facing history without claiming disease diagnosis.

Single-scene labels remain spectral screening priorities. `Nominal`, `Watch`,
`Moderate anomaly`, `High anomaly` and `Insufficient data` describe the available
multispectral evidence, not confirmed agronomic health or a specific cause.

## Verified scene catalog

The implemented catalog accepts only the three-file downlink contract, verifies
every checksum and image property, and atomically copies valid bundles into an
immutable scene store backed by SQLite. Re-ingesting the identical bundle is
safe; reusing a scene identifier for different content is rejected.

For coordinate missions, `vita-mission` in the integration package is the thin
ground transport client. It submits the validated coordinate command, polls the
payload, independently verifies the three downloaded files, then calls this
same catalog ingestion path. Earth Engine access and every cloud, crop and
condition calculation remain payload-side.

```powershell
vita-ground-ingest `
  ..\testing\runs\pastis_10425_compact_downlink\payload\downlink `
  --store .\runtime
```

The catalog supports scene, region, latest-observation and chronological-history
queries. Exact scientific values remain in the verified JSON record; image
assets are never treated as measurement sources.

## HTTP API

The implemented FastAPI service supports verified uploads, scene and region
queries, exact manifests and grid cells, immutable preview/overlay delivery and
chronological history. Its OpenAPI documentation is available at `/docs` while
the service is running. See `api/README.md` for commands and the complete route
list.

## Interactive client

The same server exposes an offline-first dashboard at `/`. It provides:

- searchable scene and region navigation;
- a zoomable/pannable RGB scene with an adjustable crop-health heat map;
- transparent masking outside valid clear crop pixels;
- queryable grid cells backed by exact JSON measurements;
- transparent score components, index summaries and evidence coverage;
- cloud-quality composition and chronological condition history;
- responsive layouts without an external map or JavaScript CDN.

The dashboard deliberately describes crop condition as spectral screening. It
does not convert display colors back into measurements or claim to diagnose a
disease.

## Container service

`deployment/ground/Dockerfile` installs only the shared and ground packages.
It contains no payload package, Earth Engine client, model artifacts or
scientific processing code. `compose.ground.yaml` mounts the project-local
`runtime/ground` catalog at `/data/ground` inside the container and publishes
only `127.0.0.1:8000`.

After independently validating the downloaded bundle, the host mission CLI
uploads exactly `scene.json`, `scene.webp` and `condition.png` to the canonical
`POST /api/v1/scenes` route. The container revalidates the bundle before
installing it, so it needs no payload job mount, model, credential or
scientific implementation.

```powershell
docker compose --env-file .env.ground -f compose.ground.yaml build
docker compose --env-file .env.ground -f compose.ground.yaml up -d
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/health"
```

The canonical health route is `/api/v1/health`; the dashboard and OpenAPI
routes remain `/` and `/docs`. See `deployment/README.md` for the complete
project-local split workflow.

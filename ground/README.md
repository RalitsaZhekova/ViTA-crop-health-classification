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
- serve the RGB preview and condition/quality overlay;
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

```powershell
vita-ground-ingest `
  --bundle ..\testing\runs\pastis_10425_compact_downlink\payload\downlink `
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

The interactive client is the next layer built on this API.

# Storage contract

The implemented local catalog stores:

- one immutable scene record per acquisition;
- region identifier, sensor, acquisition time, CRS and bounds;
- model and algorithm versions;
- cloud/analysis pixel counts;
- NDVI, EVI, GNDVI, CVI and RGB summary statistics;
- the compact RGB preview and crop-condition heat-map locations and checksums;
- the queryable interaction-grid measurements delivered in scene JSON.

Each received three-file bundle is checksum-verified, decoded to validate its
image properties, staged, and then atomically installed. A SQLite index supports
scene and region history queries while the original bundle remains immutable.
Identical retransmission is idempotent and conflicting content is rejected.

SQLite plus files on disk is sufficient for the MVP. Production should use
PostgreSQL/PostGIS and object storage. The active downlink contract is defined in
`shared/schemas/downlink_bundle.schema.json`.

Historical baselines must be keyed by stable region geometry and acquisition
date. They must not compare unrelated fields or different growth stages as if
they were one time series.

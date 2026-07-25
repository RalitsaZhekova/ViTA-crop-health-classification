# Storage contract

The first implementation should store:

- one immutable scene record per acquisition;
- region identifier, sensor, acquisition time, CRS and bounds;
- model and algorithm versions;
- cloud/analysis pixel counts;
- NDVI, EVI, GNDVI, CVI and RGB summary statistics;
- locations of classification, probability, cloud and index COG assets.

For the short demo, SQLite plus files on disk is sufficient. Production should
use PostgreSQL/PostGIS and object storage. The JSON contract is defined in
`shared/schemas/health_observation.schema.json`.

Historical baselines must be keyed by stable region geometry and acquisition
date. They must not compare unrelated fields or different growth stages as if
they were one time series.

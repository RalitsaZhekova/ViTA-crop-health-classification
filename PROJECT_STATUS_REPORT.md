# ViTA crop-condition pipeline status

Status date: 2026-07-28

## Executive status

The repository now contains a working local Sentinel development pipeline from
a georeferenced multispectral scene through a compact, web-ready downlink
package:

```text
Sentinel development GeoTIFF
-> intake and radiometric contract validation
-> semantic cloud and shadow classification
-> operational unusable mask
-> crop probability and binary crop inference
-> payload vegetation indices and RGB diagnostics
-> payload condition score and spatial anomaly assessment
-> scene.webp + condition.png + scene.json
-> DOWNLINK_READY
```

The calculations have moved from the ground package into the payload execution
path. Ground is now responsible for receiving and validating the bundle,
historical storage, API delivery, trends and the client application.

This is a verified local software pipeline, not yet a flight-qualified mission
system. Physical downlink transport, uplink commands, Balkan-1 reconstruction,
payload-computer qualification and the production web application remain open.

## Implemented components

### Training and selected crop model

- The selected payload model performs single-frame binary crop/non-crop
  segmentation from Blue, Green, Red and narrow NIR.
- Its weights-only artifact and SHA-256 identity are pinned under
  `payload/models/`.
- Training used the US IBM-NASA HLS base corpus plus optical European PASTIS
  replay data from folds 1-4. PASTIS fold 5 remains reserved.
- Current selected-model metrics are not held-out European or Balkan accuracy
  evidence.
- Training data and experiment checkpoints are excluded from the payload
  deployment package.

### Scene intake

The payload intake validates:

- GeoTIFF structure and positive dimensions;
- CRS, affine transform, bounds and resolution;
- declared band descriptions and logical roles;
- RGB and NIR availability;
- nodata and sampled numeric ranges;
- acquisition time and scene identifier;
- explicit or inferred reflectance scale.

The current Sentinel adapter prefers B8A for OmniCloudMask and also uses B8A for
the selected crop model, while retaining B08 fallback compatibility. Balkan-1
uses its native NIR band for the cloud model; its crop transfer remains
explicitly provisional.

### Cloud and unusable mask

The payload streams the scene through bounded cloud-model tiles and produces:

- semantic classes: clear, thick cloud, thin cloud and cloud shadow;
- invalid-input mask;
- final unusable mask including cloud, shadow, invalid pixels and configured
  safety post-processing;
- clear, cloud, shadow, invalid, usable and unusable statistics;
- the configurable `PROCESS`, `PROCESS_CLEAR_AREAS` or `REJECT`
  recommendation;
- georeferenced intermediate masks and detailed visualizations.

Downstream crop and condition stages use the operational unusable mask rather
than cloud percentage alone.

### Crop inference

The crop stage:

- runs only after the cloud gate passes;
- processes large rasters through bounded overlapping windows;
- preserves geospatial alignment;
- produces crop probability, confidence and binary crop intermediates;
- excludes all unusable pixels from valid outputs;
- calculates crop fraction over usable pixels;
- records the selected-model artifact, checksum and threshold.

The current crop model is binary and must not be described as exact field
boundary or crop-health classification.

### Payload crop-condition analysis

The payload now calculates condition products before downlink. The analysis mask
is:

```text
binary crop
AND crop probability >= 0.645
AND usable
AND finite source data
AND calibrated reflectance range
```

Calculated measurements include:

- NDVI;
- GNDVI;
- EVI;
- SAVI;
- CVI as a diagnostic only;
- VARI, RGB brightness and excess green.

The headline spectral-vigor score combines normalized NDVI, GNDVI, EVI and SAVI
components with default weights of 40%, 25%, 20% and 15%. The region result uses
70% median and 30% lower-quartile absolute score, then applies a bounded penalty
for robust within-crop-region anomalies.

The result records coverage, component scores, evidence quality, explanations,
limitations and one of:

- `Nominal`;
- `Watch`;
- `Moderate anomaly`;
- `High anomaly`;
- `Insufficient data`.

The value is not percentage healthy, disease probability, crop survival or a
confirmed agronomic diagnosis.

## Compact downlink package

A successful payload run ends with `DOWNLINK_READY` and produces exactly three
routine transmission files:

| File | Purpose |
| --- | --- |
| `scene.webp` | Lossy RGB overview for the base map |
| `condition.png` | Lossless aligned RGBA overlay with the condition gradient and separate thick-cloud, thin-cloud, shadow, invalid and unusable-buffer categories |
| `scene.json` | Authoritative measurements, condition explanation, georeferencing, model versions, legend, checksums and interaction grid |

The interaction grid is capped at 16 by 16 but adapts to small scenes so it does
not create cells smaller than 32 source pixels. This keeps the JSON compact while
allowing map click and hover cards to return exact stored values rather than
reverse-engineering displayed colors.

The two image references inside `scene.json` are relative. Each image record
contains its byte size, media type, dimensions and SHA-256 checksum. The package
contains no workstation-absolute paths.

Full-resolution GeoTIFFs remain payload processing intermediates and are not in
the routine bundle. A later mission policy may preserve or transmit selected
evidence chips, periodic calibration scenes or critical complete scenes.

## Verified Sentinel execution

The previously retained cloud model and Prithvi model were executed on the georeferenced
PASTIS/Sentinel development scene `pastis_10425_20190924_5band.tif`.

| Measurement | Verified result |
| --- | --- |
| Pipeline status | `COMPLETE` |
| Payload status | `DOWNLINK_READY` |
| Runtime | 14.24 seconds on the development GPU workstation |
| Unusable area | 43.05% |
| Crop over usable area | 69.45% |
| Condition | `Moderate anomaly`, 36.28/100 |
| Analyzed coverage | 23.39% |
| Downlink files | 3 |
| Downlink size | 27,690 bytes |
| Source TIFF size | 132,580 bytes |
| Package/source fraction | 20.89% |
| Adaptive query grid | 4 by 4 |

The RGB WebP and transparent condition overlay were visually inspected. All
three file sizes, both image checksums, image dimensions, relative references,
condition values and metric records were independently verified against the
payload intermediates. Exact evidence is stored in
`integration/verification.json`.

This scene belongs to training/replay data. It proves execution and interfaces,
not held-out crop accuracy, broad cloud accuracy or agronomic validity.

## What remains before a flight downlink claim

### Mission control and uplink

- define and validate the uplink command schema;
- centralize the single official mission decision controller;
- combine unusable coverage, usable crop coverage, priority and runtime budget;
- add explicit discard, insufficient-data and downlink-queue actions;
- enforce watchdog, timeout, memory, power and storage limits.

### Balkan-1 L1B and raw reconstruction

- inspect real Balkan-1 and Balkan-2 files separately;
- confirm delivered band centers, radiometry, metadata and product levels;
- reconstruct and co-register raw RGB, NIR and PAN captures;
- orthorectify using the available position, attitude, timing and camera data;
- calibrate 12-bit DN values to comparable reflectance;
- choose and validate the one-NIR mapping;
- compare native-resolution and approximately 10 m downsampled baselines;
- validate cloud, crop and condition behavior against real labelled regions.

The Sentinel cloud model was trained for L1C-style imagery while the mission
ceiling is L1B. A sensor-specific L1B-to-analysis preprocessing path is therefore
mandatory before onboard results can be trusted.

### Physical downlink transport

The software constructs a portable package locally, but does not yet implement:

- queueing and scene priority;
- radio/contact-window scheduling;
- chunking and transmission framing;
- retry, resume and acknowledgement;
- incomplete-transfer cleanup;
- checksum verification by an independent ground receiver.

### Runtime qualification

The current run time is workstation evidence, not payload-computer evidence.
The complete chain must be measured on the EnduroSat target for wall time, CPU,
accelerator availability, peak RAM, storage, energy use and thermal behavior.

### Commercial licensing

The selected OmniCloudMask `1.7.1` code and official V4 weights are recorded as
MIT licensed. Preserve upstream notices and attribution in distributed builds.

## Ground and business application work remaining

The three-file package is sufficient to start the professional web MVP. Ground
work should proceed in this order:

1. Implement checksum-verified `scene.json` ingestion.
2. Store immutable scene records and image objects.
3. Index scene footprint, region and acquisition time.
4. Expose scene, latest-region and region-history API endpoints.
5. Display the WebP base image and aligned PNG overlay in Leaflet or MapLibre.
6. Populate click/hover cards from JSON grid cells.
7. Add cloud, crop, condition, confidence and evidence-quality legends.
8. Add time-series comparisons only between compatible, co-registered
   observations of the same region.
9. Add alert rules and client tenancy/authentication.

Temporal warnings must account for crop type, season, phenology, harvest,
registration and sensor calibration. They should be presented as trend-based
early warning, not disease prediction, until validated outcome data exist.

## Current scientific claim

The defensible statement is:

> Reduced vegetation vigor or an anomalous crop-region spectral response was
> detected in the available Blue, Green, Red and NIR observations under the
> recorded masks, thresholds and calibration assumptions.

The system cannot currently identify a specific disease, pest, nutrient
deficiency, drought cause, plant-water content, thermal stress or yield loss.

## Canonical implementation map

- `payload/src/prithvi_payload/pipeline.py` — stage-gated payload orchestration.
- `payload/src/prithvi_payload/condition_stage.py` — windowed payload condition processing.
- `payload/src/prithvi_payload/downlink.py` — compact three-file package builder.
- `shared/src/prithvi_shared/health.py` — vegetation indices and RGB diagnostics.
- `shared/src/prithvi_shared/condition.py` — transparent condition and anomaly scoring.
- `shared/schemas/downlink_bundle.schema.json` — active downlink contract.
- `integration/src/vita_integration/pipeline.py` — Sentinel development orchestration.
- `integration/verification.json` — accepted real-model execution evidence.
- `ground/` — receiving, storage, history, API and client boundaries.

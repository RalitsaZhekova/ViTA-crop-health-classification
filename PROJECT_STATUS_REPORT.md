# ViTA Space Challenges 2026

## Cloud Detection, Crop Relevance, Downlink Prioritization and Crop-Condition Monitoring

**Project status report — 27 July 2026**

## 1. Purpose and status statement

This report records what is demonstrably implemented in the current repository, where the payload pipeline stops today, and what remains before an accepted scene can be packaged and sent through the downlink. It also records the ground-side work that follows downlink.

The current system has a working local prototype for:

```text
preprocessed georeferenced scene
-> bounded-memory intake validation
-> Sentinel-2 cloud and shadow classification
-> operational unusable-pixel mask
-> cloud-stage recommendation
-> cloud-gated, windowed crop/non-crop segmentation
-> crop fraction over usable pixels
-> local masks, previews, metadata and result.json
```

The code therefore reaches the **crop-classification result immediately before a final mission/downlink decision**, but it does not yet reach a real downlink. The following critical elements are still absent: Balkan-1 raw reconstruction, the requested L1B-only onboard preprocessing path, Balkan radiometric and spectral adaptation, one centralized mission controller, crop-value-based downlink acceptance, an uplink command protocol, downlink packaging and transfer management, and qualification on the actual payload computer.

The ground component already contains the core index calculations, but it is not connected to the payload result, does not yet write operational raster layers, and does not yet provide the condition scoring, database, API, visualization or temporal warning system.

The most accurate overall description is:

> A locally demonstrated Sentinel-compatible cloud-to-crop prototype with standalone ground measurement functions, not yet a flight-qualified or end-to-end mission system.

## 2. Agreed mission scope

### 2.1 Sensors and bands

The operational algorithms are restricted to:

- Blue;
- Green;
- Red;
- one Near Infrared band.

Balkan-1 will additionally deliver a panchromatic band. PAN is useful as a spatial reference during reconstruction and potentially for display products, but it is not an input to either current AI model. It should not be introduced as an unvalidated fifth model channel.

No SWIR, thermal infrared, far infrared or dedicated cirrus band is available. This limits cloud/snow discrimination and prevents direct measurement of thermal stress or plant water content.

### 2.2 Payload and ground split

The intended operational split is:

```text
UPLINK
command, target, timing, priority and processing profile
                         |
                         v
PAYLOAD
scene intake
-> raw Balkan reconstruction and correction, stopping at L1B
-> band registration and radiometric preparation
-> cloud/shadow/invalid detection
-> unusable mask
-> crop relevance
-> centralized mission decision
-> downlink package and queue
                         |
                         v
DOWNLINK
accepted bands, masks, statistics, provenance and metadata
                         |
                         v
GROUND
ingestion and reflectance preparation
-> valid crop analysis mask
-> vegetation and RGB indices
-> spatial and optional temporal anomaly analysis
-> cautious crop-condition result
-> storage, API, map and reports
```

The requested raw Balkan processing ceiling is now **Level 1B**. The onboard branch must not add an L1C product-generation stage. Existing Sentinel-2 products may still be used as development and demonstration inputs, but that must not be confused with the required Balkan raw-to-L1B payload path.

### 2.3 Scientific claim

With RGB+NIR, the defensible claim is that the system detects reduced vegetation vigor or an unusual crop-region spectral response. It does not confirm disease, pests, nutrient deficiency, drought cause, thermal stress, exact water content or yield loss.

Ground labels should therefore be `Nominal`, `Watch`, `Moderate anomaly`, `High anomaly` and `Insufficient data`, once a defensible baseline exists. The current code correctly limits itself to `MEASURED` and `INSUFFICIENT_DATA`.

### 2.4 Responsibility split

The cloud/health workstream owns cloud and shadow detection, construction and statistics of the final unusable mask, and the ground crop-condition analysis. The crop-model workstream owns crop-model training, validation and crop probability/binary output. The health stage consumes that output and must not silently replace crop detection with an index threshold. Input/output contracts, thresholds, payload deployment, end-to-end testing and the final demonstration remain shared responsibilities.

## 3. Repository and deployment separation completed

The repository has been separated into four responsibilities:

| Component | Current responsibility | Current state |
| --- | --- | --- |
| `training/`, `src/`, `configs/` | Data preparation, training, calibration and retained experiment evidence | Implemented |
| `payload/` | Inference-only weights, scene intake, cloud detection and crop inference | Prototype implemented for preprocessed Sentinel-compatible data |
| `ground/` | Crop-condition measurements and future storage/API/map boundaries | Calculations implemented; system integration pending |
| `shared/` | Band, normalization, calibration and JSON contracts | Partly implemented; canonical payload schema is not yet used by `result.json` |

Training chips and experiment outputs are not required on the satellite. The current `payload/` tree is approximately **424.2 MB**, of which almost all is the two model artifacts:

- crop weights: 382,645,915 bytes;
- CloudSEN12 weights: 41,267,324 bytes.

This is already radically smaller than the training workspace and confirms that the 30–49 GB datasets do not need to be deployed. The final installed environment or container may be larger because PyTorch, TerraTorch, raster libraries and their native dependencies have not yet been size-optimized or bundled.

## 4. Crop-model work completed

### 4.1 Selected model

The selected payload crop model is now a **pixel-wise, single-image, binary crop/non-crop segmentation model**. This supersedes older planning notes that described the current model as only a chip-level classifier.

Its contract is:

- architecture: frozen Prithvi-EO-2.0 100M backbone with a UPerNet decoder/head;
- input: `[batch, 4, 1, height, width]`;
- band order: `BLUE, GREEN, RED, NIR_NARROW`;
- output: crop probability, binary crop mask and winning-class confidence;
- ordinary crop threshold: 0.49;
- conservative health-analysis threshold: 0.645;
- selected weights checksum: `c948977bffdaeb89ecf4f4d069db13c7ed81d4f3403eec9e257c45c235b1484e`.

The weights-only artifact removes approximately 77.2 MB of optimizer and training state. A direct comparison on one real internal-validation chip recorded zero probability difference and identical binary decisions between the source checkpoint and the deployment weights.

### 4.2 European data and training status

The selected binary configuration includes a 20% PASTIS replay contribution from European folds 1–4. The local PASTIS inventory contains 2,433 metadata-linked patches; fold 5, containing 496 patches, is reserved. The selected single-frame binary training run is marked complete.

A separate three-frame, 13-class European replay model is preserved, but it is not the deployed payload model. Its automation status is marked failed because a preflight run could not find the expected exhaustive validation report at that moment, even though a checkpoint and later metrics exist. This does not invalidate the selected binary deployment artifact, but the replay workflow is not presently a clean, reproducible green run.

### 4.3 Recorded model metrics

The selected binary checkpoint was calibrated on internal validation data. The repository records:

| Metric | Result |
| --- | ---: |
| Binary accuracy at threshold 0.49 | 0.794961 |
| Balanced accuracy | 0.793647 |
| Crop F1 | 0.778128 |
| False-crop rate at threshold 0.49 | 0.188319 |
| Health-mask precision at threshold 0.645 | 0.846481 |
| Health-mask recall at threshold 0.645 | 0.632780 |
| False-crop rate at threshold 0.645 | 0.099173 |

The evaluation represents 308 source validation chips expanded to 924 single-date examples. The final held-out test has not been consumed. These figures are internal validation evidence, not a guarantee for every Sentinel-2 scene and not evidence for Balkan-1.

The reported 0.8179 crop F1 on PASTIS scene 10425 is a selected training/replay demonstration. It is useful execution evidence, but it must not be advertised as general or held-out accuracy.

## 5. Payload work completed through crop classification

### 5.1 Bounded-memory scene intake

The intake stage inspects a GeoTIFF without loading the complete scene. It checks:

- file format, dimensions, CRS, transform, finite bounds and nodata declaration;
- band descriptions and logical roles;
- approximate sampled radiometry;
- acquisition timestamp format;
- the spectral routes needed by the cloud and crop models.

It recognizes Sentinel-2 and Balkan-1 as sensor names. A preprocessed Balkan scene is expected to contain exactly RGB, NIR and PAN, but it is deliberately marked as needing Balkan cloud and spectral adapters. This is contract scaffolding, not Balkan inference support.

### 5.2 Cloud and cloud-shadow detection

The integrated cloud stage uses CloudSEN12 `dtacs4bands`, version 1.0.2, with checksum-pinned weights. The present model contract is Sentinel-2 L1C top-of-atmosphere data in this order:

```text
B08, B04, B03, B02
NIR broad, Red, Green, Blue
```

It performs windowed inference using 512-pixel tiles, 64-pixel overlap and reflected padding. It produces four semantic classes:

- 0 — clear;
- 1 — thick cloud;
- 2 — thin cloud;
- 3 — cloud shadow.

Post-processing currently:

- includes thick cloud, thin cloud and shadow as unusable;
- removes detected regions smaller than 16 pixels;
- dilates remaining detections by two pixels;
- adds invalid and nodata pixels to the unusable mask.

The integrated `scene-run` path writes:

- semantic mask GeoTIFF;
- unusable mask GeoTIFF;
- invalid-input mask GeoTIFF;
- cloud preview PNG;
- cloud metadata JSON.

Although the standalone cloud configuration declares class-score saving, the integrated windowed `scene-run` outputs do not currently contain class-score GeoTIFFs. They must not be described as canonical current outputs until connected and verified.

The cloud-stage recommendation is calculated from total unusable area:

| Unusable area | Recommendation |
| --- | --- |
| `<= 30%` | `PROCESS` |
| `> 30%` and `< 80%` | `PROCESS_CLEAR_AREAS` |
| `>= 80%` | `REJECT` |

This is a module recommendation, not yet the one official mission/downlink decision.

### 5.3 Crop inference

When the crop stage is explicitly requested and its current cloud gate passes, the pipeline loads the selected crop model and processes the scene in bounded windows:

- core tile: 224 pixels;
- halo: 16 pixels;
- batch size: 4;
- full-scene materialization: prohibited;
- current crop threshold: 0.49.

The crop executor validates that the source and unusable mask have identical width, height, CRS and affine transform. It excludes every unusable pixel from the reported crop products and encodes those pixels as nodata:

- probability: `-9999` on unusable pixels;
- binary crop: `255` on unusable pixels;
- confidence: `-9999` on unusable pixels.

It writes georeferenced probability, binary and confidence GeoTIFFs, crop metadata and a combined quicklook. It correctly calculates crop fraction using usable pixels as the denominator and returns `null` rather than dividing by zero if no usable pixels exist.

### 5.4 Canonical local run record

Every manual run writes `testing/runs/<run>/result.json`, with references to detailed intake, cloud and crop artifacts. All 73 artifact references found in the current run records resolve to existing files.

This is the correct local integration entry point for the current prototype. It is not yet a downlink manifest: paths are absolute workstation paths, the schema is `0.1-draft`, and it does not conform to the separate shared `1.0` inference schema.

## 6. Demonstrated integration evidence

The repository records the following useful demonstrations:

| Evidence | Recorded result | Limitation |
| --- | --- | --- |
| Real Sentinel-compatible cloud execution | 1,153 x 2,275 pixels, 24 tiles, 5.31 s, geospatial metadata preserved | Execution only; no cloud ground truth |
| Full cloud-to-crop run on PASTIS 40011 | crop stage completed on CUDA; 77.68% crop over usable pixels | Training/replay example, not held-out accuracy |
| Masked crop integration on PASTIS 10422 | 15.01% unusable; 2,459 masked output pixels correctly encoded as nodata | Contract test, not accuracy evidence |
| Cloud gate skip | 88.48% cloud; crop model was not loaded | Uses current 60% cloud gate, not final mission controller |
| Label comparison on PASTIS 10425 | crop precision 0.8882, recall 0.7579, F1 0.8179 | Selected training/replay scene |
| Deployment crop smoke | finite 224 x 224 probability and mask; 399.81 MiB peak GPU allocation | Laptop RTX 3060, not payload hardware |
| Cloud synthetic GPU smoke | 0.569 s and 98.92 MiB peak GPU allocation | Synthetic input; no accuracy conclusion |

Historical verification records report 13 cloud tests and 57 repository regression tests passing before test-source cleanup. The current repository no longer contains a runnable automated test suite, so these are retained records rather than reproducible current CI evidence.

The current lint check is also not fully green: one import-order issue remains in `payload/src/prithvi_payload/crop_executor.py`. It is minor, but a release report should not claim a clean lint gate until it is corrected.

## 7. Exact status at the downlink boundary

| Mission function | Status | What this means |
| --- | --- | --- |
| Receive an uplink processing command | **Not implemented** | Runs are started manually by local PowerShell/CLI commands |
| Validate a preprocessed Sentinel scene | **Implemented** | Works for georeferenced inputs with documented band descriptions |
| Reconstruct raw Balkan bands | **Not implemented** | Only a design contract exists |
| Produce the required Balkan L1B result | **Not implemented** | Current cloud path instead assumes a preprocessed L1C-like Sentinel contract |
| Detect Sentinel clouds/shadows | **Prototype implemented** | Execution is validated; broad accuracy is not |
| Detect Balkan clouds/shadows | **Not implemented/validated** | Intake intentionally blocks this route |
| Construct operational unusable mask | **Implemented** | Includes cloud, shadow, invalid/nodata and configured post-processing |
| Run crop segmentation | **Prototype implemented** | Demonstrated on Sentinel-compatible/PASTIS inputs |
| Calculate crop fraction over usable pixels | **Implemented** | Correct denominator and zero-usable handling |
| Make one final crop-aware mission decision | **Not implemented** | Only a cloud recommendation and separate cloud gate exist |
| Create a portable downlink manifest/package | **Not implemented** | Current `result.json` is a local run summary |
| Queue, prioritize, chunk and transmit accepted data | **Not implemented** | No downlink transport integration exists |
| Enforce payload runtime/resource budgets | **Not implemented** | Tiling limits memory, but flight budgets are undefined |
| Execute on EnduroSat payload hardware | **Not verified** | Current performance evidence is from a laptop GPU |

The present executable pipeline ends with `status = CROP_COMPLETE`. It never produces a final action such as `DOWNLINK_TO_GROUND`, `DISCARD_LOW_CROP` or `INSUFFICIENT_DATA`, and it does not assemble or transmit an accepted scene package.

## 8. Integration conflicts that must be resolved

### 8.1 L1B mission ceiling versus L1C-trained cloud model

The current cloud detector is explicitly configured and validated for Sentinel-2 L1C top-of-atmosphere reflectance. The new Balkan requirement is to reconstruct only through L1B and not generate L1C onboard.

The target solution must therefore either:

1. produce model-compatible, calibrated RGBNIR tensors from the L1B sensor-geometry product without claiming or generating an L1C product, and validate the domain shift; or
2. replace the cloud method with one demonstrably valid on the delivered Balkan L1B representation.

Simply relabeling the existing L1C configuration as L1B would be scientifically and operationally invalid.

### 8.2 One NIR mission contract versus two Sentinel NIR routes

The current Sentinel cloud model requires broad NIR B08, while the crop model requires narrow NIR B8A. Consequently, the current full Sentinel route expects B02, B03, B04, B08 and B8A, even though each individual model uses only four inputs.

The mission contract allows only one RGB+NIR set. A single official NIR mapping and harmonization strategy must be selected. Options include validating broad NIR for the crop model, adapting the cloud input to the delivered Balkan NIR response, or using sensor-specific transformations. Until this is resolved, the five-band Sentinel prototype is not the same interface as the four-band Balkan mission input.

### 8.3 Multiple decision rules

The cloud module recommends from **unusable percentage** using 30% and 80%. The crop-stage loader independently gates on **cloud percentage** below 60%. There is no minimum crop fraction in the final route.

These rules can disagree and must be replaced by one mission controller that considers at least:

- unusable fraction;
- absolute usable-pixel count;
- crop fraction over usable pixels;
- crop confidence/coverage;
- processing completion and data integrity;
- runtime deadline and expected package size;
- mission priority supplied by uplink.

### 8.4 Payload-result schema mismatch

The canonical run record uses schema `0.1-draft`, while `shared/schemas/inference_result.schema.json` requires schema `1.0` with top-level acquisition time, model checksum and raster assets. The current run record does not satisfy that schema. One versioned, portable schema must replace both.

### 8.5 Ground mask mismatch

The selected payload model emits binary values 0 and 1. The current ground `build_analysis_mask` default still selects fine-grained crop IDs `(2, 3, 7, 8, 10, 11)`. Unless `crop_class_ids=(1,)` is passed explicitly, it will select no pixels from the deployed binary mask. The ground interface must be changed to consume the actual binary payload contract directly.

### 8.6 Commercial license blocker

The CloudSEN12 weights are recorded as CC BY-NC 4.0. The deployment manifest correctly marks commercial release as blocked. A B2B product cannot rely on these weights without an appropriate commercial license or a commercially compatible replacement.

## 9. Work remaining before downlink

### 9.1 Freeze the common sensor and metadata contract

Define one versioned input contract containing:

- satellite and sensor version;
- scene ID and UTC acquisition time;
- RGB and the one NIR band mapping;
- PAN role;
- raw/L1B/processed product level;
- dtype, bit depth, scale, offset, units, nodata and saturation;
- dimensions, CRS/geometry information and source coordinates;
- spacecraft position/attitude and camera calibration when available;
- band-registration quality;
- model and threshold profile IDs.

Balkan-1 and any later Balkan-2 work must remain separate until real metadata proves compatibility.

### 9.2 Implement Balkan-1 raw reconstruction to L1B

For each raw capture:

1. decode each separate RGB, NIR and PAN band;
2. identify invalid, missing and saturated samples;
3. apply sensor calibration and distortion correction available at L1B;
4. choose PAN or a stable multispectral band as the spatial registration reference;
5. estimate translation, rotation, scale and, where justified, affine or local distortion from data and metadata;
6. co-register the multispectral bands in sensor geometry;
7. retain coordinates and all transform provenance;
8. calculate per-band residual registration error and quality flags;
9. output a calibrated, co-registered L1B scene plus invalid mask;
10. stop without generating an L1C cartographic product onboard.

Unknown altitude, attitude, rotation and offsets must be estimated or marked insufficient; they must not be replaced by hard-coded assumptions. Vendor-processed imagery should be used as comparison evidence, not automatically treated as ground truth.

### 9.3 Build and validate the Balkan sensor adapters

The adapter must handle:

- exact band mapping and NIR response;
- 12-bit DN interpretation;
- scale, offset, radiance/reflectance conversion and saturation;
- single-NIR harmonization for both models;
- sensor-resolution alignment;
- downsampling from approximately 1.5 m toward the model's validated scale;
- nodata and invalid masks;
- co-registration tolerances;
- sensor and processing-level provenance.

At minimum, compare native-resolution, downsampled and downsampled-plus-harmonized inference. PAN should remain excluded from model tensors. Under the current no-more-training policy, this work is calibration, adaptation and validation—not model retraining. If accuracy is unacceptable after those steps, the limitation must be escalated rather than hidden behind a confident output.

### 9.4 Implement the one official mission controller

A deterministic initial controller should follow this sequence:

```text
invalid input or no usable pixels
-> INSUFFICIENT_DATA

unusable fraction >= 0.80
-> DISCARD_TOO_CLOUDY

0.30 < unusable fraction < 0.80
-> PROCESS_CLEAR_AREAS

unusable fraction <= 0.30
-> PROCESS

after crop inference:
crop fraction over usable pixels below configured mission threshold
-> DISCARD_LOW_CROP

sufficient usable crop coverage and valid outputs
-> DOWNLINK_TO_GROUND
```

The initial crop threshold may be 0.30, but it must remain a mission profile value rather than a scientific constant. A later tile-aware override can preserve valuable clear agricultural subregions inside an otherwise cloudy scene.

The controller must be the only component allowed to emit a final mission action. Cloud and crop modules should emit measurements and recommendations only.

### 9.5 Define and implement uplink commands

The satellite must not depend on a person launching a PowerShell command. A compact versioned command envelope is required. At minimum it should support:

- command ID and schema version;
- target scene/region or capture request;
- earliest/latest execution time and expiry;
- priority;
- sensor and processing profile;
- requested stopping point (`intake`, `cloud`, `crop`, or `decide`);
- approved threshold profile rather than arbitrary unsafe numbers;
- maximum runtime and output-size budget;
- downlink content policy;
- status query and cancellation where the flight system permits them.

The handler needs authentication/signature verification, replay protection, idempotent command IDs, range validation, queueing, ACK/NACK responses, execution-state telemetry and an immutable audit record. Invalid or expired commands must fail closed without starting a model.

### 9.6 Apply hard runtime and resource constraints

Windowing is implemented, but the actual constraints are unknown. The payload team/EnduroSat must provide acceptance limits for:

- maximum wall-clock time per scene and per tile;
- CPU architecture, core/thread allowance and whether a supported accelerator exists;
- maximum RAM and accelerator memory;
- maximum temporary storage and final output size;
- power/energy budget and allowed duty cycle;
- maximum input dimensions and concurrent jobs;
- watchdog interval, command deadline and thermal constraints;
- downlink bandwidth and contact-window capacity.

The runtime must measure wall time, CPU time, peak resident RAM, accelerator memory, I/O time, tile counts, skipped tiles, temporary bytes and output bytes. It must enforce budgets rather than merely report them after completion.

Graceful failure states should include `DEADLINE_EXCEEDED`, `RESOURCE_LIMIT`, `MODEL_FAILURE`, `INCOMPLETE_OUTPUT` and `INSUFFICIENT_DATA`. A partial result must never be labeled complete. Temporary tiles must be cleaned only after a recoverable manifest is safely committed.

Current laptop timings are useful only as engineering baselines. The 5.31-second cloud run on 2.62 million pixels and sub-second 128 x 128 crop examples cannot be extrapolated to a huge 1.5 m Balkan scene or to unknown payload hardware.

### 9.7 Create the downlink package

For the first prototype, an accepted package should contain or reference:

- Blue, Green, Red and NIR data needed by the ground stage;
- unusable mask;
- crop binary and probability/confidence products;
- semantic cloud mask when useful;
- scene footprint/coordinates and acquisition time;
- usable, unusable, cloud, shadow, invalid and crop statistics;
- model checksums and versions;
- sensor, product level and radiometric scale/offset/units;
- thresholds and final decision reason;
- reconstruction and registration quality;
- runtime/resource measurements;
- checksums, byte sizes and relative asset names.

The package needs a portable manifest, atomic finalization, compression, chunking, checksums, priority, retry/resume behavior and transfer acknowledgement. Absolute workstation paths must not appear. Full accepted bands and masks are appropriate for the first end-to-end prototype; tile-only downlink can be evaluated later after bandwidth measurements.

## 10. Work remaining after downlink

### 10.1 Connect ground ingestion to payload output

Implement a reader for the final downlink manifest, verify every checksum, validate grids and mask semantics, and reject incomplete or misaligned scenes. Nearest-neighbor resampling should be used for categorical masks and an appropriate continuous method for reflectance bands.

Because the onboard raw path stops at L1B, the ground stage must explicitly produce comparable calibrated reflectance before EVI, CVI or temporal analysis. The exact ground radiometric/atmospheric process must be sensor-specific and recorded in provenance.

### 10.2 Integrate the existing health calculations

The following calculations already exist for co-registered floating-point reflectance:

- NDVI;
- EVI;
- GNDVI;
- CVI, defined in this project as `(NIR x Red) / Green^2`;
- VARI;
- excess green;
- RGB brightness.

Older planning material also proposed SAVI and a fixed NDVI/GNDVI/SAVI weighted score. SAVI and that combined score are not implemented in the current ground package. The latest requested MVP instead emphasizes NDVI, EVI, GNDVI, CVI and RGB measurements. The team should either formally remove SAVI from the active contract or add it later using calibrated reflectance; it must not be listed as a current output now.

They currently operate on an in-memory window and correctly reject integer raw DN inputs, invalid reflectance and unstable divisions. Remaining work is to:

- consume the deployed binary crop mask correctly;
- build `crop AND usable AND finite AND valid` analysis masks;
- use the conservative 0.645 crop probability threshold for health analysis unless recalibrated;
- stream complete scenes;
- write index and analysis-mask COG GeoTIFFs;
- validate the JSON observation against the shared schema;
- define a meaningful minimum coverage/pixel threshold.

### 10.3 Add spatial condition and anomaly assessment

The present code summarizes index distributions but does not yet calculate a combined 0–100 condition score, robust spatial anomaly penalty or final condition label.

The MVP should compare valid crop pixels within a crop region or regular grid using robust median/MAD statistics. It must not call this within-field analysis unless field polygons or a field-ID raster actually exist. Each result should expose component scores, coverage and confidence rather than only a single number.

### 10.4 Add temporal warnings only with valid repeat data

Temporal decline requires matching area, grid, registration, radiometry, sensor compatibility, product level, acquisition dates and usable pixels in both observations. NDVI/GNDVI decline can indicate harvest, senescence, rotation, acquisition differences or residual atmosphere as well as stress.

The first forecast should therefore be a trend-based warning, not a prediction of a disease or causal problem. Stronger future prediction would require longer time series, weather, crop stage/type and labeled outcomes.

### 10.5 Implement storage, API and visualization

The repository currently contains only contracts for these components. Remaining implementation includes:

- COGs for raster layers;
- GeoJSON or stable region geometry for summaries;
- SQLite plus disk assets for the demonstration;
- later PostGIS/object storage for production;
- API endpoints for scene ingestion, latest region condition, history and layers;
- authentication, authorization, tenancy and audit controls for B2B operation;
- a Leaflet or MapLibre interface with RGB, unusable, crop probability, indices, anomaly, coverage and history layers.

JSON should contain compact metrics and asset references, never complete pixel arrays.

## 11. Validation and release work remaining

### 11.1 Sentinel validation

- Run all planned validation scenes, not selected examples only.
- Produce the cloud-results table for snow, water, coast, urban areas, bright soil, haze, invalid pixels and borders.
- Obtain or create cloud/shadow reference labels for quantitative false-positive and false-negative evaluation.
- Validate the selected one-NIR contract.
- Use calibrated reflectance appropriate to each ground health calculation.
- Restore a runnable automated test suite and CI gate.

### 11.2 Balkan-1 validation

- Inspect every raw and processed file independently.
- Verify sensor, product level, band identities, units, bit depth, ranges, saturation, timestamps, coordinates and metadata.
- Compare raw reconstruction with processed deliveries.
- Measure band-registration residuals.
- Manually label representative Balkan cloud, shadow and crop regions.
- Test native and downsampled variants.
- Never claim Balkan accuracy from Sentinel or PASTIS evidence.

### 11.3 Payload qualification

- Build a reproducible payload bundle/container without training dependencies.
- Verify the bundle checksum and cold-start behavior.
- Test on the real payload CPU/accelerator with representative huge images.
- Measure latency, RAM, storage, output size, power and failure recovery.
- Verify uplink commands, watchdog behavior and downlink resume.
- Obtain a commercially usable cloud-model license or replacement.

## 12. Recommended progression and exit criteria

### Priority 0 — resolve contracts

1. Confirm that the L1B ceiling applies to the Balkan onboard raw branch and document the exact L1B deliverable.
2. Select one NIR contract for both models.
3. Replace the 60% cloud gate and separate recommendations with one decision specification.
4. Make one payload/downlink JSON schema canonical.
5. Obtain payload runtime and downlink budgets.

**Exit criterion:** no conflicting band, product-level, decision or schema definitions remain.

### Priority 1 — complete the pre-downlink control plane

1. Add the centralized mission controller.
2. Add the uplink command envelope, validation and state machine.
3. Add a portable downlink manifest and package builder.
4. Add enforced time, memory, storage and output limits.

**Exit criterion:** a command can deterministically produce discard, insufficient-data or downlink-ready status without manual threshold decisions.

### Priority 2 — complete Balkan-1 processing

1. Implement raw-to-L1B reconstruction.
2. Implement calibration and the single-NIR adapter.
3. Validate reconstruction, clouds and crops on real Balkan data.

**Exit criterion:** raw and vendor-processed Balkan captures produce traceable, comparable results within defined registration and runtime tolerances.

### Priority 3 — qualify the payload and downlink

1. Deploy the minimal bundle to representative/actual payload hardware.
2. Benchmark huge scenes under hard budgets.
3. Validate uplink, failure recovery, packaging and downlink transfer.
4. Resolve the cloud-weights commercial license.

**Exit criterion:** an accepted scene is actually transferred and independently verified on the ground.

### Priority 4 — complete the ground MVP

1. Connect ingestion and COG writers.
2. Integrate health measurements with the binary crop mask.
3. Add spatial anomaly scoring and cautious labels.
4. Add SQLite, lightweight API and map.
5. Add temporal warnings only after compatible repeated observations exist.

**Exit criterion:** a downlinked scene appears in the map with auditable crop coverage, measurements, condition explanation, quality/confidence and downloadable artifacts.

## 13. Challenge-level assessment

The repository demonstrates substantial progress toward the 60% prototype level, but the complete end-to-end mission is not yet demonstrated because the official mission decision, downlink transfer and ground visualization are absent.

- **60%:** partially demonstrated; complete only after the controller, downlink package and ground result are connected.
- **75%:** not yet demonstrated; the code has not been qualified on the EnduroSat payload computer.
- **90%:** not yet demonstrated; real EnduroSat/Balkan imagery has not been validated.
- **100%:** not yet demonstrated; raw Balkan reconstruction through the agreed L1B ceiling is not implemented.

## 14. Conclusion

The architecture remains sound: cloud and crop relevance belong on the payload; richer crop-condition analysis, history, storage and visualization belong on the ground. The project has moved beyond isolated models and now has a real, bounded-memory cloud-to-crop execution path with pinned artifacts and auditable outputs.

The current endpoint, however, is `CROP_COMPLETE`, not downlink. The next milestone is not another model-training cycle. It is to close the mission-control gap: finalize the L1B and one-NIR contracts, reconstruct and adapt Balkan-1 data, centralize the decision, accept uplink commands, enforce runtime limits, and build a portable downlink package. Only after that package is received and verified on the ground should the project claim an end-to-end onboard-to-ground demonstration.

## Appendix A — primary repository evidence

- [`README.md`](README.md) — selected model and training/data context.
- [`COMPONENTS.md`](COMPONENTS.md) — component ownership and storage policy.
- [`payload/README.md`](payload/README.md) — payload entry point and stated limits.
- [`payload/src/prithvi_payload/pipeline.py`](payload/src/prithvi_payload/pipeline.py) — current executable stage boundary.
- [`payload/src/prithvi_payload/scene_intake.py`](payload/src/prithvi_payload/scene_intake.py) — current sensor and band intake contract.
- [`payload/src/prithvi_payload/cloud_executor.py`](payload/src/prithvi_payload/cloud_executor.py) — streamed cloud outputs and unusable-area recommendation.
- [`payload/src/prithvi_payload/crop_stage.py`](payload/src/prithvi_payload/crop_stage.py) — current independent 60% cloud gate.
- [`payload/src/prithvi_payload/crop_executor.py`](payload/src/prithvi_payload/crop_executor.py) — windowed crop outputs and usable-pixel fraction.
- [`payload/verification.json`](payload/verification.json) — retained execution and integration evidence.
- [`payload/DEPLOYMENT_MANIFEST.yaml`](payload/DEPLOYMENT_MANIFEST.yaml) — package contents, missing components and license block.
- [`configs/selected_model.yaml`](configs/selected_model.yaml) — selected checkpoint and calibration evidence.
- [`ground/src/prithvi_ground/health.py`](ground/src/prithvi_ground/health.py) — currently implemented health measurements.
- [`ground/health_analysis_contract.md`](ground/health_analysis_contract.md) — scientific and radiometric contract.
- [`shared/schemas/inference_result.schema.json`](shared/schemas/inference_result.schema.json) — intended payload schema, not yet used by canonical runs.

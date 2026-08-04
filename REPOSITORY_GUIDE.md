# ViTA crop-intelligence repository guide

This document explains the repository from the top level down to its individual tracked files. It is written for readers who do not need to be programmers. Technical terms are introduced where they matter, and the active production path is separated from training tools, experiments, tests, and generated data.

The repository has two related purposes:

1. train and select a crop-segmentation model; and
2. operate a payload-to-ground pipeline that acquires or accepts an image, removes unusable pixels such as clouds, identifies likely cropland, estimates spectral crop condition, and publishes a small web-ready result.

The current operational inputs are:

- Sentinel-2 surface-reflectance imagery acquired from Google Earth Engine; and
- already-preprocessed Balkan-1 five-band GeoTIFFs, with a validated sensor-specific calibration sidecar.

Raw Balkan-1 L0/L1 processing is deliberately outside the active model pipeline. The payload models must receive a georeferenced, radiometrically meaningful product—not raw detector values or display-stretched colors.

## 1. The two-second Jetson requirement

### Short answer

A **warm, bounded science inference** can potentially be brought below two seconds on a sufficiently capable Jetson Orin after significant optimization and empirical validation. The current **complete full-scene pipeline cannot** meet two seconds.

The distinction is essential:

| Possible target | Under two seconds? | Reason |
|---|---:|---|
| Full 10,360 × 13,389 Balkan image, including reprojection, all condition rasters, WebP/PNG creation, and file packaging | No | This is about 139 million source pixels. CPU raster operations and file I/O dominate, and the two GPU model calls alone currently exceed two seconds on the development RTX GPU. |
| Earth Engine search and download plus inference | No | Network/provider latency is variable and cannot be controlled as a real-time payload SLA. |
| Raw Balkan L0/L1A/L1B/L1C processing plus both models | No, not under the present design | Registration, geolocation, atmospheric correction, and product construction are separate substantial workloads. |
| Preprocessed image already in memory, fixed analysis grid, resident engines, science decision only | Potentially | This removes network, process startup, model loading, full-resolution reprojection, report rasters, and image encoding from the critical path. |
| Full publication produced after the fast science result | Yes as an asynchronous follow-up, but not inside the two-second SLA | The immediate decision can be emitted first; large diagnostic and display products can finish later. |

No honest guarantee is possible until the exact Jetson module, memory size, JetPack release, power envelope, thermal solution, input dimensions, and meaning of “finished” are fixed.

### What the current timing proves

The most representative local record is the full Balkan-1 scene `3458`, processed on an NVIDIA GeForce RTX 3060 Laptop GPU. The source was 1,226,450,444 bytes and 10,360 × 13,389 pixels.

| Stage | Total | GPU inference inside the stage | Important non-model work |
|---|---:|---:|---|
| Cloud | 22.324 s | 3.400 s | 14.118 s preparing the 10 m grid and 2.236 s reprojecting masks |
| Crop | 55.315 s | 0.967 s | Balkan calibration, tile reads, blending, full-resolution raster publication, and preview work |
| Condition | 84.956 s | none | Three windowed raster passes, eight health layers, condition products, compression, and a 10.101 s preview |
| Downlink | 23.095 s | none | Reading large rasters, resizing, color rendering, WebP/PNG encoding, checksums, and JSON packaging |
| Listed stages combined | 185.689 s | 4.366 s | Most time is outside neural-network inference |

The end-to-end manual run was longer again because intake, model loading, setup, and other script work are outside those stage totals. This means “turn CUDA on” is not the solution: CUDA is already used for both neural networks, but most work still happens on the CPU, through GDAL/rasterio, or through storage.

### The only useful definition of the fast path

For engineering and acceptance, define the requirement as:

> From a preprocessed, calibrated image tile already available in memory to a cloud mask, crop-probability mask, crop/condition summary, and small overlay representation, with both model engines already loaded, at a fixed maximum input size, measured at p95 latency after thermal soak.

Exclude these from the two-second clock and measure them separately:

- boot and container startup;
- Python import and model/engine loading;
- Earth Engine query and network download;
- raw sensor processing and atmospheric correction;
- writing full-resolution GeoTIFF diagnostic layers;
- full-size PNG/WebP creation;
- downlink transmission, ground ingestion, and browser rendering.

If mission requirements say any of those must be included, the two-second requirement must either be relaxed or the product size must be drastically reduced.

### Required hardware decision

Record all of the following before optimization starts:

- exact module: Orin Nano, Orin NX 8/16 GB, or AGX Orin 32/64 GB;
- production module versus developer kit;
- available power mode and sustained wattage;
- cooling, expected ambient temperature, and enclosure airflow;
- JetPack/L4T, CUDA, cuDNN, TensorRT, PyTorch, rasterio/GDAL, and Python versions;
- storage medium and measured sequential/random bandwidth;
- camera/DMA path and whether the image can arrive in pinned or CUDA-addressable memory;
- maximum analysis width, height, bands, data type, and ground sample distance.

For this workload, an Orin NX 16 GB is a reasonable minimum development target; an AGX Orin gives more performance and memory margin. That is a planning recommendation, not a latency guarantee. A full 139-million-pixel workflow is unsuitable for a two-second goal on either system.

NVIDIA’s current JetPack must be matched to the actual Orin hardware and base image. As of this guide’s date, NVIDIA lists JetPack 7.2/Jetson Linux 39.2 for the Orin family, but the repository intentionally does not guess the deployed board’s installed stack. TensorRT engine files are hardware/software specific and should be built or validated on the target class of device.

### Target execution architecture

The fast path should become:

```text
preprocessed calibrated image in memory
                │
                ▼
one shared 10 m analysis grid (created once)
                │
       ┌────────┴────────┐
       ▼                 ▼
TensorRT cloud       TensorRT crop
engine               engine
       │                 │
       └────────┬────────┘
                ▼
GPU-resident mask fusion and vegetation indices
                │
                ▼
small summary + compact overlay representation  ← two-second boundary
                │
                ▼
asynchronous GeoTIFF diagnostics, preview encoding,
checksums, bundle publication, and downlink
```

The current implementation repeatedly crosses three expensive boundaries: native Balkan resolution ↔ 10 m grid, GPU ↔ CPU, and memory ↔ compressed files. The optimized path should cross each boundary as little as possible.

### Optimization work, in the correct order

#### Phase A — freeze the contract and establish a trustworthy baseline

1. Select the exact Jetson and its JetPack-compatible base image. Do not choose a generic CUDA image.
2. Fix a maximum science grid. A two-second service cannot accept arbitrary image dimensions. Start with 512 × 512 and test 1024 × 1024 only if the smaller target passes.
3. Add p50, p95, and p99 timings around upload/copy, preprocessing, each engine, mask fusion, condition calculation, and publication.
4. Benchmark after model warm-up and after at least 30 minutes of thermal soak.
5. Capture `tegrastats`, power mode, clocks, temperature, memory use, and throttling with every accepted benchmark.
6. Preserve the present PyTorch outputs as the numerical reference: cloud class agreement, unusable mask, crop probabilities, binary crop mask, health indices, score, and final bundle.

The repository already has `vita-payload-benchmark`, but its timing fields need to be extended for GPU preprocessing, transfers, engine enqueue, synchronization, and fast-versus-asynchronous publication.

#### Phase B — remove avoidable work without changing science

1. Keep the payload service persistent. `PayloadRuntime.initialize()` already loads models once; the manual Balkan script currently starts a new process and therefore cannot represent production latency.
2. Verify model hashes and calibration identity at service initialization or first ingest, then cache the result. Do not hash a 383 MB checkpoint or a 1.2 GB source on every fast-path call.
3. Allocate fixed reusable input/output buffers at startup. Use pinned host memory and non-blocking copies where a host copy remains necessary.
4. Reproject the source only once to the common 10 m analysis grid. Cloud, crop, masks, health indices, and condition scores should stay on that grid.
5. Do not reproject every intermediate result to the 1.5 m Balkan source grid. Preserve geospatial transform metadata so the compact result can still be located correctly.
6. Stop writing class-score GeoTIFFs, eight health GeoTIFFs, crop probability/confidence GeoTIFFs, and several condition GeoTIFFs on the real-time path. Make those optional asynchronous diagnostics.
7. Create at most one small display preview. Never read a full-resolution product merely to reduce it immediately to 1600 pixels.
8. Replace Matplotlib in the payload fast path. It is appropriate for diagnostics, not a hard real-time renderer.

This phase should produce the largest end-to-end improvement even before changing either model.

#### Phase C — compile and optimize both models

1. Export each selected model to ONNX with fixed, deployment-relevant shapes.
2. Confirm ONNX output equivalence against PyTorch on a versioned Sentinel-2 and Balkan-1 corpus.
3. Build TensorRT FP16 engines and benchmark them with `trtexec` on the target Jetson.
4. Use fixed shapes and fixed batches where possible; dynamic shapes reduce optimization opportunities and complicate memory planning.
5. Capture per-layer profiles and an Nsight Systems trace. Optimize the measured bottleneck, not an assumed one.
6. Try INT8 only after FP16 is correct. Build a representative calibration/quantization set containing both sensors, clear/cloud/thin-cloud/shadow cases, crop/non-crop cases, seasons, illumination levels, and Balkan calibration variants.
7. Compare more than pixel accuracy. Require cloud-class IoU/F1, unusable-mask agreement, crop probability error, crop precision/recall, threshold stability, condition-score difference, and final decision agreement.
8. Keep sensitive operations such as normalization and softmax in a safer precision if FP16/INT8 changes decisions.

The cloud system is an ensemble of two networks. If its TensorRT engine still cannot meet the budget, the likely production solution is a **single distilled cloud model** trained to reproduce the ensemble, followed by validation against real Sentinel and Balkan data. Silently dropping one ensemble member would change the selected model without evidence and is not acceptable.

The crop model is Prithvi EO 2.0 100M plus UPerNet. If the fixed-shape TensorRT version remains too slow, reaching two seconds will require distillation, pruning, or a smaller deployment backbone/head. That is a new trained artifact and must pass the repository’s explicit model-replacement policy.

#### Phase D — move post-processing to accelerated kernels

The current cloud post-processing uses SciPy connected-component labeling and dilation on the CPU. The condition stage uses NumPy, rasterio, repeated compressed writes, and several raster passes. For the fast path:

- implement class argmax, invalid-mask fusion, morphology, crop thresholding, confidence, and health indices as CUDA/TensorRT/CuPy/NPP operations;
- retain arrays on the GPU between models and calculations;
- combine the four scored indices in one kernel or fused tensor expression;
- use GPU reductions for counts, means, histograms/quantiles, median estimates, and anomaly percentages;
- copy only compact statistics and the final reduced overlay to the CPU;
- make exact diagnostic rasters a background task fed from the shared result, not prerequisites for the decision.

If connected-component cleanup is scientifically necessary, benchmark a GPU implementation. If it is removed or approximated, revalidate the cloud mask because small regions affect unusable coverage and therefore every later stage.

#### Phase E — acceptance and operational proof

A release passes only if all of these hold:

- p95 warm science latency is below 2.0 seconds at the chosen maximum size;
- no thermal or power throttling occurs during a representative mission-length soak;
- peak memory stays below a defined safe fraction of physical memory;
- repeated identical inputs produce equivalent decisions;
- Sentinel and Balkan regression suites meet agreed scientific tolerances;
- error paths reject invalid radiometry, bands, grids, calibrations, and oversized inputs safely;
- asynchronous product generation cannot block the real-time worker or exhaust storage;
- the original PyTorch path remains available as an offline reference until the TensorRT path is accepted.

### A provisional latency budget

This is a design target, not a prediction:

| Warm fast-path item | Target budget |
|---|---:|
| Input/grid preparation and radiometric mapping | 100 ms |
| Cloud engine | 350 ms |
| Crop engine | 650 ms |
| GPU mask/condition calculations and reductions | 250 ms |
| Compact result and overlay preparation | 200 ms |
| Scheduling/copy/safety margin | 350 ms |
| Total | 1,900 ms |

Meeting this likely requires AGX Orin or Orin NX at an appropriate sustained power mode, a bounded analysis grid, TensorRT, no large file writes, and probably cloud-model distillation. A 1024 × 1024 grid may not fit this budget with the present models; 512 × 512 is the sensible first acceptance size.

## 2. System map in plain language

Think of the system as six departments with strict responsibilities:

| Area | Plain-language responsibility | Must not do |
|---|---|---|
| `src/` + `configs/` + `training/` | Develop, train, compare, and select crop models | Ship training machinery on the satellite payload |
| `payload/` | Run trusted cloud, crop, and condition science and produce a compact bundle | Trust raw/unvalidated inputs or depend on the ground UI |
| `shared/` | Define common vocabulary, formulas, thresholds, and message shapes | Contain sensor- or deployment-specific orchestration |
| `integration/` | Connect mission commands, payload jobs, download, and ground upload | Recalculate scientific outputs |
| `ground/` | Validate, store, list, and display payload results | Secretly change cloud/crop/condition results |
| `deployment/` + compose files | Package the correct components with least privilege | Mix training data, secrets, and ground code into the flight image |

This separation is why the architecture has more than one Python package. It makes the flight boundary auditable, keeps large data and secrets out of Git, prevents the dashboard from becoming a second science implementation, and allows training to evolve without changing the deployed model automatically.

## 3. The active end-to-end pipeline

### Sentinel-2 coordinate-driven flow

```text
bounding box + dates + cloud-selection policy
                       │
                       ▼
mission CLI validates the command
                       │
                       ▼
payload API queues one GPU job
                       │
                       ▼
Earth Engine finds Sentinel-2 SR candidates
                       │
                       ▼
each candidate is downloaded to a fixed 10 m UTM grid
                       │
                       ▼
scene intake → cloud model → payload cloud decision
                       │
         accepted candidate only
                       ▼
crop model → crop mask → condition analysis
                       │
                       ▼
scene.webp + condition.png + scene.json
                       │
                       ▼
integration client verifies downloads and uploads to ground
                       │
                       ▼
ground verifies checksums/schema, installs atomically, serves dashboard
```

Earth Engine metadata cloud percentage is used to order candidates, not as the final cloud truth. The payload runs its own cloud model on downloaded pixels and decides from that result. Candidate attempts are bounded so the service cannot search forever.

### Preprocessed Balkan-1 flow

```text
preprocessed 5-band georeferenced GeoTIFF
+ source-bound calibration JSON
                       │
                       ▼
manual Balkan pipeline runner
                       │
                       ▼
intake validates bands, grid, scale, calibration identity
                       │
              ┌────────┴────────┐
              ▼                 ▼
cloud input mapped         crop input calibrated
to a 10 m R/G/NIR grid     toward Sentinel model units
              │                 │
              ▼                 ▼
         cloud masks        Prithvi crop map
              └────────┬────────┘
                       ▼
condition analysis → compact bundle → optional ground ingest
```

The original preprocessed Balkan image is used for the dashboard base image. Color stretching happens only when making the WebP; it does not modify the scientific source raster or feed display colors into the models.

### Stage gates and why they exist

1. **Intake** confirms that the scene is truly usable: supported sensor, dimensions, CRS, affine transform, required logical bands, acquisition time, numerical scale, and calibration identity.
2. **Cloud planning** translates the intake report into a precise cloud-model contract. Planning and execution are separate so the contract can be inspected and tested.
3. **Cloud execution** predicts clear, thick cloud, thin cloud, and shadow; creates an invalid mask and a combined unusable mask; and decides whether useful clear area remains.
4. **Crop planning** refuses crop inference when total predicted cloud is 60% or more, selects the sensor-specific threshold and radiometric adapter, and records all model inputs.
5. **Crop execution** runs overlapping 224 × 224 tiles, blends probabilities at tile edges, and forces all cloud/shadow/invalid pixels to no-data.
6. **Condition analysis** only analyzes pixels that are crop, usable, sufficiently high-confidence, finite, and radiometrically plausible.
7. **Downlink packaging** creates exactly three portable files and embeds provenance, quality, limitations, geospatial information, and checksums.

The pipeline is fail-closed: missing or contradictory evidence stops a stage instead of guessing.

## 4. Model inference

### Cloud model: OmniCloudMask V4

The selected cloud classifier is OmniCloudMask 1.7.1, model version 4. It is an ensemble of EdgeNeXt Small and RegNetY-004 weights. Both components and the combined identity have pinned SHA-256 checksums.

Input:

- red, green, and near-infrared bands in that exact order;
- finite, strictly positive pixels define the valid model footprint;
- Sentinel pixels are already on the 10 m acquisition grid;
- Balkan pixels are reprojected to a 10 m UTM analysis grid first.

Execution:

- FP32 CUDA in the present configuration;
- 1000 × 1000 model patches with 300-pixel overlap;
- ensemble scores are normalized and combined;
- the highest score becomes the semantic class.

Semantic output values:

| Value | Meaning |
|---:|---|
| 0 | Clear |
| 1 | Thick cloud |
| 2 | Thin cloud |
| 3 | Cloud shadow |

Post-processing first combines thick cloud, thin cloud, and shadow. Connected regions smaller than 16 pixels are removed; the remaining unusable mask is expanded by two pixels; invalid source pixels are then included. The decision is based on **all unusable pixels**, not just cloud pixels:

- up to 30% unusable: `PROCESS`;
- above 30% but below 80%: `PROCESS_CLEAR_AREAS`;
- 80% or more: `REJECT`.

Cloud results are authoritative for masking. Earth Engine’s metadata is candidate-selection information only.

### Crop model: Prithvi EO 2.0 100M + UPerNet

The selected crop model has a Prithvi EO 2.0 100M temporal/location-aware backbone and a UPerNet segmentation decoder. It is a two-class model: non-crop and crop.

Its weight-only checkpoint is 382,645,915 bytes with a pinned SHA-256 digest. The payload never automatically replaces it with a newer experiment.

Input tensor:

```text
[batch, bands, time, height, width]
[B,     4,     1,    224,    224]
```

Bands are `BLUE`, `GREEN`, `RED`, and `NIR_NARROW`. The model also receives acquisition year/day-of-year and tile-center latitude/longitude. Values are normalized with the exact means and standard deviations used during training.

The current PyTorch path uses `torch.inference_mode()` and CUDA FP16 autocast. Softmax converts two output logits into crop probability. The executor uses overlapping tiles with a 16-pixel halo, batches four tiles, and blends probabilities with edge-tapered weights to prevent seams.

Operational thresholds:

| Sensor/use | Threshold | Meaning |
|---|---:|---|
| Sentinel crop classification | 0.49 | Probability at or above this becomes crop |
| Sentinel condition eligibility | 0.645 | Stricter confidence required before health scoring |
| Balkan crop classification | 0.30 | Sensor-specific operational threshold after validated calibration |
| Balkan condition eligibility | 0.30 | Current sensor-specific threshold for condition evidence |

Output rasters use exact values: binary `0` non-crop, `1` crop, `255` unusable; floating rasters use `-9999` as no-data. Preview smoothing is display-only and never changes those values.

### Why Balkan needs calibration

The crop model learned Sentinel-like reflectance distributions. A Balkan-1 band may cover a similar wavelength but still have different response, gain, offset, atmosphere, and processing. Feeding raw Balkan numbers into Sentinel normalization can produce plausible-looking but wrong probabilities.

The calibration sidecar contains monotonic mapping curves from Balkan values to the model’s expected units. The ground-side fitter:

1. takes co-registered Balkan and Sentinel reference pixels;
2. divides them into spatial fit and held-out blocks;
3. bins source values and finds robust target medians;
4. applies isotonic/PAVA fitting so the mapping cannot run backwards;
5. reduces the curve to interpolation knots;
6. accepts it only with enough held-out pixels and correlation evidence; and
7. binds the sidecar to the exact Balkan source SHA-256 and byte size.

At inference, a simple interpolation through those knots maps each band. Sentinel reference imagery is not needed on the payload. A sidecar for one source cannot silently be used with another source.

## 5. Masks, health indices, and condition scoring

### The analysis mask

A pixel enters health analysis only when every statement is true:

```text
crop binary value is 1
AND cloud unusable mask is 0
AND crop probability is finite and above the sensor health threshold
AND source band masks say the pixel exists
AND calibrated reflectance is finite and between -0.2 and 2.0
```

This mask is the main protection against reporting vegetation statistics for cloud, shadow, image edges, missing data, non-crop land, or weak crop predictions.

### Per-pixel health measurements

The system calculates eight transparent spectral features from calibrated blue (`B`), green (`G`), red (`R`), and near-infrared (`N`) reflectance:

| Layer | Formula | Intuition |
|---|---|---|
| NDVI | `(N - R) / (N + R)` | General green vegetation vigor |
| GNDVI | `(N - G) / (N + G)` | Greenness using the green band |
| EVI | `2.5(N - R) / (N + 6R - 7.5B + 1)` | Vegetation signal with blue/red correction |
| SAVI | `1.5(N - R) / (N + R + 0.5)` | Vegetation signal with soil adjustment |
| CVI | `(N × R) / G²` | Chlorophyll-related ratio |
| VARI | `(G - R) / (G + R - B)` | Visible-band greenness |
| Excess green | `2G - R - B` | Green dominance |
| RGB brightness | `(B + G + R) / 3` | Visible brightness |

Near-zero denominators become no-data instead of infinity. The first four indices drive the prototype condition score; all eight are recorded as explainable measurements.

### Condition score

Each scored index is linearly mapped to 0–100 within a prototype reference range:

- NDVI: 0.20 to 0.80, weight 0.40;
- GNDVI: 0.15 to 0.70, weight 0.25;
- EVI: 0.10 to 0.80, weight 0.20;
- SAVI: 0.15 to 0.80, weight 0.15.

The pixel condition score is their weighted average. The region’s absolute score is:

```text
0.70 × median pixel score + 0.30 × lower-quartile pixel score
```

The system then looks for pixels substantially worse than their own region. It uses median absolute deviation, a robust statistic that is less distorted by extreme pixels. A relative anomaly requires both a robust deficit z-score of at least 2.5 and a score deficit of at least 10 points. The anomaly fraction can subtract up to 20 points.

Labels are:

- 75–100: `Nominal`;
- 55–<75: `Watch`;
- 35–<55: `Moderate anomaly`;
- below 35: `High anomaly`.

If at least 5% of analyzed pixels are alerts, an otherwise `Nominal` result is reduced to `Watch`. Fewer than 64 valid pixels or less than 0.1% coverage produces `INSUFFICIENT_DATA` rather than an unreliable score.

Evidence quality combines valid pixel count, area coverage, and average crop probability. It indicates how much trustworthy evidence exists; it is not a probability that the agronomic interpretation is correct.

The condition result is a **screening priority, not a disease diagnosis**. A single satellite image cannot reliably separate stress from harvest, senescence, fallow ground, crop type, cultivar, or growth stage without additional evidence.

### Why condition processing is currently slow

The payload implementation is memory-bounded rather than latency-optimized. It processes 512-pixel windows and writes each health layer and condition layer to compressed GeoTIFF. It then makes additional passes for robust statistics and spatial anomalies, and renders a diagnostic preview. This is safe for very large scenes but expensive. The latest full Balkan run used 567 first-pass windows and about 85 seconds.

## 6. Outputs and ground presentation

### Full payload working products

During processing, the payload can create:

- semantic cloud mask;
- combined unusable mask;
- cloud class-score rasters;
- crop probability, binary, and confidence rasters;
- eight health-index rasters;
- condition score, valid-crop, robust deficit, anomaly, low-vigor, and alert rasters;
- diagnostic PNG previews;
- stage-plan and provenance JSON files.

These are valuable for verification but too large for routine satellite downlink and inappropriate for a two-second real-time path.

### Compact downlink contract

The operational bundle contains exactly:

| File | Purpose |
|---|---|
| `scene.webp` | Portable true-color view of the supplied source scene |
| `condition.png` | Transparent color overlay showing valid crop condition |
| `scene.json` | Machine-readable measurements, provenance, map grid, asset checksums, legends, warnings, and limitations |

Images are at most 1600 pixels on their longest side. WebP quality is 82. The overlay only colors valid, clear, high-confidence crop pixels. An interaction grid, normally up to 16 × 16 with minimum 32-pixel cells, provides dashboard-friendly regional summaries without shipping the scientific rasters.

### Ground trust boundary

The ground catalog does not recompute science. It verifies:

- the expected three files and supported media types;
- schema and identifiers;
- relative rather than absolute paths;
- checksums and byte sizes;
- matching image dimensions;
- safe geospatial and percentage values;
- idempotency—re-uploading an identical scene is safe, while conflicting content is rejected.

Installation is atomic, and SQLite supplies searchable history. The API and static web application then expose regions, scenes, images, overlays, metrics, and interaction cells.

## 7. Data and security rules

- `data/`, `outputs/`, most of `runtime/`, `testing/runs/`, model weight binaries, caches, `.env` files, and credentials are ignored by Git.
- Code, small manifests, schemas, expected hashes, and documentation are tracked.
- Raw Balkan folders belong under `data/balkan1/raw/`; supplied processed scenes under `data/balkan1/preprocessed/`; derived L1A under `data/balkan1/processed/l1a/`; calibration sidecars under the ignored Balkan data tree.
- Payload output is written under `runtime/payload/` or `testing/runs/`, never back into the source image folder.
- Secrets are mounted at runtime and must never enter container layers, logs, bundle JSON, Git, or the dashboard.
- The payload container intentionally excludes training data/code and ground code. The ground container intentionally excludes model weights and Earth Engine credentials.
- API ports bind to loopback in the supplied compose files. Remote use should go through a controlled tunnel or authenticated network boundary.

## 8. How to read the repository efficiently

For a new reader, use this order:

1. `README.md` for supported operations and commands.
2. `COMPONENTS.md` for ownership boundaries.
3. This guide for the complete mental model.
4. `payload/src/prithvi_payload/pipeline.py` for manual stage order.
5. `payload/src/prithvi_payload/runtime.py` and `service.py` for production Sentinel jobs.
6. `scene_intake.py`, `cloud_stage.py`, `cloud_executor.py`, `crop_stage.py`, and `crop_executor.py` in execution order.
7. `shared/src/prithvi_shared/health.py` and `condition.py` for the scientific formulas.
8. `downlink.py`, then `ground/catalog.py`, `ground/api.py`, and the web files for delivery and display.
9. `payload/models/*.yaml`, `shared/*.yaml`, and JSON schemas for immutable contracts.
10. Only then read `src/prithvi_crop/`, training configs, and experiment records if model development is relevant.

The next section names every tracked file and explains its role.

## 9. Complete tracked-file atlas

Generated data, model binaries, local credentials, virtual environments, caches, and runtime results are intentionally not enumerated as source files. Their directory roles are described above and in the relevant `.gitignore` files.

### Top-level files and directories

| Path | What it does and why it exists |
|---|---|
| `.dockerignore` | Prevents local data, secrets, outputs, Git history, caches, and other unnecessary material from entering Docker build contexts. This is both a security and image-size control. |
| `.env.ground.example` | Documents the non-secret environment variables expected by the ground service. Copy it locally to `.env.ground`; do not commit the local copy. |
| `.env.payload.example` | Development template for payload image, base image, Earth Engine project, CUDA requirement, paths, and queue settings. |
| `.env.payload.production.example` | Production-oriented payload environment template with stronger operational expectations and no real credential values. |
| `.gitattributes` | Controls Git treatment of text and binary files, helping preserve consistent line endings and avoid unsuitable binary diffs. |
| `.gitignore` | The central rule preventing imagery, model binaries, credentials, outputs, runtime databases, caches, and virtual environments from being committed. |
| `README.md` | Main operator/developer entry point: repository purpose, supported sensor paths, environment setup, commands, deployment, and validation status. |
| `COMPONENTS.md` | Defines ownership and trust boundaries among training, payload, shared contracts, integration, and ground. |
| `PROJECT_STATUS_REPORT.md` | Evidence-oriented snapshot of what is implemented, validated, provisional, or still blocked. It prevents prototype claims from being mistaken for completed flight qualification. |
| `REPOSITORY_GUIDE.md` | This document: plain-language architecture, Jetson plan, science explanation, and complete source atlas. |
| `Makefile` | Short aliases for common setup, validation, testing, training, and pipeline commands on systems with `make`. It is convenience, not hidden business logic. |
| `pyproject.toml` | Root training/development Python package definition, dependencies, console commands, package discovery, and Ruff lint configuration. |
| `compose.full-local.yaml` | Starts both local payload and ground services together with persistent runtime volumes and loopback-only ports. |
| `compose.ground.yaml` | Builds/runs only the ground API and dashboard, with its own storage volume and optional upload key. |
| `compose.payload.local.yaml` | Builds/runs the payload on a local x86 CUDA machine for integration and regression work. |
| `compose.payload.jetson.yaml` | Jetson payload topology: NVIDIA runtime, persistent job/outbox/log volumes, secrets mount, health checks, bounded concurrency, and loopback API. A target-compatible base image must still be selected. |
| `compose.payload.production.yaml` | Compatibility production compose name that requires the operator to declare `local-x86` or `jetson`; newer Jetson work should prefer the dedicated Jetson file. |
| `data/.gitkeep` | Keeps the otherwise ignored data directory present in a fresh clone. Real datasets and images remain local. |
| `outputs/.gitkeep` | Keeps the ignored training/output root present without committing checkpoints or generated artifacts. |
| `secrets/.gitkeep` | Keeps the local secrets mount directory present while the actual credential files remain ignored. |
| `runtime/.gitkeep` | Keeps the machine-local runtime root present. |
| `runtime/ground/.gitkeep` | Placeholder for the ground SQLite database and installed scenes. |
| `runtime/payload/jobs/.gitkeep` | Placeholder for atomic payload job state, inputs, and results. |
| `runtime/payload/logs/.gitkeep` | Placeholder for protected payload logs. |
| `runtime/payload/outbox/.gitkeep` | Placeholder for payload products awaiting transfer. |

### `configs/` — training experiments and model selection

These files affect training and selection; the payload uses its own pinned deployment manifests.

| Path | What it does and why it exists |
|---|---|
| `configs/prithvi_4band_head_only.yaml` | Early experiment that trains only the segmentation head while keeping the Prithvi backbone frozen. It establishes a cheap baseline. |
| `configs/prithvi_4band_single_frame.yaml` | Single-date multiclass crop/land-cover experiment using four optical bands and a deployment-oriented model score. |
| `configs/prithvi_4band_single_frame_binary.yaml` | Selected binary crop/non-crop training configuration. It uses mixed precision, explicit checkpoints, and the two-class task now deployed. |
| `configs/prithvi_4band_augmented_refine.yaml` | Refinement experiment with training-only augmentation intended to improve robustness without altering inference inputs. |
| `configs/prithvi_4band_europe_replay.yaml` | Mixed replay experiment combining the original crop dataset with harmonized European PASTIS samples to test geographic generalization. |
| `configs/crop_binary_calibration.yaml` | Records validation-set threshold selection. It distinguishes the 0.49 classification threshold from the stricter 0.645 Sentinel health-evidence threshold. |
| `configs/selected_model.yaml` | Human-reviewed training-side selection record: source checkpoint, validation metrics, thresholds, preserved alternatives, and explicit replacement policy. |

### `deployment/` — container construction and deployment evidence

| Path | What it does and why it exists |
|---|---|
| `deployment/README.md` | Explains payload/ground image boundaries, safe base-image selection, secrets, local versus Jetson deployment, health checks, and regression procedure. |
| `deployment/payload/Dockerfile` | Builds the payload image from an operator-supplied CUDA/JetPack-compatible base, installs guarded dependencies, copies only payload/shared code and pinned model assets, and runs as a non-root user. |
| `deployment/payload/constraints.txt` | Protects ABI-sensitive GPU, NumPy, geospatial, and computer-vision package versions from unintended pip replacement. |
| `deployment/payload/entrypoint.sh` | Creates/verifies writable runtime directories, checks mounted credentials and CUDA policy, then starts the requested payload command. |
| `deployment/payload/stack_guard.py` | Records and compares the base image’s CUDA/PyTorch/OpenCV/NumPy stack before and after dependency installation so a container build fails if pip damaged the vendor GPU stack. |
| `deployment/ground/Dockerfile` | Builds a small CPU-only ground image containing shared contracts, ground API/catalog, and web assets—but no models, Earth Engine credentials, or training code. |
| `deployment/ground/entrypoint.sh` | Creates the ground store with safe ownership/permissions and then launches the API command. |
| `deployment/local_regression_reference.json` | Frozen known-good Sentinel local run: command, selected scene, measurements, artifact hashes, timings, and environment. It detects deployment regressions; it is not an accuracy benchmark. |

### `ground/` — verified storage, API, and dashboard

| Path | What it does and why it exists |
|---|---|
| `ground/README.md` | Ground component overview, install/serve/ingest commands, API routes, storage policy, and trust boundary. |
| `ground/api/README.md` | Documents versioned ground HTTP endpoints, upload authentication, and expected request/response behavior. |
| `ground/storage/README.md` | Explains the on-disk scene layout, SQLite catalog, atomic installation, checksums, and idempotency. |
| `ground/visualization/README.md` | Explains how the dashboard uses the source WebP, transparent condition PNG, legend, metrics, and interaction grid without recomputing science. |
| `ground/pyproject.toml` | Defines the standalone ground package, dependencies, `vita-ground-ingest`/`vita-ground-serve` commands, and bundled web files. |
| `ground/requirements.txt` | Installable ground dependency list for environments that do not install from the package metadata. |
| `ground/src/prithvi_ground/__init__.py` | Public package surface for bundle validation and catalog operations. |
| `ground/src/prithvi_ground/catalog.py` | Core ground trust logic: validates the exact bundle contract, safe paths, checksums, image dimensions/types, manifests, and percentages; installs bundles atomically; indexes regions/scenes in SQLite; rejects conflicting uploads. |
| `ground/src/prithvi_ground/api.py` | FastAPI application exposing health, upload, region/scene history, manifest, images, and dashboard routes. Optional constant-time API-key comparison protects uploads. |
| `ground/src/prithvi_ground/web/index.html` | Static dashboard structure and accessible UI containers. |
| `ground/src/prithvi_ground/web/styles.css` | Responsive visual design for scene cards, map/overlay presentation, metrics, legends, status, and loading/error states. |
| `ground/src/prithvi_ground/web/app.js` | Browser controller that calls the ground API, selects regions/scenes, layers `condition.png` over `scene.webp`, displays condition/quality facts, and supports interaction-grid inspection. |
| `ground/tests/test_catalog.py` | Builds controlled bundles and tests validation, checksums, unsafe paths, idempotent concurrent ingest, conflict handling, catalog queries, and atomic storage. |
| `ground/tests/test_api.py` | Exercises API health, upload authentication, listing, scene retrieval, asset serving, and dashboard availability through FastAPI’s test client. |

### `integration/` — mission orchestration outside the payload

| Path | What it does and why it exists |
|---|---|
| `integration/README.md` | Explains coordinate-driven and file-driven end-to-end demonstrations, client responsibilities, tunneling, and verification scope. |
| `integration/pyproject.toml` | Defines the integration package and its `vita-sentinel-run`, `vita-sentinel-mvp`, and `vita-mission` console commands. |
| `integration/verification.json` | Frozen evidence for a real local Sentinel payload-to-ground MVP, including hashes, GPU environment, stage results, and explicit claim limitations. |
| `integration/scripts/open_payload_tunnel.ps1` | Opens an SSH local-forward tunnel so a remote payload API can remain bound to loopback rather than exposed publicly. |
| `integration/scripts/run_sentinel_end_to_end.ps1` | PowerShell wrapper for a local Sentinel file through payload science and compact downlink generation. |
| `integration/scripts/run_sentinel_mvp.ps1` | PowerShell wrapper that continues through ground catalog ingestion and produces an MVP result. |
| `integration/src/vita_integration/__init__.py` | Marks the orchestration package; intentionally has no science implementation. |
| `integration/src/vita_integration/payload_client.py` | Strict client for payload job submission, polling, and downloading only the three approved artifacts with checksum verification and safe filenames. |
| `integration/src/vita_integration/ground_client.py` | Upload client for the canonical ground multipart endpoint, including URL validation and optional upload key. |
| `integration/src/vita_integration/mission_cli.py` | Main coordinate-driven operator command: builds a validated Earth Engine request, submits it, displays safe progress, polls to a terminal state, downloads the bundle, ingests locally or via ground API, and optionally notifies the dashboard. |
| `integration/src/vita_integration/pipeline.py` | One-command file-based Sentinel demonstration that invokes the payload pipeline, condition stage, and downlink builder while writing an atomic end-to-end result. |
| `integration/src/vita_integration/mvp.py` | Extends the file-based demonstration through verified ground ingestion and emits stable scene/dashboard links. |
| `integration/tests/test_payload_client.py` | Tests payload request/poll/download behavior, hashes, URL/path safety, and service errors without requiring the real payload. |
| `integration/tests/test_ground_client.py` | Tests canonical ground uploads, API keys, errors, and URL restrictions. |
| `integration/tests/test_pipeline.py` | Tests end-to-end orchestration, stage handoff, failures, and result manifests with controlled rasters/backends. |
| `integration/tests/test_mvp.py` | Tests payload-to-ground MVP composition and link/result generation. |
| `integration/tests/test_visualizations.py` | Verifies semantic and crop visualization category mappings so interpolated previews cannot misrepresent exact masks. |

### `payload/` — the flight-side science and service boundary

#### Payload manifests, packaging, and component documentation

| Path | What it does and why it exists |
|---|---|
| `payload/.gitignore` | Keeps downloaded model weights, temporary products, credentials, and local payload runtime artifacts out of Git while retaining small manifests. |
| `payload/README.md` | Complete payload operator/developer overview: supported inputs, stage commands, model identities, outputs, service API, benchmarks, security, and known limits. |
| `payload/DEPLOYMENT_MANIFEST.yaml` | Auditable allowlist of files permitted in the flight bundle and an explicit denylist for training, ground, data, outputs, and secrets. It also records commercial license status. |
| `payload/pyproject.toml` | Defines the payload package, pinned major dependency ranges, and commands such as `cloud-detect`, `scene-run`, `payload-condition`, `vita-payload-serve`, preflight, and benchmark. |
| `payload/requirements.txt` | Concrete pip-install dependency list for payload environments/container builds. Platform GPU packages are protected by deployment constraints and stack checks. |
| `payload/verification.json` | Versioned evidence for selected crop/cloud weights, hashes, model equivalence, real GPU smoke tests, peak memory, Sentinel/Balkan regressions, and licensing. |
| `payload/reconstruction/README.md` | States that raw-image reconstruction is outside the active payload science boundary and records what would be required before such products could become model inputs. |
| `payload/cloud_masking/README.md` | Compatibility pointer explaining that the active implementation is the `cloud_detection` package and that the former cloud masking area is not a second runtime. |
| `payload/cloud_masking/UPSTREAM.md` | Records upstream OmniCloudMask/Vita-CloudDetector provenance, commits, licenses, local integration decisions, and deviations. |

#### Cloud model documentation, configuration, and batch entry point

| Path | What it does and why it exists |
|---|---|
| `payload/cloud_detection/configs/cloud_detector.yaml` | Operational cloud contract: OmniCloudMask identity, weight directory/hash, R/G/NIR order, FP32 device policy, patch/overlap, sensor handling, class IDs, small-region removal, dilation, decisions, and output choices. |
| `payload/cloud_detection/docs/architecture.md` | Explains the cloud package’s backend, preprocessing, tiling, reconstruction, post-processing, outputs, and integration boundary. |
| `payload/cloud_detection/docs/input_output_specification.md` | Precise input band/scale/grid/nodata requirements and semantic, score, unusable, metadata, and preview outputs. |
| `payload/cloud_detection/docs/integration_guide.md` | Shows standalone and stage-gated integration patterns, environment overrides, model installation, and error handling. |
| `payload/cloud_detection/docs/model_card.md` | Model purpose, training/upstream identity, classes, supported use, limitations, license, and validation status. |
| `payload/cloud_detection/docs/technical_basis.md` | Scientific/engineering rationale for R/G/NIR cloud classification, ensemble inference, overlap, morphology, invalid masking, and decision thresholds. |
| `payload/cloud_detection/scripts/run_batch_inference.py` | Runs the standalone cloud pipeline over an explicit list/directory of GeoTIFFs and writes a JSON batch summary. It is an operator utility, not the persistent mission service. |

#### Model assets and immutable identity records

| Path | What it does and why it exists |
|---|---|
| `payload/models/README.md` | Explains which binary files must be installed locally, their source, verification hashes, and why weight binaries are not tracked in Git. |
| `payload/models/architecture.yaml` | Exact crop network recipe needed to reconstruct the model before loading weights: Prithvi EO v2 100 TL, time/location encoding, selected feature levels, UPerNet decoder, four bands, one frame, two classes. |
| `payload/models/selected_model.yaml` | Payload-side crop model manifest with filename, format, byte size, digest, source checkpoint, alternatives, metrics, thresholds, and non-automatic replacement policy. |
| `payload/models/omnicloudmask/README.md` | Installation/provenance notes for the two ignored OmniCloudMask weight files. |
| `payload/models/omnicloudmask/model.yaml` | Cloud ensemble manifest with package/model versions, class order, component filenames/sizes/hashes, combined hash, commits, and MIT license approval. |

The actual files `prithvi_crop_binary_single_frame_v1_weights.pt` and the two OmniCloudMask `.safetensors` files are intentionally ignored but required at runtime. Their manifests make missing or substituted weights detectable.

#### Payload utility scripts

| Path | What it does and why it exists |
|---|---|
| `payload/scripts/download_cloud_weights.py` | Downloads the pinned OmniCloudMask components into the expected local directory and verifies every hash before accepting them. This is an online setup action, never routine inference. |
| `payload/scripts/run_scene.ps1` | PowerShell wrapper for the manual stage-gated scene pipeline with Windows path/environment handling. |

#### `payload/src/cloud_detection/` — reusable cloud implementation

| Path | What it does and why it exists |
|---|---|
| `payload/src/cloud_detection/__init__.py` | Lazy public interface for `CloudDetectionPipeline` and its result type; laziness avoids loading heavy dependencies merely by importing the package. |
| `payload/src/cloud_detection/config.py` | Loads YAML, applies defaults/resolves relative paths, and validates cloud configuration before any model runs. |
| `payload/src/cloud_detection/types.py` | Small dataclasses representing a complete cloud result and tile/backend outputs, keeping array meanings explicit. |
| `payload/src/cloud_detection/backend.py` | Verifies ensemble/package identity, loads the two OmniCloudMask networks, calls upstream array inference, normalizes four-class scores, and provides a deterministic test backend for tests only. |
| `payload/src/cloud_detection/preprocessing.py` | Defines strict positive/finite R-G-NIR validity and reflectance normalization expected by the cloud backend. |
| `payload/src/cloud_detection/tiling.py` | Creates deterministic overlapping windows, reflection padding, and reconstruction weights so large images can be predicted without seams. |
| `payload/src/cloud_detection/postprocessing.py` | Converts semantic classes into the unusable mask, removes connected regions below 16 pixels, dilates by two pixels, and adds invalid pixels. Currently CPU/SciPy and therefore a Jetson optimization target. |
| `payload/src/cloud_detection/io.py` | Reads standalone GeoTIFF bands/metadata and writes masks/scores while preserving CRS, transform, nodata, compression, and band descriptions. |
| `payload/src/cloud_detection/preview.py` | Produces diagnostic cloud visualizations with exact category palettes, overlays, legends, and atomic file replacement. This is not model input. |
| `payload/src/cloud_detection/pipeline.py` | Coordinates standalone read → validate → tile → model → blend → semantic mask → post-process → percentages/decision → files/metadata/preview. |
| `payload/src/cloud_detection/cli.py` | `cloud-detect` argument parser and JSON-printing command-line entry point. |

#### `payload/src/prithvi_payload/acquisition/` — Sentinel provider boundary

| Path | What it does and why it exists |
|---|---|
| `payload/src/prithvi_payload/acquisition/__init__.py` | Public exports for acquisition provider, records, and safe errors. |
| `payload/src/prithvi_payload/acquisition/base.py` | Protocol describing what any payload acquisition provider must implement, allowing controlled fakes in tests without changing runtime logic. |
| `payload/src/prithvi_payload/acquisition/errors.py` | Structured error type with safe codes/messages/details suitable for job status; protected internal exceptions do not leak to clients. |
| `payload/src/prithvi_payload/acquisition/models.py` | Internal dataclasses for candidate metadata and acquired scenes, including provider provenance, grid, bands, scale, hash, size, rank, and timing. These are not routine downlink assets. |
| `payload/src/prithvi_payload/acquisition/grid.py` | Converts a WGS84 bounding box to the appropriate UTM zone and deterministic 10 m grid, enforcing a maximum 1024-pixel dimension and finite geometry. |
| `payload/src/prithvi_payload/acquisition/earth_engine.py` | Fixed Sentinel-2 SR Harmonized provider: initializes credentials, searches/validates candidates, applies selection policy, requests five bands, streams bounded GeoTIFF downloads with retries, normalizes/validates the local raster, hashes it, and records safe evidence. |

#### `payload/src/prithvi_payload/` — active stage-gated science and service

| Path | What it does and why it exists |
|---|---|
| `payload/src/prithvi_payload/__init__.py` | Lazy public crop-inference interface, avoiding expensive TerraTorch/PyTorch loading during unrelated imports and preflight checks. |
| `payload/src/prithvi_payload/scene_intake.py` | First fail-closed gate for preprocessed images: validates IDs/time, sensor, CRS/transform/dimensions, band descriptions and logical roles, nodata/finite samples, radiometric scale, source hash, and Balkan calibration binding; writes an inspectable intake report. |
| `payload/src/prithvi_payload/cloud_classifier.py` | Single source of truth for the selected cloud backend identity, expected hashes, configuration, and construction. It prevents an old cloud model from being used elsewhere. |
| `payload/src/prithvi_payload/cloud_stage.py` | Converts intake evidence into a cloud execution plan with exact band indices, scale, strategy, required outputs, and readiness/errors. Planning is separate so contracts can be audited before costly inference. |
| `payload/src/prithvi_payload/cloud_executor.py` | Executes a ready cloud plan. Sentinel uses bounded source-grid windows; Balkan builds a bounded 10 m UTM analysis grid, runs OmniCloudMask, post-processes masks, reprojects categorical results to the source grid, publishes rasters/preview/metadata, and records detailed timings. |
| `payload/src/prithvi_payload/crop_stage.py` | Applies the independent 60% total-cloud gate, resolves four crop bands and temporal coordinates, requires the Balkan adapter when applicable, selects sensor-specific crop/health thresholds, and emits a ready/blocked crop plan. |
| `payload/src/prithvi_payload/balkan_crop_calibration.py` | Validates source-bound calibration JSON, its hash/size/date/reference evidence and monotonic curves, then interpolates Balkan values into model units. It rejects stale, swapped, malformed, or weak calibration. |
| `payload/src/prithvi_payload/inference.py` | Verifies the 383 MB selected checkpoint, rebuilds the exact architecture, loads weights-only state with mmap, moves the model to CPU/CUDA, validates tensor shapes, normalizes inputs, and runs inference-mode FP16 autocast on CUDA. |
| `payload/src/prithvi_payload/crop_executor.py` | Memory-bounded crop segmentation: reads calibrated/selected bands in overlapping 224 tiles, fills invalid pixels safely, adds time/location coordinates, batches model calls, blends probabilities in memmaps, masks unusable pixels, writes probability/binary/confidence products, creates a diagnostic preview, and records inference versus total timings. Balkan execution uses a temporary 10 m calibrated crop grid and then publishes to the source grid. |
| `payload/src/prithvi_payload/condition_stage.py` | Windowed condition processor: resolves payload artifacts, builds the strict analysis mask, calibrates reflectance, calculates eight indices and four component scores, streams exact statistics, performs robust anomaly passes, writes condition/health rasters, renders a quicklook, and emits the comprehensive condition report. |
| `payload/src/prithvi_payload/downlink.py` | Reads payload/condition evidence and creates the exact three-file web bundle. It preserves original source colors for the base WebP, creates a transparent condition overlay, builds geospatial interaction cells, strips non-portable internal paths, hashes assets, and enforces a size-aware manifest. |
| `payload/src/prithvi_payload/pipeline.py` | Manual file-driven stage controller. It runs intake, cloud planning/execution, crop planning/execution, condition, and downlink with explicit stop points, stage metadata, warnings/errors, atomic result writes, and callback progress. Used by supplied Balkan processing and local file tests. |
| `payload/src/prithvi_payload/runtime.py` | Persistent production orchestration. It initializes Earth Engine and both models once under a lock, searches bounded Sentinel candidates, acquires/evaluates each with payload cloud truth, continues only an accepted candidate, updates job status, cleans rejected working data, packages results, and calculates warm/total timings. |
| `payload/src/prithvi_payload/job_store.py` | Thread-safe restart-safe job database using one directory per validated ID and atomic JSON writes. It stores commands/status/events/results, rebuilds history, serves only approved completed artifacts, and requeues interrupted jobs after restart. |
| `payload/src/prithvi_payload/service.py` | Persistent FastAPI payload service with one bounded queue and one GPU worker, health/job/artifact endpoints, safe error mapping, startup model warm initialization, and graceful stop. Serialization protects GPU memory and predictable latency. |
| `payload/src/prithvi_payload/benchmark.py` | Reads a completed persisted job and reports required timing fields plus exactly what the warm timing excludes. It prevents cold/network/UI time from being confused with science latency. |
| `payload/src/prithvi_payload/container_preflight.py` | Startup inspection that checks required directories, credentials path, CUDA policy, package/platform versions, and expected model/config identities without loading models or contacting Earth Engine. |
| `payload/src/prithvi_payload/ee_smoke.py` | Small non-interactive check that credentials can initialize Earth Engine and access the fixed Sentinel collection. It does not run a mission. |

#### Payload tests

| Path | What it protects |
|---|---|
| `payload/tests/test_scene_intake.py` | Sensor/band/grid/radiometry/time/identifier validation, readiness, no-data, scale, CUDA policy, and intake metadata. |
| `payload/tests/test_omnicloudmask_backend.py` | Exact package/weight identities, output shape/normalization, device policy, empty-valid behavior, and test-versus-real backend isolation. |
| `payload/tests/test_crop_executor.py` | Overlap tapering and probability accumulation at interior/edge/corner tiles so crop seams and uncovered pixels cannot be introduced. |
| `payload/tests/test_condition_stage.py` | Exact masks, indices, scores, insufficient-data behavior, assets, reports, source calibration, and windowed execution. |
| `payload/tests/test_balkan_crop_calibration.py` | Sidecar schema, source binding, monotonic knots, evidence thresholds, hash/size failures, and interpolation. |
| `payload/tests/test_balkan1_preprocessing.py` | Real Balkan L1A helper behavior, metadata discovery, registration/radiometry checks, and deterministic product expectations. It does not make raw processing part of the active model path. |
| `payload/tests/test_earth_engine_acquisition.py` | Candidate parsing/ordering, fixed bands/grid, safe downloads, byte limits, retries, TIFF validation, hashes, and provider error codes with fakes. |
| `payload/tests/test_earth_engine_live.py` | Opt-in live Earth Engine integration test guarded by environment flags so normal test runs never require network credentials. |
| `payload/tests/test_acquisition_runtime.py` | Candidate loop, payload-measured cloud selection, cleanup, status events, timing, model reuse, errors, and completed result contract. |
| `payload/tests/test_payload_service.py` | API queueing, conflict/full behavior, worker completion/rejection/failure, recovery, history, and exact artifact routes. |
| `payload/tests/test_benchmark.py` | Completed-job requirements, all timing fields, exclusion declaration, and malformed-record rejection. |
| `payload/tests/test_container_preflight.py` | Startup policy and safe inspection under present/missing CUDA, credentials, paths, and model identity conditions. |
| `payload/tests/test_deployment_contract.py` | Docker/compose/manifest boundaries: required model hashes, no forbidden code/data, non-root execution, secrets, ports, volumes, CUDA, and service commands. |
| `payload/tests/test_stack_guard.py` | Base-stack snapshot/comparison behavior and rejection of unintended Torch/OpenCV/NumPy/ABI changes. |

### `scripts/` — data preparation, training wrappers, and Balkan ground tools

These scripts are operator/development tools. They do not belong in the flight payload bundle unless a future reviewed design explicitly promotes a bounded function.

| Path | What it does and why it exists |
|---|---|
| `scripts/prepare_pastis.py` | Verifies the downloaded PASTIS archive, safely extracts only required optical imagery/labels/metadata, prevents path traversal, and records counts/hashes without duplicating unrelated data. |
| `scripts/run_single_frame_training.ps1` | Windows wrapper that validates environment/data/config, then launches the single-frame multiclass experiment with repeatable paths/logging. |
| `scripts/run_single_frame_binary_training.ps1` | Wrapper for the selected binary crop/non-crop training run. |
| `scripts/run_augmented_training.ps1` | Wrapper for the augmentation/refinement experiment. |
| `scripts/run_europe_replay_pipeline.ps1` | Orchestrates PASTIS validation, replay training/evaluation, and retained evidence for the European generalization experiment. |
| `scripts/balkan1/README.md` | Defines ignored Balkan directory layout, difference between raw/L1A/supplied-preprocessed data, commands, calibration requirements, visualizations, and the handoff to payload science. |
| `scripts/balkan1/preprocessors/__init__.py` | Package marker for checked-in preprocessing implementations; prevents arbitrary local scripts from being confused with maintained processors. |
| `scripts/balkan1/preprocess_collection.py` | Generic sequential runner for an explicitly supplied preprocessor command over selected scene folders, with placeholder substitution and fail-fast subprocess behavior. |
| `scripts/balkan1/process_l1a.py` | Builds the minimum real L1A product from delivered raw DN bands: discovers source/metadata, applies specified alignment and radiometric operations, preserves provenance/geospatial information, and writes validation logs/products. It does not claim L1B/L1C. |
| `scripts/balkan1/process_l1a_collection.py` | Runs `process_l1a.py` sequentially over selected/all raw scene directories with consistent output layout and failure propagation. |
| `scripts/balkan1/validate_l1a.py` | Compares a produced L1A product with the supplied processed reference using registration, homography, radiometric diagnostics, and bounded visualizations. It evaluates progress; it does not alter the payload input. |
| `scripts/balkan1/visualize_preprocessing.py` | Creates a readable before/after L1A processing report showing composites, band alignment, edge overlays, and parsed processing metrics. |
| `scripts/balkan1/visualize_collection.py` | Builds compact true-color/false-color overviews and metadata for all supplied preprocessed Balkan scenes so operators can choose cloud/crop candidates. |
| `scripts/balkan1/stage_sample.py` | Extracts a bounded geospatial chip from a real preprocessed scene into ignored testing input, preserving band descriptions/grid and writing source/hash provenance. It never creates mock pixels. |
| `scripts/balkan1/calibrate_crop_input.py` | Fits the validated, monotonic, source-bound Balkan-to-Sentinel radiometric curves from co-registered real data using spatial held-out evidence. This is the only script that creates crop calibration sidecars. |
| `scripts/balkan1/run_pipeline.py` | Operator entry point for any supplied preprocessed Balkan GeoTIFF plus its calibration. It derives/accepts scene metadata, forces sensor `balkan-1`, invokes the same stage-gated payload pipeline, records elapsed time, and can ingest the resulting compact bundle into the ground store. |

### `shared/` — sensor-neutral contracts and transparent science

#### Shared documents, YAML contracts, and JSON schemas

| Path | What it does and why it exists |
|---|---|
| `shared/README.md` | Explains which constants/formulas/contracts may be shared and why orchestration and sensor-specific code stay elsewhere. |
| `shared/band_definitions.yaml` | Human-readable model tensor layout, four logical input bands, one time step, Sentinel mapping, and Balkan requirements/warning. |
| `shared/normalization.yaml` | Exact per-band training means/standard deviations, numeric scale, normalization formula, and warning against normalizing raw Balkan DN. |
| `shared/class_mapping.yaml` | Fine land-cover class names plus reviewed mappings to binary crop/non-crop, ignored reference class, and active-crop health subsets. |
| `shared/health_analysis_contract.md` | Plain-language scientific contract for analysis masks, reflectance units, index formulas, scoring, evidence, limitations, and output claims. |
| `shared/schemas/payload_acquisition_command.schema.json` | JSON Schema for the only accepted coordinate mission command: safe IDs, bbox, dates, fixed Earth Engine provider, and cloud-selection policy/targets. |
| `shared/schemas/inference_result.schema.json` | JSON Schema for a payload inference-result record with sensor, model identity, geospatial facts, and raster assets. |
| `shared/schemas/health_observation.schema.json` | JSON Schema for measured/insufficient condition observations, quality, metrics, condition object, and raster references. |
| `shared/schemas/downlink_bundle.schema.json` | Strict schema for `scene.json`: product identity, source, assets, geospatial facts, metrics, interaction grid, legends, algorithms, warnings, limitations, and package hashes. |
| `shared/pyproject.toml` | Defines the lightweight shared Python package and its only runtime dependencies, NumPy and Pydantic. |

#### Shared Python implementation

| Path | What it does and why it exists |
|---|---|
| `shared/src/prithvi_shared/__init__.py` | Curated public exports for acquisition commands, bands, classes, model hashes/thresholds, health, and condition types/functions. |
| `shared/src/prithvi_shared/acquisition.py` | Pydantic validation for dates, geographic bounding boxes, identifiers, provider, least-cloudy/target-range policies, and complete payload command serialization. |
| `shared/src/prithvi_shared/bands.py` | Code-level four-band order, tensor dimensions, normalization arrays, and band-order validator used by both training and payload. |
| `shared/src/prithvi_shared/classes.py` | Code-level fine class names and crop/non-crop/health mappings, including safe class-name lookup. |
| `shared/src/prithvi_shared/calibration.py` | Selected crop checkpoint names/hashes and calibrated Sentinel thresholds. Centralization prevents training, payload, and tests from silently diverging. |
| `shared/src/prithvi_shared/health.py` | Exact analysis-mask validation, eight reflectance-index formulas, safe division, finite/range checks, metric summaries, and health-observation construction. |
| `shared/src/prithvi_shared/condition.py` | Validated prototype ranges/weights, pixel scoring, robust spatial anomalies, evidence quality, final labels/explanations/limitations, and an all-in-memory reference implementation. |

#### Shared tests

| Path | What it protects |
|---|---|
| `shared/tests/test_acquisition.py` | Valid/invalid bbox/date/ID/policy combinations and JSON Schema agreement. |
| `shared/tests/test_health_mask.py` | Exact inclusion/exclusion behavior for crop, no-data, cloud unusable mask, probabilities, and distinct classification/health thresholds. |
| `shared/tests/test_health_indices.py` | Numerical agreement with every documented index formula and invalid denominator/radiometry handling. |
| `shared/tests/test_condition.py` | Component/region scoring, anomaly logic, labels, evidence quality, insufficient data, explanations, and configuration validation. |

### `src/prithvi_crop/` — model development and training package

This package can create/select deployment artifacts but is not copied into the payload image.

| Path | What it does and why it exists |
|---|---|
| `src/prithvi_crop/__init__.py` | Package initialization and workspace-local cache defaults so large Hugging Face/Torch assets do not spill into arbitrary user directories. |
| `src/prithvi_crop/constants.py` | Auditable dataset ID/revision, fine classes, bands, folds/splits, ignored label, chip size, and model assumptions. |
| `src/prithvi_crop/download_data.py` | Revision-pinned IBM/NASA dataset downloader with safe tar extraction, split completeness checks, metadata installation, disk-space checks, and resumable behavior. |
| `src/prithvi_crop/validate_data.py` | Scans real imagery/labels/metadata for shape, dtype, bands, dates, labels, split leakage, duplicates, and numerical readiness; emits a report rather than fixing data silently. |
| `src/prithvi_crop/pastis_validation.py` | Validates extracted PASTIS optical arrays, masks, metadata, fold counts, dates, and paired file completeness without loading the training model. |
| `src/prithvi_crop/data.py` | TerraTorch dataset/data-module extensions implementing deterministic leakage-safe internal validation, temporal-band selection, masks, and reproducible loaders. |
| `src/prithvi_crop/europe.py` | PASTIS adapter and mixed replay datasets: conservative label harmonization, seasonal date selection, deterministic sampling, dihedral augmentation, and held-out fold protection. |
| `src/prithvi_crop/transforms.py` | Training-only geometric/radiometric augmentations that update images and masks consistently. These are never used during payload inference. |
| `src/prithvi_crop/task.py` | TerraTorch/Lightning segmentation task with persisted confusion matrices, per-class and aggregate metrics, deployment score, CSV/JSON evidence, and binary-relevant evaluation. |
| `src/prithvi_crop/binary.py` | Maps fine-class logits/targets to reviewed crop-versus-non-crop probabilities/predictions for legacy/multiclass evaluation. |
| `src/prithvi_crop/calibration.py` | Training-side copy of operating thresholds with the validation/checkpoint context that produced them. Shared deployment constants are checked against this record. |
| `src/prithvi_crop/runtime.py` | Builds configured task and data module outside LightningCLI for smoke tests, evaluation, and visualization while reusing the YAML contract. |
| `src/prithvi_crop/check_config.py` | Static, fail-fast validator for required config sections, allowed paths, model bands/classes/frames, callbacks, loaders, checkpoints, and output containment. |
| `src/prithvi_crop/preflight.py` | Prints and validates the complete training contract—software/GPU, config hash, data, model, selected folds, output paths, and resume status—before expensive training. |
| `src/prithvi_crop/smoke.py` | Loads one real batch and performs a CUDA forward/backward/optimizer step, checking shapes, finite loss/gradients, memory, frozen/trainable parameters, and class mapping. |
| `src/prithvi_crop/train.py` | Validated launcher around TerraTorch LightningCLI with automatic safe resume and explicit overrides; it refuses invalid config/data before starting. |
| `src/prithvi_crop/evaluate.py` | Held-out evaluation launcher that requires a real checkpoint and prevents evaluation without a selected artifact. |
| `src/prithvi_crop/evaluate_binary.py` | Sweeps probability thresholds on internal validation, calculates confusion/accuracy/balanced accuracy/precision/recall/F1/IoU and constrained health threshold, and writes calibration evidence without consuming held-out test data. |
| `src/prithvi_crop/visualize_examples.py` | Chooses deterministic real validation examples and renders source, truth, probability, and prediction panels plus hashes/metadata for business and technical review. |

### `testing/` — ignored real inputs and generated verification runs

| Path | What it does and why it exists |
|---|---|
| `testing/README.md` | Defines the rule that tests use real staged inputs or programmatically tiny unit fixtures, keeps large runs ignored, and explains evidence layout. |
| `testing/inputs/sentinel2/.gitkeep` | Placeholder for ignored real Sentinel test GeoTIFFs. |
| `testing/inputs/balkan1/.gitkeep` | Placeholder for ignored real Balkan test chips/scenes and calibration sidecars. |
| `testing/runs/.gitkeep` | Placeholder for ignored pipeline results, timings, previews, and regression evidence. |

### `training/` — retained data/run evidence, not the training implementation

| Path | What it does and why it exists |
|---|---|
| `training/README.md` | Explains dataset references, experiment evidence retention, checkpoint policy, and why training code lives in root `src/`/`configs/`. |
| `training/requirements.txt` | Full training dependency set for reproducible GPU development; broader and heavier than the payload or ground requirements. |
| `training/datasets/manifest.yaml` | Records local dataset paths, byte/file/chip counts, fold/split roles, replay fraction, and assurance that no duplicate dataset copy was created or originals modified. |
| `training/experiment_logs/manifest.yaml` | Retention policy and selected Europe replay checkpoint/metrics, with explicit exclusion of bulk logs, caches, and non-selected checkpoints. |
| `training/experiment_logs/europe_replay/data_readiness.json` | Small evidence record showing PASTIS pair/fold/date validation and ignored extras before replay training. |
| `training/experiment_logs/europe_replay/binary_validation_metrics.json` | Detailed threshold candidates and confusion metrics for the Europe replay model, retained for comparison even though that model is not automatically deployed. |

## 10. What is active, optional, experimental, or generated

| Category | Contents |
|---|---|
| Active runtime | `payload/src/prithvi_payload/`, `payload/src/cloud_detection/`, selected model/config manifests, `shared/src/`, payload service, integration clients, ground catalog/API/web |
| Required but ignored runtime assets | Crop `.pt` weights, two cloud `.safetensors` weights, Earth Engine credential, real source imagery, Balkan calibration JSON |
| Ground/development tools | `scripts/balkan1/`, `integration/`, data preparation and visualization scripts |
| Training only | root `src/prithvi_crop/`, `configs/`, most root dependencies, `training/`, training scripts |
| Optional diagnostics | Full GeoTIFF score/health/condition layers and Matplotlib previews |
| Generated state | `outputs/`, `runtime/`, `testing/runs/`, caches, databases, logs, temporary memmaps |
| Not yet part of operational science | Raw Balkan L0/L1 processing, complete L1B/L1C/atmospheric-correction chain, flight-qualified TensorRT fast path |

## 11. Important engineering cautions

1. **Do not judge scientific radiometry by WebP color.** Display stretch is for humans; model values come from calibrated raster bands.
2. **Do not use a Balkan calibration with a different file.** It is intentionally source-bound.
3. **Do not compare cloud metadata directly with payload cloud output.** They serve different roles and may use different footprints/definitions.
4. **Do not count masked pixels as non-crop.** Value `255` and floating `-9999` mean excluded/no-data, not a class.
5. **Do not treat condition as diagnosis.** It is transparent spectral screening with known prototype limits.
6. **Do not benchmark the manual script as a resident service.** Model loading and process startup distort production inference timing.
7. **Do not promise arbitrary-size latency.** Pixel count, number of tiles, reprojection, compression, and storage scale with the input.
8. **Do not build TensorRT engines on one stack and assume portability to another.** Version and target compatibility must be checked.
9. **Do not optimize only model calls.** Current evidence shows most time is grids, rasters, condition processing, previews, and packaging.
10. **Do not change models or precision without scientific regression.** A faster but differently classified cloud/crop mask changes every downstream percentage and score.

## 12. Recommended next implementation milestone

Create a versioned `fast_science_v1` path beside—not in place of—the current reference pipeline:

1. fixed 512 × 512, four/five-band in-memory input contract;
2. common 10 m grid and one sensor adapter;
3. resident FP16 TensorRT engines;
4. GPU-resident unusable/crop/condition calculation;
5. compact result object returned before file publication;
6. asynchronous adapter that feeds the existing downlink/ground contract;
7. an equivalence harness against the existing PyTorch/raster reference;
8. target-Jetson p50/p95/thermal/memory report.

This preserves the current working Sentinel and Balkan paths as the scientific oracle while making the latency architecture testable. Once equivalence and the two-second SLA are proven, the fast path can become the default and the reference path can remain for audit and diagnostics.

## 13. Official acceleration references

Use these primary NVIDIA sources when selecting and qualifying the target stack:

- [JetPack SDK downloads and current component versions](https://developer.nvidia.com/embedded/jetpack/downloads)
- [Jetson Orin module specifications](https://www.nvidia.com/en-eu/autonomous-machines/embedded-systems/jetson-orin/)
- [TensorRT best-practice measure/optimize workflow](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html)
- [TensorRT precision and accuracy considerations](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html)
- [TensorRT engine/version/hardware compatibility](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html)
- [Jetson Orin power, clocks, thermal behavior, and `tegrastats`](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)


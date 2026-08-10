# ViTA pipeline and optimization guide

This guide describes the two MVP pipelines, their code contracts, the warm local
workflow, every implemented performance optimization, the timing report, and the
three-file web handoff. The science path is the same locally and in the Jetson
container; only the configured inference backend and transport differ.

For NVIDIA container construction, SSH orchestration, GHCR, Jetson power settings,
and production acceptance, see [DEPLOYMENT_JETSON.md](DEPLOYMENT_JETSON.md).

## 1. Fast local workflow

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\vita.ps1 sentinel
.\vita.ps1 balkan
.\vita.ps1 web
```

Those are the normal local commands:

- `sentinel` uses `data\sentinel2\S2_20260712T170851_T14TPL_cloudy.tif`;
- `balkan` uses `data\balkan1\preprocessed\3408_L1ORT.tif` and its adjacent
  `*.crop_calibration.json` file;
- each run prints the complete payload timing breakdown and ingests its downlink;
- `web` starts the ground application and opens
  [http://127.0.0.1:8000/](http://127.0.0.1:8000/).

The first pipeline command starts one background payload service and waits until model
loading, fixed-input preparation, and CUDA warmup finish. Later commands reuse that
service. Check or stop services created by the script with:

```powershell
.\vita.ps1 health
.\vita.ps1 stop
```

Override only what differs from the proof-of-concept defaults:

```powershell
.\vita.ps1 sentinel `
  -InputPath sentinel2 `
  -Image another_scene.tif `
  -RegionId another-region

.\vita.ps1 balkan `
  -InputPath balkan1/preprocessed/another_scene.tif `
  -RegionId another-region
```

The existing interfaces are preserved:

- `vita-mvp sentinel ...` and `vita-mvp balkan ...` run a new one-shot Python process;
- `vita-payload-server` exposes `GET /healthz` and `POST /v1/jobs`;
- `scripts/ground/Invoke-VitaPayload.ps1` calls the service over SSH and downloads the
  bundle;
- the payload and ground Docker Compose deployments remain under `deploy/`.

Use the persistent service for latency measurements. A one-shot `vita-mvp` command
must import the model ecosystem and load both model objects again, so it measures a
different lifecycle.

## 2. Architecture

```text
payload-local image path or small SSH JSON request
                         |
                         v
             warm single-worker service
                         |
      +------------------+------------------+
      |                                     |
 Sentinel-2 GeoTIFF                 Balkan-1 L1ORT GeoTIFF
      |                             + verified calibration
      |                                     |
      |                             shared 10 m UTM grid
      |                                     |
      +------------------+------------------+
                         |
               OmniCloudMask cloud stage
                         |
                  cloud/invalid gate
                         |
               Prithvi crop segmentation
                         |
             health indices + condition score
                         |
          scene.webp + condition.png + scene.json
                         |
           validated ground catalog + web application
```

No source image is uploaded by the ground request. `input` and optional `image` are
relative paths below the payload's read-only data root. The routine downlink contains
only the three compact web products.

## 3. Startup lifecycle

`PayloadRuntime` in `payload/src/prithvi_payload/service.py` owns the long-lived
execution state.

1. It validates CUDA when `CUDA_REQUIRED=1`.
2. It pre-imports modules that would otherwise make the first accepted request pay
   Python import and TorchDynamo initialization costs.
3. If `VITA_BALKAN_PREPARE_INPUTS` is configured, it validates each fixed Balkan
   scene, verifies its calibration SHA-256, and creates or validates both persistent
   shared-grid caches. The former singular variable remains supported for local use.
4. It loads the two OmniCloudMask ensemble checkpoints once.
5. It loads the selected Prithvi crop model once on native CUDA.
6. It warms the resident FP16 cloud ensemble, all four exact cloud paths, and the
   fixed FP32 crop batch directly on CUDA. No production model compiler runs here.
7. Only then does `/healthz` report `status: ready`. Readiness includes a tiny
   synchronized CUDA operation on the model worker, so a stale context after a
   laptop sleep, driver reset, or GPU switch is detected before a job starts.

FastAPI is limited to one worker-thread token and `job_lock` rejects concurrent jobs.
This is deliberate: CUDA/cuDNN state can be thread-local, and competing 100M-parameter
jobs produce unstable latency and memory pressure. Model construction, warmup, and
accepted requests therefore remain on one persistent worker path.

The convenience script in `vita.ps1` starts this same service; it does not introduce a
second inference implementation.

## 4. Request lifecycle and code map

### 4.1 Payload API and orchestration

`JobRequest` in `service.py` accepts only:

- sensor;
- payload-local input path and optional Sentinel filename;
- region and unique job identifiers;
- optional acquisition/radiometry overrides;
- optional payload-local Balkan calibration path.

Absolute paths, `..`, and image bytes are rejected. `PayloadRuntime.run()` resolves
the source below `VITA_INPUT_ROOT`, then calls `run_scene()` in
`payload/src/prithvi_payload/pipeline.py`. `run_scene()` is the single stage-gated
orchestrator used by both sensors and by both the service and original CLI.

### 4.2 Scene intake

`inspect_scene()` in `scene_intake.py` reads metadata and small samples rather than
materializing the full scene. It validates:

- file existence, CRS, dimensions, transform, nodata and band count;
- band descriptions and logical band routing;
- acquisition timestamp and reflectance scale;
- sensor-specific readiness for cloud and crop stages;
- the Balkan calibration schema, band order, validation evidence, and source SHA-256.

Sentinel uses logical B02/B03/B04/B08/B8A routing. Cloud uses NIR/RED/GREEN/BLUE;
Prithvi uses BLUE/GREEN/RED/NIR_NARROW. Balkan requires exactly five preprocessed
bands—BLUE/GREEN/RED/NIR/PAN—but PAN is retained only as source evidence and is not
fed to either current model.

### 4.3 Balkan shared analysis grid

Sentinel proceeds on its existing raster because it is already small and compatible.
Balkan first calls `materialize_balkan_analysis_grid()` in `balkan_analysis.py`.

The function:

1. chooses the UTM zone containing the scene centre;
2. calculates the unchanged 10 m target grid;
3. finds overview factors common to all four reflectance bands;
4. selects the largest overview that is still finer than the target grid;
5. enforces a configurable working-memory limit;
6. reads the selected four-band overview once;
7. performs one multiband area-average reprojection;
8. writes BLUE/GREEN/RED/NIR_BROAD float32 bands with the same nodata and model routes;
9. records source, grid, overview, resampling, memory, cache, and timing provenance.

For `3408_L1ORT.tif`, factor 4 is selected. The output remains 1,739×2,132 pixels,
10 m, EPSG:32612. The original source is never modified.

The cache key includes the verified source SHA-256, algorithm version, resolution,
band indices, preparation mode, and overview factor. A cached TIFF is accepted only
when its key, algorithm, band descriptions, size, CRS, and transform match its metadata.
Compose sets `VITA_BALKAN_OVERVIEW_REQUIRED=1`; missing or oversized overviews fail
readiness instead of silently returning to the old slow path.

### 4.4 Cloud planning and execution

`build_cloud_stage_plan()` in `cloud_stage.py` validates the four-band route,
reflectance divisor, model compatibility, output contract, and cloud class mapping.

`execute_cloud_stage()` in `cloud_executor.py` then:

- reads bounded raster windows with a 150-pixel halo;
- normalizes reflectance and builds strict invalid-pixel masks;
- invokes the resident two-model mean-logit OmniCloudMask backend in FP16 directly on
  CUDA;
- requests semantic classes directly when supported, avoiding unused probability
  tensors and CPU transfers;
- converts configured cloud classes, shadows and invalid pixels into the unusable mask;
- retains semantic, unusable, invalid, source-band, and validity arrays in bounded RAM
  for routine downlink jobs; full/debug runs still write the equivalent GeoTIFFs;
- counts classes while tiles are already in memory.

The supplied Sentinel scene and shared Balkan grid each use one model tile in the
current proof of concept, but the executor remains bounded for larger supported scenes.

### 4.5 Crop planning and Prithvi execution

`build_crop_stage_plan()` in `crop_stage.py` applies the cloud gate and verifies the
selected model artifact, threshold, acquisition coordinates, spectral adapter,
reflectance multiplier, and unusable mask.

`execute_crop_stage()` in `crop_executor.py`:

- reads 224×224 tiles with a 16-pixel halo;
- skips fully unusable tiles without invoking the model;
- groups real tiles into fixed batches of 16 on the 64 GB Orin (four locally by
  default); the final batch is padded and only real outputs are retained;
- applies the validated Balkan monotonic calibration when required;
- supplies temporal year/day-of-year and geographic coordinates to Prithvi;
- pads only the final incomplete batch to the fixed optimized shape;
- blends overlapping probability tiles with deterministic linear edge weights;
- retains probability, binary decision, calibrated bands, and masks in bounded RAM on
  the routine path; full/debug runs write the equivalent rasters;
- calculates summary statistics during the existing output pass.

`PayloadCropModel` in `inference.py` owns checkpoint validation, source export caching,
accepted-plan loading, and backend reporting. `tensorrt_builder.py` separately exports
ONNX, validates it, builds direct plans with `trtexec`, and publishes a checksum-sealed
manifest only after real-scene parity passes. Service startup never compiles and never
imports Torch-TensorRT.

### 4.6 Health and condition analysis

`run_payload_condition()` in `condition_stage.py` consumes the canonical source,
cloud, crop-probability and crop-binary grids. The transparent formulas live in
`shared/src/prithvi_shared/health.py` and `shared/src/prithvi_shared/condition.py`.

It calculates NDVI, GNDVI, EVI, SAVI, CVI, VARI, excess green and RGB brightness, then
combines absolute and relative evidence into the condition score, anomaly layers,
quality label and screening label. Production uses one bounded 4,096-pixel window for
the reviewed grids and exact all-valid-pixel statistics:

- pass 1 calculates indices, exact moments, and exact percentile values;
- the statistics step derives robust centres/scales;
- the spatial pass produces relative anomaly and final condition layers.

The original deterministic bounded reservoir remains available for larger offline
inputs, but production uses exact percentiles because they are both more accurate and
faster at this MVP size. Outputs are screening priorities, not disease diagnoses.

### 4.7 Downlink packaging

`build_downlink_bundle()` in `downlink.py` validates that all source rasters share the
same grid and produces exactly:

- `scene.webp`: compact RGB context;
- `condition.png`: transparent condition overlay;
- `scene.json`: checksums, summaries, provenance, region history fields, and a compact
  interaction grid used by the browser.

The routine interaction grid reads the already computed arrays in RAM. RGB rendering
starts early and overlaps model/science work; overlay preparation, grid aggregation,
WebP, and lossless PNG encoding are concurrent. SHA-256 and byte counts are returned
for every file.

Sentinel context uses a combined RGB 2–98% display stretch. For calibrated Balkan-1
scenes, the WebP renderer first applies the same validated Balkan-to-Sentinel monotonic
curves used by the science path, then performs the combined stretch. This display-only
step removes native channel-gain color casts without changing inference rasters,
probabilities, masks, indices, or condition scores.

### 4.8 Ground ingest and web application

`vita-ingest` calls `SceneCatalog.ingest()` in `ground/src/prithvi_ground/catalog.py`.
It verifies the schema, identifiers, relative paths, checksums, media types, dimensions,
geographic values, and the exact three-file contract before atomically publishing a
scene to `runtime/ground`.

`vita-dashboard` serves the catalog API and static application from
`ground/src/prithvi_ground/api.py` and `ground/src/prithvi_ground/web/`. The browser
loads the WebP, overlays the PNG, and uses `scene.json` for summaries, interaction and
region history. Ground ingest and visualization occur after the payload response and
are not counted in `payload_seconds`.

## 5. Implemented optimization techniques

### Resident models instead of process-per-request loading

The service constructs both model objects once. This removes repeated imports,
checkpoint reads, graph construction, CUDA allocation and backend initialization from
accepted requests. `/healthz` exposes startup costs separately.

### Explicit warmup and correct warmup shapes

Cloud warmup includes batch 1 at 1,000 pixels for Sentinel, batches 1 and 4 at 869
pixels for Balkan, and every additional static shape encountered by the exact four
scene paths. Crop warmup uses fixed batch 16 at 224×224. The crop model runs before
the final exact cloud warmups because it can displace convolution/workspace state.
Warmup predictions are discarded; no scientific output is cached as a substitute for
inference.

### Single-worker CUDA ownership

FastAPI initialization and jobs use one long-lived worker thread. A nonblocking job
lock returns HTTP 409 for concurrent work instead of letting two GPU pipelines compete.
This eliminated severe first-request and multi-process timing variance observed during
local validation.

### Native FP16 cloud CUDA, semantic-only inference, and batching

The reviewed two-model mean-logit ensemble remains resident in PyTorch, runs in FP16
on CUDA, and is warmed for batch 1 at 1,000 pixels plus batches 1 and 4 at 869 pixels.
`predict_semantic()` returns only class IDs to the CPU. Production does not export or
compile cloud engines during startup or timed requests.

The optional TensorRT experiment compiled after a narrow zero-channel graph rewrite,
but the real-scene check then changed 0.299141% of cloud classes against the source
model, above the unchanged 0.1% gate. Raising the tolerance would hide an accuracy
failure, so fail-closed deployment rejects the TensorRT cloud backend, any active cloud
engine/profile, and TensorRT CUDA graph mode. The experimental compiler code remains
isolated for future requalification; production never enters it.

### Prithvi fixed batching on native CUDA

Crop tiles run in batches of 16 on Orin. The final short batch is padded to the same
static shape and only real outputs are retained. The exact exported FP32 Prithvi graph
runs on CUDA through PyTorch. TensorRT 10.8 conversion is not a production backend for
this model on the pinned stack: full, precision-controlled, calibrated, and attempted
hybrid builds all changed about 2.2% of thresholded decisions on the packaged-scene
batch. No calibration or relaxed tolerance is applied to conceal that mismatch. The
cloud model likewise remains on native CUDA, using the reviewed FP16 source ensemble.

### One Balkan grid instead of repeated preprocessing

Cloud, crop, condition and packaging share one 10 m product. The original approach
decoded and warped four full-resolution bands separately, costing roughly 12.7 seconds
for the supplied 1.16 GiB TIFF. The current overview-plus-multiband path measured about
1.43–1.76 seconds on the development machine. A verified persistent hit normally costs
only metadata validation.

The remaining reprojection measured roughly 0.4 seconds, so it stays in GDAL's
area-average CPU implementation. A GPU linear remap would change the scientific
resampling contract for little possible gain. GPU effort is reserved for the neural
networks, where it materially changes latency.

### Bounded, shared in-memory raster operations

`raster_ops.py` contains the single UTM-zone and padded-window implementations used by
the relevant stages. The reviewed grids fit explicit memory limits, so routine jobs
pass source bands, masks, probabilities, indices, and condition layers directly
between stages. Arbitrary/full debug paths remain windowed and disk-backed. Temporary
files are replaced atomically, and output directories are job-specific.

### Work avoidance inside raster stages

- Fully unusable crop tiles never reach Prithvi.
- Class, probability and confidence statistics reuse arrays already being written.
- Production condition statistics use exact moments and every valid value for
  percentiles; the bounded deterministic sample remains an offline fallback.
- Display limits reuse the prepared Balkan grid instead of rereading the 1.16 GiB
  source.
- The interaction grid uses the already computed RAM arrays instead of raster reads.
- RGB work overlaps neural/science work; overlay, grid, WebP, and PNG work are
  concurrent. WebP method 0 and lossless PNG level 1 minimize packaging latency.

### Persistent, identity-bound caches

The process caches a source digest only while path, size, modification time and change
time remain unchanged. The Balkan raster cache additionally binds source SHA-256 and
algorithm/grid identity. The production deployment removes VITA's rejected TensorRT
model/timing caches before startup. No accepted cache uses only a filename as proof of
equivalence.

### Compact routine downlink

Only WebP, PNG and JSON are written by the timed routine path and cross the ground
link. Non-downlinked science layers remain ephemeral arrays. Full stage/debug commands
still produce scientific GeoTIFFs for audit and diagnosis.

## 6. Reading the timing report correctly

`payload_seconds` begins immediately before `run_scene()` and ends when the three-file
bundle is complete. It includes:

- intake;
- Balkan shared-grid lookup/build when applicable;
- cloud and crop planning;
- cloud and crop execution;
- condition analysis;
- downlink packaging.

It excludes service startup/model loading/warmup, request path resolution, response
checksum calculation, SSH/SCP, ground ingest, and the web application.

The top-level rows that add back to `payload_seconds` are:

```text
intake
+ shared analysis grid
+ cloud plan
+ cloud stage
+ crop plan
+ crop stage
+ condition stage
+ downlink packaging
+ orchestration remainder
```

`cloud_inference_seconds` and `cloud_mask_processing_seconds` are already inside
`cloud_stage_seconds`. `crop_inference_seconds` is already inside
`crop_stage_seconds`. Do not add nested rows again.

For meaningful acceptance:

1. start one service and wait for `/healthz`;
2. use AC/max-performance mode and adequate cooling;
3. ensure no second Python/CUDA service is alive;
4. let deployment run every exact scene three times (increase repetitions for release
   characterization);
5. require every repetition, not merely the median, to remain below two seconds;
6. retain stack/backend fields with the timings.

A cold, never-verified 1.16 GiB Balkan file must still pay full SHA-256 verification.
The fixed MVP scene is legitimately prepared on the payload before readiness; this is
not equivalent to claiming that every unseen image has warm-cache latency.

## 7. Runtime files and repository cleanliness

Tracked source, deployments and docs do not depend on generated experiment folders.
The following are deliberately ignored and regenerable:

- `runtime/`: jobs, caches, engines, logs, ground catalog and dashboard assets;
- `outputs/`: legacy training experiments, if recreated externally;
- `testing/`: manual validation inputs/runs, if recreated externally;
- Python/tool caches such as `__pycache__`, `.pytest_cache` and `.ruff_cache`.

Do not delete:

- `data/` unless the imagery owner explicitly approves it;
- `payload/models/` model binaries;
- Balkan `*.crop_calibration.json` sidecars;
- `secrets/` or deployment environment files without replacing their configuration.

Stop local services before deleting `runtime/`:

```powershell
.\vita.ps1 stop
Remove-Item -LiteralPath .\runtime -Recurse -Force
```

The next local command recreates the necessary runtime layout.

## 8. Accuracy and operational boundaries

- Sentinel follows its native validated band/radiometry contract.
- Balkan crop inference requires the source-bound validated monotonic calibration.
- The factor-4 Balkan overview path is an MVP speed/accuracy trade, not bitwise
  equivalence to full-resolution averaging; measured parity and limitations are
  recorded in the Jetson guide.
- FP16 and any future compiler backend require comparison against the FP32/PyTorch
  reference on the same scenes; production currently uses native FP16 CUDA for cloud
  and native FP32 CUDA for crop.
- INT8/quantization remains disabled until a representative calibration dataset and
  mission tolerance exist. The optional ModelOpt warning does not affect FP16.
- The two-second target applies to a ready payload and the approved four-scene image-size
  envelope. It is not a model-loading, network-transfer or arbitrary-image guarantee.

## 9. Deployment path

Local operation proves contracts and provides a development baseline. Production uses:

1. the NVIDIA Jetson-compatible PyTorch iGPU base;
2. the NVIDIA container runtime;
3. code-only GHCR images built natively on ARM64;
4. read-only data/model mounts and persistent runtime/engine mounts;
5. loopback-only payload HTTP reached through an authenticated SSH tunnel;
6. SHA-256-verified three-file SCP downlink;
7. loopback-only ground dashboard.

Run the complete preflight, build, health, benchmark and release procedure in
[DEPLOYMENT_JETSON.md](DEPLOYMENT_JETSON.md).

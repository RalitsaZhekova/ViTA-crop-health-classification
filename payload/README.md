# Payload component

This directory is the flight-side boundary. It is intentionally independent of
training datasets and experiment logs.

## Included now

- `models/`: one pinned checkpoint, its SHA-256 identity and architecture;
- `src/prithvi_payload/`: training-free crop and cloud classifier loaders;
- shared vegetation-index, RGB-diagnostic and transparent condition scoring;
- `reconstruction/`: the Balkan-1 raw-band registration contract;
- `src/cloud_detection/`: complete reviewed cloud-detection runtime;
- `cloud_detection/`: its configs, operational scripts and documentation;
- `cloud_masking/`: the boundary between the standalone mask and crop inference;
- `requirements.txt`: runtime-only Python dependencies.

The base model receives normalized batches shaped `[batch, 4, 1, height, width]` in
the exact `BLUE, GREEN, RED, NIR_NARROW` order. The inference wrapper accepts
inputs on the training numeric scale and performs the recorded normalization.
In an installed deployment, set `PRITHVI_MODEL_DIR` to the directory containing
the checkpoint and `architecture.yaml`; source checkouts resolve
`payload/models/` automatically.

The selected base model emits crop/non-crop output only. The preserved crop-type
models remain ground-side under `outputs/` and are not part of the flight bundle.

## Earth Engine acquisition service

The persistent payload service accepts only the shared acquisition command
schema and uses the fixed project `vita-503208`, collection
`COPERNICUS/S2_SR_HARMONIZED`, band order `B02,B03,B04,B08,B8A`, 10 metre UTM
grid and reflectance scale 10000. Regions larger than 1024 pixels in either
dimension are rejected instead of resized. Callers cannot provide collection
names, bands, expressions, commands, modules, output paths or executables.

Application Default Credentials are initialized once; interactive
`ee.Authenticate()` is never used:

```text
EE_PROJECT_ID=vita-503208
GOOGLE_APPLICATION_CREDENTIALS=/run/secrets/earth_engine_credentials
EE_MAX_CANDIDATES=50
EE_MAX_SCENE_ATTEMPTS=5
CUDA_REQUIRED=1
```

Verify credential and collection access without printing credential data or a
signed download URL:

```powershell
python -m prithvi_payload.ee_smoke
```

Start the long-lived API after mounting the runtime credential and model files:

```powershell
python -m prithvi_payload.service --host 127.0.0.1 --port 8081
```

The service initializes Earth Engine, CUDA, CloudSEN and the crop model once,
warms both models once, and serializes GPU work through a bounded queue. Job
state is written atomically below `VITA_JOB_ROOT` and completed jobs survive
restart. Each `status.json` retains safe state-transition events and candidate
attempts. The store also maintains an atomic `history.json` containing all prior
jobs, requests and completed analysis summaries; it is available from
`GET /v1/jobs`. The remaining fixed endpoints are `GET /health`, `POST /v1/jobs`,
individual job status, and one endpoint for each of the three routine artifacts.
This operational history is not added to the routine three-file downlink.

For `target_cloud_range`, Earth Engine scene metadata must first be within the
requested 15-35% demonstration range. Up to five ordered candidates are then
downloaded one at a time and evaluated by the existing CloudSEN stage over the
actual AOI. A candidate outside the payload-measured range is recorded safely,
its unnecessary raster/model intermediates are removed, and the next candidate
is tried. The accepted candidate continues through the existing crop, condition
and downlink stages from the same cloud result. The 60% crop cloud gate, cloud
thresholds, masks and all scientific calculations remain unchanged.

Container deployment uses one application Dockerfile with separate local and
Jetson Compose contracts:

- [`deployment/payload/Dockerfile`](../deployment/payload/Dockerfile);
- [`compose.payload.local.yaml`](../compose.payload.local.yaml);
- [`compose.payload.jetson.yaml`](../compose.payload.jetson.yaml);
- [`../.env.payload.example`](../.env.payload.example).

`VITA_PAYLOAD_BASE_IMAGE` is deliberately required. The image copies only the
shared and payload runtime packages, the cloud configuration and the two
selected checksum-pinned model artifacts. It does not copy training, ground,
integration, secrets or generated runtime data. The Earth Engine key is mounted
only at runtime and the API is published only on host loopback.

Before selecting a Jetson base image, a human operator must run on that target:

```bash
uname -m
id -u
id -g
cat /etc/os-release
cat /etc/nv_tegra_release 2>/dev/null
dpkg-query --show nvidia-jetpack 2>/dev/null
docker version
docker compose version
docker info
```

Choose an NVIDIA-supported `aarch64` PyTorch image matching the exact
JetPack/L4T and CUDA ABI. Do not reuse the local x86 CUDA image. After manual
placement of `secrets/earth-engine.json`, set the non-root container UID/GID in
`.env.payload` from `id -u` and `id -g`, then use the reviewed Jetson commands:

```bash
docker compose \
  --env-file .env.payload \
  -f compose.payload.jetson.yaml \
  build

docker compose \
  --env-file .env.payload \
  -f compose.payload.jetson.yaml \
  up -d

curl --fail http://127.0.0.1:8081/health
```

These commands require no `sudo`; stop if the current user cannot access
Docker. See [`deployment/README.md`](../deployment/README.md) for secret checks,
local validation, rollback, health criteria and the SSH-tunnel handoff. Jetson
deployment is always a human-reviewed step and is never performed by repository
scripts.

`scene-run` inspects a preprocessed Sentinel-2 or Balkan-1 GeoTIFF, resolves its
declared band order, and runs cloud detection through bounded 512-pixel windows.
When explicitly requested, scenes below the 60% cloud gate continue through
bounded crop segmentation and windowed condition processing. Every cloud-shadow,
cloud and invalid pixel in the operational unusable mask is excluded from crop
and condition outputs. The Balkan-1 route is provisional until real imagery is
radiometrically and spectrally validated.

Real Balkan-1 collections remain under ignored `data/` storage or at an
external path; they are not stored in this component. The versioned local
preprocessing and handoff utilities are documented in
[`scripts/balkan1/README.md`](../scripts/balkan1/README.md). They can supply an
explicit band order for existing L1ORT products that have no GeoTIFF band
descriptions, without copying or rewriting the multi-gigabyte source.

Balkan crop transfer remains blocked by default. An explicit
`--allow-provisional-balkan-crop` execution-only adapter is available for
accelerator and interface demonstrations. Its metadata records the broad-NIR to
narrow-NIR transfer as unvalidated; its outputs must not be presented as Balkan
crop accuracy evidence.

Run only from an explicit command:

```powershell
.\payload\scripts\run_scene.ps1 `
  -InputPath path\to\preprocessed_scene.tif `
  -Sensor sentinel-2 `
  -AcquiredAt 2026-07-26T12:00:00Z `
  -StopAfter cloud `
  -Output testing\runs\sentinel2_demo
```

Continue through crop classification:

```powershell
.\payload\scripts\run_scene.ps1 `
  -InputPath path\to\preprocessed_scene.tif `
  -Sensor sentinel-2 `
  -AcquiredAt 2026-07-26T12:00:00Z `
  -StopAfter crop `
  -MaxCropCloudPercentage 60 `
  -Output testing\runs\sentinel2_crop_demo
```

Continue through payload condition processing:

```powershell
.\payload\scripts\run_scene.ps1 `
  -InputPath path\to\preprocessed_scene.tif `
  -Sensor sentinel-2 `
  -AcquiredAt 2026-07-26T12:00:00Z `
  -StopAfter condition `
  -ReflectanceScale 10000 `
  -Output testing\runs\sentinel2_condition_demo
```

The condition stage uses bounded windows and deterministic scene-wide robust
statistics. It records all index summaries, condition components, evidence
quality, limitations and geospatial provenance under `condition_analysis/`.

The compact downlink builder reduces a completed condition result to exactly
three web-ready files:

- `scene.webp`: an 82-quality RGB overview, at most 1600 pixels on its longest side;
- `condition.png`: a lossless aligned RGBA crop-condition heat map; only valid,
  clear crop pixels are colored and every cloud, shadow, invalid, buffered,
  non-crop or unmeasured pixel is transparent;
- `scene.json`: exact metrics, score explanations, evidence quality,
  georeferencing, checksums and an adaptive interaction grid of up to 16 by 16
  cells, without creating cells smaller than 32 source pixels.

The bundle is deliberately separate from payload GeoTIFF intermediates. The web
application reads numbers from JSON and uses the images only for presentation.

Produce the terminal routine downlink package:

```powershell
.\payload\scripts\run_scene.ps1 `
  -InputPath path\to\preprocessed_scene.tif `
  -Sensor sentinel-2 `
  -AcquiredAt 2026-07-26T12:00:00Z `
  -StopAfter downlink `
  -ReflectanceScale 10000 `
  -Output testing\runs\sentinel2_downlink_demo
```

A successful run ends with `DOWNLINK_READY` and records the three absolute local
paths in payload `result.json`. The portable `scene.json` itself contains only
relative image references and checksums.

The crop route requires both Sentinel-2 `B08` for CloudSEN12 and `B8A` for the
selected Prithvi model. It writes crop probability, binary crop and confidence
GeoTIFFs, crop metadata and a combined PNG. A scene at or above the configured
cloud percentage is stopped before the crop model is loaded.

Cloud previews show RGB, the exact semantic classes, a display-feathered RGB
overlay, and the exact operational unusable mask. Thick cloud, thin cloud,
cloud shadow and invalid/nodata use distinct colors and report their pixel
fractions. Crop previews show RGB, continuous probability, a display-feathered
probability overlay with the accepted-crop edge, and the exact binary mask.
Feathering applies only to PNG presentation; all GeoTIFF masks, probabilities,
thresholds and statistics remain unchanged.

Each run writes one canonical `result.json` containing the summarized metadata
for every completed stage and links to detailed stage JSON, GeoTIFF masks and
PNG visualizations. API and database work should read `result.json`; the stage
files remain available for debugging and audit.

## Explicitly excluded

- the 49 GiB datasets;
- augmentation and training code;
- checkpoints other than the selected model;
- TensorBoard and training logs;
- databases, API and client application.

## Current limits

Crop inference is ready for tiled arrays, and the cloud stage streams large
GeoTIFFs without loading a full scene. Balkan-1 raw-band reconstruction and
real Balkan-1 cloud/crop validation are not yet implemented.

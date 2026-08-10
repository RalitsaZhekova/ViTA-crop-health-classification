# Jetson AGX Orin payload deployment

This is the end-to-end MVP deployment for a ground Windows computer and an NVIDIA Jetson AGX Orin 64 GB payload at `/data/code/VITA`. The four fixed demo scenes are provisioned once; source imagery is never uploaded as part of a job. A ground job transmits only sensor, payload-local path, region, and job identifiers through an SSH tunnel. The payload returns exactly three verified files: `scene.webp`, `condition.png`, and `scene.json`.

For the sensor contracts, stage-by-stage code map, local three-command workflow,
timing definitions, runtime artifacts, and detailed explanation of every optimization,
read the [pipeline and optimization guide](PIPELINE_GUIDE.md) first.

## Architecture and deployment choices

```text
Ground Windows                       Jetson AGX Orin 64 GB
----------------                    ---------------------------------
Invoke-VitaPayload.ps1              /data/code/VITA/data (read-only)
        |                           /data/code/VITA/payload/models (ro)
        | SSH local forward          |
        +===========================>| 127.0.0.1:8090
        |     small JSON request     | warm FastAPI worker
        |                            |  cloud: accepted direct TensorRT FP16
        |                            |  crop: accepted mixed-FP16 TensorRT
        |                            |  health/indices: CPU + GDAL threads
        |                            |  package exactly 3 artifacts
        | SCP + SHA-256              |
        <============================+
validate + catalog ingest
ground dashboard on 127.0.0.1:8000
```

The payload image extends `nvcr.io/nvidia/pytorch:25.01-py3-igpu`, matching the stack verified inside the target Orin's NVIDIA runtime: NumPy 1.26.4, PyTorch `2.6.0a0+ecf3bae40a.nv25.01`, CUDA 12.8, and TensorRT 10.8.0.40. This PyTorch build and the image's OpenCV binaries use the NumPy 1.x ABI, so the payload retains NumPy 1.26.4. TerraTorch 1.1.1 supports that ABI and pins segmentation-models-pytorch 0.5.0, which also satisfies OmniCloudMask 1.7.1. TorchGeo 0.7.1 is paired with Lightly 1.5.22 because later Lightly releases eagerly import a distributed-training API omitted by this Jetson PyTorch build; neither package's training path is used during VITA inference. The real 95,484,420-parameter Prithvi checkpoint was strict-loaded against this model stack with every key matched. The image adds pinned ONNX 1.17 and ONNX Script 0.1 for offline export and verifies TensorRT's Python runtime plus `trtexec`. Torch-TensorRT is not imported by the builder or service. Because standard Docker builds do not receive the NVIDIA runtime, `deploy.sh` performs ONNX export, TensorRT parsing, plan construction, and scientific parity on the target GPU before starting the service. Dependency, parser, plan, parity, readiness, and performance failures are fatal.

The target host is JetPack 7.2 / L4T R39.2. NVIDIA's published Jetson PyTorch compatibility table originally paired container 25.01 with JetPack 6.1, so this is not a vendor-certified version pairing. It is retained for the MVP because both the unmodified NGC image and the local NumPy-1.26 derivative passed a live `--runtime nvidia` CUDA check on this exact Orin, while moving to PyTorch 2.11 would change the inference/compiler stack. Record this exception in the release evidence and requalify against a JetPack-7.2-supported framework image before a mission production release.

This follows NVIDIA's [Jetson container tutorial](https://developer.nvidia.com/embedded/learn/tutorials/jetson-container): use an NVIDIA Jetson/NGC base, launch it with the NVIDIA runtime, and bind-mount payload data. It also adopts the relevant guidance from NVIDIA's [DeepStream Docker documentation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_docker_containers.html): Jetson and generic ARM images are distinct, the NVIDIA container toolkit is required, and the image must match the payload platform. The TIFFs, calibration sidecars, weights, runtime results, exported PyTorch graph, and Balkan grid cache are bind-mounted rather than copied into the application image.

DeepStream itself is deliberately not in this image. DeepStream is a GStreamer/video analytics pipeline; these inputs are scientific multi-band GeoTIFF rasters. It would add video codecs and plugins without accelerating Rasterio reprojection, spectral-index calculations, or this PyTorch segmentation graph. The NVIDIA PyTorch iGPU image is the smaller and better-matched base. If the project later ingests live video, DeepStream should be a separate service.

The acceleration policy is:

| Stage | MVP execution | Reason |
|---|---|---|
| OmniCloudMask ensemble | Direct FP16 TensorRT plans: fixed batch 1 at 700 px and batch 4 with exact logical-batch padding at 869/891 px | Offline acceptance derives every fixed-scene patch size and applies the unchanged 0.1% class-mismatch gate before publishing any plan |
| Prithvi crop segmentation | Direct weakly typed mixed-FP16 TensorRT plan, FP32 I/O, TF32 disabled, fixed batch 16 | Offline acceptance compares balanced real-scene tiles with the established CUDA-autocast FP16 source using unchanged 0.2% decision and 0.5% mean-probability gates; logits and thresholds are not calibrated or altered |
| Balkan 10 m preparation | Embedded overview read, one multiband average GDAL warp, checksum-keyed persistent grid | Avoids decoding four full-resolution bands separately while preserving the existing 10 m UTM, band-order, nodata, and reflectance contracts |
| Health indices and packaging | Exact vectorized NumPy statistics in RAM, concurrent RGB/overlay/grid/codec work | Routine runs avoid non-downlinked science rasters; lossless PNG level 1 and WebP method 0 favor the two-second latency contract |

The optimized Balkan warp remains on the CPU deliberately. Its measured reprojection
is only about 0.41 seconds after the overview read, while NVIDIA
[VPI dynamic remap](https://docs.nvidia.com/vpi/group__VPI__DynamicRemap.html) does not
provide area-average interpolation. Replacing the scientific average with a GPU
linear remap would change the product for a small possible saving. NVIDIA
[nvTIFF](https://docs.nvidia.com/cuda/nvtiff/) can decode Deflate float32 TIFF data on
Jetson and is a sensible later C++ optimization if profiling on Orin still identifies
TIFF decode as material; it does not itself implement the GeoTIFF reprojection or the
Python Rasterio contract used by the models.

ModelOpt is not required for FP16 TensorRT. The direct path does not import Torch-TensorRT, so its optional quantization warning is not part of this workflow. INT8 remains out of scope until a representative calibration set and a separate accuracy acceptance test exist.

The release profile uses `trtexec` builder optimization level 5 and
precision-specific timing caches. Crop is built and parity-qualified before the three
cloud plans, so a rejected crop candidate stops before the more expensive cloud tactic
search. The previous checksum-sealed accepted manifest remains untouched unless every
new parity gate passes.

Deployment builds TensorRT plans only in the offline builder container on the target Orin because serialized engines are tied to the TensorRT version and target GPU. It first exports and validates every ONNX graph, rejects zero-sized constants, parses all graphs before tactic construction, releases the PyTorch source models from GPU memory, and then runs `trtexec`. Plans and precision-specific timing caches live below `runtime/engines/tensorrt/direct`; repository images contain code and dependencies only—not imagery, weights, or engine plans.

The image creates a non-login service account with the numeric UID/GID of the account executing `deploy.sh`, which matches the owner of the VITA runtime directories; the target defaults are `2002:2002`. This gives Python and PyTorch a valid passwd identity while the NVIDIA runtime supplies the required video/render device groups. A private tmpfs home keeps library and compiler caches writable while the image filesystem remains read-only and all Linux capabilities remain dropped. The VITA-owned `runtime/payload` and `runtime/engines` bind mounts also use Docker's private `:Z` SELinux relabeling when SELinux is active. Before model loading, deployment creates and removes a probe file in each mount. All writable paths remain scoped below `/data/code/VITA` or private container tmpfs; no other team's directory or container is modified.

An installed TensorRT SDK or `trtexec` binary is not itself an accepted model engine. The service loads only plans named in a checksum-sealed `accepted.json` whose exact GPU/software signature, model hashes, I/O contracts, plan hashes, and parity evidence all validate. Candidate `.plan` files are inert if the builder fails, and the accepted manifest is replaced atomically only after every gate passes.

## 1. Provision exactly four demo scenes

The reviewed payload package is recorded in `deploy/payload/demo-assets.sha256` and contains exactly:

- Sentinel-2 Bulgaria/Thrace and Brazil/Mato Grosso GeoTIFFs;
- Balkan-1 3370 and 3408 GeoTIFFs;
- the adjacent validated calibration JSON for each Balkan image.

Sentinel metadata--acquisition time, reflectance scale, CRS, nodata, and band order--is embedded in each GeoTIFF, so Sentinel does not use a separate sidecar JSON. Balkan needs its `.crop_calibration.json` for scientific harmonisation and for the corrected web preview.

From the ground checkout, this command verifies all six data files and the three required model files, skips any identical remote files, copies only missing assets through SSH, verifies each temporary upload, and atomically installs it:

```powershell
.\scripts\ground\Install-VitaPayloadAssets.ps1 `
  -SshTarget payload-user@payload-host
```

Use `-SshPort` or `-IdentityFile` when needed. A mismatched existing remote file is never overwritten unless `-Force` is explicitly supplied. The two Balkan TIFFs total about 2.44 GB; no other imagery is transferred. The three model weights add about 440 MB and are skipped when their remote hashes already match. Use `-SkipModels` only after independently provisioning those exact weights.
Run `.\scripts\ground\Install-VitaPayloadAssets.ps1 -ValidateOnly` to verify the complete local package without opening an SSH connection.

On the Orin, confirm that the project, all four scenes, calibrations, and weights exist:

```bash
cd /data/code/VITA
test -f data/sentinel2/S2_20260610T091331_T35TLG_bulgaria-thrace.tif
test -f data/sentinel2/S2_20260115T135659_T21LXF_brazil-mato-grosso.tif
test -f data/balkan1/preprocessed/3370_L1ORT.tif
test -f data/balkan1/preprocessed/3370_L1ORT.crop_calibration.json
test -f data/balkan1/preprocessed/3408_L1ORT.tif
test -f data/balkan1/preprocessed/3408_L1ORT.crop_calibration.json
test -f payload/models/prithvi_crop_binary_single_frame_v1_weights.pt
```

The fast Balkan path requires a common embedded overview in the four reflectance
bands. Both proof-of-concept TIFFs contain factors 4, 8, 16, and 32. If GDAL is
available on the host, verify both files before deployment:

```bash
gdalinfo data/balkan1/preprocessed/3370_L1ORT.tif | grep -m 4 'Overviews:'
gdalinfo data/balkan1/preprocessed/3408_L1ORT.tif | grep -m 4 'Overviews:'
```

Compose sets `VITA_BALKAN_OVERVIEW_REQUIRED=1`, so an input without a suitable
overview fails service readiness instead of silently returning to the 12--13 second
full-resolution path. The implementation retains that exact, slower fallback for
development by setting `VITA_BALKAN_OVERVIEW_REQUIRED=0`. It never creates or changes
the payload source TIFF.

The two OmniCloudMask `.safetensors` files must be under `payload/models/omnicloudmask`. These assets are intentionally excluded from Git and the Docker build context. Compose mounts `data` and `payload/models` read-only at runtime.

Find the exact installed NVIDIA image tag:

```bash
docker image ls --format '{{.Repository}}:{{.Tag}}' | grep -E 'nvidia/.+pytorch|pytorch'
```

If it is not literally `nvcr.io/nvidia/pytorch:25.01-py3-igpu`, copy the example and set the actual tag:

```bash
cp deploy/payload.env.example deploy/payload.env
# Edit VITA_PAYLOAD_BASE_IMAGE in deploy/payload.env to the installed tag.
```

Do not choose an image based only on CUDA version. It must be a Jetson iGPU image compatible with the host JetPack/L4T release. The preflight prints `/etc/nv_tegra_release`, the installed JetPack package, Docker runtimes, and the complete in-container GPU stack.

Make the payload helpers executable after the first checkout (Git normally preserves these bits):

```bash
chmod +x deploy/payload/*.sh
./deploy/payload/probe.sh
./deploy/payload/preflight.sh
```

`probe.sh` is read-only and can run before the data package is present. Save its complete output for the deployment record. The preflight is successful only when the machine is a 64 GB Jetson Orin running `aarch64`, at least 10 GiB is free, the exact six-file demo manifest and model checksums pass, the exact validated ARM64 base-image ID is local, the NVIDIA Docker runtime works, CUDA is visible, and the expected NumPy/PyTorch/CUDA/TensorRT versions import. It also installs the required Ubuntu packages inside a disposable container, verifies `gdal-config`, performs Python dependency resolution including the pinned ONNX exporter in that same temporary environment, and validates the resolved Compose configuration before building. These checks do not change the base image, host packages, or payload files; Docker removes the temporary container when the check exits.

For timing runs, inspect the current Orin power policy:

```bash
sudo nvpmodel -q --verbose
sudo jetson_clocks --show
```

Select the approved maximum-performance profile for this specific module/carrier and run `sudo jetson_clocks` before benchmarking. Profile IDs vary by JetPack/device configuration, so this guide intentionally does not hard-code an `nvpmodel -m` number. Ensure adequate cooling; otherwise thermal throttling makes a two-second acceptance result meaningless.

## 2. Build and start the payload

From `/data/code/VITA`:

```bash
cp deploy/payload.env.example deploy/payload.env  # skip if already configured
./deploy/payload/deploy.sh
```

The script requires both model backends to be either `pytorch` or `tensorrt` and rejects the retired TensorRT CUDA-graph switch. In direct TensorRT mode it runs preflight, builds the image, validates CUDA/ONNX/TensorRT and all four sensor contracts, runs the offline builder, and starts Compose only after the builder publishes an accepted manifest. Service startup only deserializes the accepted plans; it never compiles. Unchanged ONNX, target, precision, and build arguments reuse checksum-verified plans, so a later parity retry does not repeat tactic construction. The script then runs fail-closed acceleration checks and the configured timed repetitions of all four scenes.

The last complete performance qualification, including every stage timing, is written
atomically to `runtime/payload/performance-acceptance.json` on both success and SLO
failure.

If startup fails or the container restarts, the script prints the last 200 log lines and
brings down only the explicitly named `vita-payload` Compose project. It retains the
bind-mounted exported PyTorch graph and Balkan-grid caches; these are validated reusable
deployment artifacts, not failed containers.

Before building or starting the service, the script removes the exact rejected VITA
crop/cloud TensorRT cache, artifact, and timing-cache paths from earlier experiments.
It never invokes a shared Docker or BuildKit prune and never touches another Compose
project.

Check status and logs:

```bash
curl --fail http://127.0.0.1:8090/healthz
./deploy/payload/logs.sh
docker compose -f deploy/compose.payload.yaml ps
```

A production-ready health response must show:

- `cuda_available: true`;
- the expected PyTorch/CUDA stack (the base still reports its installed optional
  TensorRT packages for release evidence);
- `crop_backend: "tensorrt"`, `crop_device: "cuda"`, `crop_inference_dtype: "fp32"`,
  `crop_tensorrt_precision: "mixed-fp16"`, `crop_tf32: false`, and
  `crop_batch_size: 16`;
- one checksum-bound direct crop TensorRT engine with its accepted parity record;
- `tensorrt_cudagraphs: false`;
- `cloud_backend: "omnicloudmask_tensorrt_fp16"` and `cloud_batch_size: 4`;
- three checksum-bound direct cloud TensorRT engines at 700, 869, and 891 px;
- cloud warmup profiles for batch 1 at 700 px and batches 1 and 4 at 869/891 px;
- four distinct `cloud_scene_warmup_profiles`, each with `kind: "fixed_input_profile"` and `prediction_retained: false`;
- two `balkan_analysis_caches` entries (a first build may report `cache_hit: false`; later startups report true);
- each Balkan cache reports `preprocessing_mode: "embedded_overview_then_average"` and `overview_factor: 4`.

The two fixed Sentinel scenes use 700 px model patches. OmniCloudMask's reviewed
no-data rule reduces the prepared 3408 and 3370 Balkan grids to 869 and 891 px. The
unused 1,000 px backend base is not built for this checksum-pinned release. The service
also reads each fixed input and runs a discarded cloud prediction so CUDA/cuDNN sees
every exact Sentinel and Balkan mosaic shape before readiness. Model construction,
warmup, and inference are pinned
to one long-lived worker thread because CUDA/cuDNN setup includes thread-local state;
warming on the application thread and inferring on a different FastAPI worker made the
first request pay about four extra seconds locally. No prediction, semantic mask, crop
result, condition map, or downlink is retained from warmup. If a fixed scene changes,
update `VITA_BALKAN_PREPARE_INPUTS` or `VITA_DEMO_SENTINEL_IMAGES`, the checksum
manifest, and the provisioning script inputs, then rebuild and rerun acceptance.

The service has one worker and rejects a concurrent job with HTTP 409. This prevents two 100M-parameter pipelines from competing for GPU memory and corrupting latency measurements. It binds to Jetson `127.0.0.1`; do not expose port 8090 in the firewall.

## 3. Prepare SSH from ground

Use key authentication and connect once interactively so the real payload host key is stored in `known_hosts`:

```powershell
ssh -p 22 payload-user@payload-host
```

The orchestration uses `BatchMode=yes` and never disables host-key checking. If a private key is not in the normal OpenSSH location, pass `-IdentityFile`.

On the ground checkout, install the current project so `vita-ingest` is available and build the dashboard image once:

```powershell
Set-Location D:\ML_ComputerVision\prithvi_crop_head_starter
.\.venv\Scripts\python.exe -m pip install -e .
docker compose -f deploy\compose.ground.yaml build
```

Only TCP SSH needs to be reachable from ground to payload. The job API travels inside the SSH local forward and the three outputs travel via SCP.

## 4. Run the proof-of-concept jobs from ground

Sentinel-2 Bulgaria/Thrace:

```powershell
.\scripts\ground\Invoke-VitaPayload.ps1 `
  -SshTarget payload-user@payload-host `
  -Sensor sentinel-2 `
  -Input sentinel2 `
  -Image S2_20260610T091331_T35TLG_bulgaria-thrace.tif `
  -RegionId sentinel-bulgaria-thrace
```

Sentinel-2 Brazil/Mato Grosso:

```powershell
.\scripts\ground\Invoke-VitaPayload.ps1 `
  -SshTarget payload-user@payload-host `
  -Sensor sentinel-2 `
  -Input sentinel2 `
  -Image S2_20260115T135659_T21LXF_brazil-mato-grosso.tif `
  -RegionId sentinel-brazil-mato-grosso
```

Balkan-1 3370:

```powershell
.\scripts\ground\Invoke-VitaPayload.ps1 `
  -SshTarget payload-user@payload-host `
  -Sensor balkan-1 `
  -Input balkan1/preprocessed/3370_L1ORT.tif `
  -RegionId balkan-3370
```

Balkan-1 3408:

```powershell
.\scripts\ground\Invoke-VitaPayload.ps1 `
  -SshTarget payload-user@payload-host `
  -Sensor balkan-1 `
  -Input balkan1/preprocessed/3408_L1ORT.tif `
  -RegionId balkan-3408
```

`Input` is relative to the payload's `/data/code/VITA/data` mount, so it must not start with `data/`. The Balkan calibration sidecar is discovered beside the TIFF; use `-CropCalibration` only for a different payload-local relative path.

Useful connection overrides are `-SshPort`, `-IdentityFile`, `-LocalTunnelPort`, and `-RemoteProjectRoot`. Every job gets a unique ID. Supplying `-JobId` is supported, but an existing payload or ground job is never overwritten.

For each invocation the script:

1. creates a temporary local SSH forward to payload loopback;
2. checks the warm payload service and acceleration stack;
3. uplinks only the JSON job request;
4. waits for cloud, crop, health, and packaging stages;
5. closes the tunnel;
6. SCPs exactly the three downlink files;
7. verifies each SHA-256 against the authenticated payload response;
8. validates the bundle contract and ingests it into `runtime/ground`;
9. starts the loopback-only ground dashboard container.

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/). Use `-SkipDashboard` if the existing local `vita-dashboard` process should remain in charge instead.

Artifacts are retained at:

```text
Payload full run:   /data/code/VITA/runtime/payload/runs/<job-id>/
Payload model cache: /data/code/VITA/runtime/engines/
Payload grid cache: /data/code/VITA/runtime/payload/cache/balkan-analysis/
Ground receipt:     runtime/downlink/<job-id>/
Ground catalog:     runtime/ground/scenes/<job-id>/
```

## 5. Two-second performance acceptance

The returned `payload_seconds` is the ready-service execution from scene intake through three-file packaging. It excludes SSH setup, service startup/model load/warmup, SCP, and ground ingest. `under_two_seconds` is computed from that value. No prediction or output is cached: each request still executes cloud, crop, condition, and packaging. Check `/healthz` before starting the clock; a request before readiness is a cold-start test.

`./deploy/payload/deploy.sh` automatically validates one accepted crop plan, two accepted
cloud plans, their sealed target/parity record, fixed batches, four discarded scene
warmups, both Balkan grids, and every required RAM fast path. It then runs all four
scenes three times and fails if
**any** run is 2.0 seconds or slower. Rerun the gate manually with:

```bash
docker compose --env-file deploy/payload.env -f deploy/compose.payload.yaml exec -T payload \
  python -m prithvi_payload.performance_acceptance
```

`VITA_SKIP_PERFORMANCE_ACCEPTANCE=1` is for diagnosis only; a deployment started with
it has not passed production acceptance.

Do not use the one-shot `vita-mvp` command as the payload latency benchmark. A local
profile attributed about 3.37 seconds of a 4.39-second cloud load to importing
OmniCloudMask, segmentation-models-pytorch, timm, torchvision, and TorchDynamo; the
actual two-checkpoint construction/load was about 0.81 seconds. A new Python process
must pay those imports again. The single-worker `vita-payload-server` is therefore the
latency architecture: it loads and warms once, reports ready, then reuses both models.

The automatic gate uses three repetitions per scene. Increase
`VITA_PERFORMANCE_REPETITIONS` for release characterization and retain the JSON report.
Watch the payload concurrently:

```bash
tegrastats --interval 500
```

Use the stage timings in the response and payload `result.json`, not only the total:

- `reported_stage_total_seconds` sums the eight top-level stages and
  `orchestration_seconds` is the measured remainder, so they add back to
  `payload_seconds`; cloud/crop inference and mask-processing values are nested inside
  their corresponding stage totals and must not be added a second time;
- high `cloud_inference_seconds`: validate all three accepted FP16 plans, the batch-1
  700 and batch-4 869/891 contracts, and completed exact-scene warmups in
  `/healthz`;
- high `crop_inference_seconds`: validate one accepted mixed-FP16/no-TF32 direct plan
  and confirm that the service did not fall back to PyTorch;
- high `shared_analysis_grid_seconds`: verify `/healthz` reports
  `embedded_overview_then_average`, factor 4, the expected Balkan cache key, and a
  persistent writable `runtime/payload/cache/balkan-analysis` directory;
- high condition or packaging time: inspect their detailed internal timings; the routine path uses exact in-memory science products, concurrent preparation/encoding, WebP method 0, and lossless PNG level 1;
- high total only on the first request: warmup or engine caching is incomplete.

Earlier Torch-TensorRT experiments produced about 2.2% crop decision disagreement and
0.299141% cloud class disagreement, so those engines remain rejected and none of their
calibration or partition workarounds is used. The direct path exports ONNX itself,
builds complete TensorRT plans with `trtexec`, preserves the source logits and
thresholds, and applies the same unchanged parity gates before publication. This is a
new qualification path, not evidence that the historical engines became accurate; the
target Orin build must still pass before the service can report direct TensorRT ready.

Historical optimization measurements on the RTX 3060 development machine showed that
the reviewed FP32 change reduced the already
materialized cloud stage to about 0.31 seconds for the 526×681 Sentinel scene and
1.19 seconds for the 1,739×2,132 Balkan grid. Both optimized semantic rasters matched
the saved FP32 class masks pixel-for-pixel in the controlled validation run. A warm
local payload-service Sentinel run completed the full payload pipeline and three-file
bundle in about 1.7–1.9 seconds. After the cache and same-thread warmup changes, the
first accepted local Balkan request completed in about 6.66 seconds: intake 0.03,
shared-grid lookup 0.004, cloud 1.16, crop 1.32, condition 2.58, packaging 1.51, and
reported orchestration 0.05 seconds. The generated WebP, PNG, interaction grid,
metrics, and condition result matched the prior optimized bundle exactly. These are
diagnostic results, not Orin acceptance
numbers. The current compact FP32/PyTorch upper baseline is about 3.3 seconds for
Balkan 3408 (cloud 1.27, crop 0.74, exact condition 0.76, packaging 0.42) and below one
second for Sentinel. The local native-FP16 parity pass changed
43 of 358,206 valid Sentinel classes (0.0120%) and 3 of 2,077,729 valid Balkan classes
(0.00014%) relative to the saved FP32 masks. Those results support the strict target
gate; mission-wide FP16 approval beyond the four packaged size/scene envelope remains
a separate scientific acceptance decision.

Changing the fixed crop batch from 8 to 16 changes CUDA reduction order slightly but
not the model or threshold. On Balkan 3408 the measured crop-coverage delta was
0.00130 percentage points, the condition-score delta was 0.00472 points, and the label
remained `High anomaly`. The production parity probe uses the same fixed batch-16
contract as inference and draws four deterministic tiles from each packaged scene.

Input dimensions fundamentally bound runtime. The fixed two-second service-level objective therefore applies to the four checksum-pinned scenes and their reviewed size envelope; the code bounds the Balkan cloud analysis grid at 25 million pixels and acceptance records the exact dimensions and bytes.

The two packaged Sentinel scenes are each 700 x 700 pixels and 3.3--3.5 MiB. Balkan `3370_L1ORT.tif` is 10,943 x 13,627 pixels and 1.14 GiB; its shared 10 m grid is 1,783 x 2,190 pixels. Balkan `3408_L1ORT.tif` is 10,745 x 13,340 pixels and 1.13 GiB; its shared 10 m grid is 1,739 x 2,132 pixels. The original local 3408 response was about 30.03 seconds, including 13.09 seconds of grid preparation, 5.70 seconds of condition work, and 2.53 seconds of packaging.

The new cold grid build reads the TIFF's existing factor-4 overview once and performs
one four-band average reprojection. On the RTX 3060 development machine repeated cold
grid builds took about 1.43--1.76 seconds versus 12.73 seconds for the previous four
full-resolution warps, an 86--89% reduction. Full source SHA-256 verification remains
in intake and took about 3.0--4.2 seconds locally for this 1.16 GiB file; it is
intentionally not skipped or hidden. Therefore a genuinely unseen cold local file is
still expected to take roughly 10--11 seconds end to end, while the fixed
checksum-verified startup cache keeps normal requests on the prepared grid. Only the
automatic repeated Orin acceptance can establish whether every fixed scene is below
two seconds.

This is a controlled speed/accuracy trade, not a claim of bitwise equivalence. Against
the previous full-resolution average grid on the supplied scene, reflectance-band
correlations were 0.978--0.985. A full FP32/PyTorch pipeline comparison found 0.047%
invalid-mask disagreement, 0.832% cloud-class disagreement on common-valid pixels,
1.800% crop-mask disagreement, crop-probability correlation 0.9955, and condition-map
correlation 0.9971. The aggregate condition score moved from 32.40 to 34.54 while the
final condition label remained `High anomaly`. Establish
mission tolerances on more scenes before treating the overview path as scientifically
qualified beyond this MVP. The output metadata records the selected overview,
resampling, grid, band order, memory estimate, and detailed preparation timings so the
choice is auditable.

## 6. GitHub Container Registry

`.github/workflows/containers.yml` publishes two code-only images:

- `ghcr.io/<owner>/vita-ground:<commit-sha>` on a normal GitHub-hosted AMD64 runner;
- `ghcr.io/<owner>/vita-payload:<commit-sha>` on a self-hosted Jetson runner labeled `self-hosted`, `Linux`, `ARM64`, and `jetson`.

The payload is built natively because its NVIDIA iGPU base is ARM64/Jetson-specific. QEMU builds cannot validate the GPU runtime and make native dependency failures harder to diagnose. The workflow also launches the built image with `--runtime nvidia` and verifies CUDA before publishing it.

For deployment by immutable Git SHA:

```bash
docker login ghcr.io
docker pull ghcr.io/<owner>/vita-payload:<commit-sha>
cp deploy/payload.env.example deploy/payload.env
# Set VITA_PAYLOAD_IMAGE=ghcr.io/<owner>/vita-payload:<commit-sha>
VITA_SKIP_BUILD=1 ./deploy/payload/deploy.sh
```

Keep the NGC base and GHCR app image immutable in a release record. Never publish `data`, model weights, `runtime`, credentials, or TensorRT caches; `.dockerignore` enforces this boundary.

## 7. Troubleshooting

`CUDA_REQUIRED=1 but torch.cuda.is_available() is false` means the container was not launched through the NVIDIA runtime, the NVIDIA container toolkit is not configured, or the base is incompatible with host L4T. Run `deploy/payload/preflight.sh` before changing Python packages.

`deploy/payload.env must set VITA_CROP_BACKEND=pytorch` means an older copied payload
environment is still requesting the rejected crop TensorRT path. Change that one value
to `pytorch`; native CUDA is enforced separately by deployment acceptance.

`deploy/payload.env must set VITA_CLOUD_BACKEND=pytorch` or
`VITA_TRT_CUDAGRAPHS=0` means an older copied environment still requests the rejected
cloud compiler route. Set those exact values before rebuilding. If logs contain
`TensorRT Conversion Context`, cloud engine compilation, or cloud compiler parity,
the running image/environment is not this production revision; stop it and verify the
checked-out commit plus these three backend settings. Do not relax parity tolerances.

`PERFORMANCE_SLO_FAILED` includes every repetition and the minimum, median, and maximum
for each scene. Use the returned stage breakdown and `tegrastats` to diagnose the
failure. Do not raise the target or skip the gate to label the deployment production.

The ModelOpt quantization warning can be ignored if it appears during optional package
inspection in preflight. The production service does not import the cloud compiler or
request quantization. Do not install ModelOpt merely to suppress the warning.

To audit deployment residue without touching another team's Docker objects, use:

```bash
docker ps -a --filter label=com.docker.compose.project=vita-payload \
  --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
docker image ls --filter reference='vita-payload*' \
  --format 'table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.ID}}'
du -sh runtime/payload runtime/engines
```

Deployment removes only rejected VITA TensorRT cache names and stale `accept-*`
directories directly below `runtime/payload/runs`. Each completed performance request
also removes its own disposable acceptance directory after recording the timing result.

Do not run `docker system prune`, `docker builder prune`, or an unfiltered image prune
on the shared Jetson. BuildKit's default cache is shared by every team and cannot be
safely attributed to VITA from its cache IDs. Failed `docker compose run --rm` probes
remove their own containers; the successful app layers are referenced by
`vita-payload:1.0.0` and should be retained.

HTTP 422 with a cloud-gate status means the scientific pipeline did not produce a complete downlink; inspect that job's payload `result.json`. It is not a transport failure.

An SSH tunnel error should be resolved by testing `ssh -p <port> user@host`, confirming the host key, key permissions, and that the payload service is healthy. Do not work around it with `StrictHostKeyChecking=no`.

If port 18090 is occupied on ground, pass a different `-LocalTunnelPort`. If port 8000 is occupied, start the dashboard with another port:

```powershell
.\scripts\ground\Start-VitaDashboard.ps1 -Port 8001
```

## Release acceptance checklist

- Payload preflight passes with the recorded JetPack/L4T and base-image digest.
- Payload Docker build succeeds without replacing the NVIDIA PyTorch/CUDA/TensorRT stack.
- Health reports direct TensorRT mixed-FP16 crop inference with FP32 I/O and TF32
  disabled, direct TensorRT FP16 cloud inference, one crop engine, three cloud engines,
  the exact batch-1/batch-4 profiles, and four discarded scene warmups.
- All four fixed input scenes complete every configured acceptance repetition without errors.
- Every measured `payload_seconds` value is below 2.0 seconds for the agreed pixel-size envelope.
- Scientific outputs retain the accepted native CUDA source-model contracts.
- Ground receives exactly WebP, PNG, and JSON and verifies all SHA-256 values.
- Ground catalog validation passes and all four scenes render in the dashboard.
- Only SSH is network-reachable; payload and dashboard HTTP ports remain loopback-only.
- GHCR images are pinned by commit SHA/digest; models and runtime caches remain outside images.

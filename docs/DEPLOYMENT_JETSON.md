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
        |                            |  cloud: Torch-TensorRT FP16
        |                            |  crop: Torch-TensorRT FP16
        |                            |  health/indices: CPU + GDAL threads
        |                            |  package exactly 3 artifacts
        | SCP + SHA-256              |
        <============================+
validate + catalog ingest
ground dashboard on 127.0.0.1:8000
```

The payload image extends `nvcr.io/nvidia/pytorch:25.01-py3-igpu`, matching the stack verified inside the target Orin's NVIDIA runtime: NumPy 1.26.4, PyTorch `2.6.0a0+ecf3bae40a.nv25.01`, CUDA 12.8, TensorRT 10.8.0.40, and Torch-TensorRT 2.6.0a0. This PyTorch build and the image's OpenCV binaries use the NumPy 1.x ABI, so the payload retains NumPy 1.26.4. TerraTorch 1.1.1 supports that ABI and pins segmentation-models-pytorch 0.5.0, which also satisfies OmniCloudMask 1.7.1. TorchGeo 0.7.1 is paired with Lightly 1.5.22 because later Lightly releases eagerly import a distributed-training API omitted by this Jetson PyTorch build; neither package's training path is used during VITA inference. The real 95,484,420-parameter Prithvi checkpoint was strict-loaded against this model stack with every key matched. The build runs `pip check`, imports every CPU-safe dependency, and asserts the installed Torch-TensorRT distribution version. Because standard Docker builds do not receive the NVIDIA runtime, `deploy.sh` then launches the completed image with `runtime: nvidia` and requires CUDA, TensorRT, and Torch-TensorRT to import successfully before starting the service. Dependency, GPU-runtime, model-engine, parity, and performance failures are all fatal.

The target host is JetPack 7.2 / L4T R39.2. NVIDIA's published Jetson PyTorch compatibility table originally paired container 25.01 with JetPack 6.1, so this is not a vendor-certified version pairing. It is retained for the MVP because both the unmodified NGC image and the local NumPy-1.26 derivative passed a live `--runtime nvidia` CUDA check on this exact Orin, while moving to PyTorch 2.11 would change the inference/compiler stack. Record this exception in the release evidence and requalify against a JetPack-7.2-supported framework image before a mission production release.

This follows NVIDIA's [Jetson container tutorial](https://developer.nvidia.com/embedded/learn/tutorials/jetson-container): use an NVIDIA Jetson/NGC base, launch it with the NVIDIA runtime, and bind-mount payload data. It also adopts the relevant guidance from NVIDIA's [DeepStream Docker documentation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_docker_containers.html): Jetson and generic ARM images are distinct, the NVIDIA container toolkit is required, and the image must match the payload platform. The TIFFs, calibration sidecars, weights, runtime results, and target-built TensorRT caches are bind-mounted rather than copied into the application image.

DeepStream itself is deliberately not in this image. DeepStream is a GStreamer/video analytics pipeline; these inputs are scientific multi-band GeoTIFF rasters. It would add video codecs and plugins without accelerating Rasterio reprojection, spectral-index calculations, or this PyTorch segmentation graph. The NVIDIA PyTorch iGPU image is the smaller and better-matched base. If the project later ingests live video, DeepStream should be a separate service.

The acceleration policy is:

| Stage | MVP execution | Reason |
|---|---|---|
| OmniCloudMask ensemble | FP16 Torch-TensorRT, fixed static profiles, batch up to 4, engine/timing cache | The exact two-model mean-logit graph is compiled and parity-checked for every demo-scene shape before readiness |
| Prithvi crop segmentation | FP16 Torch-TensorRT, fixed batch 16, engine/timing cache | The 64 GB Orin amortizes dispatch across more tiles; a deterministic probability/decision parity probe runs before readiness |
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

ModelOpt is not required for FP16 TensorRT. The observed `Unable to import quantization op` warning is harmless here: the deployment never requests a quantized graph. INT8 should only be introduced after a representative calibration set and an accuracy acceptance test exist.

TensorRT engines are created on the Orin and persisted in `runtime/engines`. Do not build or publish those caches from another GPU: NVIDIA documents that serialized engines are tied to their platform, TensorRT version, and target GPU, and JetPack does not support TensorRT hardware-compatibility mode. The repository's GitHub images contain code and dependencies only—not imagery, weights, or engine plans.

The service runs with the numeric UID/GID of the account executing `deploy.sh`, which matches the owner of the VITA runtime directories; the target defaults are `2002:2002`. The NVIDIA runtime supplies the required video/render device groups. A private tmpfs home keeps library caches writable while the image filesystem remains read-only and all Linux capabilities remain dropped. The VITA-owned `runtime/payload` and `runtime/engines` bind mounts also use Docker's private `:Z` SELinux relabeling when SELinux is active. Before model loading, deployment creates and removes a probe file in each mount. All writable paths remain scoped below `/data/code/VITA` or private container tmpfs; no other team's directory or container is modified.

An installed TensorRT SDK or `trtexec` binary is not itself a model engine. If the payload also contains a `.engine`/`.plan` file, reuse it only after proving that it was built from this exact crop/cloud graph and weights on this Orin with TensorRT 10.8. The deployment therefore builds model-specific Torch-TensorRT partitions rather than silently trusting an unidentified plan.

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

`probe.sh` is read-only and can run before the data package is present. Save its complete output for the deployment record. The preflight is successful only when the machine is a 64 GB Jetson Orin running `aarch64`, at least 10 GiB is free, the exact six-file demo manifest and model checksums pass, the exact validated ARM64 base-image ID is local, the NVIDIA Docker runtime works, CUDA is visible, and the expected NumPy/PyTorch/CUDA/TensorRT/Torch-TensorRT versions import. It also installs the required Ubuntu packages inside a disposable container, verifies `gdal-config`, performs Python dependency resolution in that same temporary environment, and validates the resolved Compose configuration before building. These checks do not change the base image, host packages, or payload files; Docker removes the temporary container when the check exits.

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

The script runs preflight, builds the app layer on the already-installed NVIDIA image, validates all four sensor contracts from inside the final image, starts Compose with `runtime: nvidia`, and waits up to 90 minutes for first-time export, all target-built TensorRT profiles, parity validation, both Balkan analysis grids, and exact-scene warmup. It then runs fail-closed acceleration checks and three timed repetitions of all four scenes. First startup can be slow; subsequent restarts reuse export, engine, timing, and checksum-keyed Balkan grid caches.

Check status and logs:

```bash
curl --fail http://127.0.0.1:8090/healthz
./deploy/payload/logs.sh
docker compose -f deploy/compose.payload.yaml ps
```

A production-ready health response must show:

- `cuda_available: true`;
- the expected PyTorch/CUDA/TensorRT/Torch-TensorRT versions;
- `crop_backend: "tensorrt"`;
- `crop_tensorrt_engine_count` greater than zero;
- `crop_batch_size: 16` and a crop parity record within the configured decision/probability tolerances;
- `tensorrt_cudagraphs: true`, captured during warmup for lower fixed-shape launch overhead;
- `cloud_backend: "omnicloudmask_tensorrt_fp16"`;
- `cloud_batch_size: 4`, a positive cloud TensorRT engine count, and a parity record for every static profile;
- cloud warmup profiles for batch 1 at 1,000 px and batches 1 and 4 at 869 px, plus profiles discovered by exact-scene warmup;
- four distinct `cloud_scene_warmup_profiles`, each with `kind: "fixed_input_profile"` and `prediction_retained: false`;
- two `balkan_analysis_caches` entries (a first build may report `cache_hit: false`; later startups report true);
- each Balkan cache reports `preprocessing_mode: "embedded_overview_then_average"` and `overview_factor: 4`.

The 1,000 px profile is the fixed Sentinel path. For the proof-of-concept Balkan grid,
OmniCloudMask's reviewed no-data rule reduces its model patch to 869 px. The service
also reads each fixed input and runs a discarded cloud prediction so that every exact
Sentinel and Balkan mosaic shape has a validated static engine. Model construction,
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
Payload TRT cache:  /data/code/VITA/runtime/engines/
Payload grid cache: /data/code/VITA/runtime/payload/cache/balkan-analysis/
Ground receipt:     runtime/downlink/<job-id>/
Ground catalog:     runtime/ground/scenes/<job-id>/
```

## 5. Two-second performance acceptance

The returned `payload_seconds` is the ready-service execution from scene intake through three-file packaging. It excludes SSH setup, service startup/model load/target engine build/warmup, SCP, and ground ingest. `under_two_seconds` is computed from that value. No prediction or output is cached: each request still executes cloud, crop, condition, and packaging. Check `/healthz` before starting the clock; a request before readiness is a cold-start test.

`./deploy/payload/deploy.sh` automatically validates both TensorRT backends, fixed
batches, compiler parity, four discarded scene warmups, both Balkan grids, and every
required RAM fast path. It then runs all four scenes three times and fails if **any**
run is 2.0 seconds or slower. Rerun the gate manually with:

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
- high `cloud_inference_seconds`: validate FP16 TensorRT, batch 4, positive engine counts, and parity for every exact static profile in `/healthz`;
- high `crop_inference_seconds`: confirm the health response reports TensorRT engine partitions and cache hits appear in logs;
- high `shared_analysis_grid_seconds`: verify `/healthz` reports
  `embedded_overview_then_average`, factor 4, the expected Balkan cache key, and a
  persistent writable `runtime/payload/cache/balkan-analysis` directory;
- high condition or packaging time: inspect their detailed internal timings; the routine path uses exact in-memory science products, concurrent preparation/encoding, WebP method 0, and lossless PNG level 1;
- high total only on the first request: warmup or engine caching is incomplete.

Before accepting FP16/TensorRT, run the same scenes with `VITA_CROP_BACKEND=pytorch` and `VITA_CLOUD_INFERENCE_DTYPE=fp32`, then compare class percentages, crop percentage, condition score/label, and visual masks against the accelerated output. Quantization is out of scope until this parity check and a representative calibration dataset are formalized. Do not enable `torch.backends.cudnn.benchmark` without measuring startup as well as steady state; fixed-shape autotuning can make service warmup much longer.

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
second for Sentinel. TensorRT can only be validated on the target Orin. The local FP16
parity pass changed
43 of 358,206 valid Sentinel classes (0.0120%) and 3 of 2,077,729 valid Balkan classes
(0.00014%) relative to the saved FP32 masks. The project still requires an explicit
mission accuracy tolerance before FP16 is declared scientifically accepted.

Changing the fixed crop batch from 8 to 16 changes CUDA reduction order slightly but
not the model or threshold. On Balkan 3408 the measured crop-coverage delta was
0.00130 percentage points, the condition-score delta was 0.00472 points, and the label
remained `High anomaly`. The production parity probe uses the same fixed batch-16
contract as inference.

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

`Crop model TensorRT compilation failed` is intentionally fatal with `VITA_TRT_STRICT=1`. Inspect payload logs for an unsupported operator or memory failure. For diagnosis only, set `VITA_TRT_STRICT=0`; the health endpoint will then report `crop_backend: pytorch`, which does not satisfy TensorRT acceptance.

`Torch-TensorRT produced no TensorRT engine partitions` means compilation technically returned but did not accelerate any graph segment. This is treated as failure rather than silently claiming TensorRT.

`Cloud TensorRT received an unwarmed profile after readiness` means the request shape
was not one of the four accepted MVP paths. Add the new checksum-pinned scene to
startup warmup and re-run parity/latency acceptance; never compile a new engine inside
a timed request.

`PERFORMANCE_SLO_FAILED` includes every repetition and the minimum, median, and maximum
for each scene. Use the returned stage breakdown and `tegrastats` to diagnose the
failure. Do not raise the target or skip the gate to label the deployment production.

The ModelOpt quantization warning can be ignored for FP16. Do not install ModelOpt merely to suppress the warning.

HTTP 422 with a cloud-gate status means the scientific pipeline did not produce a complete downlink; inspect that job's payload `result.json`. It is not a transport failure.

An SSH tunnel error should be resolved by testing `ssh -p <port> user@host`, confirming the host key, key permissions, and that the payload service is healthy. Do not work around it with `StrictHostKeyChecking=no`.

If port 18090 is occupied on ground, pass a different `-LocalTunnelPort`. If port 8000 is occupied, start the dashboard with another port:

```powershell
.\scripts\ground\Start-VitaDashboard.ps1 -Port 8001
```

## Release acceptance checklist

- Payload preflight passes with the recorded JetPack/L4T and base-image digest.
- Payload Docker build succeeds without replacing the NVIDIA PyTorch/CUDA/TensorRT stack.
- Health reports CUDA, FP16 TensorRT cloud and crop execution, positive engine counts, fixed batch 4/16, and passing compiler parity.
- All four fixed input scenes complete every configured acceptance repetition without errors.
- Every measured `payload_seconds` value is below 2.0 seconds for the agreed pixel-size envelope.
- Accelerated scientific outputs pass the approved FP32/PyTorch parity tolerances.
- Ground receives exactly WebP, PNG, and JSON and verifies all SHA-256 values.
- Ground catalog validation passes and all four scenes render in the dashboard.
- Only SSH is network-reachable; payload and dashboard HTTP ports remain loopback-only.
- GHCR images are pinned by commit SHA/digest; models and target-built TensorRT caches remain outside images.

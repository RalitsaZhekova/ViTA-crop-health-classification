# Balkan-1 local imagery workflow

This directory contains only versioned workflow and preprocessing code. Full
Balkan-1 acquisitions stay outside `payload/`, outside container build contexts
and outside Git.

## Local data layout

The default workspace is:

```text
data/balkan1/
  raw/
    <scene-id>/              delivered TIFF and acquisition metadata
  preprocessed/
    <scene-id>_L1ORT.tif     georeferenced five-band analysis product

testing/inputs/balkan1/      optional bounded proof chips only
testing/runs/                cloud/crop/condition/downlink run products
```

Everything below `data/`, `testing/inputs/balkan1/` and `testing/runs/` is
ignored by Git. `data` and `testing/inputs` are also excluded from Docker build
contexts. The workflow accepts external roots, so an existing multi-gigabyte
collection does not need to be copied into the checkout.

The inspected collection has raw bands named `Band_1`, `Band_2`, `Band_3`,
`Band_7`, `Band_0`. Its imager metadata records central wavelengths of 490,
560, 665 and 842 nm for the blue, green, red and NIR channels; the remaining
625 nm channel is the panchromatic product. The existing five-band L1ORT files
retain this order but have no GeoTIFF band descriptions. The launchers therefore
declare `BLUE GREEN RED NIR PAN` explicitly. This mapping is collection-specific
and must be reviewed again for another delivery.

Every inspected acquisition includes `position.csv`, `attitude.csv`, the raw
TIFF and either decoded JSON or a packet-extraction log. The extraction logs
retain per-line exposure timestamps, PPS samples and UTC user-data anchors, so
the processor recovers those values even for the six scenes whose JSON and
packet stream are no longer present. This is sufficient evidence to build an
initial line-by-line navigation model.

The delivered raw folders do not contain DSNU/PRNU or absolute-gain tables, a
DEM, an independent map reference, or atmospheric state. The supplied
production logs show that those were loaded from separate paths such as
`/code/assets/balkan/GIPPs/GIPP_BLK1.yml`, per-scene `l1c_dem/merged.tif` and
`l1c_reference_s2/s2_ref.tif`. Do not claim absolute L1B radiance/reflectance,
terrain-corrected L1C, or surface reflectance merely from the navigation files.

## Add a preprocessor

Place the Python implementation under `scripts/balkan1/preprocessors/` so it is
versioned independently from imagery. The generic launcher invokes only scenes
selected by `--scene-id` unless `--all` is supplied explicitly.

The default preprocessor contract is:

```text
python your_preprocessor.py --input <raw-scene-folder> --output <output-tiff>
```

Run one scene from the default local layout:

```powershell
.venv\Scripts\python.exe scripts\balkan1\preprocess_collection.py `
  scripts\balkan1\preprocessors\your_preprocessor.py `
  --scene-id 3036
```

For an existing external collection, point to its current roots without moving
or duplicating it:

```powershell
.venv\Scripts\python.exe scripts\balkan1\preprocess_collection.py `
  scripts\balkan1\preprocessors\your_preprocessor.py `
  --raw-root D:\path\to\collection `
  --preprocessed-root D:\path\to\collection\SEN2_OUTPUT `
  --scene-id 3036
```

If the preprocessor uses another CLI, put its arguments after `--`. The
launcher expands `{scene_id}`, `{scene_dir}`, `{raw_root}`,
`{preprocessed_root}` and `{output}`. Use `--dry-run` to inspect the exact
command first. Quote arguments containing placeholders in PowerShell.

Only production implementations backed by real mission inputs belong under
`preprocessors/`. Mock calibration, synthetic orbit fallbacks and educational
reference processors are intentionally excluded from this workflow.

## Real L0R validation and minimum L1A

The supported non-mock starting point is `process_l1a.py`. It validates the
delivered packet/JSON/TIFF reconstruction and uses both documented detector
margins. On every line and band it measures the dark bias from the 68 real
calibration pixels at the left and right edges, interpolates the cross-track
dark plane, subtracts it, removes the remaining 88-pixel inactive margin,
estimates only high-frequency fixed-pattern striping from the real scene, and
registers all bands to Red using real SIFT feature matches with a RANSAC quality
gate, then accepts a dense phase-correlation translation only when it improves
the feature residual. The delivered `DarkOffset` hardware setting is recorded but is not
misinterpreted as a black level. `--black-level-dn` is an explicit constant
override for the measured dark-reference model and should be used only when an
actual calibration source requires it.

Validate the delivered L0 reconstruction without writing an image:

```powershell
.venv\Scripts\python.exe scripts\balkan1\process_l1a.py 3036 --validate-only
```

Build one minimum L1A product:

```powershell
.venv\Scripts\python.exe scripts\balkan1\process_l1a.py 3036
```

Run the dense dark correction, fixed-pattern correction, affine resampling and
validity-mask resampling on CUDA while retaining bounded CPU raster I/O and
feature-based control:

```powershell
.venv\Scripts\python.exe scripts\balkan1\process_l1a.py 3036 --device cuda
```

CUDA is opt-in and fails closed when unavailable. The L1A manifest records the
GPU identity, CUDA version, peak allocation, accelerated operations and phase
timings. The default remains CPU so existing workflows are unchanged.

Regenerate only the comparison image with a brighter or darker display gamma
without rewriting the L1A TIFF:

```powershell
.venv\Scripts\python.exe scripts\balkan1\process_l1a.py 3036 `
  --preview-only --preview-gamma 0.65
```

Gamma affects the PNG only. Values below 1 brighten midtones; the L1A pixels
and metadata remain unchanged. This normal preview deliberately keeps the raw
and L1A products in their native sensor orientation and does not add the
north-up L1ORT panel.

Validate the L1A against the supplied L1ORT without making that reference an
input to preprocessing:

```powershell
.venv\Scripts\python.exe scripts\balkan1\validate_l1a.py 3036
```

This flips and robustly aligns bounded overviews for QA, reports feature-match
and per-band correlation/error metrics, and writes a three-panel aligned
comparison in sensor-row orientation. Its per-scene affine fits are diagnostics
only; they are never applied to L1A data and must not be reused as physical
calibration.

Outputs are written to `data/balkan1/derived/l1a/`: a validated L0R manifest,
the five-band `*_L1A_MIN.tif`, a detailed L1A manifest and a comparison PNG.
The product is real corrected DN in registered sensor space. It is explicitly
not radiance, reflectance, georeferenced or orthorectified and must not yet be
sent to the cloud/crop models.

Validate every delivered scene, then process the collection sequentially:

```powershell
.venv\Scripts\python.exe scripts\balkan1\process_l1a_collection.py `
  --all --validate-only

.venv\Scripts\python.exe scripts\balkan1\process_l1a_collection.py --all
```

Sequential execution is intentional because every product is approximately a
gigabyte and concurrent readers/writers would contend for disk and memory.

## Visualize preprocessing before inference

Each `<scene-id>_Raw.tif` already stores five band planes in one TIFF. It is
not yet a combined map product: the planes are raw DN, displaced in time and
sensor geometry, and have no CRS. The L1ORT product combines them only after
correction, band registration and orthorectification.

Create a preprocessing-only before/after explanation from an existing raw and
trusted L1ORT pair:

```powershell
.venv\Scripts\python.exe scripts\balkan1\visualize_preprocessing.py 3036
```

The command reads bounded windows and writes an ignored PNG plus a JSON
manifest under `testing/runs/balkan1_preprocessing_3036/`. It shows the five
raw bands, raw RGB misregistration, registered L1ORT RGB, NIR false color and
panchromatic detail. It does not invoke cloud detection, crop classification,
or modify either source TIFF.

## Stage a bounded proof chip

Review every delivered L1ORT at bounded resolution before selecting a proof
window:

```powershell
.venv\Scripts\python.exe scripts\balkan1\visualize_collection.py
```

The ignored contact sheet shows true-color and NIR false-color panels. Its
NDVI threshold is only a vegetation-coverage aid for human scene selection; it
is not a crop prediction or accuracy result.

Full L1ORT products are roughly gigabyte-scale. Create an ignored, bounded chip
for fast pipeline and target-acceleration tests:

```powershell
.venv\Scripts\python.exe scripts\balkan1\stage_sample.py `
  D:\path\to\SEN2_OUTPUT\3036_L1ORT.tif `
  --size 1024
```

The default output is `testing/inputs/balkan1/3036_L1ORT_sample.tif`. The script
preserves georeferencing, writes explicit band descriptions and records the
source window and output SHA-256 in an ignored manifest. It also writes a
true-color, NIR false-color and NDVI review PNG beside the chip; this is a
selection aid, not model output. Use `--window X Y W H` to select a reviewed
region. It refuses to write proof imagery below `payload/`.

## Run the payload stages

Cloud detection is the safe default and remains a Sentinel-to-Balkan transfer:

```powershell
.venv\Scripts\python.exe scripts\balkan1\run_pipeline.py `
  testing\inputs\balkan1\3036_L1ORT_sample.tif `
  --stop-after cloud `
  --reflectance-scale 1
```

Only use a scale of 1 after confirming that the preprocessed values are
reflectance. Raw DN must not be passed to either model. If the scale is omitted,
scene intake reports its sampled inference; that inference is not a calibration
certificate.

Crop inference must not consume the preprocessed pixels directly: Balkan-1 has
a finer native ground sampling distance and a different radiometric response
than the Sentinel/HLS data used to train the Prithvi head. For each scene,
prepare a co-registered Sentinel reference containing B02, B03, B04 and B8A in
model scale units, then fit the ground-side calibration:

```powershell
.venv\Scripts\python.exe scripts\balkan1\calibrate_crop_input.py `
  data\balkan1\preprocessed\3408_L1ORT.tif `
  D:\path\to\3408_sentinel2_reference.tif `
  --acquired-at 2026-06-16T18:40:43Z
```

The inputs must overlap and be co-registered. The fitter uses alternating
spatial blocks for fit and held-out validation and refuses to write a sidecar
unless every band correlation is at least 0.75 and their mean is at least 0.80.
Its default output is adjacent to the
Balkan image as `3408_L1ORT.crop_calibration.json`. The sidecar contains no
imagery but stays with the ignored local data rather than under `payload/` or
Git. Do not reuse a sidecar for a different file or acquisition.

Once that sidecar exists, crop, condition and downlink run through the validated
Balkan route:

```powershell
.venv\Scripts\python.exe scripts\balkan1\run_pipeline.py `
  data\balkan1\preprocessed\3408_L1ORT.tif `
  --acquired-at 2026-06-16T18:40:43Z `
  --stop-after crop `
  --reflectance-scale 1
```

The runtime validates the sidecar against the source SHA-256, creates a
temporary calibrated 10 m model raster, runs the same Prithvi model, and maps
the probability back to the full original grid for the website. Metadata
records the adapter, reference provenance, held-out quality, devices and
timings. The Sentinel reference is not copied into the payload.

The calibrated Balkan route records its operational `0.30` crop and health
thresholds in the crop plan. Sentinel-2 continues to use its independently
validated `0.49` and `0.645` thresholds; changing one sensor route does not
silently recalibrate the other. Independent Balkan crop-label accuracy remains
a separate validation task.

If no valid sidecar is available, crop remains blocked. There is no unvalidated
fallback into the production crop model.

For an acquisition-lineage execution proof, first build the real raw-derived
L1A with `--device cuda`, then stage a reviewed chip from the paired delivered
L1ORT bearing the same scene ID and run the payload command above. Preserve the
L0R/L1A manifest, L1ORT chip manifest and payload `result.json` together. The
L1A proves reconstruction and registration; the paired L1ORT supplies the
reflectance/geolocation that the models require. Do not silently substitute
uncalibrated L1A DN as reflectance.

# Local Balkan-1 band-alignment integration

The local integration adapts the PAN-referenced seeded global-shift method from
[`RalitsaZhekova/balkan1-band-alignment`](https://github.com/RalitsaZhekova/balkan1-band-alignment),
pinned to commit `5ba3076f5d8a198248067512dbbff2728dc2e35b`.

The corrected algorithm tries direct phase correlation to PAN, retries a weak
match using gradient magnitude, and then bridges a still-failed band through
any other band that aligned successfully. It fails closed if no path reaches
the configured confidence threshold.

It is a geometry-only preprocessing stage. It does not create georeferencing,
orthorectify a raw image, repair detector defects, or calibrate digital numbers.

## Supported local paths

### Correct an already georeferenced Balkan product

This is the path that can continue into the current cloud and crop pipeline:

```powershell
.\.venv\Scripts\python.exe -m prithvi_payload.balkan_alignment `
  data\balkan1\preprocessed\3408_L1ORT.tif `
  runtime\aligned\3408_L1ORT_aligned.tif `
  --band-order BLUE GREEN RED NIR PAN
```

The command writes:

- `3408_L1ORT_aligned.tif`, ordered as
  `BLUE, GREEN, RED, NIR_BROAD, PANCHROMATIC`;
- `3408_L1ORT_aligned.alignment.json`, containing the source/output checksums,
  fitted shifts, residuals, confidence values, and readiness status.

Run the local MVP with the original scene calibration. The calibration loader
accepts it only after verifying that the adjacent alignment report binds the
new file to the exact calibrated parent and that the operation was
radiometry-preserving:

```powershell
.\.venv\Scripts\vita-mvp.exe balkan `
  runtime\aligned\3408_L1ORT_aligned.tif `
  --crop-calibration data\balkan1\preprocessed\3408_L1ORT.crop_calibration.json `
  --region-id balkan-test-3408
```

After reinstalling the editable package, `vita-balkan-align.exe` is an alias for
the Python module command.

### Align and run in one local command

For a georeferenced input, the one-shot CLI can prepare the aligned product and
then enter the existing pipeline:

```powershell
.\.venv\Scripts\vita-mvp.exe balkan `
  data\balkan1\preprocessed\3408_L1ORT.tif `
  --align-bands-to runtime\aligned\3408_L1ORT_aligned.tif `
  --crop-calibration data\balkan1\preprocessed\3408_L1ORT.crop_calibration.json `
  --region-id balkan-test-3408
```

Use `--overwrite` only when both the alignment product and run output may be
replaced.

### Create an alignment-only raw intermediate

Delivered `*_Raw.tif` files have no CRS or orthorectification model. They can be
aligned for inspection using the L0 manifest's `BandStartRow` metadata:

```powershell
.\.venv\Scripts\python.exe -m prithvi_payload.balkan_alignment `
  data\balkan1\raw\3408\3408_Raw.tif `
  runtime\aligned\3408_L0_registered.tif `
  --metadata data\balkan1\derived\l1a\3408_L0R_manifest.json `
  --band-start-axis column `
  --band-start-row-scale 0.19 `
  --allow-ungeoreferenced
```

That output is tagged `PIPELINE_READY=FALSE`. Model intake intentionally rejects
it because band alignment does not supply the missing Earth-location transform.

### Experimental raw-to-model run

Scene 3408 revealed that delivered `BandStartRow` offsets map to raster
**columns**, at approximately `0.19` raster pixel per detector-row unit. This
was measured from the raw raster itself: the broad central correlation peaks
were BLUE `+234 px`, GREEN `+156 px`, RED `+70 px`, and NIR `-67 px`. The
metadata is an approximate search seed; the correlation result, not the seed,
defines the final shift.

Build the aligned raw raster:

```powershell
.\.venv\Scripts\python.exe -m prithvi_payload.balkan_alignment `
  data\balkan1\raw\3408\3408_Raw.tif `
  runtime\raw-experiment\3408_raw_aligned.tif `
  --metadata data\balkan1\derived\l1a\3408_L0R_manifest.json `
  --band-start-axis column `
  --band-start-row-scale 0.19 `
  --allow-ungeoreferenced `
  --measurement-tile-size 512 `
  --global-search-radius 96 `
  --minimum-confidence 0.25 `
  --device auto `
  --warp-tile-size 2048 `
  --compression zstd
```

`--device auto` selects CUDA when PyTorch can see a CUDA device and otherwise
falls back to the bounded CPU implementation. The CUDA path batches the four
direct phase correlations and any bridge candidates, then fuses all four
moving-band warps into one operation per output tile. Correlation uses a
bounded central tile rather than materializing the complete raw scene on the
GPU.

On scene 3086 and an RTX 3060 Laptop GPU, the corrected alignment completed in
`24.74 s`. Registration, including BLUE's bridge through GREEN, took `3.07 s`;
the fused warp and GeoTIFF write took `10.09 s`; overviews took `7.46 s`; and
the output SHA-256 took `2.42 s`. The resulting shifts were BLUE `(38, 100)`
via GREEN, GREEN `(26, 70)`, RED `(12, 33)`, and NIR `(-5, -30)` in `(dy, dx)`
order.

The experimental adapter performs dark-reference subtraction on every native
raw line, removes the 88 inactive detector columns on each side, applies the
available L1A-to-L1ORT QA affine fits, and warps directly from native 1.5 m raw
pixels to one north-up 10 m UTM grid. There is no intermediate 10 m image and
no second analysis-grid resampling. Detector columns are encoded right-of-track,
which is the geospatial equivalent of the flip-x established by the offline
L1A QA:

```powershell
.\.venv\Scripts\python.exe -m prithvi_payload.balkan_raw_proxy `
  runtime\raw-experiment\3408_raw_aligned.tif `
  runtime\raw-experiment\3408_raw_model_proxy.tif `
  --alignment-report runtime\raw-experiment\3408_raw_aligned.alignment.json `
  --radiometric-diagnostics data\balkan1\derived\l1a\3408_L1A_reference_validation.json `
  --position data\balkan1\raw\3408\position.csv `
  --attitude data\balkan1\raw\3408\attitude.csv `
  --parent-calibration data\balkan1\preprocessed\3408_L1ORT.crop_calibration.json
```

Crop inference requires an explicit opt-in so an experimental proxy can never
silently enter the operational path:

```powershell
.\.venv\Scripts\vita-mvp.exe balkan `
  runtime\raw-experiment\3408_raw_model_proxy.tif `
  --crop-calibration runtime\raw-experiment\3408_raw_model_proxy.crop_calibration.json `
  --experimental-raw-proxy `
  --region-id balkan-3408-raw-experimental `
  --scene-id 3408-raw-experimental `
  --output runtime\raw-experiment\pipeline `
  --ground-store runtime\raw-experiment\ground
```

Area averaging removes more spatial detail than the supplied L1ORT product.
That blur was the main cause of cloud false positives: the old path resampled
twice and reported `48.78%` cloud on scene 3086. The proxy now passes through
the shared analysis stage without another warp. A nodata-aware unsharp filter
is applied only to RED/GREEN/NIR at cloud-model input (`sigma=1.2`,
`amount=4.0`); the stored reflectance and visible image remain untouched. Crop
inference uses the same idea on all four model bands with the milder
`amount=1.75` and retains the operational `0.30` threshold when a same-scene
crop adapter exists. If the offline raw/L1ORT QA has a minimum pairwise band
correlation below `0.80`, the more conservative profile uses cloud amount
`5.0` and disables crop sharpening; scene 3215 showed that this avoids bright-
ground cloud errors and crop undercounting caused by a poorly constrained
spatial fit.

The delivered attitude/position telemetry initially placed some raw scenes
several kilometres from their supplied references. A small roll/pitch ground
offset model fitted across the 11 supplied scene centers now corrects the
translation before the north-up warp. Its leave-one-scene-out position RMSE is
`4.20 km`; it is useful for crop location embeddings, but it is not an
orthorectification or navigation solution.

Validation results after these changes are:

| Scene | Product | Cloud | Shadow | Crop of usable | Health score |
|---|---:|---:|---:|---:|---:|
| 3086 | raw-derived | 30.16% | 9.08% | 1.68% | 0.00 |
| 3086 | supplied L1ORT | 26.75% | 10.61% | 1.94% | 2.03 |
| 3215 | raw-derived | 0.21% | 0.00% | 33.78% | 7.32 |
| 3215 | supplied L1ORT | 0.04% | 0.00% | 35.17% | 6.43 |
| 3283 | raw-derived | 1.60% | 0.12% | 16.02% | experimental |
| 3283 | supplied L1ORT | 2.10% | 0.80% | 17.02% | 2.94 |
| 3370 | raw-derived | 0.01% | 0.00% | 41.26% | 50.10 |
| 3370 | supplied L1ORT | 0.02% | 0.00% | 40.83% | 53.74 |
| 3408 | raw-derived | 0.36% | 0.42% | 43.54% | 23.66 |
| 3408 | supplied L1ORT | 2.09% | 0.94% | 41.93% | 29.25 |

Scene 3086 has no same-scene Sentinel crop calibration. Its inherited adapter
is explicitly marked as an unqualified cross-scene fallback, so the raw proxy
uses a conservative `0.50` crop/health threshold; this reduces urban false
positives and matches the equally unqualified L1ORT comparison. It must not be
treated as a general threshold for calibrated scenes.

The raw preview uses independent 1st-to-99th percentile RGB stretches, low
saturation, and requires every visible band to be positive. This produces the
neutral, no-rainbow appearance of `data/band_alignment.png` and hides colored
nodata fringes. It is display-only. Cloud and crop detail restoration are also
model-input-only. Render readable result overlays with:

```powershell
.\.venv\Scripts\python.exe scripts\balkan1\render_pipeline_results.py `
  runtime\raw-cloud-3086\pipeline-final-working\result.json `
  runtime\raw-cloud-3086\final-visualizations
```

The proxy, calibration, intake, crop plan, and downlink metadata retain
`EXPERIMENTAL_RAW_PROXY` / `UNQUALIFIED_ENGINEERING_EXPERIMENT` provenance.
All model pixels in these runs originate from the raw TIFF after alignment;
the supplied L1ORT files are used only offline to derive/check the currently
missing absolute radiometric calibration. These outputs are working
engineering results, not science-qualified measurements.

## Processing and safety contract

- PAN is the reference and remains band 5, but the cloud and crop models do not
  consume it.
- Each moving band first attempts a seeded direct phase correlation to PAN.
- A weak direct match is retried using gradient magnitude. A still-failed band
  is tried through every already-aligned non-reference band in deterministic
  order, and the two accepted shifts are composed.
- The upstream `0.25` PSR confidence gate is retained; a weak direct peak is not
  accepted merely because it is the best available peak.
- Any band that cannot pass alignment fails the complete operation; an unwarped
  band is never silently forwarded.
- Raster reads and bilinear writes are windowed, avoiding a full-scene dense
  displacement grid.
- CRS, affine transform, nodata, and original dataset tags are preserved.
- The output includes tiled storage and average overviews required by the fast
  Balkan analysis-grid path.

## Current qualification boundary

Synthetic tests cover known constant shifts, row- and column-oriented metadata
seeds, the upstream-style generic bridge, CUDA/CPU parity, GeoTIFF metadata,
pipeline intake, calibration inheritance, report tampering, and failure
behavior. The real raw scene-3086 check exercises the bridge path and passes a
per-band `0.80` correlation gate against the supplied L1ORT image.

This proves local engineering integration, not scientific or Jetson latency
acceptance. Before replacing an operational input, compare the corrected
product against manual control points or a trusted reference and rerun
cloud/crop parity. The experimental proxy provides an engineering fallback
when the operational raw orthorectification/geolocation assets are unavailable.
A science-qualified raw-to-final onboard product still needs the missing
camera, terrain, and absolute calibration stages.

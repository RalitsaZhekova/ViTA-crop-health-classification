# Local Balkan-1 band-alignment integration

The local integration adapts the PAN-referenced local shift-field method from
[`balkan1-band-alignment` Test-2](https://github.com/stelspirou-git/balkan1-band-alignment/tree/Test-2),
pinned to commit `5688c564f733c754601240e6263b48f2792aafeb`.

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
  --allow-ungeoreferenced
```

That output is tagged `PIPELINE_READY=FALSE`. Model intake intentionally rejects
it because band alignment does not supply the missing Earth-location transform.

### Experimental raw-to-model run

Scene 3408 revealed that delivered `BandStartRow` offsets map to raster
**columns**, at approximately `0.19` raster pixel per detector-row unit. This
was measured from the raw raster itself: the broad central correlation peaks
were BLUE `+234 px`, GREEN `+156 px`, RED `+70 px`, and NIR `-67 px`. With the
correct seed orientation, the upstream local-field method obtains 19--30
accepted control tiles per band instead of falling back to weak global peaks.

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
  --minimum-confidence 0.20
```

The experimental adapter then performs line-wise dark-reference subtraction,
removes the 88 inactive detector columns on each side, averages to 10 m,
applies the available L1A-to-L1ORT QA affine fits, and constructs an approximate
rotated UTM grid from ECEF position and midpoint roll/pitch. Detector columns
are encoded right-of-track, which is the geospatial equivalent of the flip-x
established by the offline L1A QA:

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

The improved local experiment on 2026-08-18 produced `DOWNLINK_READY` and:

- central BLUE-to-RED residual: `165.0 px -> 5.8 px`;
- central GREEN-to-RED residual: `86.0 px -> 5.0 px`;
- cloud estimate: `5.73%` (qualified L1ORT baseline: `2.09%`);
- crop over usable pixels: `43.46%` (L1ORT baseline: `41.93%`);
- condition-analysis coverage: `23.68%` (L1ORT baseline: `22.54%`);
- condition score: `36.21`, labelled `Moderate anomaly` (L1ORT: `29.25`);
- warm model pipeline: `9.35 s`; total command including model load: `16.43 s`.

The raw proxy uses an experimental crop probability threshold of `0.55` and a
more conservative condition-analysis threshold of `0.65`. These replace the
parent model's `0.30` threshold only for an explicitly enabled raw proxy. The
values were selected from a scene-3408 parity sweep: `0.55` gives crop coverage
close to the qualified L1ORT run while visibly reducing desert false positives.
They are not validated for a second raw scene and must not be presented as a
general Balkan-1 threshold calibration.

The experimental web preview also uses a wider combined RGB stretch with
reduced saturation. The condition overlay has lower opacity and muted colors.
Both changes are display-only; neither modifies the reflectance passed to the
cloud or crop models. A before/improved/reference visualization is written to
`runtime/raw-experiment/3408_raw_v2_v3_l1ort_comparison.png`.

The proxy, calibration, intake, crop plan, and downlink metadata all retain
`EXPERIMENTAL_RAW_PROXY` / `UNQUALIFIED_ENGINEERING_EXPERIMENT` provenance.
The map location is approximate (about 5 km from the supplied reference center
on scene 3408), and the radiometric QA fits are not an absolute calibration.
These outputs demonstrate that the software can run from raw-derived bands;
they are not science-qualified measurements.

## Processing and safety contract

- PAN is the reference and remains band 5, but the cloud and crop models do not
  consume it.
- Local phase-correlation measurements are fitted to a smooth bilinear field.
  A deterministic spatial-consensus pass rejects inconsistent control points.
- A global direct/gradient shift remains the fallback when a local field cannot
  pass the residual gates.
- Any band that cannot pass alignment fails the complete operation; an unwarped
  band is never silently forwarded.
- Raster reads and bilinear writes are windowed, avoiding a full-scene dense
  displacement grid.
- CRS, affine transform, nodata, and original dataset tags are preserved.
- The output includes tiled storage and average overviews required by the fast
  Balkan analysis-grid path.

## Current qualification boundary

Synthetic tests cover known constant and metadata-seeded shifts, GeoTIFF
metadata, pipeline intake, calibration inheritance, report tampering, and
failure behavior. A read-only check on the real `3408_L1ORT` scene found local
fields for BLUE, GREEN, and RED; NIR selected the documented global-gradient
fallback.

The real `3408_L1ORT` local acceptance run completed on 2026-08-17:

- alignment product: 1,482,501,587 bytes, produced in 291.93 seconds;
- pipeline status: `DOWNLINK_READY` / `MVP_READY`;
- model execution: PyTorch CUDA;
- post-alignment pipeline: 15.51 seconds;
- one-shot end to end including model loads: 23.35 seconds;
- downlink bundle: 688,931 bytes across three files.

This proves local engineering integration, not scientific or Jetson latency
acceptance. Before replacing an operational input, compare the corrected
product against manual control points or a trusted reference and rerun
cloud/crop parity. The five-minute alignment implementation must also be
optimized or moved outside the time-critical payload path before onboard use.
The experimental proxy provides an engineering fallback when the operational
raw orthorectification/geolocation assets are unavailable. A science-qualified
raw-to-final onboard product still needs the missing camera, terrain, and
absolute calibration stages.

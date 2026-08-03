# Cloud detection and masking boundary

The payload uses the reviewed ViTA Cloud Detector baseline: OmniCloudMask
`1.7.1`, model V4. The backend is isolated behind the existing four-class cloud
interface, so crop inference, condition scoring, downlink packaging and the
ground API retain their previous contracts.

## Inputs

- pipeline array layout: `[bands, height, width]`;
- retained cloud-stage order: `[NIR, Red, Green, Blue]`;
- native OmniCloudMask order inside the adapter: `[Red, Green, NIR]`;
- Sentinel-2 prefers `B8A` and falls back to `B08` when needed;
- Balkan-1 uses L1ORT bands `Red=3`, `Green=2`, `NIR=4`; PAN is retained but
  not sent to the cloud model;
- finite, strictly positive Red/Green/NIR pixels define the valid model
  footprint.

OmniCloudMask dynamically normalizes each patch, so the reflectance divisor
does not set the model distribution. The pipeline still requires a verified
scale because later crop and vegetation-index stages depend on calibrated
values.

The supplied Balkan benchmark is defined at 10 m. The executor therefore
reprojects Balkan L1ORT bands independently to a temporary 10 m UTM analysis
grid, runs the V4 ensemble with its baseline 1000 px / 300 px overlap settings,
and maps categorical masks back to the untouched source grid. Source imagery is
never rewritten and the temporary grid is never downlinked.

## Outputs

- semantic classes: clear, thick cloud, thin cloud and cloud shadow;
- normalized model-confidence scores;
- georeferenced semantic, invalid-input and operational unusable masks;
- cloud, shadow, usable and unusable percentages;
- `PROCESS`, `PROCESS_CLEAR_AREAS` or `REJECT` decision;
- JSON metadata and a headless preview PNG.

By default, thick cloud, thin cloud, cloud shadow and invalid input are
unusable. Existing small-region filtering and dilation remain configured in
`payload/cloud_detection/configs/cloud_detector.yaml`.

## Crop integration

The operational unusable mask remains the authoritative exclusion input to
Prithvi. Masked probability, binary and confidence pixels are written as
nodata, and the existing 60% cloud gate can stop crop inference before the crop
model is loaded.

Sentinel cloud inference prefers `B8A`, which is also the crop model's native
NIR input. Balkan crop execution still requires
`--allow-provisional-balkan-crop`: OmniCloudMask is Balkan-validated, but the
Prithvi Balkan NIR transfer remains execution-only and unvalidated.

## Download verified model files

```powershell
.\.venv\Scripts\python.exe payload\scripts\download_cloud_weights.py
```

The two `.safetensors` files are ignored by Git and verified against component
and ensemble SHA-256 values before model loading. See `UPSTREAM.md` for exact
provenance, benchmark scope and licensing.

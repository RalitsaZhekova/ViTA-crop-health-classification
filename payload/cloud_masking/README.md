# Cloud detection and masking boundary

The complete reviewed CloudSEN12 pipeline is integrated as an isolated payload
component. It can classify and post-process a co-registered Sentinel-2 L1C
GeoTIFF without loading or calling crop inference.

Current input:

- array layout `[bands, height, width]`;
- exact band order `[B08, B04, B03, B02]`;
- Sentinel-2 L1C top-of-atmosphere digital numbers scaled by 10,000, or
  pre-scaled TOA reflectance with an explicit scale of 1;
- all-band zero and non-finite pixels reported separately as invalid input.

Current output:

- semantic classes: clear, thick cloud, thin cloud and cloud shadow;
- uncalibrated class confidence scores;
- georeferenced class-score and semantic GeoTIFFs;
- invalid-input and binary unusable-pixel maps;
- cloud, shadow, usable and unusable percentages;
- `PROCESS`, `PROCESS_CLEAR_AREAS` or `REJECT` decision;
- JSON metadata and a headless preview PNG.

By default, thick cloud, thin cloud, cloud shadow and invalid input are
unusable. Small detections are removed and remaining unusable areas are
dilated according to `payload/cloud_detection/configs/cloud_detector.yaml`.

## Integrated scene route

`prithvi_payload.pipeline` now connects scene intake to cloud detection for a
single preprocessed GeoTIFF. It reorders declared source bands without creating
a full-scene intermediate array and writes semantic, unusable and invalid-input
masks window by window. The older `cloud-detect` command remains available for
the original exact four-band Sentinel-2 contract.

For Balkan-1, a five-band `RED, GREEN, BLUE, NIR, PAN` product is required. PAN
is retained but not sent to CloudSEN12. Reflectance-calibrated input is accepted;
raw 12-bit DN input is blocked unless a verified calibration scale is supplied.
Cloud results remain provisional until tested against real Balkan-1 imagery.

## Crop integration intentionally not connected yet

The standalone pipeline now creates `cloud_unusable`, but it is not yet applied
to Prithvi crop output. Real Sentinel-2 scenes must be inspected before that
connection is enabled.

Cloud classification uses Sentinel-2 `B08`; the crop model uses `B8A`. These
must not be silently substituted. A scene-ingestion adapter will need to supply
both NIR choices or perform an explicitly validated resampling/conversion.

Before Balkan-1 use, the classifier must be validated for its 1.5 m resolution,
spectral response and 12-bit calibration. Shared RGB/NIR labels alone do not
establish compatibility.

See `UPSTREAM.md` for provenance, validation limits and the non-commercial
weights licence.

## Run the original standalone detector

```powershell
$env:PYTHONPATH="payload/src;shared/src"
.\.venv\Scripts\python.exe -m cloud_detection.cli `
  --input path\to\sentinel2_l1c.tif `
  --output outputs\cloud_detection
```

# Ground component

`ground/` receives georeferenced payload results and turns them into auditable
crop-condition observations.

## Included now

- sensor-neutral NDVI, EVI, GNDVI, SAVI, CVI and RGB measurements;
- conservative crop/cloud/no-data analysis masking;
- transparent absolute-vigor and robust within-crop-region anomaly scoring;
- cautious `Nominal`, `Watch`, `Moderate anomaly`, `High anomaly` and
  `Insufficient data` screening labels;
- strict JSON-safe observation records;
- storage, API and visualization contracts.

## Deliberately not claimed

A single image does not provide a defensible healthy/stressed diagnosis. The
current labels express spectral screening priority using documented prototype
ranges and spatial evidence. Crop-, region- and growth-stage-aware temporal
baselines are still required before these labels can support trend claims.

The API, database adapter and operational map are boundaries, not completed
services. This prevents a demo stub from being mistaken for production code.

## Run Phase 2 from a payload result

The scene processor requires a payload `result.json` with `CROP_COMPLETE`
status. It validates the source, unusable mask, crop binary mask and crop
probability grid before processing any pixels.

```powershell
$env:PYTHONPATH = "ground/src;shared/src"
python -m prithvi_ground.scene `
  testing/runs/example/result.json `
  --output testing/runs/example/ground
```

The processor uses bounded windows and three passes for a measured scene:

1. calculate and write vegetation/RGB indices and absolute pixel scores;
2. calculate a whole-region median absolute deviation;
3. write robust deficit, relative anomaly, low-vigor and combined alert layers.

Outputs include compressed tiled GeoTIFFs, a portable JSON report with relative
asset paths, and a PNG quicklook. Means and standard deviations use every valid
analysis pixel. Percentiles use a deterministic priority-reservoir sample of at
most 50,000 spatially identified pixels, so results do not depend on tile size.

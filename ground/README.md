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

# Ground component

`ground/` receives georeferenced payload results and turns them into auditable
crop-condition observations.

## Included now

- sensor-neutral NDVI, EVI, GNDVI, CVI and RGB measurements;
- conservative crop/cloud/no-data analysis masking;
- strict JSON-safe observation records;
- storage, API and visualization contracts.

## Deliberately not claimed

A single image does not provide a defensible healthy/stressed diagnosis. The
current code emits measurements and `MEASURED`/`INSUFFICIENT_DATA` states.
Normal/watch/stressed labels require several dates and a region-, crop- and
growth-stage-aware baseline.

The API, database adapter and operational map are boundaries, not completed
services. This prevents a demo stub from being mistaken for production code.

# Pipeline testing workspace

This is the only location for manual end-to-end pipeline tests.

```text
testing/
  inputs/
    sentinel2/   preprocessed Sentinel-2 GeoTIFFs
    balkan1/     preprocessed or reconstructed Balkan-1 GeoTIFFs
  runs/
    <run-name>/  one self-contained result per command
```

Input imagery and generated runs are ignored by Git. Payload code, configs and
models remain under `payload/`; runtime data must not be stored there.

The full Balkan-1 collection belongs under `data/balkan1/` or at an external
path. Only bounded, explicitly selected proof chips should be placed in
`testing/inputs/balkan1/`. See
[`scripts/balkan1/README.md`](../scripts/balkan1/README.md) for preprocessing,
sampling and the guarded Balkan payload handoff.

Run a Sentinel-2 scene:

```powershell
.\payload\scripts\run_scene.ps1 `
  -InputPath "testing\inputs\sentinel2\your_scene.tif" `
  -Sensor sentinel-2 `
  -AcquiredAt "2026-07-27T12:00:00Z" `
  -Output "testing\runs\your_run"
```

The primary API-facing artifact is `<run-name>/result.json`. Detailed stage
metadata, masks and visualizations are stored beside it in named subfolders.

Run cloud detection and then crop classification when cloud coverage is below
60%:

```powershell
.\payload\scripts\run_scene.ps1 `
  -InputPath "testing\inputs\sentinel2\your_five_band_scene.tif" `
  -Sensor sentinel-2 `
  -AcquiredAt "2026-07-27T12:00:00Z" `
  -StopAfter crop `
  -Output "testing\runs\your_crop_run"
```

The Sentinel-2 file must declare `B02`, `B03`, `B04`, `B08` and `B8A` band
descriptions. Band order is resolved from those descriptions. Crop outputs are
written to `crop_maps/` and the combined preview to `visualisations/`.

Run a staged Balkan-1 proof chip through the same payload without replacing the
Sentinel-2 route:

```powershell
.venv\Scripts\python.exe scripts\balkan1\run_pipeline.py `
  testing\inputs\balkan1\your_L1ORT_sample.tif `
  --acquired-at 2026-07-27T12:00:00Z `
  --reflectance-scale 1 `
  --output testing\runs\your_balkan_run
```

Sentinel-2 and Balkan-1 inputs and results coexist under separate sensor/run
folders. Sensor selection is per scene, so the calibrated Balkan route does not
alter Sentinel-2 band resolution or model inputs. Balkan crop inference requires
an adjacent validated `*.crop_calibration.json` sidecar.

Run the complete local payload-to-ground demonstration:

```powershell
.\integration\scripts\run_sentinel_end_to_end.ps1 `
  -InputPath "testing\inputs\sentinel2\your_five_band_scene.tif" `
  -AcquiredAt "2026-07-27T12:00:00Z" `
  -ReflectanceScale 10000 `
  -Output "testing\runs\your_end_to_end_run"
```

This creates `end_to_end_result.json`, a self-contained `payload/` result tree,
and a `ground/` tree containing the crop-condition report, index rasters,
condition/anomaly masks and quicklook. It is a local software handoff, not a
simulation or implementation of the eventual satellite downlink.

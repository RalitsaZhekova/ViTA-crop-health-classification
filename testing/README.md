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

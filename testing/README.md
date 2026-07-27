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

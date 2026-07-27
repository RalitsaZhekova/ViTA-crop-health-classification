# End-to-end demonstration

This component orchestrates the payload and ground packages without adding the
ground health engine to the flight bundle. A complete run has this layout:

```text
<run>/
  end_to_end_result.json
  payload/
    result.json
    cloud_masks/
    crop_maps/
    metadata/
    visualisations/
  ground/
    crop_condition_report.json
    health_layers/
    condition/
    visualisations/
```

Run a Sentinel-2 five-band development scene from PowerShell:

```powershell
.\integration\scripts\run_sentinel_end_to_end.ps1 `
  -InputPath testing\inputs\sentinel2\scene.tif `
  -AcquiredAt 2026-07-27T12:00:00Z `
  -ReflectanceScale 10000 `
  -Output testing\runs\scene_end_to_end
```

The input currently needs B02, B03, B04, B08 and B8A descriptions because the
Sentinel cloud and crop models use different NIR bands. This is a development
interface, not the final single-NIR Balkan contract.

`end_to_end_result.json` uses relative references to the payload and ground
records. Ground raster references are also relative to the ground report. The
orchestrator stops without starting Phase 2 when the payload cloud gate skips
crop inference, and it protects an existing result unless `-Overwrite` is set.

The accepted real-model execution evidence is recorded in
[`verification.json`](verification.json). That scene is a PASTIS
training/replay example, so it validates execution and interfaces—not held-out
crop accuracy, broad cloud accuracy or agronomic diagnosis.

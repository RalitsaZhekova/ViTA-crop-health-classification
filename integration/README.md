# End-to-end demonstration

This component orchestrates the full Sentinel development flow. Cloud masking,
crop inference and condition calculations now all execute inside the payload
boundary; ground work starts from the compact downlink product.

Current intermediate layout:

```text
<run>/
  end_to_end_result.json
  payload/
    result.json
    cloud_masks/
    crop_maps/
    condition_analysis/
    metadata/
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
Sentinel cloud and crop models use different NIR bands. This remains a Sentinel
development interface, not the final single-NIR Balkan contract.

The orchestrator stops before crop and condition processing when the payload
cloud gate rejects the scene. Existing run summaries are protected unless
`-Overwrite` is supplied. Accepted real-model execution evidence is recorded in
[`verification.json`](verification.json); it validates execution and interfaces,
not held-out crop accuracy, broad cloud accuracy or agronomic diagnosis.

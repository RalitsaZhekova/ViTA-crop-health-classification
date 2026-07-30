# End-to-end Sentinel MVP

This component orchestrates the full Sentinel development flow. Cloud masking,
crop inference and condition calculations now all execute inside the payload
boundary; ground work starts from the compact downlink product. The complete MVP
continues through verified ground ingestion, the API and interactive client.

Current intermediate layout:

```text
<run>/
  end_to_end_result.json
  mvp_result.json
  payload/
    result.json
    cloud_masks/
    crop_maps/
    condition_analysis/
    downlink/
      scene.webp
      condition.png
      scene.json
    metadata/
    visualisations/
```

## Complete payload-to-ground MVP

Run the complete flow and keep the web application open:

```powershell
.\integration\scripts\run_sentinel_mvp.ps1 `
  -InputPath testing\inputs\sentinel2\scene.tif `
  -AcquiredAt 2026-07-27T12:00:00Z `
  -SceneId field_42_20260727 `
  -RegionId field_42 `
  -ReflectanceScale 10000 `
  -Output testing\runs\scene_mvp `
  -GroundStore testing\runs\ground_mvp `
  -Serve
```

Open `http://127.0.0.1:8000` for the dashboard or `/docs` for the API. Omit
`-Serve` for a non-blocking batch run that finishes once the scene is API-ready.
Use a unique `SceneId` per acquisition and the same stable `RegionId` for later
observations of one field. Together with the fixed `GroundStore`, this builds
chronological history instead of isolated reports.

The MVP result ends with `MVP_READY` and records the payload, downlink,
ground-catalog and API-ready stages. The receiver revalidates all three files,
checks image properties and checksums, and installs them atomically before the
scene can appear in the client.

## Payload-to-downlink only

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

Successful payload-only execution ends with `DOWNLINK_READY`; complete MVP
execution ends with `MVP_READY`. Both validate software interfaces, not radio
transfer, contact-window scheduling, retry/resume or ground acknowledgement.

## Coordinate mission client

`vita-mission` is the ground-side client for the persistent payload acquisition
API. It validates the shared command, creates a collision-resistant job ID,
checks payload health, submits coordinates, displays safe candidate progress,
and polls to a terminal state. On completion it downloads only the three routine
files, verifies every payload-status checksum, invokes the existing bundle
validator (including image checksums), and uses `SceneCatalog.ingest` for the
canonical atomic ground installation.

```powershell
vita-mission run-region `
  --bbox "23.10,42.50,23.15,42.55" `
  --start "2026-07-01" `
  --end "2026-07-29" `
  --region-id "field_42" `
  --selection-policy "target_cloud_range" `
  --target-cloud-min 15 `
  --target-cloud-max 35 `
  --target-cloud-ideal 25 `
  --payload-url "http://127.0.0.1:8081" `
  --ground-store "ground/runtime"
```

The printed result distinguishes Earth Engine metadata cloud percentage from
payload-measured cloud, shadow and unusable percentages. No cloud, crop or
condition result is recomputed by this client.

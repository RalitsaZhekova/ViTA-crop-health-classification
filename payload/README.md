# Payload component

This directory is the flight-side boundary. It is intentionally independent of
training datasets and experiment logs.

## Included now

- `models/`: one pinned checkpoint, its SHA-256 identity and architecture;
- `src/prithvi_payload/`: training-free crop and cloud classifier loaders;
- `reconstruction/`: the Balkan-1 raw-band registration contract;
- `src/cloud_detection/`: complete reviewed cloud-detection runtime;
- `cloud_detection/`: its configs, operational scripts and documentation;
- `cloud_masking/`: the boundary between the standalone mask and crop inference;
- `requirements.txt`: runtime-only Python dependencies.

The base model receives normalized batches shaped `[batch, 4, 1, height, width]` in
the exact `BLUE, GREEN, RED, NIR_NARROW` order. The inference wrapper accepts
inputs on the training numeric scale and performs the recorded normalization.
In an installed deployment, set `PRITHVI_MODEL_DIR` to the directory containing
the checkpoint and `architecture.yaml`; source checkouts resolve
`payload/models/` automatically.

The selected base model emits crop/non-crop output only. The preserved crop-type
models remain ground-side under `outputs/` and are not part of the flight bundle.

`scene-run` inspects a preprocessed Sentinel-2 or Balkan-1 GeoTIFF, resolves its
declared band order, and runs cloud detection through bounded 512-pixel windows.
When explicitly requested, scenes below the 60% cloud gate continue through
bounded crop segmentation. Every cloud-shadow, cloud and invalid pixel in the
operational unusable mask is excluded from the crop outputs. The Balkan-1 route
is provisional until real imagery is radiometrically and spectrally validated.

Run only from an explicit command:

```powershell
.\payload\scripts\run_scene.ps1 `
  -InputPath path\to\preprocessed_scene.tif `
  -Sensor sentinel-2 `
  -AcquiredAt 2026-07-26T12:00:00Z `
  -StopAfter cloud `
  -Output testing\runs\sentinel2_demo
```

Continue through crop classification:

```powershell
.\payload\scripts\run_scene.ps1 `
  -InputPath path\to\preprocessed_scene.tif `
  -Sensor sentinel-2 `
  -AcquiredAt 2026-07-26T12:00:00Z `
  -StopAfter crop `
  -MaxCropCloudPercentage 60 `
  -Output testing\runs\sentinel2_crop_demo
```

The crop route requires both Sentinel-2 `B08` for CloudSEN12 and `B8A` for the
selected Prithvi model. It writes crop probability, binary crop and confidence
GeoTIFFs, crop metadata and a combined PNG. A scene at or above the configured
cloud percentage is stopped before the crop model is loaded.

Cloud previews show RGB, the exact semantic classes, a display-feathered RGB
overlay, and the exact operational unusable mask. Thick cloud, thin cloud,
cloud shadow and invalid/nodata use distinct colors and report their pixel
fractions. Crop previews show RGB, continuous probability, a display-feathered
probability overlay with the accepted-crop edge, and the exact binary mask.
Feathering applies only to PNG presentation; all GeoTIFF masks, probabilities,
thresholds and statistics remain unchanged.

Each run writes one canonical `result.json` containing the summarized metadata
for every completed stage and links to detailed stage JSON, GeoTIFF masks and
PNG visualizations. API and database work should read `result.json`; the stage
files remain available for debugging and audit.

## Explicitly excluded

- the 49 GiB datasets;
- augmentation and training code;
- checkpoints other than the selected model;
- TensorBoard and training logs;
- health calculations, databases, API and visualization.

## Current limits

Crop inference is ready for tiled arrays, and the cloud stage streams large
GeoTIFFs without loading a full scene. Balkan-1 raw-band reconstruction and
real Balkan-1 cloud/crop validation are not yet implemented.

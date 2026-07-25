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

The model receives normalized batches shaped `[batch, 4, 3, height, width]` in
the exact `BLUE, GREEN, RED, NIR_NARROW` order. The inference wrapper accepts
inputs on the training numeric scale and performs the recorded normalization.
In an installed deployment, set `PRITHVI_MODEL_DIR` to the directory containing
the checkpoint and `architecture.yaml`; source checkouts resolve
`payload/models/` automatically.

Cloud detection is independent of crop inference. It accepts Sentinel-2 L1C
bands in exact `[B08, B04, B03, B02]` order and returns semantic classes,
confidence rasters, an unusable mask, metadata and a preview. That mask is not
yet applied to crop output.

## Explicitly excluded

- the 49 GiB datasets;
- augmentation and training code;
- checkpoints other than the selected model;
- TensorBoard and training logs;
- health calculations, databases, API and visualization.

## Current limits

Crop inference is ready for tiled arrays, and the standalone Sentinel-2 cloud
pipeline supports GeoTIFF input/output. Connecting its mask to crop output,
streaming very large scenes without loading the full raster, Balkan-1 raw-band
reconstruction and Balkan-1 classifier adaptation are not yet implemented.

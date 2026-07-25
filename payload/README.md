# Payload component

This directory is the flight-side boundary. It is intentionally independent of
training datasets and experiment logs.

## Included now

- `models/`: one pinned checkpoint, its SHA-256 identity and architecture;
- `src/prithvi_payload/`: a training-free tile inference loader;
- `reconstruction/`: the Balkan-1 raw-band registration contract;
- `cloud_masking/`: the cloud-mask integration contract;
- `requirements.txt`: runtime-only Python dependencies.

The model receives normalized batches shaped `[batch, 4, 3, height, width]` in
the exact `BLUE, GREEN, RED, NIR_NARROW` order. The inference wrapper accepts
inputs on the training numeric scale and performs the recorded normalization.
In an installed deployment, set `PRITHVI_MODEL_DIR` to the directory containing
the checkpoint and `architecture.yaml`; source checkouts resolve
`payload/models/` automatically.

## Explicitly excluded

- the 49 GiB datasets;
- augmentation and training code;
- checkpoints other than the selected model;
- TensorBoard and training logs;
- health calculations, databases, API and visualization.

## Current limits

The inference core is ready for tiled tensors, but scene tiling/GeoTIFF writing,
Balkan-1 raw-band reconstruction and Balkan-1 cloud-mask adaptation are not yet
implemented. Their contracts are recorded here so they can be added without
mixing ground or training concerns.

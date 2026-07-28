# Training component

This directory records the data and retained experiment summaries needed to
understand the model-development work without shipping them to the satellite.

## Contents

- `datasets/manifest.yaml`: paths, file counts and sizes for the existing
  datasets; the data is not duplicated.
- `experiment_logs/`: compact JSON summaries for the selected European replay
  experiment.

The canonical training implementation is `src/prithvi_crop/`, configurations
are under `configs/`, and launchers are under `scripts/`. Git history preserves
older copies and external integration provenance. No training code belongs in
`payload/`.

## Dataset policy

Datasets stay in their current location to avoid an unnecessary 49 GiB copy.
They can later be mounted under a dedicated training volume without changing
payload or ground packages.

## Model policy

Training is frozen for now. The selected payload model is the single-frame
binary checkpoint recorded in [`../configs/selected_model.yaml`](../configs/selected_model.yaml).
It was initialized through the preserved single-frame crop-type checkpoint,
which was itself adapted from the European-replay checkpoint
`epoch=02-macro_f1=0.5062.ckpt`. The final selected weights-only binary export is
deployed from `payload/models/`; the European replay and other earlier stage
checkpoints remain under `outputs/` for provenance and safe rollback.

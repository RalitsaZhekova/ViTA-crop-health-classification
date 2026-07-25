# Training archive

This directory preserves everything needed to understand or reproduce the
model-development work without shipping it to the satellite.

## Contents

- `datasets/manifest.yaml`: paths, file counts and sizes for the existing
  datasets; the 49 GiB of data is not duplicated.
- `configs/`: snapshots of the three training stages and binary calibration.
- `scripts/`: the data-preparation and training launchers used in the project.
- `source_snapshot/prithvi_crop/`: a copy of the validated training/evaluation
  implementation at extraction time.
- `experiment_logs/`: compact TensorBoard/config/log artifacts for the
  refinement and European replay experiments.

The source snapshot retains the original `prithvi_crop` imports so it remains
an exact recovery copy. The working legacy package under `src/prithvi_crop`
continues to run the current tests. No training code belongs in `payload/`.

## Dataset policy

Datasets stay in their current location to avoid an unnecessary 49 GiB copy.
They can later be mounted under a dedicated training volume without changing
payload or ground packages.

## Model policy

Training is frozen for now. The selected model is
`epoch=02-macro_f1=0.5062.ckpt`; its verified payload copy is the only
checkpoint intended for deployment. Other checkpoints remain untouched in
`outputs/` for rollback and forensic comparison.

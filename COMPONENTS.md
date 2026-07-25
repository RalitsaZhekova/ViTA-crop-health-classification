# Component layout

This repository is being separated without moving or deleting the validated
legacy implementation. The original `src/prithvi_crop`, `data`, `outputs`,
`configs`, and `scripts` paths remain intact while the four deployment domains
are assembled alongside them.

```text
training/  -> shared/
payload/   -> shared/
ground/    -> shared/ and payload result records
shared/    -> no project-domain dependency
```

## Responsibilities

| Component | Runs where | Owns | Must not own |
| --- | --- | --- | --- |
| `training/` | Development workstation | Dataset references, augmentation, training/evaluation code, configs and experiment logs | Flight runtime or customer API |
| `payload/` | Balkan-1 payload computer | Pinned model, tiled inference, raw-band reconstruction boundary and cloud-mask boundary | Training data, experiment logs, health history or web UI |
| `ground/` | Ground infrastructure | Crop-condition calculations, observations, storage contract, API contract and visualization | Training loop or satellite reconstruction |
| `shared/` | All components | Band order, normalization, class mapping, thresholds and JSON schemas | Sensor processing or business logic |

## Non-destructive extraction policy

- No original file is moved, renamed or deleted.
- The 49 GiB datasets are referenced from `training/datasets/manifest.yaml`;
  they are not duplicated.
- Training source under `training/source_snapshot/` is an immutable recovery
  snapshot of the validated implementation.
- Only the selected checkpoint is copied into `payload/models/`.
- A SHA-256 digest pins the payload copy to the validated checkpoint.
- Missing reconstruction, Balkan-1 cloud masking, storage, API and operational
  visualization implementations are documented as explicit boundaries rather
  than represented as finished code.

The legacy paths can be retired only after the extracted components have been
accepted and independently packaged. That later cleanup is intentionally
outside this change.

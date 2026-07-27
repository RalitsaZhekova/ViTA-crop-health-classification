# Component layout

This repository separates training, payload, ground, and shared responsibilities
while keeping one canonical copy of every implementation and configuration.

```text
training/  -> shared/
payload/   -> shared/
ground/    -> shared/ and payload result records
integration/ -> payload/, ground/ and shared/ for ground demonstrations only
shared/    -> no project-domain dependency
```

## Responsibilities

| Component | Runs where | Owns | Must not own |
| --- | --- | --- | --- |
| `training/` | Development workstation | Dataset references, augmentation, training/evaluation code, configs and experiment logs | Flight runtime or customer API |
| `payload/` | Balkan-1 payload computer | Pinned model, tiled inference, raw-band reconstruction boundary and cloud-mask boundary | Training data, experiment logs, health history or web UI |
| `ground/` | Ground infrastructure | Crop-condition calculations, observations, storage contract, API contract and visualization | Training loop or satellite reconstruction |
| `integration/` | Development/ground demonstration | One-command payload-to-ground orchestration and final run summary | Flight bundle or duplicated model/business logic |
| `shared/` | All components | Band order, normalization, class mapping, thresholds and JSON schemas | Sensor processing or business logic |

## Storage policy

- Datasets remain under `data/` and are indexed by
  `training/datasets/manifest.yaml`; they are never duplicated into packages.
- Training implementation, configuration, and launchers have one canonical
  home in `src/prithvi_crop/`, `configs/`, and `scripts/`.
- Required training checkpoints stay under `outputs/`; payload deployment uses
  checksum-pinned, weights-only artifacts in `payload/models/`.
- Reproducible previews, caches, raw logs, source snapshots, and test
  scaffolding are not retained in the working repository.
- Missing reconstruction, Balkan-1 cloud masking, storage, API and operational
  visualization implementations are documented as explicit boundaries rather
  than represented as finished code.

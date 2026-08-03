# Component layout

This repository separates training, payload, ground, and shared responsibilities
while keeping one canonical copy of every implementation and configuration.

```text
training/  -> shared/
payload/   -> shared/
ground/    -> shared/ and compact downlink records
integration/ -> payload/, ground/ and shared/ for development demonstrations only
shared/    -> no project-domain dependency
```

## Responsibilities

| Component | Runs where | Owns | Must not own |
| --- | --- | --- | --- |
| `training/` | Development workstation | Dataset references, augmentation, training/evaluation code, configs and experiment logs | Flight runtime or customer API |
| `payload/` | Balkan-1 payload computer | Pinned models, tiled cloud/crop inference, condition calculations, compact downlink packaging and raw-band reconstruction boundary | Training data, experiment logs, historical analytics or web UI |
| `ground/` | Ground infrastructure | Downlink validation, observations, historical storage, API contract and visualization | Training loop, satellite reconstruction or alternate condition scoring |
| `integration/` | Development workstation | One-command payload-to-ground orchestration and final run summary | Flight transport or duplicated model/business logic |
| `shared/` | All components | Band order, normalization, class mapping, thresholds, scientific formulas and JSON schemas | Sensor processing, orchestration or UI logic |

## Storage policy

- Datasets remain under `data/` and are indexed by
  `training/datasets/manifest.yaml`; they are never duplicated into packages.
- Local Balkan-1 acquisitions use ignored `data/balkan1/` storage or an external
  root. Only bounded proof chips may be staged in ignored
  `testing/inputs/balkan1/`; neither location enters the payload image.
- Training implementation, configuration, and launchers have one canonical
  home in `src/prithvi_crop/`, `configs/`, and `scripts/`.
- Required training checkpoints stay under `outputs/`; payload deployment uses
  checksum-pinned, weights-only artifacts in `payload/models/`.
- Reproducible previews, caches, raw logs, source snapshots, and test
  scaffolding are not retained in the working repository.
- Missing Balkan-1 reconstruction and calibration, flight-specific cloud-model
  validation, physical downlink transport, production authentication and
  multi-tenant deployment are documented as explicit boundaries rather than
  represented as finished code.

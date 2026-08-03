# Prithvi four-band crop and land-cover model

This repository trains pixel-wise crop models using a frozen
`Prithvi-EO-2.0-100M-TL` backbone and a trainable UPerNet task head. The
selected payload model classifies crop/non-crop from one image; preserved
experiments classify crop type from one or three observations. The only inputs
are `BLUE`, `GREEN`, `RED`, and
`NIR_NARROW`, in that order. No SWIR channel is passed to the model and no
synthetic, copied, or imputed SWIR channel is created.

The selected binary checkpoint was trained with both the US IBM–NASA HLS corpus
and optical European PASTIS replay data. Preserved 13-class checkpoints predict
crop and land-cover classes, but none of these learned models infer crop health:
suitable supervised health targets and evaluation labels are not present in the
training datasets.

The payload-side rule-based condition stage calculates NDVI, EVI, GNDVI, SAVI,
CVI and RGB diagnostics only on clear, confident crop pixels. It combines the
most defensible normalized vigor components into an auditable screening score,
then applies robust within-crop-region anomaly analysis. Its reflectance,
masking, interpretation and JSON output rules are documented in
[`shared/health_analysis_contract.md`](shared/health_analysis_contract.md).

## Extracted components

The selected model is frozen for the current project phase. Its checksum-pinned
runtime and the remaining system responsibilities are separated into:

- [`training/`](training/README.md): dataset references and retained experiment
  summaries, with canonical training code and configs at the repository root;
- [`payload/`](payload/README.md): cloud and crop inference, condition
  calculations, compact downlink packaging and the Balkan-1 reconstruction boundary;
- [`ground/`](ground/README.md): downlink validation, storage, history, API and
  visualization boundaries;
- [`integration/`](integration/README.md): one-command Sentinel payload-to-ground
  MVP orchestration;
- [`shared/`](shared/README.md): bands, normalization, classes, thresholds and
  exchange schemas.

[`COMPONENTS.md`](COMPONENTS.md) defines ownership and canonical storage
locations.

## Local Balkan-1 imagery

Raw and preprocessed Balkan-1 imagery belongs under the ignored
`data/balkan1/` workspace, or may remain in an external collection referenced by
path. It is never copied into `payload/` or a container image. Versioned
preprocessing, proof-chip and payload-handoff utilities live under
[`scripts/balkan1/`](scripts/balkan1/README.md). Bounded, explicitly selected
proof chips may be staged under the already ignored `testing/inputs/balkan1/`
when local or target acceleration evidence is needed.

The current Balkan cloud route and opt-in crop route are sensor-transfer
execution tests. They do not establish cloud or crop accuracy on Balkan-1.

## Complete Sentinel MVP

The complete local command runs intake, cloud/shadow masking, crop segmentation,
streamed payload crop-condition analysis, compact packaging, checksum-verified
ground ingestion, the API and the interactive client:

```powershell
.\integration\scripts\run_sentinel_mvp.ps1 `
  -InputPath testing\inputs\sentinel2\your_five_band_scene.tif `
  -AcquiredAt 2026-07-27T12:00:00Z `
  -SceneId field_42_20260727 `
  -RegionId field_42 `
  -ReflectanceScale 10000 `
  -Output testing\runs\your_mvp_run `
  -GroundStore testing\runs\ground_mvp `
  -Serve
```

Open `http://127.0.0.1:8000` for the client and
`http://127.0.0.1:8000/docs` for the OpenAPI interface. Omit `-Serve` when a
batch run should stop after verified ground ingestion.

The Sentinel development route currently requires described B02, B03, B04,
B08 and B8A bands for compatibility with historical inputs. OmniCloudMask
prefers B8A and the crop model also uses B8A. Balkan-1 retains its RGB, NIR and
PAN L1ORT contract; PAN is not sent to either selected model.

The payload reaches `DOWNLINK_READY`; the complete receiver flow finishes with
`MVP_READY`. Routine transmission consists of
exactly `scene.webp`, `condition.png` and `scene.json`; full-resolution GeoTIFFs
remain local processing intermediates. The JSON contains exact measurements,
score explanations, evidence quality, georeferencing, asset checksums and an
interactive query grid. A condition label is a single-scene spectral screening
priority—not a disease diagnosis. See
[`integration/verification.json`](integration/verification.json) for the exact
accepted Sentinel/PASTIS execution evidence and its limitations.

## Coordinate-driven Earth Engine missions

The persistent payload API can acquire the fixed
`COPERNICUS/S2_SR_HARMONIZED` five-band Sentinel-2 input directly from Google
Earth Engine. Candidate metadata is ordered on the payload, but the existing
OmniCloudMask stage remains authoritative for cloud acceptance over the requested
region. An accepted candidate continues from the same cloud-stage result; cloud
inference and masks are not recreated.

```powershell
vita-mission run-region `
  --bbox "23.10,42.50,23.15,42.55" `
  --start "2026-07-01" `
  --end "2026-07-29" `
  --region-id "field_42" `
  --selection-policy target_cloud_range `
  --target-cloud-min 15 --target-cloud-max 35 --target-cloud-ideal 25 `
  --payload-url "http://127.0.0.1:8081" `
  --ground-store "ground/runtime"
```

The ground command submits coordinates, polls safe progress, downloads exactly
`scene.json`, `scene.webp` and `condition.png`, verifies independent payload and
manifest checksums, and calls the existing ground catalog ingestion. Runtime
credentials are mounted only into the payload service. See
[`payload/README.md`](payload/README.md) for service and deployment details.

## Training datasets and model contract

The selected payload checkpoint is based on a controlled two-source training
mixture:

| Source | Coverage | Role in the selected model |
| --- | --- | --- |
| IBM–NASA multi-temporal crop classification | 3,854 HLS S30 chips from the contiguous United States in 2022 | Base training corpus, internal validation and untouched held-out test |
| Optical PASTIS | European Sentinel-2 patches | Training replay only; eligible folds 1–4 are sampled so PASTIS contributes 20% of combined training examples |

PASTIS fold 5, containing 496 patches, remains reserved and is not included in
the reported selected-model metrics. The current internal-validation results
therefore should not be presented as a held-out European performance estimate.

### IBM–NASA HLS base corpus

The official IBM-NASA dataset is hosted at
<https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification>
and contains 3,854 HLS S30 chips from the contiguous United States in 2022.
Downloads default to the audited repository revision
`f285bb27c8f623a0fb6a44a6fd953c3ad34007d6`.
Each 224 x 224 input GeoTIFF has three dates and six bands per date:

```text
BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2
```

TerraTorch selects indices 0-3 from each date, producing:

```text
[B, 4, 3, 224, 224]
```

The data are 30 m HLS S30 observations in EPSG:5070. Masks contain no-data
class 0 plus 13 target classes 1-13. The loader reduces labels by one, so the
training ignore index is `-1` and target classes are 0-12.

For the IBM–NASA HLS corpus, final evaluation is kept independent of model
selection using:

- train: deterministic 90% of the official training chips;
- validation: the remaining deterministic 10% of official training chips;
- test: every official validation chip, untouched during training.

A chip's three dates stay together. The validator also rejects duplicate chip
IDs and overlapping footprints across the official source partitions. The
published data are randomly sampled rather than a geographic holdout, so
nearby-chip spatial autocorrelation remains a scientific limitation.

## Environment

Python 3.11 is recommended. On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip install -e .\shared
python -m pip install -e .\payload
python -m pip install -e .\ground
python -m pip install -e .\integration
```

On Linux/macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip install -e ./shared
python -m pip install -e ./payload
python -m pip install -e ./ground
python -m pip install -e ./integration
```

Do not replace PyTorch without checking the installed build, driver, and CUDA
compatibility:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

The training configuration explicitly requires one GPU; it does not silently
fall back to CPU.

## Download and validate

Confirm at least 30 GiB is free, then download the complete 11.8 GB archive
set:

```powershell
prithvi-download --root data/multi_temporal_crop --delete-archives
```

`huggingface_hub` resumes interrupted downloads. Extraction occurs in a
staging directory, rejects traversal, links, devices, and other special tar
members, verifies every image/mask against the official manifests, and only
then replaces an incomplete split and deletes the local archive. Re-running is
idempotent.

Run exhaustive validation:

```powershell
prithvi-validate --root data/multi_temporal_crop --device cuda
prithvi-check-config --config configs/prithvi_4band_head_only.yaml
```

Validation scans all 3,854 images and masks, not a sample. It checks counts and
manifests, dimensions, the official band contract, finite values, CRS,
image-mask transforms, geographic centers and footprint leakage, labels,
class support, dates, metadata coverage, logical train/validation/test splits,
and a real TerraTorch batch contract. Its machine-readable report is
`outputs/dataset_validation.json`.

## Lint and smoke test

```powershell
python -m ruff check src scripts shared/src payload/src ground/src
python -m prithvi_crop.smoke --config configs/prithvi_4band_head_only.yaml
```

If GNU Make is installed, `make lint` and `make smoke` are equivalent. GNU
Make is not bundled with Windows.

The smoke test loads one real batch (at least two chips because of UPerNet
BatchNorm), normalizes it, moves image and metadata to CUDA, executes
forward/loss/backward/optimizer steps, verifies the backbone has no gradients
or parameter changes, confirms downstream parameters update, and prints peak
allocated GPU memory.

## Train

The validated launcher prints the GPU, software versions, batch settings,
precision, parameter counts, split sizes, class distribution, cache, and
output paths before invoking TerraTorch:

```powershell
prithvi-train --config configs/prithvi_4band_head_only.yaml
```

It automatically resumes the newest `last.ckpt` under `outputs/`. To start a
new run explicitly:

```powershell
prithvi-train --config configs/prithvi_4band_head_only.yaml --resume none
```

Best and latest checkpoints are kept at the fixed, resume-safe path
`outputs/prithvi_4band_head_only/checkpoints/`; TensorBoard event logs are
versioned separately under `outputs/logs/`.

### European PASTIS replay refinement

The European refinement reuses the best augmented checkpoint, keeps every
original HLS training sample, and makes optical PASTIS data 20% of the combined
training examples. The selected single-frame binary configuration uses this
replay directly. Only unambiguous PASTIS crop labels receive one of the existing
fine-grained classes; all other supported agricultural labels provide
crop/non-crop supervision. No output classes are added.

On Windows, the resumable background pipeline downloads the official archive,
verifies its checksum, extracts only Sentinel-2 arrays, masks and metadata,
runs preflight and smoke checks, and then launches training:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_europe_replay_pipeline.ps1
```

The archive, extracted data, checkpoints and logs remain under the ignored
`data/` and `outputs/` directories. The reusable adapter and configuration stay
versioned in Git.

When PASTIS is already extracted, validate all metadata-linked NumPy image
headers and semantic targets without loading the model:

```powershell
prithvi-validate-europe --root data/europe/pastis
```

Extra image arrays without metadata are ignored. The validated inventory has
2,433 metadata-linked patches: 1,937 in replay-eligible folds 1–4 and 496 in
reserved fold 5. The loader deterministically samples only the number of
eligible fold-1–4 examples needed to maintain the configured 20% replay share;
it does not put all 2,433 patches into training. PASTIS is not added to the HLS
internal-validation or held-out-test loaders. The replay pipeline detects a
valid extracted dataset and will not download the archive again.

### Selected single-image payload model

`configs/prithvi_4band_single_frame_binary.yaml` trains the selected direct
crop/non-crop model from one image. Each source date becomes an independent
example while the split remains grouped by chip. The frozen Prithvi backbone
uses its native one-frame positional encoding and warm-started downstream
weights. The one- and three-frame crop-type checkpoints remain preserved under
`outputs/` but are not shipped in the base payload bundle.

The direct equivalent is shown below. Set the cache variables first when you
want the direct CLI to use the same repository-local caches as the launcher:

```powershell
$env:HF_HOME = (Resolve-Path outputs/cache/huggingface)
$env:MPLCONFIGDIR = (Resolve-Path outputs/cache/matplotlib)
terratorch fit --config configs/prithvi_4band_head_only.yaml
```

The validated RTX 3060 6 GiB profile uses batch size 8, four workers, FP16
mixed precision, and no gradient accumulation. UPerNet's training-time
BatchNorm requires a batch of at least 2. For a more constrained GPU, preserve
the effective batch size of 8 with:

```powershell
prithvi-train --config configs/prithvi_4band_head_only.yaml `
  --batch-size 2 --num-workers 4 `
  --trainer.accumulate_grad_batches=4
```

Pretrained weights are downloaded automatically from
<https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-100M-TL>. The
repository launcher defaults the cache to `outputs/cache/huggingface/hub`
(ignored by Git), unless `HF_HOME` or `HF_HUB_CACHE` is set before launch.

## Evaluate

Choose the checkpoint with the highest validation macro F1 from the checkpoint
directory; do not tune against the held-out test results:

```powershell
prithvi-evaluate --config configs/prithvi_4band_head_only.yaml --checkpoint outputs/path/to/best.ckpt
```

Direct TerraTorch equivalent:

```powershell
terratorch test --config configs/prithvi_4band_head_only.yaml --ckpt_path outputs/path/to/best.ckpt
```

Evaluation reports held-out loss, overall and balanced accuracy, macro
precision/recall/F1, mIoU, per-class precision/recall/F1, and class support.
It writes JSON metrics plus CSV and PNG confusion matrices under
`outputs/prithvi_4band_head_only/evaluation/`.

## Classes

```text
0  Natural Vegetation
1  Forest
2  Corn
3  Soybeans
4  Wetlands
5  Developed / Barren
6  Open Water
7  Winter Wheat
8  Alfalfa
9  Fallow / Idle Cropland
10 Cotton
11 Sorghum
12 Other
```

## Scope and next model stage

The IBM–NASA base corpus is US-only, 2022-only HLS S30 at 30 m, while the
selected model also has European Sentinel-2 exposure through PASTIS replay.
That mixture is broader than US-only training, but it is still not a global,
pan-European or Balkan benchmark. No held-out European result is currently
reported. A Balkan validation program still needs local multi-season labels,
geographic and temporal holdouts, sensor/radiometric harmonization, and checks
for the exact narrow-NIR spectral response.

For crop health, keep the same four-band temporal input and shared model
pipeline, define an agronomically meaningful health target, collect
field-linked or independently validated health labels, and add a health head
or multitask objective. Evaluate health by crop, growth stage, geography, and
season, with calibration and uncertainty reporting. Do not derive a health
claim from the current crop-type masks alone.

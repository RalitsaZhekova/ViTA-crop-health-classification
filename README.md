# Prithvi four-band crop and land-cover model

This repository trains pixel-wise crop models using a frozen
`Prithvi-EO-2.0-100M-TL` backbone and a trainable UPerNet task head. The
selected payload model classifies crop/non-crop from one image; preserved
experiments classify crop type from one or three observations. The only inputs
are `BLUE`, `GREEN`, `RED`, and
`NIR_NARROW`, in that order. No SWIR channel is passed to the model and no
synthetic, copied, or imputed SWIR channel is created.

The first checkpoint predicts 13 crop and land-cover classes. The same model
pipeline is intended to gain crop-health responsibility later through a
separately supervised health or multitask head. This crop-type checkpoint does
not infer health: suitable health targets, definitions, and evaluation data are
not present in this dataset.

The separate rule-based condition stage calculates NDVI, EVI, GNDVI, SAVI,
CVI and RGB diagnostics only on clear, confident crop pixels. It combines the
most defensible normalized vigor components into an auditable screening score,
then applies robust within-crop-region anomaly analysis. Its reflectance,
masking, interpretation and JSON output rules are documented in
[`ground/health_analysis_contract.md`](ground/health_analysis_contract.md).

## Extracted components

The selected model is frozen for the current project phase. Its checksum-pinned
runtime and the remaining system responsibilities are separated into:

- [`training/`](training/README.md): dataset references and retained experiment
  summaries, with canonical training code and configs at the repository root;
- [`payload/`](payload/README.md): weights-only model inference plus Balkan-1
  reconstruction and cloud-mask boundaries;
- [`ground/`](ground/README.md): condition measurements, storage, API and
  visualization boundaries;
- [`integration/`](integration/README.md): one-command Sentinel payload-to-ground
  demonstration orchestration;
- [`shared/`](shared/README.md): bands, normalization, classes, thresholds and
  exchange schemas.

[`COMPONENTS.md`](COMPONENTS.md) defines ownership and canonical storage
locations.

## Complete Sentinel demonstration

The current local end-to-end command runs intake, cloud/shadow masking, crop
segmentation and streamed ground crop-condition analysis:

```powershell
.\integration\scripts\run_sentinel_end_to_end.ps1 `
  -InputPath testing\inputs\sentinel2\your_five_band_scene.tif `
  -AcquiredAt 2026-07-27T12:00:00Z `
  -ReflectanceScale 10000 `
  -Output testing\runs\your_end_to_end_run
```

The Sentinel development route currently requires described B02, B03, B04,
B08 and B8A bands because the retained cloud and crop models use different NIR
responses. The final Balkan mission contract remains RGB plus one NIR; this
five-band development adapter must not be mistaken for that unresolved sensor
interface.

The result directory contains separate payload and ground records plus a
portable `end_to_end_result.json`. A condition label is a single-scene spectral
screening priority—not a disease diagnosis. See
[`integration/verification.json`](integration/verification.json) for the exact
accepted Sentinel/PASTIS execution evidence and its limitations.

## Data and model contract

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

To keep final evaluation independent of model selection, this repository uses:

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
```

On Linux/macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
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

### European replay refinement

The European refinement reuses the best augmented checkpoint, keeps every
original training sample, and adds a conservative 20% replay share from the
optical PASTIS dataset. Only unambiguous PASTIS crop labels receive one of the
existing fine-grained classes; all other supported agricultural labels provide
crop/non-crop supervision. No output classes are added, and PASTIS fold 5
remains reserved.

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

Extra image arrays without metadata are ignored. Training uses only the 2,433
patch IDs that have metadata and matching semantic targets. The replay
pipeline detects a valid extracted dataset and will not download the archive
again.

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

This dataset is US-only, 2022-only, HLS S30 at 30 m, and not a global or Balkan
benchmark. A Balkan validation program needs local multi-season labels,
geographic and temporal holdouts, sensor/radiometric harmonization, and checks
for the exact narrow-NIR spectral response.

For crop health, keep the same four-band temporal input and shared model
pipeline, define an agronomically meaningful health target, collect
field-linked or independently validated health labels, and add a health head
or multitask objective. Evaluate health by crop, growth stage, geography, and
season, with calibration and uncertainty reporting. Do not derive a health
claim from the current crop-type masks alone.

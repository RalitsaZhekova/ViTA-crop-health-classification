# ViTA Phase 1 — Cloud Detection

Production-oriented cloud filtering for the Space Challenges 2026 Crop Health Monitoring pipeline.

## Chosen model

The module uses the official pretrained CloudSEN12 **`dtacs4bands`** model.

Exact input order:

1. `B08` — Near Infrared
2. `B04` — Red
3. `B03` — Green
4. `B02` — Blue

The checkpoint was trained on Sentinel-2 **L1C top-of-atmosphere reflectance**. The Earth Engine
notebook therefore exports from `COPERNICUS/S2_HARMONIZED`, and preprocessing divides the original
integer values by 10,000. No training is required for the first prototype.

## Outputs

- semantic mask: clear, thick cloud, thin cloud, cloud shadow;
- binary unusable-pixel mask;
- combined cloud score map;
- cloud, shadow and usable percentages;
- `PROCESS`, `PROCESS_CLEAR_AREAS` or `REJECT` decision;
- georeferenced GeoTIFFs, JSON metadata and visual preview.

The cloud score is a softmax model confidence score. It is **not claimed to be calibrated probability**.

## Repository layout

```text
configs/          Runtime configuration
notebooks/        Google Earth Engine export and demonstration
src/              Production package
scripts/          Inference and model-download commands
tests/            Unit and external integration tests
docs/             Architecture, evaluation and integration documentation
```

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -e ".[model,dev,gee]"
```

## Workflow

### 1. Export Sentinel-2 scenes

Open `notebooks/01_gee_dataset_builder.ipynb`, authenticate Earth Engine, select the areas and
start the Drive export tasks. Do not cloud-mask the imagery before inference.

### 2. Download the official checkpoint

```bash
python scripts/download_weights.py
```

### 3. Run one scene

```bash
python scripts/run_inference.py \
  --input data/raw/cloud_scene_01.tif \
  --output outputs
```

### 4. Run unit tests

```bash
pytest -q -m "not integration"
```

The real model smoke test is intentionally marked `integration` because it downloads/loads external
weights:

```bash
pytest -q -m integration
```

## Docker

Download the weights first so the container can run fully offline, then:

```bash
docker build -t vita-cloud-detector .
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/outputs:/app/outputs" \
  -v "$(pwd)/models:/app/models" \
  vita-cloud-detector \
  --input /app/data/raw/cloud_scene_01.tif \
  --output /app/outputs
```

## Honest validation status

The repository includes verified local unit tests for tiling, geospatial I/O, post-processing,
metrics and the complete pipeline through a deterministic test backend. Quantitative accuracy will
only be reported after running the official checkpoint on expert-labelled CloudSEN12 validation data.

## Upstream licences

The upstream package is LGPL-3.0. CloudSEN12 data and pretrained models are released under
non-commercial Creative Commons terms. Attribution and licence review are required.

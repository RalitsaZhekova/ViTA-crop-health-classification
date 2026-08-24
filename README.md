# ViTA crop-intelligence MVP — local Windows setup

ViTA runs cloud detection, crop classification, crop-condition analysis, and web
packaging for Sentinel-2 and Balkan-1 imagery. This tutorial is for local execution on
a Windows PC with an NVIDIA GPU. A Jetson, SSH, Docker, and TensorRT are not required.

The Git repository does **not** include model binaries, Sentinel imagery, Balkan
imagery, telemetry, or credentials. Each user must obtain those files independently
and place them in the exact folders described below.

## 1. Prerequisites

- Windows 10 or 11 with PowerShell;
- Git;
- 64-bit Python 3.11;
- an NVIDIA GPU with a driver compatible with CUDA 12.6;
- sufficient free disk space for the model files, input GeoTIFFs, and generated jobs.

Clone the repository and open it in PowerShell:

```powershell
git clone <REPOSITORY-URL> prithvi_crop_head_starter
cd prithvi_crop_head_starter
```

## 2. Create the virtual environment and install every dependency

Run all Python commands inside the project virtual environment:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.13.0+cu126 torchvision==0.28.0+cu126 `
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e ".[dev,earth-engine]"
python -m pip check
```

`pip check` must finish with `No broken requirements found`. Keep this PowerShell
window open so the virtual environment remains active.

## 3. Install the model weights

### Fine-tuned Prithvi crop model

Obtain the project's fine-tuned crop checkpoint from the project owner or its model
artifact release. A generic Prithvi foundation checkpoint is **not** a replacement.
Place it at:

```text
payload/models/prithvi_crop_binary_single_frame_v1_weights.pt
```

Required file properties:

```text
Size:    382645915 bytes
SHA-256: c948977bffdaeb89ecf4f4d069db13c7ed81d4f3403eec9e257c45c235b1484e
```

Verify it from PowerShell:

```powershell
Get-FileHash `
  .\payload\models\prithvi_crop_binary_single_frame_v1_weights.pt `
  -Algorithm SHA256
```

### OmniCloudMask cloud models

Obtain the two OmniCloudMask V4 checkpoints and place them in
`payload/models/omnicloudmask/` with these exact names:

```text
PM_model_OCM_7.97_R_G_NIR_3_smp_edgenext_small.usi_in1k_PT_state.safetensors
PM_model_OCM_7.97_R_G_NIR_3_smp_regnety_004.pycls_in1k_PT_state.safetensors
```

The application checks all model checksums and fails rather than using a missing or
different checkpoint.

## 4. Obtain and place the input imagery

Images are **not supplied by this repository**. Users must acquire or create their own
properly licensed inputs and preserve the following folder and filename contracts.

### Stored Sentinel-2

Place the prepared five-band Sentinel GeoTIFF here:

```text
data/sentinel2/S2_20260814T104624_T31UFU_flevoland-latest-2026.tif
```

It must contain bands `B02`, `B03`, `B04`, `B08`, and `B8A` in that order, use the
expected Sentinel reflectance scale of 10000, and contain georeferencing. A raw SAFE
download is not automatically interchangeable with this prepared GeoTIFF.

### Preprocessed Balkan-1

Place the preprocessed image and its matching calibration sidecar here:

```text
data/balkan1/preprocessed/3408_L1ORT.tif
data/balkan1/preprocessed/3408_L1ORT.crop_calibration.json
```

### Raw Balkan-1

Raw processing requires more than the raw TIFF. Obtain the matching telemetry,
manifest, radiometric validation, and parent calibration, then create this layout:

```text
data/balkan1/raw/3408/3408_Raw.tif
data/balkan1/raw/3408/position.csv
data/balkan1/raw/3408/attitude.csv
data/balkan1/derived/l1a/3408_L0R_manifest.json
data/balkan1/derived/l1a/3408_L1A_reference_validation.json
data/balkan1/preprocessed/3408_L1ORT.crop_calibration.json
```

The first run of an unchanged raw scene creates a preprocessing cache under
`runtime/payload-local/cache/raw-preprocess`. Later runs reuse the verified aligned
and model-grid products while still rerunning all model and output stages.

### Google Earth Engine bounding-box selection

To enable live Sentinel selection, create or obtain an Earth Engine-enabled Google
service-account key and place its JSON file at:

```text
secrets/earth-engine.json
```

The JSON must contain its `project_id`, and the service account must have Earth Engine
access. Do not commit this credential.

## 5. Run locally

Make sure no obsolete Jetson target is set in the current PowerShell session:

```powershell
Remove-Item Env:VITA_JETSON_SSH_TARGET -ErrorAction SilentlyContinue
Remove-Item Env:VITA_RAW_SSH_TARGET -ErrorAction SilentlyContinue
```

The first analysis command starts and warms one local CUDA service. Useful commands:

```powershell
.\vita.ps1 sentinel
.\vita.ps1 balkan
.\vita.ps1 raw
.\vita.ps1 web
.\vita.ps1 health
.\vita.ps1 stop
```

The dashboard is available at [http://127.0.0.1:8000/](http://127.0.0.1:8000/).

## Final commands and input values

Check that PyTorch can use CUDA:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); print('Torch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)"
```

Run the web application:

```powershell
.\vita.ps1 web
```

Enter or pass these input values:

```text
Preprocessed Balkan input path: balkan1/preprocessed/3408_L1ORT.tif
Raw Balkan scene ID:            3408
Sentinel input folder:          sentinel2
Sentinel image filename:        S2_20260814T104624_T31UFU_flevoland-latest-2026.tif
```

Equivalent command-line examples:

```powershell
.\vita.ps1 balkan -InputPath balkan1/preprocessed/3408_L1ORT.tif
.\vita.ps1 raw -InputPath 3408
.\vita.ps1 sentinel `
  -InputPath sentinel2 `
  -Image S2_20260814T104624_T31UFU_flevoland-latest-2026.tif
```

For implementation details and optimization notes, see
[docs/PIPELINE_GUIDE.md](docs/PIPELINE_GUIDE.md).

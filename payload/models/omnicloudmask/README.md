# OmniCloudMask V4 weights

This directory is the local, ignored checkpoint cache for the production cloud
backend. Download the two pinned V4 ensemble members with:

```powershell
.\.venv\Scripts\python.exe payload\scripts\download_cloud_weights.py
```

The script downloads through OmniCloudMask `1.7.1` and verifies both component
SHA-256 values plus the stable ensemble fingerprint recorded in `model.yaml`.
Checkpoint files use the `.safetensors` extension and are intentionally excluded
from Git. They are copied into payload container builds only from a prepared local
model directory.

Review `payload/cloud_masking/UPSTREAM.md` before distribution or deployment.

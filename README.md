# ViTA crop-intelligence MVP

ViTA runs cloud detection, crop classification, crop-condition analysis, and compact
web packaging for local Sentinel-2 and preprocessed Balkan-1 GeoTIFFs.

## Local quick start

Install the project once from PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Then use the warm, optimized payload service through three short commands:

```powershell
.\vita.ps1 sentinel
.\vita.ps1 balkan
.\vita.ps1 web
```

The first pipeline command starts the payload service, loads both models, prepares the
verified Balkan cache, warms CUDA, and waits for readiness. Every run prints the full
execution-time breakdown and imports its WebP, PNG, and JSON bundle into
`runtime\ground`. The `web` command starts the dashboard and opens
[http://127.0.0.1:8000/](http://127.0.0.1:8000/).

Useful commands:

```powershell
.\vita.ps1 health
.\vita.ps1 stop
.\vita.ps1 sentinel -Image another_scene.tif -RegionId another-region
.\vita.ps1 balkan -InputPath balkan1/preprocessed/another_scene.tif
```

The original `vita-mvp`, `vita-payload-server`, REST API, SSH orchestration, and Docker
commands remain supported. Use the persistent service above for latency measurements;
the one-shot CLI must reload models in every process.

The Jetson production path builds an FP16 TensorRT cloud engine and an FP32 TensorRT
crop engine on the target Orin, validates them against PyTorch, and runs all four
packaged demo scenes three times. Deployment fails unless every warm
`payload_seconds` result is below two seconds; it never substitutes a cached prediction
for inference.

## Documentation

- [Pipeline and optimization guide](docs/PIPELINE_GUIDE.md): architecture, execution
  contracts, code map, timings, outputs, local operation, and every implemented
  optimization.
- [Jetson deployment guide](docs/DEPLOYMENT_JETSON.md): NVIDIA container build,
  CUDA/TensorRT configuration, SSH uplink/downlink, GHCR publishing, validation, and
  production troubleshooting.

Inputs and model binaries are intentionally excluded from Git. Generated jobs, caches,
engine plans, logs, dashboard receipts, and previews live under `runtime/` and can be
removed safely while the local services are stopped.

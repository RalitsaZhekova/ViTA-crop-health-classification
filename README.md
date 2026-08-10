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

The Jetson production path builds direct TensorRT 10.8 plans offline on the target
Orin and loads them through TensorRT's native Python runtime. Crop retains FP32 I/O
with qualified mixed-FP16 tactics. Cloud uses strongly typed FP16 for the 869/891 px
Balkan plans and FP32 for the numerically sensitive 1000 px Sentinel plan; Sentinel
uses physical batch one while the Balkan plans use batch four.
The checksum-sealed plans remain unavailable until balanced crop parity and
all-four-scene cloud parity pass against the operational PyTorch CUDA references.
Deployment then requires every repeated scene run to remain below two seconds and
never substitutes a cached prediction for inference.

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

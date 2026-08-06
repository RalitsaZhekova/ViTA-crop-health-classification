# ViTA crop-intelligence MVP

This repository runs cloud detection, crop classification, crop-condition analysis, and compact web packaging for local Sentinel-2 and preprocessed Balkan-1 GeoTIFFs.

For the NVIDIA Jetson AGX Orin payload deployment, SSH-tunnel ground command, CUDA/TensorRT configuration, GitHub Container Registry workflow, performance validation, and troubleshooting, use the [complete Jetson deployment guide](docs/DEPLOYMENT_JETSON.md).

The existing local commands remain supported:

```powershell
vita-mvp balkan data\balkan1\preprocessed\3408_L1ORT.tif --region-id balkan-test-3408
vita-mvp sentinel .\data\sentinel2 --image S2_20260712T170851_T14TPL_cloudy.tif --region-id sentinel-local-cloudy
vita-dashboard --store runtime\ground --host 127.0.0.1 --port 8000
```


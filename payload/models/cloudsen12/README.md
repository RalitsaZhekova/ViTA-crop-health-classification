# Cloud model

Run the download helper from the repository root:

```powershell
.\.venv\Scripts\python.exe payload\scripts\download_cloud_weights.py
```

It downloads `dtacs4bands.pt` through the pinned official package and verifies
SHA-256 `37205adce72fbbb65a3cfa8f47676c84ebf9b1555a27a3838d584072c954b22d`.
The large checkpoint is ignored by Git but included in the local prototype
payload manifest.

The current weights are non-commercial. See
`payload/cloud_masking/UPSTREAM.md` before distribution or deployment.

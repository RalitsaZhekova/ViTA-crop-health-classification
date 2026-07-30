from __future__ import annotations

import pytest

from prithvi_payload import container_preflight


MODEL_HASHES = {
    "crop_model_sha256": "c" * 64,
    "cloud_model_sha256": "d" * 64,
}


def test_safe_diagnostics_rejects_required_cuda_without_fallback(monkeypatch) -> None:
    monkeypatch.setattr(container_preflight, "verify_model_artifacts", lambda: MODEL_HASHES)
    monkeypatch.setattr(container_preflight.torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("CUDA_REQUIRED", "1")

    with pytest.raises(RuntimeError, match="cannot access CUDA"):
        container_preflight.safe_diagnostics()


def test_safe_diagnostics_rejects_multiple_gpu_jobs(monkeypatch) -> None:
    monkeypatch.setattr(container_preflight, "verify_model_artifacts", lambda: MODEL_HASHES)
    monkeypatch.setattr(container_preflight.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(container_preflight.torch.cuda, "get_device_name", lambda _index: "GPU")
    monkeypatch.setenv("CUDA_REQUIRED", "1")
    monkeypatch.setenv("VITA_MAX_CONCURRENT_JOBS", "2")

    with pytest.raises(RuntimeError, match="must be 1"):
        container_preflight.safe_diagnostics()


def test_safe_diagnostics_contains_only_safe_runtime_fields(monkeypatch) -> None:
    monkeypatch.setattr(container_preflight, "verify_model_artifacts", lambda: MODEL_HASHES)
    monkeypatch.setattr(container_preflight.torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("CUDA_REQUIRED", "0")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "sensitive-path")

    result = container_preflight.safe_diagnostics()

    assert result["status"] == "ok"
    assert result["cuda_available"] is False
    assert result["gpu_name"] is None
    assert set(MODEL_HASHES.items()) <= set(result.items())
    assert "sensitive-path" not in str(result)
    assert "credential" not in str(result).lower()

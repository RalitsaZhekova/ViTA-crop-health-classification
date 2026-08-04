from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch
from cloud_detection import backend as backend_module
from cloud_detection.backend import OmniCloudMaskBackend, ensemble_sha256


def test_ensemble_fingerprint_verifies_every_component(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    components = {"first.safetensors": b"first", "second.safetensors": b"second"}
    expected = {
        filename: hashlib.sha256(content).hexdigest() for filename, content in components.items()
    }
    monkeypatch.setattr(backend_module, "OMNICLOUDMASK_MODEL_FILES", expected)
    for filename, content in components.items():
        (tmp_path / filename).write_bytes(content)

    digest = hashlib.sha256()
    for filename, component_sha256 in sorted(expected.items()):
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(component_sha256))

    assert ensemble_sha256(tmp_path) == digest.hexdigest()


def test_backend_reorders_existing_contract_and_preserves_score_contract() -> None:
    captured: dict[str, object] = {}

    def predict(image: np.ndarray, **kwargs: object) -> np.ndarray:
        captured["image"] = image.copy()
        captured["kwargs"] = kwargs
        scores = np.full((4, *image.shape[1:]), 0.001, dtype=np.float32)
        scores[2] = 1.001
        return scores

    backend = OmniCloudMaskBackend.__new__(OmniCloudMaskBackend)
    backend.device = torch.device("cpu")
    backend.inference_dtype = "fp32"
    backend.patch_size = 1000
    backend.patch_overlap = 300
    backend.batch_size = 1
    backend.models = [object(), object()]
    backend._predict_from_array = predict

    tile = np.empty((4, 32, 32), dtype=np.float32)
    tile[0] = 0.80  # NIR
    tile[1] = 0.30  # Red
    tile[2] = 0.20  # Green
    tile[3] = 0.10  # Blue, deliberately not sent to the model
    tile[1, 0, 0] = 0.0

    result = backend.predict(tile)

    image = captured["image"]
    assert isinstance(image, np.ndarray)
    assert np.allclose(image[:, 1:, 1:], np.array([0.30, 0.20, 0.80])[:, None, None])
    assert np.all(image[:, 0, 0] == 0.0)
    assert result.scores.shape == (4, 32, 32)
    assert np.max(np.abs(result.scores.sum(axis=0) - 1.0)) < 1e-6
    assert np.all(result.scores.argmax(axis=0) == 2)
    assert captured["kwargs"]["custom_models"] == backend.models  # type: ignore[index]


def test_backend_skips_inference_for_an_all_invalid_tile() -> None:
    backend = OmniCloudMaskBackend.__new__(OmniCloudMaskBackend)
    tile = np.zeros((4, 32, 32), dtype=np.float32)

    result = backend.predict(tile)

    assert np.all(result.scores[0] == 1.0)
    assert np.count_nonzero(result.scores[1:]) == 0

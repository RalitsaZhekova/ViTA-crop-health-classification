from __future__ import annotations

import pytest
import torch
from cloud_detection.tensorrt_backend import (
    CloudTensorRTRouter,
    _CloudEnsemble,
    _engine_count,
)
from torch import nn


class _Scale(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image * self.value


def test_cloud_tensorrt_wrapper_preserves_mean_logit_ensemble() -> None:
    ensemble = _CloudEnsemble([_Scale(2.0), _Scale(4.0)])
    image = torch.ones((1, 3, 8, 8), dtype=torch.float32)

    torch.testing.assert_close(ensemble(image), image * 3.0)


def test_cloud_tensorrt_parity_records_logit_error_without_class_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VITA_CLOUD_TRT_MAX_CLASS_MISMATCH", "0")
    router = CloudTensorRTRouter.__new__(CloudTensorRTRouter)
    router._source = _CloudEnsemble([_Scale(1.0), _Scale(1.0)])
    image = torch.arange(1 * 3 * 4 * 4, dtype=torch.float32).reshape(1, 3, 4, 4)
    accelerated = router._source(image) + 1e-4

    parity = router._validate_output(image, accelerated)

    assert parity["class_mismatch_fraction"] == 0.0
    assert parity["mean_absolute_logit_error"] == pytest.approx(1e-4, rel=0.01)


def test_cloud_tensorrt_engine_detection_rejects_plain_pytorch() -> None:
    assert _engine_count(nn.Sequential(nn.Conv2d(3, 4, 1))) == 0

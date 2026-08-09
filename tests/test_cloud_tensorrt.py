from __future__ import annotations

import pytest
import torch
from cloud_detection.tensorrt_backend import (
    CloudTensorRTRouter,
    _CloudEnsemble,
    _engine_count,
    _remove_zero_channel_cat_noops,
)
from torch import nn


class _Scale(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image * self.value


class _ZeroChannelSkip(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        empty = torch.empty(
            (image.shape[0], 0, image.shape[2], image.shape[3]),
            device=image.device,
            dtype=image.dtype,
        )
        return torch.cat([image, empty], dim=1)


class _NonEmptySkip(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.cat([image, image], dim=1)


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


def test_cloud_tensorrt_removes_exact_zero_channel_cat_noop() -> None:
    image = torch.randn((1, 3, 8, 8), dtype=torch.float32)
    exported = torch.export.export(_ZeroChannelSkip(), (image,), strict=False)
    expected = exported.module()(image)

    assert _remove_zero_channel_cat_noops(exported) == 1
    actual = exported.module()(image)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all("empty" not in str(node.target) for node in exported.graph.nodes)


def test_cloud_tensorrt_keeps_nonempty_cat() -> None:
    image = torch.randn((1, 3, 8, 8), dtype=torch.float32)
    exported = torch.export.export(_NonEmptySkip(), (image,), strict=False)

    assert _remove_zero_channel_cat_noops(exported) == 0
    assert any("cat" in str(node.target) for node in exported.graph.nodes)

from __future__ import annotations

import threading

import numpy as np
import pytest
import torch
from cloud_detection.backend import _replace_zero_nodata_for_fixed_patch
from cloud_detection.tensorrt_backend import (
    CLOUD_MAX_AGGREGATE_CLASS_MISMATCH,
    CLOUD_MAX_SCENE_CLASS_MISMATCH,
    REVIEWED_CLOUD_SCENE_BATCH_SIZES,
    REVIEWED_CLOUD_SCENE_PRECISIONS,
    CloudTensorRTRouter,
    _CloudEnsemble,
    _remove_zero_channel_cat_noops,
)
from torch import nn


def _empty_router() -> CloudTensorRTRouter:
    router = CloudTensorRTRouter.__new__(CloudTensorRTRouter)
    nn.Module.__init__(router)
    return router


class _Scale(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image * self.value


class _FixedBatchScale(nn.Module):
    def __init__(self, batch_size: int, value: float) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.value = value

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        assert image.shape[0] == self.batch_size
        return image * self.value


class _CaptureDtype(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen: torch.dtype | None = None

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.seen = image.dtype
        return image


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


def test_raw_fixed_cloud_patch_keeps_nodata_inert_without_exact_zeros() -> None:
    image = np.ones((3, 40, 50), dtype=np.float32)
    image[:, :20] = 0.0

    prepared = _replace_zero_nodata_for_fixed_patch(image)

    assert np.count_nonzero(prepared == 0.0) == 0
    assert np.all(prepared[:, :20] > 0.0)
    np.testing.assert_array_equal(prepared[:, 20:], image[:, 20:])
    assert np.all(image[:, :20] == 0.0)


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


def test_cloud_tensorrt_router_satisfies_omnicloudmask_custom_model_contract() -> None:
    from omnicloudmask.cloud_mask import collect_models

    router = _empty_router()
    router._plans = {(8, 8): _Scale(1.0)}

    collected = collect_models(
        custom_models=[router],
        inference_device=torch.device("cpu"),
        inference_dtype=torch.float32,
        source="hugging_face",
    )

    assert collected == [router]
    assert isinstance(router, nn.Module)


def test_cloud_tensorrt_router_uses_module_forward_dispatch() -> None:
    router = _empty_router()
    router._plans = {(8, 8): _Scale(2.0)}
    router._used_profiles = set()
    router._lock = threading.Lock()
    image = torch.ones((1, 3, 8, 8))

    torch.testing.assert_close(router(image), image * 2.0)
    assert router._used_profiles == {(1, 8, 8)}


def test_cloud_tensorrt_router_pads_to_fixed_engine_batch_and_slices_output() -> None:
    router = _empty_router()
    router._plans = {(8, 8): _FixedBatchScale(4, 2.0)}
    router._batch_contracts = {(8, 8): (1, 4, 4)}
    router._used_profiles = set()
    router._lock = threading.Lock()
    image = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8)

    output = router(image)

    assert output.shape == image.shape
    torch.testing.assert_close(output, image * 2.0)
    assert router._used_profiles == {(2, 8, 8)}


def test_cloud_tensorrt_router_rejects_batch_outside_manifest_contract() -> None:
    router = _empty_router()
    router._plans = {(8, 8): _FixedBatchScale(4, 2.0)}
    router._batch_contracts = {(8, 8): (1, 4, 4)}
    router._used_profiles = set()
    router._lock = threading.Lock()

    with torch.no_grad(), pytest.raises(
        RuntimeError,
        match="outside the accepted contract",
    ):
        router(torch.ones((5, 3, 8, 8)))


def test_cloud_tensorrt_router_promotes_only_the_selected_profile() -> None:
    plan = _CaptureDtype()
    router = _empty_router()
    router._plans = {(8, 8): plan}
    router._batch_contracts = {(8, 8): (1, 1, 1)}
    router._input_dtypes = {(8, 8): torch.float32}
    router._used_profiles = set()
    router._lock = threading.Lock()

    output = router(torch.ones((1, 3, 8, 8), dtype=torch.float16))

    assert plan.seen == torch.float32
    assert output.dtype == torch.float32


def test_reviewed_cloud_batches_match_operational_payload_tiles() -> None:
    assert dict(REVIEWED_CLOUD_SCENE_BATCH_SIZES) == {869: 4, 891: 4, 1000: 1}
    assert dict(REVIEWED_CLOUD_SCENE_PRECISIONS) == {
        869: "fp16",
        891: "fp16",
        1000: "fp32",
    }
    assert CLOUD_MAX_AGGREGATE_CLASS_MISMATCH == 0.001
    assert CLOUD_MAX_SCENE_CLASS_MISMATCH == 0.002


def test_cloud_tensorrt_warmup_profiles_follow_physical_engines() -> None:
    router = _empty_router()
    router._records = {
        (700, 700): {"patch_size": 700},
        (891, 891): {"patch_size": 891},
    }
    router._batch_contracts = {
        (700, 700): (1, 1, 1),
        (891, 891): (1, 4, 4),
    }

    assert router.warmup_profiles == ((1, 700), (1, 891), (4, 891))

from __future__ import annotations

import torch
from prithvi_payload.inference import (
    PayloadCropModel,
    _functionalize_prithvi_export_for_tensorrt,
)
from torch import Tensor, nn


class _BatchRecordingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_size = 0

    def forward(
        self,
        image: Tensor,
        temporal_coords: Tensor,
        location_coords: Tensor,
    ) -> Tensor:
        self.batch_size = image.shape[0]
        height, width = image.shape[-2:]
        logits = torch.zeros((image.shape[0], 2, height, width), device=image.device)
        logits[:, 1] = 1
        return logits


class _FourTemporaryDivisions(nn.Module):
    def forward(self, image: Tensor) -> Tensor:
        vectors = []
        for _ in range(4):
            omega = torch.arange(4, dtype=image.dtype, device=image.device)
            omega /= 2.0
            vectors.append(omega)
        return image + torch.stack(vectors).sum(dim=0)


def test_optimized_model_pads_to_fixed_batch_and_returns_only_real_tiles() -> None:
    backend = _BatchRecordingModel()
    model = PayloadCropModel(
        backend,
        device=torch.device("cpu"),
        fixed_batch_size=4,
    )
    image = torch.zeros((1, 4, 1, 224, 224))
    temporal = torch.tensor([[[2026.0, 187.0]]])
    location = torch.zeros((1, 2))

    result = model.predict(
        image,
        temporal_coords=temporal,
        location_coords=location,
    )

    assert backend.batch_size == 4
    assert result.crop_probability.shape == (1, 224, 224)


def test_tensorrt_prediction_uses_unmodified_model_logits() -> None:
    backend = _BatchRecordingModel()
    model = PayloadCropModel(
        backend,
        device=torch.device("cpu"),
        fixed_batch_size=4,
        backend="tensorrt",
    )
    image = torch.zeros((1, 4, 1, 224, 224))
    temporal = torch.tensor([[[2026.0, 187.0]]])
    location = torch.zeros((1, 2))

    result = model.predict(
        image,
        temporal_coords=temporal,
        location_coords=location,
    )

    expected = torch.softmax(torch.tensor([0.0, 1.0]), dim=0)[1]
    torch.testing.assert_close(
        result.crop_probability,
        torch.full_like(result.crop_probability, expected),
    )


def test_prithvi_export_functionalization_preserves_output_and_decomposes() -> None:
    sample = torch.ones(4)
    exported = torch.export.export(_FourTemporaryDivisions(), (sample,), strict=False)
    reference = exported.module()(sample)

    replacements = _functionalize_prithvi_export_for_tensorrt(exported)
    result = exported.module()(sample)

    assert replacements == 4
    torch.testing.assert_close(result, reference)
    assert all(
        node.target != torch.ops.aten.div_.Tensor
        for node in exported.graph_module.graph.nodes
    )
    exported.run_decompositions()

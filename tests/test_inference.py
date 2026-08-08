from __future__ import annotations

import pytest
import torch
from prithvi_payload.inference import (
    PayloadCropModel,
    _fit_tensorrt_logit_calibration,
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


def test_tensorrt_prediction_applies_validated_logit_calibration() -> None:
    backend = _BatchRecordingModel()
    model = PayloadCropModel(
        backend,
        device=torch.device("cpu"),
        fixed_batch_size=4,
        backend="tensorrt",
        tensorrt_logit_scale=1.5,
        tensorrt_logit_bias=-0.25,
    )
    image = torch.zeros((1, 4, 1, 224, 224))
    temporal = torch.tensor([[[2026.0, 187.0]]])
    location = torch.zeros((1, 2))

    result = model.predict(
        image,
        temporal_coords=temporal,
        location_coords=location,
    )

    expected = torch.sigmoid(torch.tensor(1.25))
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


def test_tensorrt_logit_calibration_is_fit_on_disjoint_tiles() -> None:
    batch_size, height, width = 8, 4, 5
    reference_margin = torch.linspace(-2.0, 2.0, steps=height * width).view(
        1, height, width
    )
    reference_margin = reference_margin.expand(batch_size, -1, -1).clone()
    reference_margin += torch.arange(batch_size).view(-1, 1, 1) * 0.03
    expected_scale = 1.12
    expected_bias = -0.08
    accelerated_margin = (reference_margin - expected_bias) / expected_scale
    reference = torch.stack((torch.zeros_like(reference_margin), reference_margin), dim=1)
    accelerated = torch.stack(
        (torch.zeros_like(accelerated_margin), accelerated_margin), dim=1
    )
    valid_mask = torch.ones((batch_size, height, width), dtype=torch.bool)
    thresholds = torch.full((batch_size, 2), 0.3)

    parity, scale, bias = _fit_tensorrt_logit_calibration(
        reference,
        accelerated,
        valid_mask,
        thresholds,
        maximum_mismatch=0.002,
        maximum_mean_error=0.01,
    )

    assert scale == pytest.approx(expected_scale, abs=1e-5)
    assert bias == pytest.approx(expected_bias, abs=1e-5)
    assert parity["class_mismatch_fraction"] == 0.0
    assert parity["mean_absolute_probability_error"] < 1e-6
    assert parity["calibration_tile_count"] == 4.0
    assert parity["validation_tile_count"] == 4.0
    assert parity["validation_pixel_count"] == 80.0

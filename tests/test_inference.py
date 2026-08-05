from __future__ import annotations

import torch
from prithvi_payload.inference import PayloadCropModel
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

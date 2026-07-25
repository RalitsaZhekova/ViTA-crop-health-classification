import torch

from prithvi_crop.task import _cross_entropy_or_zero


def test_all_ignored_binary_target_produces_finite_zero_loss() -> None:
    logits = torch.randn(2, 2, 4, 4, requires_grad=True)
    target = torch.full((2, 4, 4), -1, dtype=torch.long)

    loss = _cross_entropy_or_zero(logits, target, ignore_index=-1)

    assert torch.isfinite(loss)
    assert loss.item() == 0
    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad).item() == 0

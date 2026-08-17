import pytest
import torch

from openpi.models_pytorch.patch16.model import Patch16Encoder


def test_resize_patch_tokens_bilinearly_restores_guidedvla_grid():
    tokens = torch.randn(2, 14 * 14, 768)
    resized = Patch16Encoder.resize_patch_tokens(tokens)
    assert resized.shape == (2, 16 * 16, 768)
    assert torch.isfinite(resized).all()


def test_resize_patch_tokens_rejects_wrong_grid():
    with pytest.raises(RuntimeError, match="Patch16 feature"):
        Patch16Encoder.resize_patch_tokens(torch.randn(1, 256, 768))

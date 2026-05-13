import torch

from openpi.models_pytorch.depth.token_merging import TokenMerging2D


def test_token_merging_accepts_inference_tensor_inputs():
    module = TokenMerging2D(patch_size=2, in_dim=8, out_dim=16)

    with torch.inference_mode():
        features = torch.randn(2, 16, 8)

    assert features.is_inference()

    merged = module(features)
    loss = merged.square().mean()
    loss.backward()

    assert module.norm.weight.grad is not None
    assert module.merge.weight.grad is not None

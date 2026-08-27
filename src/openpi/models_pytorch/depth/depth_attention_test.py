import torch

from openpi.models_pytorch.depth.depth_attention import DepthTokenKVProjector


def test_headwise_xavier_initializes_each_narrow_input_head_independently():
    torch.manual_seed(0)
    projector = DepthTokenKVProjector(
        hidden_size=48,
        num_heads=8,
        head_dim=256,
        num_groups=4,
        depth_head_indices=[4, 5],
        headwise_xavier_init=True,
    )
    weight = projector.k_projectors[0].weight.view(8, 256, 48)
    # Each block receives Xavier(48 -> 256), whose bound is far above the
    # monolithic Xavier(48 -> 2048) bound (~0.0535).
    assert all(head.abs().max() > 0.1 for head in weight)
    tokens = torch.randn(2, 256, 48)
    outputs = projector((tokens, tokens, tokens, tokens))
    assert len(outputs) == 4
    assert outputs[0].depth_token_k.shape == (2, 8, 256, 256)

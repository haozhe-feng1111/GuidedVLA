import pytest
import torch

from openpi.models_pytorch.sam2.model import Sam2Encoder


def test_prepare_fpn_tokens_resizes_all_levels_to_shared_grid():
    feature_maps = tuple(torch.randn(2, 256, side, side + 2) for side in (8, 16, 32, 64))

    token_groups = Sam2Encoder.prepare_fpn_tokens(
        feature_maps,
        feature_dim=256,
        token_grid_size=8,
    )

    assert len(token_groups) == 4
    assert all(tokens.shape == (2, 64, 256) for tokens in token_groups)


def test_prepare_fpn_tokens_rejects_wrong_level_count():
    with pytest.raises(RuntimeError, match="4 FPN levels"):
        Sam2Encoder.prepare_fpn_tokens(
            (torch.randn(1, 256, 8, 8),) * 3,
            feature_dim=256,
            token_grid_size=8,
        )


def test_prepare_fpn_tokens_rejects_wrong_channel_dim():
    with pytest.raises(RuntimeError, match="must have 256 channels"):
        Sam2Encoder.prepare_fpn_tokens(
            (torch.randn(1, 128, 8, 8),) * 4,
            feature_dim=256,
            token_grid_size=8,
        )

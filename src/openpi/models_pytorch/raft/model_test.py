import pytest
import torch

from openpi.models_pytorch.raft.model import RaftFeatureEncoder


class _FakeFeatureEncoder(torch.nn.Module):
    def forward(self, images):
        assert images.shape == (2, 3, 224, 224)
        assert images.dtype == torch.float32
        return torch.ones(2, 256, 28, 28, device=images.device)


def test_resize_stride8_feature_to_256_native_tokens():
    feature_map = torch.randn(2, 256, 28, 28)
    tokens = RaftFeatureEncoder.resize_feature_map(feature_map)
    assert tokens.shape == (2, 256, 256)
    assert torch.isfinite(tokens).all()


def test_resize_rejects_non_raft_feature_contract():
    with pytest.raises(RuntimeError, match="RAFT-Large fnet feature"):
        RaftFeatureEncoder.resize_feature_map(torch.randn(1, 128, 28, 28))


def test_forward_processes_one_image_and_reuses_one_final_feature():
    encoder = RaftFeatureEncoder.__new__(RaftFeatureEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.feature_encoder = _FakeFeatureEncoder()

    outputs = encoder(torch.zeros(2, 3, 224, 224, dtype=torch.bfloat16, requires_grad=True))

    assert len(outputs) == 4
    assert all(output.shape == (2, 256, 256) for output in outputs)
    assert all(output.dtype == torch.bfloat16 for output in outputs)
    assert all(output is outputs[0] for output in outputs)
    assert not outputs[0].requires_grad


def test_forward_rejects_image_pairs_or_wrong_resolution():
    encoder = RaftFeatureEncoder.__new__(RaftFeatureEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.feature_encoder = _FakeFeatureEncoder()
    with pytest.raises(RuntimeError, match="current-frame RGB"):
        encoder(torch.zeros(2, 2, 3, 224, 224))

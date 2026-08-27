import pytest
import torch

from openpi.models_pytorch.wan22.model import Wan22VAEEncoder


class _FakeVAE(torch.nn.Module):
    def encode(self, video, scale):
        assert video.shape == (2, 3, 1, 224, 224)
        assert len(scale) == 2
        return torch.ones(2, 48, 1, 14, 14, device=video.device, dtype=video.dtype)


def test_resize_single_frame_latent_to_256_native_tokens():
    latents = torch.randn(2, 48, 1, 14, 14)
    tokens = Wan22VAEEncoder.resize_latents(latents)
    assert tokens.shape == (2, 256, 48)
    assert torch.isfinite(tokens).all()


def test_resize_rejects_non_wan22_latent_contract():
    with pytest.raises(RuntimeError, match="single-frame latent"):
        Wan22VAEEncoder.resize_latents(torch.randn(1, 16, 1, 14, 14))


def test_forward_reuses_one_final_feature_for_four_independent_projectors():
    encoder = Wan22VAEEncoder.__new__(Wan22VAEEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.vae = _FakeVAE()
    encoder.compute_dtype = torch.float32
    encoder.register_buffer("latent_mean", torch.zeros(1, 48, 1, 1, 1))
    encoder.register_buffer("latent_inv_std", torch.ones(1, 48, 1, 1, 1))

    outputs = encoder(torch.zeros(2, 3, 224, 224, requires_grad=True))

    assert len(outputs) == 4
    assert all(output.shape == (2, 256, 48) for output in outputs)
    assert all(output is outputs[0] for output in outputs)
    assert not outputs[0].requires_grad

from __future__ import annotations

import torch

from metrics import compute_feature_metrics
from sae import SAEConfig
from sae import TopKAuxKSAE


def test_sae_topk_and_decoder_norm() -> None:
    model = TopKAuxKSAE(SAEConfig(input_dim=8, num_features=16, topk=3, auxk=4))
    output = model(torch.randn(11, 8))
    assert output.feature_activations.shape == (11, 16)
    assert torch.all((output.feature_activations > 0).sum(dim=1) <= 3)
    torch.testing.assert_close(torch.linalg.vector_norm(model.decoder.weight, dim=0), torch.ones(16))
    assert torch.isfinite(output.total_loss)


def test_episode_metrics_match_known_patterns() -> None:
    activations = torch.tensor(
        [
            [0.0, 0.2],
            [0.2, 0.2],
            [0.0, 0.2],
            [0.3, 0.2],
            [0.0, 0.0],
            [0.4, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    episode_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    metrics = compute_feature_metrics(activations, episode_ids)
    torch.testing.assert_close(metrics.episode_coverage, torch.tensor([1.0, 0.5]))
    torch.testing.assert_close(metrics.mean_onset_count, torch.tensor([1.5, 1.0]))
    torch.testing.assert_close(metrics.mean_activation_magnitude, torch.tensor([0.35, 0.2]))
    torch.testing.assert_close(metrics.relative_run_length, torch.tensor([0.25, 1.0]))


def test_episode_ids_must_be_contiguous() -> None:
    try:
        compute_feature_metrics(torch.ones(3, 2), torch.tensor([0, 1, 0]))
    except ValueError as error:
        assert "contiguous" in str(error)
    else:
        raise AssertionError("Expected non-contiguous episodes to fail")

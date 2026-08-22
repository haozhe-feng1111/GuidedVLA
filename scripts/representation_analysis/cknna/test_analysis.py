from __future__ import annotations

import torch
import torch.nn.functional as F

import build_manifest
import compute_cknna


def official_hsic_unbiased(kernel_a: torch.Tensor, kernel_b: torch.Tensor) -> torch.Tensor:
    m = kernel_a.shape[0]
    a = kernel_a.clone().fill_diagonal_(0)
    b = kernel_b.clone().fill_diagonal_(0)
    value = (
        torch.sum(a * b.T)
        + torch.sum(a) * torch.sum(b) / ((m - 1) * (m - 2))
        - 2 * torch.sum(torch.mm(a, b)) / (m - 2)
    )
    return value / (m * (m - 3))


def official_cknna(features_a: torch.Tensor, features_b: torch.Tensor, topk: int) -> torch.Tensor:
    n = features_a.shape[0]
    kernel_a = features_a @ features_a.T
    kernel_b = features_b @ features_b.T

    def similarity(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left_hat = left.clone().fill_diagonal_(float("-inf"))
        right_hat = right.clone().fill_diagonal_(float("-inf"))
        left_indices = torch.topk(left_hat, topk, dim=1).indices
        right_indices = torch.topk(right_hat, topk, dim=1).indices
        left_mask = torch.zeros(n, n).scatter_(1, left_indices, 1)
        right_mask = torch.zeros(n, n).scatter_(1, right_indices, 1)
        mask = left_mask * right_mask
        return official_hsic_unbiased(mask * left, mask * right)

    sim_ab = similarity(kernel_a, kernel_b)
    sim_aa = similarity(kernel_a, kernel_a)
    sim_bb = similarity(kernel_b, kernel_b)
    return sim_ab / (torch.sqrt(sim_aa * sim_bb) + 1e-6)


def test_vectorized_cknna_matches_official() -> None:
    generator = torch.Generator().manual_seed(123)
    features_a = F.normalize(torch.randn(40, 31, generator=generator), dim=-1)
    features_b = F.normalize(torch.randn(40, 17, generator=generator), dim=-1)
    expected = official_cknna(features_a, features_b, topk=10)
    kernel_a = (features_a @ features_a.T).unsqueeze(0)
    kernel_b = features_b @ features_b.T
    actual = compute_cknna.cknna_from_kernels(kernel_a, kernel_b, topk=10)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_vectorized_cknna_batch_shape() -> None:
    generator = torch.Generator().manual_seed(456)
    features_a = F.normalize(torch.randn(7, 30, 19, generator=generator), dim=-1)
    features_b = F.normalize(torch.randn(30, 13, generator=generator), dim=-1)
    kernel_a = torch.bmm(features_a, features_a.transpose(1, 2))
    kernel_b = features_b @ features_b.T
    scores = compute_cknna.cknna_from_kernels(kernel_a, kernel_b, topk=5)
    assert scores.shape == (7,)
    assert torch.isfinite(scores).all()


def test_most_specific_suite_match_resolves_short_goal() -> None:
    classification = {
        "libero_goal": [{"name": "turn_on_the_stove_table_1"}],
        "libero_10": [{"name": "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_table_1"}],
        "libero_object": [],
        "libero_spatial": [],
    }
    assert build_manifest.assign_suite("turn on the stove", classification) == ("libero_goal", 2)

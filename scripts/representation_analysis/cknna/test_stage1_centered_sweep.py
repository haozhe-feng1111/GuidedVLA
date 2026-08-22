from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import compute_vision_cknna as original
import compute_vision_stage1_centered_sweep as sweep


def test_siglip_feature_stack_allows_projector_dimension_change() -> None:
    data = {
        "siglip_block_features": torch.randn(27, 5, 11),
        "siglip_projector_features": torch.randn(5, 17),
    }
    features = sweep.feature_stack(data, "siglip")
    assert len(features) == 28
    assert features[0].shape == (5, 11)
    assert features[-1].shape == (5, 17)


def test_grid_matches_nested_scalar_cknna() -> None:
    generator = torch.Generator().manual_seed(271828)
    stream = F.normalize(torch.randn(5, 40, 17, generator=generator), dim=-1)
    reference = F.normalize(torch.randn(3, 40, 13, generator=generator), dim=-1)
    stream_kernel = torch.stack([value @ value.T for value in stream])
    reference_kernel = torch.stack([value @ value.T for value in reference])
    episode_ids = torch.arange(40) // 2
    allowed = original.allowed_mask(episode_ids, 40, torch.device("cpu"))
    actual = sweep.cknna_grid(stream_kernel, reference_kernel, allowed)
    expected = torch.empty(3, 5)
    for reference_layer in range(3):
        for stream_layer in range(5):
            expected[reference_layer, stream_layer] = original.cknna(
                stream_kernel[stream_layer], reference_kernel[reference_layer], sweep.TOPK, allowed
            )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_stage1_centered_max_t_intervals() -> None:
    rows = []
    rng = np.random.default_rng(11)
    offsets = (("stage1", 0.0), ("nohead", 0.01), ("da3", 0.02), ("dinov2", 0.03))
    for policy, offset in offsets:
        for reference in sweep.REFERENCES:
            for stream, count in sweep.STREAM_LAYERS.items():
                for task in range(40):
                    for reference_layer in range(12):
                        for policy_layer in range(count):
                            rows.append({
                                "policy": policy, "reference": reference, "stream": stream,
                                "task_index": task, "suite": "synthetic",
                                "reference_layer": reference_layer, "policy_layer": policy_layer,
                                "score": float(offset + rng.normal(scale=0.02)),
                            })
    indices = np.random.default_rng(sweep.BOOTSTRAP_SEED).integers(0, 40, size=(200, 40))
    paired, families = sweep.paired_inference(rows, indices)
    assert len(families) == 12
    assert {row["comparison"] for row in paired} == {
        "nohead_minus_stage1", "da3_minus_stage1", "dinov2_minus_stage1"
    }
    for row in paired:
        point_width = row["pointwise_ci_high"] - row["pointwise_ci_low"]
        simultaneous_width = row["simultaneous_ci_high"] - row["simultaneous_ci_low"]
        assert simultaneous_width >= point_width * 0.95
        if row["simultaneous_significant"]:
            assert row["pointwise_significant"]

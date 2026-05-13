import pytest
import torch

from openpi.models_pytorch.attention.attn_paths import attn_path_guided
from openpi.models_pytorch.attention.attn_paths import run_sdpa


def test_attn_path_guided_treats_missing_depth_kv_as_standard_heads():
    batch_size = 2
    num_heads = 4
    query_length = 3
    kv_length = 5
    head_dim = 8

    query = torch.randn(batch_size, num_heads, query_length, head_dim)
    key = torch.randn(batch_size, num_heads, kv_length, head_dim)
    value = torch.randn(batch_size, num_heads, kv_length, head_dim)

    output, attention_probs = attn_path_guided(
        query,
        key,
        value,
        attention_mask=None,
        scaling=1.0,
        dropout_p=0.0,
        distill_heads=2,
        training=True,
        depth_token_k=None,
        depth_token_v=None,
        depth_head_indices=(1, 3),
    )

    assert output.shape == (batch_size, query_length, num_heads, head_dim)
    assert attention_probs is not None
    assert attention_probs.shape == (batch_size, 2, query_length, kv_length)


def test_attn_path_guided_rejects_half_present_depth_kv():
    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 2, 3, 4)
    value = torch.randn(1, 2, 3, 4)
    depth_k = torch.randn(1, 2, 2, 4)

    with pytest.raises(ValueError, match="depth_token_k and depth_token_v"):
        attn_path_guided(
            query,
            key,
            value,
            attention_mask=None,
            scaling=1.0,
            dropout_p=0.0,
            distill_heads=0,
            training=True,
            depth_token_k=depth_k,
            depth_token_v=None,
            depth_head_indices=(0,),
        )


def test_attn_path_guided_keeps_object_attention_buffer_shape_when_depth_heads_are_exported():
    batch_size = 2
    num_heads = 5
    query_length = 3
    kv_length = 7
    depth_length = 2
    head_dim = 4

    query = torch.randn(batch_size, num_heads, query_length, head_dim)
    key = torch.randn(batch_size, num_heads, kv_length, head_dim)
    value = torch.randn(batch_size, num_heads, kv_length, head_dim)
    depth_k = torch.randn(batch_size, num_heads, depth_length, head_dim)
    depth_v = torch.randn(batch_size, num_heads, depth_length, head_dim)

    output, attention_probs = attn_path_guided(
        query,
        key,
        value,
        attention_mask=None,
        scaling=1.0,
        dropout_p=0.0,
        distill_heads=4,
        training=True,
        depth_token_k=depth_k,
        depth_token_v=depth_v,
        depth_head_indices=(2,),
    )

    assert output.shape == (batch_size, query_length, num_heads, head_dim)
    assert attention_probs is not None
    assert attention_probs.shape == (batch_size, 4, query_length, kv_length)
    assert torch.count_nonzero(attention_probs[:, 2]).item() == 0


def _reference_contiguous_distill_probs(query, key, value, scaling, distill_heads):
    """Reference softmax-based attention probs for the leading ``distill_heads`` heads."""
    q_d = query[:, :distill_heads]
    k_d = key[:, :distill_heads]
    v_d = value[:, :distill_heads]
    scores = torch.matmul(q_d, k_d.transpose(-2, -1)) * scaling
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
    return probs, torch.matmul(probs.to(v_d.dtype), v_d)


def test_attn_path_guided_proposal_f_direct_probs_assignment_is_numerically_exact():
    torch.manual_seed(0)
    batch_size, num_heads, query_length, kv_length, head_dim = 2, 6, 4, 7, 8
    distill_heads = 4  # contiguous leading [0, 4), no overlap with depth heads below

    query = torch.randn(batch_size, num_heads, query_length, head_dim)
    key = torch.randn(batch_size, num_heads, kv_length, head_dim)
    value = torch.randn(batch_size, num_heads, kv_length, head_dim)

    scaling = head_dim**-0.5

    output, attention_probs = attn_path_guided(
        query,
        key,
        value,
        attention_mask=None,
        scaling=scaling,
        dropout_p=0.0,
        distill_heads=distill_heads,
        training=False,
        depth_head_indices=(5,),  # depth head is outside [0, 4)
        depth_token_k=torch.randn(batch_size, num_heads, 2, head_dim),
        depth_token_v=torch.randn(batch_size, num_heads, 2, head_dim),
    )

    assert attention_probs is not None
    assert attention_probs.shape == (batch_size, distill_heads, query_length, kv_length)

    ref_probs, ref_attn = _reference_contiguous_distill_probs(query, key, value, scaling, distill_heads)
    torch.testing.assert_close(attention_probs, ref_probs, rtol=1e-5, atol=1e-6)

    output_heads_first = output.transpose(1, 2)
    torch.testing.assert_close(output_heads_first[:, :distill_heads], ref_attn, rtol=1e-5, atol=1e-6)


def test_run_sdpa_casts_additive_mask_to_query_dtype():
    query = torch.randn(1, 2, 3, 4, dtype=torch.float32)
    key = torch.randn(1, 2, 5, 4, dtype=torch.float32)
    value = torch.randn(1, 2, 5, 4, dtype=torch.float32)
    attention_mask = torch.zeros(1, 1, 3, 5, dtype=torch.bfloat16)

    output = run_sdpa(
        query,
        key,
        value,
        attention_mask=attention_mask,
        scaling=1.0,
        dropout_p=0.0,
    )

    assert output.shape == query.shape
    assert output.dtype == query.dtype

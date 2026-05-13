from types import SimpleNamespace

import pytest
import torch
from torch import nn

from openpi.models import pi0_config
from openpi.models_pytorch.control_attention import ControlAwareAttention
from openpi.models_pytorch.control_attention import get_trainable_control_params
from openpi.models_pytorch.gemma_pytorch import SupervisedHeadStates
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


def _make_attn_stub(dtype: torch.dtype):
    return SimpleNamespace(q_proj=SimpleNamespace(weight=torch.empty(1, dtype=dtype)))


def _make_pi0_stub(*, prefix_dtype: torch.dtype, expert_dtype: torch.dtype):
    stub = object.__new__(PI0Pytorch)
    stub.paligemma_with_expert = SimpleNamespace(
        paligemma=SimpleNamespace(
            language_model=SimpleNamespace(layers=[SimpleNamespace(self_attn=_make_attn_stub(prefix_dtype))])
        ),
        gemma_expert=SimpleNamespace(
            model=SimpleNamespace(layers=[SimpleNamespace(self_attn=_make_attn_stub(expert_dtype))])
        ),
    )
    return stub


def _align_prefix(model, tensor: torch.Tensor) -> torch.Tensor:
    return model._align_prefix_embeddings_dtype(tensor)[0]  # noqa: SLF001


def _align_expert(model, tensor: torch.Tensor) -> torch.Tensor:
    return model._align_expert_embeddings_dtype(tensor)[0]  # noqa: SLF001


def test_prefix_and_suffix_embeddings_align_to_their_own_backbones():
    model = _make_pi0_stub(prefix_dtype=torch.bfloat16, expert_dtype=torch.float32)
    prefix = torch.randn(1, 8, 4, dtype=torch.float32)
    suffix = torch.randn(1, 5, 4, dtype=torch.bfloat16)

    prefix_aligned = _align_prefix(model, prefix)
    suffix_aligned = _align_expert(model, suffix)

    assert prefix_aligned.dtype == torch.bfloat16
    assert suffix_aligned.dtype == torch.float32


def test_alignment_helpers_are_noops_when_input_dtype_is_already_correct():
    model = _make_pi0_stub(prefix_dtype=torch.float32, expert_dtype=torch.float32)
    prefix = torch.randn(1, 3, 4, dtype=torch.float32)
    suffix = torch.randn(1, 2, 4, dtype=torch.float32)

    prefix_aligned = _align_prefix(model, prefix)
    suffix_aligned = _align_expert(model, suffix)

    assert prefix_aligned.dtype == prefix.dtype
    assert suffix_aligned.dtype == suffix.dtype


class _TinyAttention(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        projection_dim = config.num_attention_heads * config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, projection_dim)
        self.k_proj = nn.Linear(config.hidden_size, projection_dim)
        self.v_proj = nn.Linear(config.hidden_size, projection_dim)
        self.o_proj = nn.Linear(projection_dim, config.hidden_size)


def test_control_attention_freeze_origin_keeps_control_branch_trainable():
    config = SimpleNamespace(hidden_size=8, num_attention_heads=2, head_dim=4)
    original_attn = _TinyAttention(config)

    attn = ControlAwareAttention(
        original_attn,
        hidden_size=config.hidden_size,
        num_control_heads=1,
        freeze_origin=True,
        use_headwise_gate=False,
    )

    assert all(not param.requires_grad for param in attn.origin.parameters())
    assert all(param.requires_grad for param in attn.object_branch.parameters())
    assert attn.zero_conv.weight.requires_grad
    assert attn.zero_conv.bias.requires_grad


def test_control_attention_origin_is_trainable_by_default():
    config = SimpleNamespace(hidden_size=8, num_attention_heads=2, head_dim=4)
    original_attn = _TinyAttention(config)

    attn = ControlAwareAttention(
        original_attn,
        hidden_size=config.hidden_size,
        num_control_heads=1,
        use_headwise_gate=False,
    )

    assert all(param.requires_grad for param in attn.origin.parameters())


def test_control_attention_copy_weights_without_gate_uses_origin_head_count():
    config = SimpleNamespace(hidden_size=8, num_attention_heads=2, head_dim=4)
    original_attn = _TinyAttention(config)

    attn = ControlAwareAttention(
        original_attn,
        hidden_size=config.hidden_size,
        num_control_heads=1,
        copy_weights=True,
        use_headwise_gate=False,
    )

    assert attn.num_control_heads == config.num_attention_heads
    assert attn.q_expand_linear is None
    assert not attn.has_q_expansion
    assert not attn.use_headwise_gate

    hidden_states = torch.randn(2, 3, config.hidden_size)
    _, (q_branch, _, _), q_branch_gate = attn.compute_dual_path_qkv(hidden_states)
    assert q_branch.shape == (2, config.num_attention_heads, 3, config.head_dim)
    assert q_branch_gate is None


def test_control_attention_copy_weights_with_gate_preserves_query_projection():
    config = SimpleNamespace(hidden_size=8, num_attention_heads=2, head_dim=4)
    original_attn = _TinyAttention(config)
    query_size = config.num_attention_heads * config.head_dim

    attn = ControlAwareAttention(
        original_attn,
        hidden_size=config.hidden_size,
        num_control_heads=1,
        copy_weights=True,
        use_headwise_gate=True,
    )

    assert attn.num_control_heads == config.num_attention_heads
    assert attn.q_expand_linear is None
    assert attn.use_headwise_gate
    assert attn.object_branch.q_proj.out_features == query_size + config.num_attention_heads
    torch.testing.assert_close(attn.object_branch.q_proj.weight[:query_size], original_attn.q_proj.weight)
    torch.testing.assert_close(attn.object_branch.q_proj.bias[:query_size], original_attn.q_proj.bias)
    assert torch.count_nonzero(attn.object_branch.q_proj.weight[query_size:]) > 0
    assert torch.count_nonzero(attn.object_branch.q_proj.bias[query_size:]) == 0

    hidden_states = torch.randn(2, 3, config.hidden_size)
    _, (q_branch, _, _), q_branch_gate = attn.compute_dual_path_qkv(hidden_states)
    assert q_branch.shape == (2, config.num_attention_heads, 3, config.head_dim)
    assert q_branch_gate.shape == (2, config.num_attention_heads, 3, 1)


def test_control_attention_reduced_head_expansion_is_trainable_control_param():
    config = SimpleNamespace(hidden_size=8, num_attention_heads=2, head_dim=4)
    original_attn = _TinyAttention(config)
    attn = ControlAwareAttention(
        original_attn,
        hidden_size=config.hidden_size,
        num_control_heads=1,
        use_headwise_gate=False,
    )
    model = nn.Module()
    model.attn = attn

    control_params = set(get_trainable_control_params(model))

    assert attn.q_expand_linear is not None
    assert attn.q_expand_linear.weight in control_params
    assert attn.q_expand_linear.bias in control_params


def _make_attention_probs(batch, num_heads, t_q, num_keys):
    """Row-stochastic attention probabilities of shape [B, H, T_q, K]."""
    raw = torch.randn(batch, num_heads, t_q, num_keys)
    return torch.softmax(raw, dim=-1)


def _reference_hard_object_loss(
    attention_probs: torch.Tensor,
    action_query_start_index: int,
    object_head_indices: tuple[int, ...],
    target_maps: torch.Tensor,
    target_masks: torch.Tensor,
    head_aggregation: str = "mean_heads",
) -> torch.Tensor:
    """Object-mass loss oracle, independent of the production kernel."""
    views, patches = target_maps.shape[-2:]
    num_image_keys = views * patches
    batch, _, _, num_total_keys = attention_probs.shape

    per_pixel_weight = target_maps * target_masks.to(target_maps.dtype).unsqueeze(-1)
    weights = torch.zeros(batch, num_total_keys, dtype=target_maps.dtype, device=target_maps.device)
    weights[:, :num_image_keys] = per_pixel_weight.flatten(1)

    if head_aggregation == "per_head":
        per_head = []
        for head in object_head_indices:
            attn = attention_probs[:, head, action_query_start_index:, :]
            object_mass = (attn * weights[:, None, :]).sum(dim=-1)
            loss = -torch.log(object_mass.clamp_min(1e-6))
            per_head.append(loss)
        loss_per_entry = torch.stack(per_head, dim=1)
    elif head_aggregation == "mean_heads":
        attn = attention_probs[:, list(object_head_indices), action_query_start_index:, :].mean(dim=1)
        object_mass = (attn * weights[:, None, :]).sum(dim=-1)
        loss_per_entry = -torch.log(object_mass.clamp_min(1e-6))
    else:
        raise ValueError(head_aggregation)

    valid_batch = weights.amax(dim=-1) > 0
    valid_mask = valid_batch.view(-1, *([1] * (loss_per_entry.ndim - 1))).expand_as(loss_per_entry)
    num_valid = valid_mask.sum().clamp_min(1).to(loss_per_entry.dtype)
    return (loss_per_entry * valid_mask.to(loss_per_entry.dtype)).sum() / num_valid


def test_hard_object_loss_matches_independent_reference():
    torch.manual_seed(0)
    model = object.__new__(PI0Pytorch)

    batch, heads, t_q, num_keys = 2, 4, 20, 32
    views, patches = 3, 4
    assert views * patches <= num_keys
    action_query_start = t_q - 5

    attn = _make_attention_probs(batch, heads, t_q, num_keys)
    object_head_indices = (0, 2)
    target_maps = (torch.rand(batch, views, patches) > 0.5).to(torch.float32)
    target_masks = torch.tensor([[True, True, False], [True, False, False]])

    got = model.compute_object_mass_loss(attn, action_query_start, object_head_indices, target_maps, target_masks)
    expected = _reference_hard_object_loss(attn, action_query_start, object_head_indices, target_maps, target_masks)
    assert got.shape == torch.Size([])
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


def test_hard_object_loss_mean_heads_averages_attention_before_supervision():
    model = object.__new__(PI0Pytorch)

    attn = torch.tensor(
        [
            [
                [[1.0, 0.0]],
                [[0.0, 1.0]],
            ]
        ],
        dtype=torch.float32,
    )
    target_maps = torch.ones(1, 1, 1)
    target_masks = torch.ones(1, 1, dtype=torch.bool)
    object_head_indices = (0, 1)

    got = model.compute_object_mass_loss(
        attn,
        0,
        object_head_indices,
        target_maps,
        target_masks,
        head_aggregation="mean_heads",
    )
    expected = _reference_hard_object_loss(
        attn,
        0,
        object_head_indices,
        target_maps,
        target_masks,
        head_aggregation="mean_heads",
    )
    per_head = model.compute_object_mass_loss(
        attn,
        0,
        object_head_indices,
        target_maps,
        target_masks,
        head_aggregation="per_head",
    )

    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)
    assert got < per_head


def test_compute_object_loss_uses_configured_head_aggregation():
    model = object.__new__(PI0Pytorch)
    model.config = SimpleNamespace(object_loss_head_aggregation="mean_heads")
    model.object_use_control = True

    attn = torch.tensor(
        [
            [
                [[1.0, 0.0]],
                [[0.0, 1.0]],
            ]
        ],
        dtype=torch.float32,
    )
    target_maps = torch.ones(1, 1, 1)
    target_masks = torch.ones(1, 1, dtype=torch.bool)
    object_head_indices = (0, 1)

    got = model.compute_object_loss(
        all_supervised_states=[(0, SupervisedHeadStates(attention_probs=attn))],
        action_query_start_index=0,
        object_head_indices=object_head_indices,
        object_targets={"object_maps": target_maps, "object_masks": target_masks},
    )
    expected = _reference_hard_object_loss(
        attn,
        0,
        object_head_indices,
        target_maps,
        target_masks,
        head_aggregation="mean_heads",
    )

    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


def test_hard_object_loss_gradient_reaches_object_heads_and_action_queries_only():
    torch.manual_seed(1)
    model = object.__new__(PI0Pytorch)

    batch, heads, t_q, num_keys = 1, 4, 12, 20
    views, patches = 3, 4
    action_query_start = t_q - 4

    attn = _make_attention_probs(batch, heads, t_q, num_keys)
    attn.requires_grad_(requires_grad=True)
    object_head_indices = (1, 3)
    target_maps = torch.ones(batch, views, patches)
    target_masks = torch.ones(batch, views, dtype=torch.bool)

    loss = model.compute_object_mass_loss(attn, action_query_start, object_head_indices, target_maps, target_masks)
    loss.backward()
    grad = attn.grad

    non_object = [h for h in range(heads) if h not in object_head_indices]
    assert grad[:, non_object].abs().max().item() == 0.0
    assert grad[:, list(object_head_indices), action_query_start:].abs().sum().item() > 0
    assert grad[:, :, :action_query_start].abs().max().item() == 0.0


def test_hard_object_loss_shape_and_dtype_are_scalar_fp32():
    torch.manual_seed(2)
    model = object.__new__(PI0Pytorch)

    attn = _make_attention_probs(1, 2, 6, 8)
    target_maps = torch.ones(1, 1, 4)
    target_masks = torch.ones(1, 1, dtype=torch.bool)
    got = model.compute_object_mass_loss(attn, 2, (0,), target_maps, target_masks)

    assert got.shape == torch.Size([])
    assert got.dtype == torch.float32


def test_hard_object_loss_init_sanity_is_in_expected_range():
    """Under uniform attention, L_obj = -log(object_fraction).

    Picking object_fraction = 2/8 = 0.25: expected ≈ -log(0.25) = 1.386.
    This guards against silent sign flips or scale bugs in the formula.
    """
    model = object.__new__(PI0Pytorch)

    batch, heads, t_q, num_keys = 4, 2, 8, 8
    views, patches = 1, 2

    attn = torch.full((batch, heads, t_q, num_keys), 1.0 / num_keys)
    target_maps = torch.ones(batch, views, patches)
    target_masks = torch.ones(batch, views, dtype=torch.bool)

    got = model.compute_object_mass_loss(attn, 0, (0,), target_maps, target_masks)
    assert 1.0 < got.item() < 2.5, f"init sanity out of range: got {got.item()}"


def test_pi0_config_defaults_to_mean_heads_object_loss_aggregation():
    config = pi0_config.Pi0Config()

    assert config.object_loss_head_aggregation == "mean_heads"


def test_pi0_config_rejects_unknown_object_loss_aggregation():
    with pytest.raises(ValueError, match="object_loss_head_aggregation"):
        pi0_config.Pi0Config(object_loss_head_aggregation="unknown")

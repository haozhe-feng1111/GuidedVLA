from types import SimpleNamespace

import pytest
import torch
from torch import nn

from openpi.models import pi0_config
from openpi.models_pytorch import pi0_pytorch
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


class _TinyPaliGemmaWithExpert(nn.Module):
    def __init__(self, *_args, **_kwargs):
        super().__init__()
        self.paligemma = SimpleNamespace(language_model=SimpleNamespace(config=SimpleNamespace()))
        self.gemma_expert = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace()))


def test_depth_inference_off_does_not_construct_or_execute_depth_modules(monkeypatch):
    tiny_config = SimpleNamespace(width=8, head_dim=4)
    monkeypatch.setattr(pi0_pytorch._gemma, "get_config", lambda _variant: tiny_config)
    monkeypatch.setattr(pi0_pytorch, "PaliGemmaWithExpertModel", _TinyPaliGemmaWithExpert)
    monkeypatch.setattr(pi0_pytorch.torch, "compile", lambda function, **_kwargs: function)

    def fail_if_constructed(*_args, **_kwargs):
        pytest.fail("depth module must not be constructed when depth inference is disabled")

    monkeypatch.setattr(pi0_pytorch, "DepthEncoder", fail_if_constructed)
    monkeypatch.setattr(pi0_pytorch, "DepthTokenKVProjector", fail_if_constructed)

    from transformers.models.siglip import check

    monkeypatch.setattr(check, "check_whether_transformers_replace_is_installed_correctly", lambda: True)

    config = pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        use_depth=True,
        disable_depth_at_inference=True,
    )
    model = PI0Pytorch(config)

    assert not model.use_depth
    assert not hasattr(model, "depth_module")
    assert not hasattr(model, "depth_token_proj")
    assert model.compute_depth_key_values([torch.zeros(1)]) is None


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


def test_depth_guided_layers_default_to_supervision_layers():
    config = pi0_config.Pi0Config(guided_layer_indices=[12, 9, 11, 10])

    assert config.guided_layer_indices == [9, 10, 11, 12]
    assert config.depth_guided_layer_indices == [9, 10, 11, 12]


def test_depth_guided_layers_can_be_decoupled_from_supervision_layers():
    config = pi0_config.Pi0Config(
        guided_layer_indices=[9, 10, 11, 12],
        depth_guided_layer_indices=[8, 5, 7, 6],
    )

    assert config.guided_layer_indices == [9, 10, 11, 12]
    assert config.depth_guided_layer_indices == [5, 6, 7, 8]


def test_dinov2_shallow_guidance_config_only_moves_external_kv_layers():
    from openpi.training import config as training_config

    config = training_config.get_config("pi0_libero_object_dinov2_base_skill_shallow_guidance").model

    assert config.guided_layer_indices == [9, 10, 11, 12]
    assert config.depth_guided_layer_indices == [5, 6, 7, 8]
    assert config.object_head_indices == [0, 1]
    assert config.depth_head_indices == [4, 5]
    assert config.skill_head_indices == [6, 7]


def test_wan22_training_config_uses_native_features_and_deep_layers():
    from openpi.training import config as training_config

    config = training_config.get_config("pi0_libero_object_wan22_vae_skill").model

    assert config.use_wan22_encoder
    assert config.wan22_dtype == "bfloat16"
    assert config.depth_guided_layer_indices == [9, 10, 11, 12]
    assert config.wan22_head_indices == [4, 5]


def test_raft_training_config_uses_single_final_fnet_feature_and_deep_layers():
    from openpi.training import config as training_config

    config = training_config.get_config("pi0_libero_object_raft_large_fnet_skill").model

    assert config.use_raft_encoder
    assert config.depth_guided_layer_indices == [9, 10, 11, 12]
    assert config.raft_head_indices == [4, 5]
    assert config.object_head_indices == [0, 1]
    assert config.skill_head_indices == [6, 7]


def test_raft_config_rejects_other_external_encoders_and_head_overlap():
    with pytest.raises(ValueError, match="mutually exclusive"):
        pi0_config.Pi0Config(
            use_depth=True,
            use_raft_encoder=True,
            raft_checkpoint_path="raft.pth",
        )

    with pytest.raises(ValueError, match="object_head_indices and raft_head_indices overlap"):
        pi0_config.Pi0Config(
            control_attention_enabled=True,
            use_raft_encoder=True,
            raft_checkpoint_path="raft.pth",
            raft_head_indices=[0],
        )


def test_joint_backbone_routes_external_kv_to_depth_guided_layers():
    captured = {}

    def fake_backbone(**kwargs):
        captured.update(kwargs)
        return (None, kwargs["inputs_embeds"][1]), None, []

    model = object.__new__(PI0Pytorch)
    object.__setattr__(model, "paligemma_with_expert", fake_backbone)
    object.__setattr__(model, "guided_layer_indices", (9, 10, 11, 12))
    object.__setattr__(model, "depth_guided_layer_indices", (5, 6, 7, 8))
    prefix_embs = torch.zeros(1, 2, 4)
    suffix_embs = torch.zeros(1, 3, 4)
    depth_kv = (object(),) * 4

    suffix_out, supervised_states = model.run_joint_backbone(
        prefix_embs=prefix_embs,
        suffix_embs=suffix_embs,
        attention_mask_4d=torch.zeros(1, 1, 5, 5),
        position_ids=torch.arange(5).unsqueeze(0),
        adarms_cond=None,
        depth_kv=depth_kv,
        head_supervision_config=None,
    )

    assert model.get_guided_layers() == [9, 10, 11, 12]
    assert captured["guided_layer_indices"] == (5, 6, 7, 8)
    assert captured["depth_kv"] is depth_kv
    assert suffix_out is suffix_embs
    assert supervised_states == []

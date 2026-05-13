from types import SimpleNamespace

import torch

from openpi.models_pytorch.gemma_pytorch import HeadSupervisionConfig
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel


def _make_model() -> PaliGemmaWithExpertModel:
    config = SimpleNamespace(
        width=64,
        depth=4,
        mlp_dim=128,
        num_heads=8,
        num_kv_heads=1,
        head_dim=16,
    )
    return PaliGemmaWithExpertModel(config, config, precision="float32")


def _run_joint_forward(
    model: PaliGemmaWithExpertModel,
    *,
    head_supervision_config: HeadSupervisionConfig | None = None,
):
    batch_size = 1
    prefix_len = 2
    suffix_len = 3
    hidden_size = model.paligemma.config.text_config.hidden_size
    total_len = prefix_len + suffix_len

    prefix = torch.randn(batch_size, prefix_len, hidden_size, dtype=torch.float32)
    suffix = torch.randn(batch_size, suffix_len, hidden_size, dtype=torch.float32)
    attention_mask = torch.zeros(batch_size, 1, total_len, total_len, dtype=torch.float32)
    position_ids = torch.arange(total_len, dtype=torch.long).unsqueeze(0)

    return model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix, suffix],
        use_cache=False,
        adarms_cond=[None, None],
        head_supervision_config=head_supervision_config,
    )


def test_gradient_checkpointing_flag_survives_eval_forward():
    model = _make_model()
    model.set_gradient_checkpointing(enabled=True)
    model.eval()

    _run_joint_forward(model)

    assert model.use_gradient_checkpointing is True


def test_gradient_checkpointing_wraps_layers_only_when_training_enabled(monkeypatch):
    model = _make_model()
    checkpoint_calls = 0
    original_checkpoint = torch.utils.checkpoint.checkpoint

    def counting_checkpoint(func, *args, **kwargs):
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        return original_checkpoint(func, *args, **kwargs)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", counting_checkpoint)

    model.train()
    model.set_gradient_checkpointing(enabled=True)
    _run_joint_forward(model)
    assert checkpoint_calls > 0

    checkpoint_calls = 0
    model.set_gradient_checkpointing(enabled=False)
    _run_joint_forward(model)
    assert checkpoint_calls == 0


def test_supervised_states_export_attention_outputs_for_losses():
    model = _make_model()
    model.train()

    (_, _), _, all_supervised_states = _run_joint_forward(
        model,
        head_supervision_config=HeadSupervisionConfig(
            supervised_layers=(0,),
            num_export_heads=2,
            num_probs_export_heads=2,
            include_skill_states=True,
            skill_head_indices=(0,),
        ),
    )

    assert len(all_supervised_states) == 1
    _, supervised_states = all_supervised_states[0]
    assert supervised_states.attention_probs is not None
    assert supervised_states.skill_features is not None

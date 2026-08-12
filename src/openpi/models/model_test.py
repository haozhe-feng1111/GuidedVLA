from flax import nnx
import jax
import pytest
import torch

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import pi0_fast
from openpi.shared import download
from openpi.shared import nnx_utils


def test_pi0_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_lora_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_fast_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)


def test_pi0_fast_lora_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)

    lora_filter = nnx_utils.PathRegex(".*lora.*")
    model_state = nnx.state(model)

    lora_state_elems = list(model_state.filter(lora_filter))
    assert len(lora_state_elems) > 0


def test_observation_from_dict_accepts_uint8_torch_images_without_normalizing():
    observation = _model.Observation.from_dict(
        {
            "image": {
                "base_0_rgb": torch.randint(0, 256, (2, 224, 224, 3), dtype=torch.uint8),
            },
            "image_mask": {
                "base_0_rgb": torch.ones(2, dtype=torch.bool),
            },
            "state": torch.randn(2, 8, dtype=torch.float32),
        },
        normalize_torch_images=False,
    )

    assert observation.images["base_0_rgb"].dtype == torch.uint8


class _TinyTokenMerging(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.merge = torch.nn.Linear(3, 2)
        self.norm = torch.nn.LayerNorm(3)


class _TinyDepthHost(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.depth_module = torch.nn.Module()
        self.depth_module.token_merging_model = torch.compile(_TinyTokenMerging(), backend="eager")


def _canonical_token_merging_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    compiled_state_dict = model.state_dict()
    return {
        key.replace(".token_merging_model._orig_mod.", ".token_merging_model."): torch.full_like(value, index + 1)
        for index, (key, value) in enumerate(compiled_state_dict.items())
    }


@pytest.mark.parametrize("source_is_compiled", [False, True])
def test_compiled_depth_adapter_loads_canonical_checkpoint_keys(source_is_compiled: bool):
    model = _TinyDepthHost()
    checkpoint_state_dict = _canonical_token_merging_state_dict(model)
    if source_is_compiled:
        checkpoint_state_dict = {
            key.replace(".token_merging_model.", ".token_merging_model._orig_mod."): value
            for key, value in checkpoint_state_dict.items()
        }
    checkpoint_state_dict = _model.normalize_pytorch_state_dict_for_loading(
        checkpoint_state_dict,
        source_label="test checkpoint",
    )

    with _model.temporarily_unwrap_compiled_modules_for_state_dict(model) as model_to_load:
        missing_keys, unexpected_keys = model_to_load.load_state_dict(checkpoint_state_dict, strict=False)

    _model._validate_depth_token_merging_weights_loaded(checkpoint_state_dict, missing_keys, unexpected_keys)
    assert missing_keys == []
    assert unexpected_keys == []
    assert hasattr(model.depth_module.token_merging_model, "_orig_mod")

    for key, expected in checkpoint_state_dict.items():
        compiled_key = key.replace(".token_merging_model.", ".token_merging_model._orig_mod.")
        torch.testing.assert_close(model.state_dict()[compiled_key], expected)


def test_depth_adapter_guard_rejects_silent_load_failure():
    model = _TinyDepthHost()
    checkpoint_state_dict = _canonical_token_merging_state_dict(model)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint_state_dict, strict=False)

    with pytest.raises(RuntimeError, match="depth token-merging weights"):
        _model._validate_depth_token_merging_weights_loaded(
            checkpoint_state_dict,
            missing_keys,
            unexpected_keys,
        )


@pytest.mark.manual
def test_model_restore():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    model = config.load(
        _model.restore_params(download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params"))
    )

    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = model.sample_actions(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)

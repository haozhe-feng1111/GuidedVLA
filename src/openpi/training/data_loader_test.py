import dataclasses

import jax
import numpy as np
import torch

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_pytorch_collate_fn_copies_numpy_leaves():
    first = np.arange(12, dtype=np.float32).reshape(3, 4)
    second = first + 100
    first.setflags(write=False)
    second.setflags(write=False)

    batch = _data_loader._pytorch_collate_fn([{"image": first}, {"image": second}])  # noqa: SLF001

    assert isinstance(batch["image"], torch.Tensor)
    np.testing.assert_allclose(batch["image"].numpy(), np.stack([first, second], axis=0))

    batch["image"][0, 0, 0] = -1
    assert first[0, 0] == 0


def test_pytorch_collate_fn_preserves_ragged_object_maps_for_later_packing():
    first = np.ones((16, 16), dtype=np.float32)
    second = np.ones((1, 16, 16), dtype=np.float32) * 2

    batch = _data_loader._pytorch_collate_fn(  # noqa: SLF001
        [
            {"attention_map": {"cam_high_attention_object": first}},
            {"attention_map": {"cam_high_attention_object": second}},
        ]
    )

    view_items = batch["attention_map"]["cam_high_attention_object"]
    assert isinstance(view_items, list)
    assert len(view_items) == 2
    np.testing.assert_allclose(view_items[0], first)
    np.testing.assert_allclose(view_items[1], second)

    loader = _data_loader.DataLoaderImpl(None, [], framework="pytorch")
    object_maps = loader._extract_object_maps(batch)  # noqa: SLF001
    packed_maps, packed_mask = loader._pack_maps_to_tensor(object_maps)  # noqa: SLF001

    assert packed_maps is not None
    assert packed_mask is not None
    assert packed_maps.shape == (2, _data_loader.MAX_VIEWS, _data_loader.NUM_PATCHES)
    assert packed_mask.shape == (2, _data_loader.MAX_VIEWS)
    assert bool(packed_mask[0, _data_loader.VIEW_NAME_TO_ID["base_0"]])
    assert bool(packed_mask[1, _data_loader.VIEW_NAME_TO_ID["base_0"]])


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_multi_datasets_config():
    # This config should be defined in your repo: pi0_base_aloha_robotwin_full_multi
    config = _config.get_config("pi0_base_aloha_robotwin_full_multi")
    # Use small batch for tests
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        skip_norm_stats=True,  # tests may not have norm stats available
        num_batches=2,
        shuffle=False,
    )

    # Ensure multi_datasets was present in the data config
    assert loader.data_config().multi_datasets is not None

    batches = list(loader)
    assert len(batches) == 2

    for item in batches:
        # Support both (obs, actions) and (obs, actions, attn_maps) returned tuples
        if isinstance(item, tuple) and len(item) == 3:
            _, actions, _ = item
        else:
            _, actions = item
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)

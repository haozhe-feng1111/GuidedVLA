from types import SimpleNamespace

import pytest
import torch

from openpi.training.data_loader import DataLoaderImpl


def _batch_with_object_maps() -> dict:
    return {
        "image": {"base_0": torch.zeros(2, 3, 224, 224)},
        "image_mask": {"base_0": torch.ones(2, dtype=torch.bool)},
        "state": torch.zeros(2, 32),
        "actions": torch.zeros(2, 50, 32),
        "attention_map": {
            "agentview_attention_object_mask": torch.ones(2, 16, 16),
        },
    }


def test_extract_object_maps_supports_robotwin_attention_object_columns():
    loader = DataLoaderImpl(None, [], framework="pytorch")
    batch = {
        "cam_high_attention_object": torch.ones(2, 16, 16),
        "cam_left_wrist_attention_object": torch.ones(2, 16, 16) * 2,
        "cam_right_wrist_attention_object": torch.ones(2, 16, 16) * 3,
    }

    result = loader._extract_object_maps(batch)  # noqa: SLF001

    assert set(result) == {"base_0", "left_wrist_0", "right_wrist_0"}
    torch.testing.assert_close(result["base_0"], batch["cam_high_attention_object"])
    torch.testing.assert_close(result["left_wrist_0"], batch["cam_left_wrist_attention_object"])
    torch.testing.assert_close(result["right_wrist_0"], batch["cam_right_wrist_attention_object"])


def test_extract_object_maps_supports_libero_attention_object_dict():
    loader = DataLoaderImpl(None, [], framework="pytorch")
    batch = {
        "attention_map": {
            "agentview_attention_object_mask": torch.ones(2, 16, 16),
            "wrist_attention_object_mask": torch.ones(2, 16, 16) * 2,
        }
    }

    result = loader._extract_object_maps(batch)  # noqa: SLF001

    assert set(result) == {"base_0", "left_wrist_0"}
    torch.testing.assert_close(result["base_0"], batch["attention_map"]["agentview_attention_object_mask"])
    torch.testing.assert_close(result["left_wrist_0"], batch["attention_map"]["wrist_attention_object_mask"])


def test_extract_object_maps_normalizes_model_attention_map_keys():
    loader = DataLoaderImpl(None, [], framework="pytorch")
    batch = {
        "attention_map": {
            "base_0_attn": torch.ones(2, 16, 16),
            "right_wrist_0_attn": torch.ones(2, 16, 16) * 3,
        }
    }

    result = loader._extract_object_maps(batch)  # noqa: SLF001

    assert set(result) == {"base_0", "right_wrist_0"}
    torch.testing.assert_close(result["base_0"], batch["attention_map"]["base_0_attn"])
    torch.testing.assert_close(result["right_wrist_0"], batch["attention_map"]["right_wrist_0_attn"])


def test_iterator_skips_object_targets_when_data_config_disables_object_loss():
    loader = DataLoaderImpl(SimpleNamespace(use_object_loss=False), [_batch_with_object_maps()], framework="pytorch")

    _, _, object_targets = next(iter(loader))

    assert object_targets is None


def test_iterator_packs_object_targets_when_data_config_enables_object_loss():
    loader = DataLoaderImpl(SimpleNamespace(use_object_loss=True), [_batch_with_object_maps()], framework="pytorch")

    _, _, object_targets = next(iter(loader))

    assert object_targets is not None
    assert object_targets["object_maps"].shape == (2, 3, 256)
    assert object_targets["object_masks"].shape == (2, 3)
    assert bool(object_targets["object_masks"][0, 0])


def test_iterator_errors_when_object_loss_enabled_without_object_maps():
    loader = DataLoaderImpl(
        SimpleNamespace(use_object_loss=True),
        [
            {
                "image": {"base_0": torch.zeros(2, 3, 224, 224)},
                "image_mask": {"base_0": torch.ones(2, dtype=torch.bool)},
                "state": torch.zeros(2, 32),
                "actions": torch.zeros(2, 50, 32),
            }
        ],
        framework="pytorch",
    )

    with pytest.raises(ValueError, match="contains no object-map supervision"):
        next(iter(loader))


def test_iterator_errors_when_object_maps_are_all_zero():
    batch = _batch_with_object_maps()
    batch["attention_map"]["agentview_attention_object_mask"] = torch.zeros(2, 16, 16)
    loader = DataLoaderImpl(SimpleNamespace(use_object_loss=True), [batch], framework="pytorch")

    with pytest.raises(ValueError, match="every valid packed map is all zero"):
        next(iter(loader))


def test_iterator_errors_when_object_maps_have_invalid_shape():
    batch = _batch_with_object_maps()
    batch["attention_map"]["agentview_attention_object_mask"] = torch.ones(2, 8, 8)
    loader = DataLoaderImpl(SimpleNamespace(use_object_loss=True), [batch], framework="pytorch")

    with pytest.raises(ValueError, match="none could be packed"):
        next(iter(loader))

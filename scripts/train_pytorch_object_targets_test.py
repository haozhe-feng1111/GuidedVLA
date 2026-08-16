from types import SimpleNamespace

import pytest
import torch

from scripts.train_pytorch import (
    _guided_training_config_enabled,
    _is_new_head_parameter,
    prepare_object_targets,
    split_missing_keys,
    validate_training_model_config,
)


def test_prepare_object_targets_returns_none_when_object_loss_disabled():
    assert prepare_object_targets(None, torch.device("cpu"), 2, use_object_loss=False) is None


def test_prepare_object_targets_errors_when_object_loss_enabled_without_targets():
    with pytest.raises(RuntimeError, match="did not provide object_targets"):
        prepare_object_targets(None, torch.device("cpu"), 2, use_object_loss=True)


def test_prepare_object_targets_errors_when_required_key_is_missing():
    object_targets = {
        "object_maps": torch.ones(2, 3, 256),
    }

    with pytest.raises(RuntimeError, match="missing 'object_masks'"):
        prepare_object_targets(object_targets, torch.device("cpu"), 2, use_object_loss=True)


def test_split_missing_keys_treats_sam2_adapters_as_expected():
    expected, unexpected = split_missing_keys(
        [
            "sam2_module.token_merging_model.conv.weight",
            "sam2_token_proj.key_proj.weight",
            "depth_module.encoder.weight",
            "skill_head.weight",
            "paligemma_with_expert.weight",
        ]
    )

    assert expected == [
        "sam2_module.token_merging_model.conv.weight",
        "sam2_token_proj.key_proj.weight",
        "depth_module.encoder.weight",
        "skill_head.weight",
    ]
    assert unexpected == ["paligemma_with_expert.weight"]


def test_sam2_projector_is_a_new_head_parameter():
    assert _is_new_head_parameter("sam2_token_proj.key_proj.weight")
    assert _is_new_head_parameter("sam2_module.token_merging_model.conv.weight")
    assert not _is_new_head_parameter("paligemma_with_expert.vision_tower.weight")


def test_depth_inference_off_config_is_rejected_for_training():
    with pytest.raises(ValueError, match="inference-only ablation"):
        validate_training_model_config(SimpleNamespace(disable_depth_at_inference=True))

    validate_training_model_config(SimpleNamespace(disable_depth_at_inference=False))
    validate_training_model_config(SimpleNamespace())


def test_sam2_config_enables_guided_training_warning():
    assert _guided_training_config_enabled(SimpleNamespace(use_sam2=True))
    assert not _guided_training_config_enabled(SimpleNamespace())

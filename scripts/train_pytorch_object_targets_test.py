import pytest
import torch

from scripts.train_pytorch import prepare_object_targets


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

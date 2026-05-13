import numpy as np

from openpi.models import model as _model
from openpi.policies import libero_policy


def _make_input_batch() -> dict:
    return {
        "observation/state": np.zeros((8,), dtype=np.float32),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "test prompt",
    }


def test_libero_inputs_maps_attention_object_keys_to_model_names():
    transform = libero_policy.LiberoInputs(model_type=_model.ModelType.PI0)
    data = _make_input_batch()
    data["attention_map"] = {
        "agentview_attention_object_mask": np.ones((16, 16), dtype=np.float32),
        "wrist_attention_object_mask": np.full((16, 16), 2.0, dtype=np.float32),
    }

    result = transform(data)

    assert set(result["attention_map"]) == {"base_0_attn", "left_wrist_0_attn"}
    np.testing.assert_array_equal(
        result["attention_map"]["base_0_attn"],
        data["attention_map"]["agentview_attention_object_mask"],
    )
    np.testing.assert_array_equal(
        result["attention_map"]["left_wrist_0_attn"],
        data["attention_map"]["wrist_attention_object_mask"],
    )


def test_libero_inputs_zero_pads_missing_attention_object_views():
    transform = libero_policy.LiberoInputs(model_type=_model.ModelType.PI0)
    data = _make_input_batch()
    data["attention_map"] = {
        "agentview_attention_object_mask": np.ones((16, 16), dtype=np.float32),
    }

    result = transform(data)

    np.testing.assert_array_equal(
        result["attention_map"]["base_0_attn"],
        data["attention_map"]["agentview_attention_object_mask"],
    )
    np.testing.assert_array_equal(
        result["attention_map"]["left_wrist_0_attn"],
        np.zeros((16, 16), dtype=np.float32),
    )

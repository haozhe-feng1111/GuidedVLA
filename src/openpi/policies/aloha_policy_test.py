import numpy as np

from openpi.policies import aloha_policy


def _make_input_batch() -> dict:
    image = np.random.randint(256, size=(3, 224, 224), dtype=np.uint8)
    return {
        "state": np.zeros((14,), dtype=np.float32),
        "images": {
            "cam_high": image,
            "cam_left_wrist": image,
            "cam_right_wrist": image,
        },
        "prompt": "test prompt",
    }


def test_aloha_inputs_maps_robotwin_attention_dict_to_canonical_names():
    transform = aloha_policy.AlohaInputs(adapt_to_pi=False)
    data = _make_input_batch()
    data["attention_map"] = {
        "cam_high_attention_object": np.ones((16, 16), dtype=np.float32),
        "cam_left_wrist_attention_object": np.full((16, 16), 2.0, dtype=np.float32),
    }

    result = transform(data)

    assert set(result["attention_map"]) == {"base_0_attn", "left_wrist_0_attn"}
    np.testing.assert_array_equal(
        result["attention_map"]["base_0_attn"],
        data["attention_map"]["cam_high_attention_object"],
    )
    np.testing.assert_array_equal(
        result["attention_map"]["left_wrist_0_attn"],
        data["attention_map"]["cam_left_wrist_attention_object"],
    )


def test_aloha_inputs_accepts_robotwin_attention_object_columns_directly():
    transform = aloha_policy.AlohaInputs(adapt_to_pi=False)
    data = _make_input_batch()
    data["cam_high_attention_object"] = np.ones((16, 16), dtype=np.float32)
    data["cam_right_wrist_attention_object"] = np.full((16, 16), 5.0, dtype=np.float32)

    result = transform(data)

    assert set(result["attention_map"]) == {"base_0_attn", "right_wrist_0_attn"}
    np.testing.assert_array_equal(result["attention_map"]["base_0_attn"], data["cam_high_attention_object"])
    np.testing.assert_array_equal(
        result["attention_map"]["right_wrist_0_attn"],
        data["cam_right_wrist_attention_object"],
    )

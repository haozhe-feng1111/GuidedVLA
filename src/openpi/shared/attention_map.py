from collections.abc import Mapping
from typing import Any

SUPPORTED_OBJECT_VIEWS = ("base_0", "left_wrist_0", "right_wrist_0")
SUPPORTED_OBJECT_VIEW_SET = frozenset(SUPPORTED_OBJECT_VIEWS)

LIBERO_OBJECT_MAP_KEY_TO_VIEW = {
    "agentview_attention_object_mask": "base_0",
    "wrist_attention_object_mask": "left_wrist_0",
}

ROBOTWIN_OBJECT_MAP_KEY_TO_VIEW = {
    "cam_high_attention_object": "base_0",
    "cam_left_wrist_attention_object": "left_wrist_0",
    "cam_right_wrist_attention_object": "right_wrist_0",
}

DATASET_OBJECT_MAP_KEY_TO_VIEW = {
    **LIBERO_OBJECT_MAP_KEY_TO_VIEW,
    **ROBOTWIN_OBJECT_MAP_KEY_TO_VIEW,
}

VIEW_TO_MODEL_ATTENTION_KEY = {view_name: f"{view_name}_attn" for view_name in SUPPORTED_OBJECT_VIEWS}
MODEL_ATTENTION_KEY_TO_VIEW = {model_key: view_name for view_name, model_key in VIEW_TO_MODEL_ATTENTION_KEY.items()}


def normalize_object_map_keys(attention_map: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize raw or model attention-map keys to canonical view names."""
    normalized: dict[str, Any] = {}
    for key, value in attention_map.items():
        if key in SUPPORTED_OBJECT_VIEW_SET:
            normalized[key] = value
        elif key in MODEL_ATTENTION_KEY_TO_VIEW:
            normalized[MODEL_ATTENTION_KEY_TO_VIEW[key]] = value
        elif key in DATASET_OBJECT_MAP_KEY_TO_VIEW:
            normalized[DATASET_OBJECT_MAP_KEY_TO_VIEW[key]] = value
    return normalized


def to_model_attention_map_keys(attention_map: Mapping[str, Any]) -> dict[str, Any]:
    """Convert raw or canonical attention-map keys to the model's ``*_attn`` names."""
    return {
        VIEW_TO_MODEL_ATTENTION_KEY[view_name]: value
        for view_name, value in normalize_object_map_keys(attention_map).items()
    }

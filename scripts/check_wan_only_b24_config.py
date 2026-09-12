"""Parse the real training CLI and check its resolved config without training."""
import dataclasses
import json

import train_pytorch


def check(config, lambda_object=None, lambda_skill=None):
    model = config.model
    data = config.data.base_config
    actual = {
        "name": config.name,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "num_train_steps": config.num_train_steps,
        "precision": config.pytorch_training_precision,
        "gradient_checkpointing": config.use_gradient_checkpointing,
        "resume": config.resume,
        "use_wan22_encoder": model.use_wan22_encoder,
        "other_encoders": any((model.use_depth, model.use_sam2,
                               model.use_patch16_encoder, model.use_raft_encoder)),
        "object_loss": model.use_object_loss,
        "skill_loss": model.use_skill_loss,
        "data_object_loss": data.use_object_loss,
        "data_skill_loss": data.use_skill_loss,
        "object_weight": config.object_loss_weight if lambda_object is None else lambda_object,
        "skill_weight": config.skill_loss_weight if lambda_skill is None else lambda_skill,
        "disable_image_augmentation": model.disable_image_augmentation,
        "wan_heads": model.wan22_head_indices,
        "guided_layers": model.guided_layer_indices,
        "vision_layers": model.depth_guided_layer_indices,
        "wan_dtype": model.wan22_dtype,
        "control": model.control_attention_enabled and model.wan22_use_control,
        "attention_heads": model.control_attention_num_heads,
        "extra_delta_transform": config.data.extra_delta_transform,
    }
    expected = dict(
        name="pi0_libero_wan22_vae_only_b24", batch_size=24,
        gradient_accumulation_steps=1, num_train_steps=30000,
        precision="float32", gradient_checkpointing=True, resume=False,
        use_wan22_encoder=True, other_encoders=False,
        object_loss=False, skill_loss=False, data_object_loss=False, data_skill_loss=False,
        object_weight=0.0, skill_weight=0.0, disable_image_augmentation=True,
        wan_heads=[4, 5], guided_layers=[9, 10, 11, 12], vision_layers=[9, 10, 11, 12],
        wan_dtype="bfloat16", control=True, attention_heads=8, extra_delta_transform=True,
    )
    errors = {key: (expected[key], value) for key, value in actual.items() if value != expected[key]}
    if errors:
        raise ValueError(f"Wan-only B24 contract mismatch (expected, actual): {errors}")
    print("WAN_ONLY_CONTRACT " + json.dumps(actual, sort_keys=True))
    print("RESOLVED_CONFIG " + json.dumps(dataclasses.asdict(config), default=str, sort_keys=True))


if __name__ == "__main__":
    # Only replace the training callback: main still uses the production parser
    # and applies the exact same overrides. No model/data loader/DDP is created.
    train_pytorch.train_loop = check
    train_pytorch.main()

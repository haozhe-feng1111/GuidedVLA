import dataclasses
from dataclasses import field
from typing import TYPE_CHECKING, Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


DEFAULT_OBJECT_SUPERVISION_VIEWS: tuple[str, ...] = tuple(key.removesuffix("_rgb") for key in _model.IMAGE_KEYS)


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # Shared transformer layers used by object, skill, and depth guidance.
    guided_layer_indices: list[int] = field(default_factory=lambda: [9, 10, 11, 12])

    # Enable ControlNet-style attention (dual-branch, zero_conv fusion) on the action expert.
    control_attention_enabled: bool = False
    # Number of attention heads for control branch (default=2, random init)
    # Set to None or -1 for full copy mode (same heads as original)
    control_attention_num_heads: int | None = 8
    # If True, copy weights from original; if False, random initialize
    control_attention_copy_weights: bool = False
    # If True, freeze the original action-expert attention branch while training control branches.
    control_attention_freeze_origin: bool = False
    # If True, only inject control branches into the guided layers; otherwise inject into all layers.
    control_attention_distill_only: bool = False
    # Whether to add per-head sigmoid gate to the control branch Q projection.
    control_attention_use_headwise_gate: bool | None = None

    # Object-mask supervision settings.
    use_object_loss: bool = False
    # Optional list/tuple of view names to supervise with object masks (H_object).
    # Defaults to all available views; set to None to use every object map.
    object_supervision_views: tuple[str, ...] | None = DEFAULT_OBJECT_SUPERVISION_VIEWS
    # Head indices used for object-mask supervision.
    object_head_indices: list[int] = field(default_factory=lambda: [0, 1])
    # Whether object-mask supervision should use the control branch (if enabled).
    object_use_control: bool = False
    # How to aggregate multiple object heads for L_object:
    # - "mean_heads": average object-head attention probabilities first, then supervise once
    #   (paper / reported-results default).
    # - "per_head": supervise each object head independently, then average losses.
    object_loss_head_aggregation: Literal["per_head", "mean_heads"] = "mean_heads"

    # Depth
    use_depth: bool = False
    # Inference-only ablation for checkpoints trained with depth guidance. The
    # checkpoint still has to declare ``use_depth=True``; the PyTorch runtime
    # then omits the depth branch and its checkpoint weights explicitly.
    disable_depth_at_inference: bool = False
    depth_encoder_type: Literal["da3", "dinov2_base"] = "da3"
    depth_model_name: str = None
    depth_head_indices: list[int] = field(default_factory=lambda: [4])
    # Whether depth attention modifications should be applied to the control branch.
    depth_use_control: bool = False

    # SAM 2.1 external visual encoder. This arm is intentionally mutually
    # exclusive with the DA3 depth encoder so an encoder comparison changes one
    # external feature source at a time.
    use_sam2: bool = False
    # Hydra config name resolved from the vendored official SAM2 package.
    sam2_model_config: str | None = None
    # Local official SAM2.1 Tiny checkpoint path.
    sam2_checkpoint_path: str | None = None
    # Action-attention heads assigned to SAM2 K/V injection.
    sam2_head_indices: list[int] = field(default_factory=lambda: [4])
    # Whether SAM2 K/V injection should use the control branch.
    sam2_use_control: bool = False
    # SAM2.1 Hiera models are pretrained at this square image resolution.
    sam2_image_size: int = 1024
    # Each FPN level is resized to this square token grid before the shared 4x4 merger.
    sam2_token_grid_size: int = 16

    # Shared frozen ViT-B/16 arm for MAE and DINOv3 encoder ablations.
    use_patch16_encoder: bool = False
    patch16_encoder_kind: Literal["mae", "dinov3"] | None = None
    patch16_source_root: str | None = None
    patch16_checkpoint_path: str | None = None
    patch16_intermediate_layers: list[int] = field(default_factory=lambda: [5, 7, 9, 11])
    patch16_head_indices: list[int] = field(default_factory=lambda: [4])
    patch16_use_control: bool = False

    # Skill (H_skill)
    use_skill_loss: bool = False
    # Number of discrete skill classes available in the dataset / model head.
    skill_num_classes: int = 8
    # Head indices used for the skill auxiliary loss (L_skill).
    skill_head_indices: list[int] = field(default_factory=lambda: [2, 3])
    # Whether skill supervision should use the control branch (if enabled).
    skill_use_control: bool = False
    # Fixed time value used when computing skill logits during inference (deterministic mode).
    skill_infer_time: float = 0.001

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

        if self.depth_encoder_type not in ("da3", "dinov2_base"):
            raise ValueError("depth_encoder_type must be 'da3' or 'dinov2_base'")

        # Control-branch sanity checks.
        enabled_external_encoders = sum((self.use_depth, self.use_sam2, self.use_patch16_encoder))
        if enabled_external_encoders > 1:
            raise ValueError("depth, SAM2, and Patch16 encoder arms are mutually exclusive")

        if self.use_sam2:
            if not self.sam2_model_config:
                raise ValueError("sam2_model_config must be set when use_sam2 is True")
            if not self.sam2_checkpoint_path:
                raise ValueError("sam2_checkpoint_path must be set when use_sam2 is True")
            if self.sam2_image_size <= 0:
                raise ValueError("sam2_image_size must be positive")
            if self.sam2_token_grid_size <= 0:
                raise ValueError("sam2_token_grid_size must be positive")
            if len(self.sam2_head_indices) == 0:
                raise ValueError("sam2_head_indices must be non-empty when use_sam2 is True")

        if self.use_patch16_encoder:
            if self.patch16_encoder_kind not in ("mae", "dinov3"):
                raise ValueError("patch16_encoder_kind must be 'mae' or 'dinov3'")
            if not self.patch16_checkpoint_path:
                raise ValueError("patch16_checkpoint_path must be set when use_patch16_encoder is True")
            if self.patch16_encoder_kind == "dinov3" and not self.patch16_source_root:
                raise ValueError("patch16_source_root must be set for DINOv3")
            if len(self.patch16_intermediate_layers) != 4:
                raise ValueError("patch16_intermediate_layers must contain exactly four layers")
            if len(self.patch16_head_indices) == 0:
                raise ValueError("patch16_head_indices must be non-empty")

        if not self.control_attention_enabled and (
            self.object_use_control
            or self.skill_use_control
            or self.depth_use_control
            or self.sam2_use_control
            or self.patch16_use_control
        ):
            raise ValueError(
                "control_attention_enabled is False but a *_use_control flag is True. "
                "Either enable control attention or disable the *_use_control flags."
            )

        if self.disable_depth_at_inference and not self.use_depth:
            raise ValueError(
                "disable_depth_at_inference is only valid for a checkpoint configured with use_depth=True."
            )

        # Validate head indices - allow empty lists to disable distillation.
        for name, indices in (
            ("object_head_indices", self.object_head_indices),
            ("skill_head_indices", self.skill_head_indices),
            ("depth_head_indices", self.depth_head_indices),
            ("sam2_head_indices", self.sam2_head_indices),
            ("patch16_head_indices", self.patch16_head_indices),
        ):
            if any(idx < 0 for idx in indices):
                raise ValueError(f"{name} must contain non-negative indices")
        if len(set(self.object_head_indices)) != len(self.object_head_indices):
            raise ValueError("object_head_indices contains duplicate entries")
        if self.object_loss_head_aggregation not in ("per_head", "mean_heads"):
            raise ValueError('object_loss_head_aggregation must be either "per_head" or "mean_heads"')
        if len(set(self.skill_head_indices)) != len(self.skill_head_indices):
            raise ValueError("skill_head_indices contains duplicate entries")
        if len(set(self.depth_head_indices)) != len(self.depth_head_indices):
            raise ValueError("depth_head_indices contains duplicate entries")
        if len(set(self.sam2_head_indices)) != len(self.sam2_head_indices):
            raise ValueError("sam2_head_indices contains duplicate entries")
        if len(set(self.patch16_head_indices)) != len(self.patch16_head_indices):
            raise ValueError("patch16_head_indices contains duplicate entries")
        if any(idx < 0 for idx in self.guided_layer_indices):
            raise ValueError("guided_layer_indices must contain non-negative indices")
        if len(self.guided_layer_indices) != 4:
            raise ValueError("guided_layer_indices must contain exactly 4 layer indices")
        if len(set(self.guided_layer_indices)) != len(self.guided_layer_indices):
            raise ValueError("guided_layer_indices contains duplicate entries")
        object.__setattr__(self, "guided_layer_indices", sorted(self.guided_layer_indices))
        if self.skill_num_classes <= 0:
            raise ValueError("skill_num_classes must be positive")
        if self.use_skill_loss and len(self.skill_head_indices) == 0:
            raise ValueError("skill_head_indices must be non-empty when use_skill_loss is True")

        overlapping_object_skill = sorted(set(self.object_head_indices) & set(self.skill_head_indices))
        if overlapping_object_skill:
            raise ValueError(
                "object_head_indices and skill_head_indices must be disjoint. "
                f"Overlapping heads: {overlapping_object_skill}"
            )
        overlapping_object_depth = sorted(set(self.object_head_indices) & set(self.depth_head_indices))
        if overlapping_object_depth:
            raise ValueError(
                "object_head_indices and depth_head_indices must be disjoint. "
                f"Overlapping heads: {overlapping_object_depth}"
            )
        overlapping_skill_depth = sorted(set(self.skill_head_indices) & set(self.depth_head_indices))
        if overlapping_skill_depth:
            raise ValueError(
                "skill_head_indices and depth_head_indices must be disjoint. "
                f"Overlapping heads: {overlapping_skill_depth}"
            )
        if self.use_sam2:
            overlapping_object_sam2 = sorted(set(self.object_head_indices) & set(self.sam2_head_indices))
            if overlapping_object_sam2:
                raise ValueError(
                    "object_head_indices and sam2_head_indices must be disjoint. "
                    f"Overlapping heads: {overlapping_object_sam2}"
                )
            overlapping_skill_sam2 = sorted(set(self.skill_head_indices) & set(self.sam2_head_indices))
            if overlapping_skill_sam2:
                raise ValueError(
                    "skill_head_indices and sam2_head_indices must be disjoint. "
                    f"Overlapping heads: {overlapping_skill_sam2}"
                )
        if self.use_patch16_encoder:
            for label, occupied in (("object", self.object_head_indices), ("skill", self.skill_head_indices)):
                overlap = sorted(set(occupied) & set(self.patch16_head_indices))
                if overlap:
                    raise ValueError(f"{label}_head_indices and patch16_head_indices overlap: {overlap}")

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)

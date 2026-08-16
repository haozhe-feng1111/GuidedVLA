import abc
import contextlib
from collections.abc import Sequence
import dataclasses
import enum
import logging
import pathlib
import typing
from typing import Generic, TypeVar

import augmax
from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch

from openpi.models_pytorch import pi0_pytorch
from openpi.shared import image_tools
import openpi.shared.array_typing as at

logger = logging.getLogger("openpi")

# Type variable for array types (JAX arrays, PyTorch tensors, or numpy arrays)
ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)


class ModelType(enum.Enum):
    """Supported model types."""

    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    PI05 = "pi05"


# The model always expects these images
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


# This may need change if we release a small model.
IMAGE_RESOLUTION = (224, 224)

_DEPTH_TOKEN_MERGING_PREFIX = "depth_module.token_merging_model."
_DEPTH_INFERENCE_ABLATION_PREFIXES = ("depth_module.", "depth_token_proj.")


def normalize_pytorch_state_dict_for_loading(
    state_dict: dict[str, torch.Tensor],
    *,
    source_label: str,
) -> dict[str, torch.Tensor]:
    """Normalize torch.compile wrapper segments and tied weights for PyTorch checkpoints."""
    embed_tokens_key = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    lm_head_key = "paligemma_with_expert.paligemma.lm_head.weight"

    normalized_state_dict = {}
    normalized_keys = 0
    for key, value in state_dict.items():
        normalized_key = ".".join(part for part in key.split(".") if part != "_orig_mod")
        if normalized_key != key:
            normalized_keys += 1
        if normalized_key in normalized_state_dict:
            raise ValueError(
                f"{source_label} contains duplicate logical key '{normalized_key}' "
                f"after stripping compile wrapper segments."
            )
        normalized_state_dict[normalized_key] = value

    if normalized_keys > 0:
        logger.info(
            "Normalized %d compiled key(s) from %s for compatibility.",
            normalized_keys,
            source_label,
        )

    if embed_tokens_key not in normalized_state_dict and lm_head_key in normalized_state_dict:
        logger.info(f"Copying tied weight: {lm_head_key} -> {embed_tokens_key}")
        normalized_state_dict[embed_tokens_key] = normalized_state_dict[lm_head_key]

    return normalized_state_dict


@contextlib.contextmanager
def temporarily_unwrap_compiled_modules_for_state_dict(model: torch.nn.Module):
    """Temporarily unwrap nested ``torch.compile`` modules for checkpoint I/O.

    Checkpoints are saved with canonical parameter names after compiled wrappers are
    removed. Loading them directly into an ``OptimizedModule`` would instead expect
    ``_orig_mod`` key segments and can silently skip weights under ``strict=False``.
    """
    root_model = model
    while hasattr(root_model, "_orig_mod"):
        root_model = root_model._orig_mod  # noqa: SLF001

    replaced_children: list[tuple[torch.nn.Module, str, torch.nn.Module]] = []

    def unwrap_children(module: torch.nn.Module) -> None:
        for child_name, child_module in list(module.named_children()):
            child_to_visit = child_module
            if hasattr(child_module, "_orig_mod"):
                replaced_children.append((module, child_name, child_module))
                child_to_visit = child_module._orig_mod  # noqa: SLF001
                setattr(module, child_name, child_to_visit)
            unwrap_children(child_to_visit)

    unwrap_children(root_model)
    try:
        yield root_model
    finally:
        for parent_module, child_name, compiled_child in reversed(replaced_children):
            setattr(parent_module, child_name, compiled_child)


def _validate_depth_token_merging_weights_loaded(
    checkpoint_state_dict: dict[str, torch.Tensor],
    missing_keys: list[str],
    unexpected_keys: list[str],
) -> None:
    """Fail closed when a checkpoint's trained depth adapter was not loaded."""
    checkpoint_has_adapter = any(key.startswith(_DEPTH_TOKEN_MERGING_PREFIX) for key in checkpoint_state_dict)
    if not checkpoint_has_adapter:
        return

    adapter_missing = [key for key in missing_keys if key.startswith(_DEPTH_TOKEN_MERGING_PREFIX)]
    adapter_unexpected = [key for key in unexpected_keys if key.startswith(_DEPTH_TOKEN_MERGING_PREFIX)]
    if not adapter_missing and not adapter_unexpected:
        return

    details = []
    if adapter_missing:
        details.append(f"missing={adapter_missing}")
    if adapter_unexpected:
        details.append(f"unexpected={adapter_unexpected}")
    raise RuntimeError(
        "Checkpoint contains trained depth token-merging weights that were not fully loaded; "
        "refusing to continue with an untrained depth adapter (" + "; ".join(details) + ")."
    )


def filter_depth_weights_for_inference_ablation(
    checkpoint_state_dict: dict[str, torch.Tensor],
    *,
    enabled: bool,
) -> dict[str, torch.Tensor]:
    """Remove only complete depth branch weights for the explicit RGB-only ablation."""

    if not enabled:
        return checkpoint_state_dict

    missing_prefixes = [
        prefix
        for prefix in _DEPTH_INFERENCE_ABLATION_PREFIXES
        if not any(key.startswith(prefix) for key in checkpoint_state_dict)
    ]
    if missing_prefixes:
        raise RuntimeError(
            "Depth-inference-off ablation requires a depth-trained checkpoint with weights for "
            f"{_DEPTH_INFERENCE_ABLATION_PREFIXES}; missing prefix(es): {missing_prefixes}."
        )

    removed_keys = [
        key for key in checkpoint_state_dict if key.startswith(_DEPTH_INFERENCE_ABLATION_PREFIXES)
    ]
    logger.warning(
        "Depth-inference-off ablation: intentionally skipped %d depth checkpoint tensor(s).",
        len(removed_keys),
    )
    return {
        key: value
        for key, value in checkpoint_state_dict.items()
        if not key.startswith(_DEPTH_INFERENCE_ABLATION_PREFIXES)
    }


# Data format
#
# Data transforms produce the model input as a nested dictionary which is later converted
# into `Obesrvation` and `Actions` objects. See below.
#
# In the dictory form, this data should look like:
# {
#     # Observation data.
#     "image": {
#         "base_0_rgb": (float32|uint8)[*b, h, w, 3],  # RGB image in [-1, 1] or [0, 255]
#         ...  # Additional camera views
#     },
#     "image_mask": {
#         "base_0_rgb": bool[*b],  # True if image is valid
#         ...  # Masks for additional views
#     },
#     "state": float32[*b, s],  # Low-dimensional robot state
#     "tokenized_prompt": int32[*b, l],  # Optional, tokenized language prompt
#     "tokenized_prompt_mask": bool[*b, l],  # Optional, mask for tokenized prompt
#     "token_ar_mask": int32[*b, l],  # Optional, autoregressive mask for FAST model
#     "token_loss_mask": bool[*b, l],  # Optional, loss mask for FAST model
#
#      # Actions data.
#      "actions": float32[*b ah ad]
# }
# where:
#   *b = batch dimensions
#   h,w = image height/width
#   s = state dimension
#   l = sequence length
#
ImageArray: typing.TypeAlias = at.Float[ArrayT, "*b h w c"] | at.UInt8[ArrayT, "*b h w c"]


def _contains_cuda_tensor(tree: typing.Any) -> bool:
    if isinstance(tree, dict):
        return any(_contains_cuda_tensor(value) for value in tree.values())
    if isinstance(tree, list | tuple):
        return any(_contains_cuda_tensor(value) for value in tree)
    return isinstance(tree, torch.Tensor) and tree.is_cuda


@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """Holds observations, i.e., inputs to the model.

    See `Observation.from_dict` to see the expected dictionary form. This is the format
    that should be produced by the data transforms.
    """

    # Images, either normalized float32 in [-1, 1] or raw uint8 before PyTorch device transfer.
    images: dict[str, ImageArray]
    # Image masks, with same keys as images.
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    # Low-dimensional robot state.
    state: at.Float[ArrayT, "*b s"]

    # Optional discrete skill id per batch element (chunk-level or aggregated per frame).
    # Only used in certain fine-tuning settings (e.g., skill-level auxiliary heads);
    # ignored by models that do not consume this field.
    skill_id: at.Int[ArrayT, "*b"] | None = None

    # Optional chunk-level soft skill label distribution for soft-label supervision.
    # Shape is typically [*b, K] where K is the number of skill classes.
    skill_soft: at.Float[ArrayT, "*b k"] | None = None

    # Tokenized prompt.
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    # Tokenized prompt mask.
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    # pi0-fast model specific fields.

    # Token auto-regressive mask (for FAST autoregressive model).
    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    # Token loss mask (for FAST autoregressive model).
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None

    @classmethod
    def from_dict(
        cls,
        data: at.PyTree[ArrayT],
        *,
        normalize_torch_images: bool = True,
    ) -> "Observation[ArrayT]":
        """This method defines the mapping between unstructured data (i.e., nested dict) to the structured Observation format."""
        # Ensure that tokenized_prompt and tokenized_prompt_mask are provided together.
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
        # If images are uint8, convert them to [-1, 1] float32.
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
            elif (
                normalize_torch_images
                and hasattr(data["image"][key], "dtype")
                and data["image"][key].dtype == torch.uint8
            ):
                data["image"][key] = data["image"][key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
            elif (
                normalize_torch_images and isinstance(data["image"][key], torch.Tensor) and data["image"][key].ndim == 4
            ):
                image = data["image"][key].to(torch.float32)
                if image.shape[-1] == 3:
                    image = image.permute(0, 3, 1, 2)
                data["image"][key] = image
        kwargs = {
            "images": data["image"],
            "image_masks": data["image_mask"],
            "state": data["state"],
            "skill_id": data.get("skill_id"),
            "skill_soft": data.get("skill_soft"),
            "tokenized_prompt": data.get("tokenized_prompt"),
            "tokenized_prompt_mask": data.get("tokenized_prompt_mask"),
            "token_ar_mask": data.get("token_ar_mask"),
            "token_loss_mask": data.get("token_loss_mask"),
        }
        # jaxtyping/beartype can reject valid CUDA torch.bool masks inside generic
        # dataclass construction even though the same tensors work downstream.
        # Keep runtime typechecking everywhere else, but bypass it for this GPU-only path.
        if _contains_cuda_tensor(kwargs):
            with at.disable_typechecking():
                return cls(**kwargs)
        return cls(**kwargs)

    def to_dict(self) -> at.PyTree[ArrayT]:
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        return result


# Defines the format of the actions. This field is included as "actions" inside the dictionary
# produced by the data transforms.
Actions = at.Float[ArrayT, "*b ah ad"]


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> Observation:
    """Preprocess the observations by performing image augmentations (if train=True), resizing (if necessary), and
    filling in a default image mask (if necessary).
    """

    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad(image, *image_resolution)

        if train:
            # Convert from [-1, 1] to [0, 1] for augmax.
            image = image / 2.0 + 0.5

            transforms = []
            if "wrist" not in key:
                height, width = image.shape[1:3]
                transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    augmax.Resize(width, height),
                    augmax.Rotate((-5, 5)),
                ]
            transforms += [
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]
            sub_rngs = jax.random.split(rng, image.shape[0])
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

            # Back to [-1, 1].
            image = image * 2.0 - 1.0

        out_images[key] = image

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        skill_id=observation.skill_id,
        skill_soft=observation.skill_soft,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )


@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """Configuration shared by all models. Specific models should inherit from this class, and implement the `create`
    method to create the corresponding model.
    """

    # Action space dimension.
    action_dim: int
    # Action sequence length.
    action_horizon: int
    # Tokenized prompt maximum length.
    max_token_len: int

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """The model type."""

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """Create a new model, initializing parameters."""

    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "BaseModel":
        """Create a model with the given parameters."""
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)
        at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    def load_pytorch(self, train_config, weight_path: str):
        logger.info(f"train_config: {train_config}")

        # IMPORTANT: Create model config with float32 to avoid precision loss during weight loading.
        # The model will be converted to bfloat16 AFTER weights are loaded by policy_config.py.
        model_config = dataclasses.replace(train_config.model, dtype="float32")

        model = pi0_pytorch.PI0Pytorch(config=model_config)

        # Inject ControlAttention BEFORE loading weights (inject-then-load).
        # For inference the checkpoint was trained with CA, so the saved keys already have
        # .origin./* and object_branch./* structure — matching the injected model.
        if getattr(model_config, "control_attention_enabled", False):
            model.enable_control_attention()

        # Normalize state dict: strip compile wrapper segments and fix tied embed_tokens weight.
        state_dict = safetensors.torch.load_file(weight_path)
        new_state_dict = normalize_pytorch_state_dict_for_loading(
            state_dict,
            source_label="PyTorch policy checkpoint",
        )
        depth_disabled_at_inference = getattr(model_config, "disable_depth_at_inference", False)
        new_state_dict = filter_depth_weights_for_inference_ablation(
            new_state_dict,
            enabled=depth_disabled_at_inference,
        )

        with temporarily_unwrap_compiled_modules_for_state_dict(model) as model_to_load:
            missing_keys, unexpected_keys = model_to_load.load_state_dict(new_state_dict, strict=False)
        _validate_depth_token_merging_weights_loaded(new_state_dict, missing_keys, unexpected_keys)

        # Split missing keys into expected (depth/skill modules may not be in checkpoint) and unexpected.
        expected_missing, unexpected_missing = [], []
        for key in missing_keys:
            if key.startswith(("depth", "skill_head.")):
                expected_missing.append(key)
            else:
                unexpected_missing.append(key)

        if expected_missing:
            logger.debug(f"Missing keys (expected, new modules not yet in checkpoint): {expected_missing}")

        # Raise an error on control-attention structural mismatches between config and checkpoint.
        ca_missing = [k for k in unexpected_missing if ".origin." in k or "object_branch" in k]
        ca_unexpected = [k for k in unexpected_keys if "object_branch" in k]
        if ca_missing or ca_unexpected:
            lines = []
            if ca_missing:
                lines.append(
                    f"  Config has control_attention_enabled=True but checkpoint is missing "
                    f"{len(ca_missing)} control-attention key(s), e.g. '{ca_missing[0]}'."
                )
            if ca_unexpected:
                lines.append(
                    f"  Config has control_attention_enabled=False but checkpoint contains "
                    f"{len(ca_unexpected)} control-attention key(s), e.g. '{ca_unexpected[0]}'."
                )
            raise ValueError(
                "Checkpoint structure does not match config:\n"
                + "\n".join(lines)
                + "\nUpdate the config's control_attention_enabled to match the checkpoint."
            )

        other_missing = [k for k in unexpected_missing if k not in ca_missing]
        other_unexpected = [k for k in unexpected_keys if k not in ca_unexpected]
        if other_missing:
            logger.warning(f"Missing keys (unexpected): {other_missing}")
        if other_unexpected:
            logger.warning(f"Unexpected keys in checkpoint: {other_unexpected}")

        logger.info("Model loaded successfully.")
        return model

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """Returns the input specification for the model. Values are jax.ShapeDtypeStruct."""

    def fake_obs(self, batch_size: int = 1) -> Observation:
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int = 1) -> Actions:
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)


@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    """Base class for all model implementations. Specific models should inherit from this class. They should call
    super().__init__() to initialize the shared attributes (action_dim, action_horizon, and max_token_len).
    """

    action_dim: int
    action_horizon: int
    max_token_len: int

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]: ...

    @abc.abstractmethod
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation, **kwargs) -> Actions: ...


def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """Restores unstructured params PyTree from a checkpoint.

    This works with checkpoints saved with `save_state` during openpi training (see `training/checkpoints.py`) as
    well as pre-trained checkpoints released for openpi.

    Args:
        params_path: The local path to the checkpoint directory.
        restore_type: The type to restore the params as. Can be set to `np.ndarray` to load the params as a numpy array.
        dtype: The dtype to restore all params as. If not provided, will use the original dtype from the checkpoint.
        sharding: The sharding to use for the params. If not provided, the params will be replicated across all devices.

    Returns:
        The restored params.
    """
    params_path = pathlib.Path(params_path).resolve() if not str(params_path).startswith("gs://") else params_path

    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item
                ),
            ),
        )["params"]

    # If the params were saved with `save_state` during openpi training, every key path will end with "value", which is
    # added by `nnx.State`. We remove the "value" suffix here and always return what NNX calls a "pure dict".
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)

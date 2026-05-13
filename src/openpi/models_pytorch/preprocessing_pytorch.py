from collections.abc import Sequence
import logging
import math

import torch

from openpi.shared import image_tools

logger = logging.getLogger("openpi")

# Constants moved from model.py
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)

IMAGE_RESOLUTION = (224, 224)

# Cache default (all-True) image masks by (batch_shape, device) to avoid a
# fresh torch.ones allocation on every forward when an image lacks an explicit mask.
_DEFAULT_MASK_CACHE: dict[tuple, torch.Tensor] = {}


def _default_image_mask(batch_shape: torch.Size, device: torch.device) -> torch.Tensor:
    key = (tuple(batch_shape), device)
    mask = _DEFAULT_MASK_CACHE.get(key)
    if mask is None:
        mask = torch.ones(batch_shape, dtype=torch.bool, device=device)
        _DEFAULT_MASK_CACHE[key] = mask
    return mask


def preprocess_observation_pytorch(
    observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
):
    """Torch.compile-compatible version of preprocess_observation_pytorch with simplified type annotations.

    This function avoids complex type annotations that can cause torch.compile issues.
    """
    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]

        # Handle both [B, C, H, W] and [B, H, W, C] formats.
        is_channels_first = image.shape[1] == 3  # Check if channels are in dimension 1

        if is_channels_first:
            # Convert [B, C, H, W] to [B, H, W, C] for processing
            image = image.permute(0, 2, 3, 1)

        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad_torch(image, *image_resolution)

        if train:
            # Convert from [-1, 1] to [0, 1] for PyTorch augmentations
            image = image / 2.0 + 0.5

            # Apply PyTorch-based augmentations
            if "wrist" not in key:
                # Geometric augmentations for non-wrist cameras
                height, width = image.shape[1:3]

                # Random crop and resize
                crop_height = int(height * 0.95)
                crop_width = int(width * 0.95)

                # Random crop
                max_h = height - crop_height
                max_w = width - crop_width
                if max_h > 0 and max_w > 0:
                    # Sample on CPU so slice bounds are plain Python ints — using GPU
                    # 0-d tensors as slice bounds forces an implicit .item() and a
                    # cudaStreamSynchronize. This path is eager (not compiled).
                    start_h = int(torch.randint(0, max_h + 1, (1,)).item())
                    start_w = int(torch.randint(0, max_w + 1, (1,)).item())
                    image = image[:, start_h : start_h + crop_height, start_w : start_w + crop_width, :]

                # Resize back to original size
                image = torch.nn.functional.interpolate(
                    image.permute(0, 3, 1, 2),  # [b, h, w, c] -> [b, c, h, w]
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

                # Random rotation (small angles) — sampled on CPU so the
                # significance check is a pure Python bool (no GPU→CPU sync).
                angle_deg = float(torch.rand(1).item()) * 10.0 - 5.0
                if abs(angle_deg) > 0.1:
                    angle_rad = angle_deg * math.pi / 180.0
                    cos_a = math.cos(angle_rad)
                    sin_a = math.sin(angle_rad)

                    # Apply rotation using grid_sample
                    grid_x = torch.linspace(-1, 1, width, device=image.device)
                    grid_y = torch.linspace(-1, 1, height, device=image.device)

                    # Create meshgrid
                    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")

                    # Expand to batch dimension
                    grid_x = grid_x.unsqueeze(0).expand(image.shape[0], -1, -1)
                    grid_y = grid_y.unsqueeze(0).expand(image.shape[0], -1, -1)

                    # Apply rotation transformation
                    grid_x_rot = grid_x * cos_a - grid_y * sin_a
                    grid_y_rot = grid_x * sin_a + grid_y * cos_a

                    # Stack and reshape for grid_sample
                    grid = torch.stack([grid_x_rot, grid_y_rot], dim=-1)

                    image = torch.nn.functional.grid_sample(
                        image.permute(0, 3, 1, 2),  # [b, h, w, c] -> [b, c, h, w]
                        grid,
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=False,
                    ).permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

            # Color augmentations: batch the three random draws into one kernel.
            color_rand = torch.rand(3, device=image.device)

            # Random brightness in [0.7, 1.3]
            image = image * (0.7 + color_rand[0] * 0.6)

            # Random contrast in [0.6, 1.4]
            contrast_factor = 0.6 + color_rand[1] * 0.8
            mean = image.mean(dim=[1, 2, 3], keepdim=True)
            image = (image - mean) * contrast_factor + mean

            # Random saturation in [0.5, 1.5]
            saturation_factor = 0.5 + color_rand[2] * 1.0
            gray = image.mean(dim=-1, keepdim=True)
            image = gray + (image - gray) * saturation_factor

            # Clamp values to [0, 1]
            image = torch.clamp(image, 0, 1)

            # Back to [-1, 1]
            image = image * 2.0 - 1.0

        # .contiguous() keeps dim-1 stride stable across train/eval paths — without it,
        # torch.compile embed_image recompiles mid-run and desyncs DDP ranks.
        if is_channels_first:
            image = image.permute(0, 3, 1, 2).contiguous()

        out_images[key] = image

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default — cached all-True tensor, safe to share (read-only)
            out_masks[key] = _default_image_mask(batch_shape, observation.state.device)
        else:
            # Canonicalize to the training-time mask shape `[*batch]`. This is a
            # cheap view for already-correct masks and also strips accidental
            # singleton suffix dims from some inference paths, e.g. `[B, 1] -> [B]`.
            out_masks[key] = observation.image_masks[key].reshape(batch_shape)

    # Create a simple object with the required attributes instead of using the complex Observation class
    class SimpleProcessedObservation:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    return SimpleProcessedObservation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        skill_id=getattr(observation, "skill_id", None),
        skill_soft=getattr(observation, "skill_soft", None),
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )

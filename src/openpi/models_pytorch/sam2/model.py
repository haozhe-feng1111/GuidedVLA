"""Frozen SAM2.1 Hiera encoder with a trainable fixed-budget token adapter."""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from openpi.models_pytorch.depth.token_merging import TokenMerging2D


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_SAM2_FPN_DIM = 256
_SAM2_FPN_LEVELS = 4


class Sam2Encoder(nn.Module):
    """Adapt frozen official SAM2.1 Hiera FPN features to GuidedVLA K/V tokens.

    Inputs are GuidedVLA tensors in ``[-1, 1]``. SAM2.1 Hiera-Tiny is run at
    its native 1024px resolution. All four 256-dimensional FPN levels are
    resized to a 64x64 grid and then passed through the same trainable 4x4
    merger used by the DA3 arm, producing four groups of 256 1024-dimensional
    tokens. Thus SAM2 changes the frozen visual encoder, not the injection
    token count or GuidedVLA head layout.
    """

    def __init__(
        self,
        sam2_model_config: str,
        sam2_checkpoint_path: str,
        *,
        feature_dim: int = 1024,
        image_size: int = 1024,
        token_grid_size: int = 64,
        freeze_sam2_model: bool = True,
    ):
        super().__init__()

        sam2_model_config = os.environ.get("OPENPI_SAM2_MODEL_CONFIG", sam2_model_config)
        sam2_checkpoint_path = os.environ.get("OPENPI_SAM2_CHECKPOINT_PATH", sam2_checkpoint_path)
        if not sam2_model_config:
            raise ValueError("sam2_model_config must be provided")
        if not sam2_checkpoint_path or not os.path.isfile(sam2_checkpoint_path):
            raise FileNotFoundError(f"SAM2 checkpoint not found: {sam2_checkpoint_path}")
        if image_size <= 0 or token_grid_size <= 0:
            raise ValueError("image_size and token_grid_size must be positive")

        try:
            from sam2.build_sam import build_sam2
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "SAM2 is required when use_sam2=True. Add both the official SAM2 source root and "
                "its isolated hydra/iopath dependency root to PYTHONPATH."
            ) from exc

        # SAM2's Tiny config defaults to ``scalp: 1`` for mask decoding, which
        # drops one FPN level. Keep all four levels for a matched multi-scale
        # external-encoder comparison with the DA3 arm.
        self.sam2_model = build_sam2(
            config_file=sam2_model_config,
            ckpt_path=sam2_checkpoint_path,
            device="cpu",
            mode="eval",
            hydra_overrides_extra=["++model.image_encoder.scalp=0"],
            apply_postprocessing=False,
        )
        self.image_size = image_size
        self.token_grid_size = token_grid_size
        self.feature_dim = feature_dim
        self.freeze_sam2_model = freeze_sam2_model

        if freeze_sam2_model:
            self._freeze_sam2_model()

        self.token_merging_model = TokenMerging2D(
            patch_size=4,
            in_dim=_SAM2_FPN_DIM,
            out_dim=feature_dim,
        )
        self.register_buffer("img_mean", torch.tensor(_IMAGENET_MEAN).view(3, 1, 1))
        self.register_buffer("img_std", torch.tensor(_IMAGENET_STD).view(3, 1, 1))

        # Only the compact trainable adapter is compiled. Hiera-Tiny itself is
        # deliberately left eager, matching the official config's guidance.
        self.token_merging_model = torch.compile(self.token_merging_model, mode="default", dynamic=False)
        self.forward = torch.compiler.disable(self.forward)

    def _freeze_sam2_model(self) -> None:
        for parameter in self.sam2_model.parameters():
            parameter.requires_grad = False
        self.sam2_model.eval()

    def freeze_unused_weight(self) -> None:
        self._freeze_sam2_model()

    def train(self, mode: bool = True):
        """Keep the frozen SAM2 backbone in eval mode during policy training."""
        super().train(mode)
        if self.freeze_sam2_model:
            self.sam2_model.eval()
        return self

    @staticmethod
    def prepare_fpn_tokens(
        fpn_features: tuple[torch.Tensor, ...] | list[torch.Tensor],
        *,
        feature_dim: int,
        token_grid_size: int,
    ) -> tuple[torch.Tensor, ...]:
        """Resize SAM2 FPN maps and convert them to ``[B, N, C]`` token groups."""
        if len(fpn_features) != _SAM2_FPN_LEVELS:
            raise RuntimeError(
                f"SAM2 image encoder must return {_SAM2_FPN_LEVELS} FPN levels, got {len(fpn_features)}."
            )

        token_groups = []
        for level, feature_map in enumerate(fpn_features):
            if feature_map.ndim != 4:
                raise RuntimeError(
                    f"SAM2 FPN level {level} must have shape [B, C, H, W], got {tuple(feature_map.shape)}."
                )
            if feature_map.shape[1] != feature_dim:
                raise RuntimeError(
                    f"SAM2 FPN level {level} must have {feature_dim} channels, got {feature_map.shape[1]}."
                )
            resized = F.interpolate(
                feature_map,
                size=(token_grid_size, token_grid_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            # Keep the adapter's autograd boundary valid even when an upstream
            # SAM2 implementation produces inference-mode tensors.
            token_groups.append(resized.flatten(2).transpose(1, 2).clone())
        return tuple(token_groups)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return four trainable-adapter token groups from a frozen SAM2 forward."""
        x = (images + 1.0) * 0.5
        x = (x - self.img_mean) / self.img_std
        x = F.interpolate(
            x,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

        with torch.no_grad():
            image_encoder_output = self.sam2_model.image_encoder(x)
        fpn_tokens = self.prepare_fpn_tokens(
            image_encoder_output["backbone_fpn"],
            feature_dim=_SAM2_FPN_DIM,
            token_grid_size=self.token_grid_size,
        )
        return tuple(self.token_merging_model(tokens) for tokens in fpn_tokens)

"""Frozen MAE/DINOv3 ViT-B/16 encoders with the existing GuidedVLA adapter."""

from __future__ import annotations

from functools import partial
import os
import sys

import torch
from torch import nn
import torch.nn.functional as F

from openpi.models_pytorch.depth.token_merging import TokenMerging2D


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_SUPPORTED_ENCODERS = ("mae", "dinov3")
_BACKBONE_DIM = 768
_SOURCE_GRID_SIZE = 14
_TARGET_GRID_SIZE = 16


class Patch16Encoder(nn.Module):
    """Adapt four frozen ViT-B/16 intermediate features to GuidedVLA K/V tokens.

    Both backbones produce a 14x14 patch grid at 224px. Each selected feature
    map is bilinearly resized to 16x16 before the repository's existing 4x4
    token merger, preserving four groups of 16 1024-dimensional tokens.
    """

    def __init__(
        self,
        encoder_kind: str,
        checkpoint_path: str,
        *,
        source_root: str | None = None,
        intermediate_layers: tuple[int, ...] = (5, 7, 9, 11),
        feature_dim: int = 1024,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        if encoder_kind not in _SUPPORTED_ENCODERS:
            raise ValueError(f"encoder_kind must be one of {_SUPPORTED_ENCODERS}, got {encoder_kind!r}")
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"{encoder_kind} checkpoint not found: {checkpoint_path}")
        if len(intermediate_layers) != 4 or sorted(set(intermediate_layers)) != list(intermediate_layers):
            raise ValueError("intermediate_layers must contain four unique ascending layer indices")
        if intermediate_layers[0] < 0 or intermediate_layers[-1] >= 12:
            raise ValueError("ViT-B intermediate layer indices must be in [0, 11]")

        self.encoder_kind = encoder_kind
        self.intermediate_layers = tuple(intermediate_layers)
        self.freeze_backbone = freeze_backbone
        self.backbone = self._build_backbone(source_root, checkpoint_path)
        if freeze_backbone:
            self._freeze_backbone()

        self.token_merging_model = TokenMerging2D(
            patch_size=4,
            in_dim=_BACKBONE_DIM,
            out_dim=feature_dim,
        )
        self.register_buffer("img_mean", torch.tensor(_IMAGENET_MEAN).view(3, 1, 1))
        self.register_buffer("img_std", torch.tensor(_IMAGENET_STD).view(3, 1, 1))
        self.token_merging_model = torch.compile(self.token_merging_model, mode="default", dynamic=False)
        self.forward = torch.compiler.disable(self.forward)

    def _build_backbone(self, source_root: str | None, checkpoint_path: str) -> nn.Module:
        if self.encoder_kind == "dinov3":
            if not source_root or not os.path.isfile(os.path.join(source_root, "hubconf.py")):
                raise FileNotFoundError(f"official DINOv3 source root not found: {source_root}")
            if source_root not in sys.path:
                sys.path.insert(0, source_root)
            from dinov3.hub.backbones import dinov3_vitb16

            return dinov3_vitb16(pretrained=True, weights=checkpoint_path)

        try:
            from timm.models.vision_transformer import VisionTransformer
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("timm is required for the official MAE ViT-B/16 architecture") from exc
        backbone = VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=_BACKBONE_DIM,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=0,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("model", checkpoint)
        encoder_prefixes = ("cls_token", "pos_embed", "patch_embed.", "blocks.", "norm.")
        encoder_state = {key: value for key, value in state_dict.items() if key.startswith(encoder_prefixes)}
        missing, unexpected = backbone.load_state_dict(encoder_state, strict=False)
        allowed_missing = {"head.weight", "head.bias"}
        if set(missing) - allowed_missing or unexpected:
            raise RuntimeError(f"MAE encoder checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        return backbone

    def _freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()

    def freeze_unused_weight(self) -> None:
        self._freeze_backbone()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    @staticmethod
    def resize_patch_tokens(tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1:] != (_SOURCE_GRID_SIZE**2, _BACKBONE_DIM):
            raise RuntimeError(
                "Patch16 feature must have shape "
                f"[B, {_SOURCE_GRID_SIZE**2}, {_BACKBONE_DIM}], got {tuple(tokens.shape)}"
            )
        feature_map = tokens.transpose(1, 2).reshape(-1, _BACKBONE_DIM, _SOURCE_GRID_SIZE, _SOURCE_GRID_SIZE)
        resized = F.interpolate(
            feature_map,
            size=(_TARGET_GRID_SIZE, _TARGET_GRID_SIZE),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return resized.flatten(2).transpose(1, 2).clone()

    def _mae_intermediate_features(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.backbone.patch_embed(images)
        x = torch.cat((self.backbone.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.backbone.pos_embed
        x = self.backbone.pos_drop(x)
        features = []
        for index, block in enumerate(self.backbone.blocks):
            x = block(x)
            if index in self.intermediate_layers:
                features.append(self.backbone.norm(x)[:, 1:])
        return tuple(features)

    def _intermediate_features(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.encoder_kind == "mae":
            return self._mae_intermediate_features(images)
        features = self.backbone.get_intermediate_layers(
            images,
            n=list(self.intermediate_layers),
            reshape=False,
            return_class_token=False,
            norm=True,
        )
        return tuple(features)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = (images + 1.0) * 0.5
        x = (x - self.img_mean) / self.img_std
        with torch.no_grad():
            features = self._intermediate_features(x)
        if len(features) != 4:
            raise RuntimeError(f"Expected four {self.encoder_kind} feature groups, got {len(features)}")
        resized = tuple(self.resize_patch_tokens(feature) for feature in features)
        return tuple(self.token_merging_model(tokens) for tokens in resized)

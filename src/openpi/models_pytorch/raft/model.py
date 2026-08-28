"""Frozen single-frame encoder backed by torchvision RAFT-Large fnet."""

from __future__ import annotations

import os

import torch
from torch import nn
import torch.nn.functional as F


_FEATURE_DIM = 256
_SOURCE_GRID_SIZE = 28
_TARGET_GRID_SIZE = 16
_EXPECTED_IMAGE_SIZE = 224


class RaftFeatureEncoder(nn.Module):
    """Encode one current RGB frame into 256 native 256-d RAFT fnet tokens.

    GuidedVLA images already use the ``[-1, 1]`` input range required by the
    torchvision RAFT weights transform, so no second normalization is applied.
    The correlation, context, update, and flow-prediction modules are discarded.
    """

    def __init__(self, checkpoint_path: str) -> None:
        super().__init__()
        checkpoint_path = os.environ.get("OPENPI_RAFT_CHECKPOINT_PATH", checkpoint_path)
        checkpoint_path = os.path.expanduser(checkpoint_path)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"RAFT-Large C_T_SKHT_V2 checkpoint not found: {checkpoint_path}")

        from torchvision.models.optical_flow import raft_large

        raft = raft_large(weights=None, progress=False)
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        missing, unexpected = raft.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"RAFT-Large checkpoint mismatch: missing={missing}, unexpected={unexpected}")

        # Only fnet is retained. This makes it impossible for the policy path to
        # accidentally consume pairwise flow, correlation, or recurrent state.
        self.feature_encoder = raft.feature_encoder.eval().requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.feature_encoder.eval()
        return self

    def freeze_unused_weight(self) -> None:
        self.feature_encoder.eval().requires_grad_(False)

    @staticmethod
    def resize_feature_map(feature_map: torch.Tensor) -> torch.Tensor:
        if feature_map.ndim != 4 or feature_map.shape[1:] != (
            _FEATURE_DIM,
            _SOURCE_GRID_SIZE,
            _SOURCE_GRID_SIZE,
        ):
            raise RuntimeError(
                "RAFT-Large fnet feature must have shape "
                f"[B, {_FEATURE_DIM}, {_SOURCE_GRID_SIZE}, {_SOURCE_GRID_SIZE}], got {tuple(feature_map.shape)}"
            )
        resized = F.interpolate(
            feature_map,
            size=(_TARGET_GRID_SIZE, _TARGET_GRID_SIZE),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return resized.flatten(2).transpose(1, 2).contiguous()

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if images.ndim != 4 or images.shape[1:] != (3, _EXPECTED_IMAGE_SIZE, _EXPECTED_IMAGE_SIZE):
            raise RuntimeError(
                "RAFT feature encoder expects [B, 3, 224, 224] current-frame RGB, "
                f"got {tuple(images.shape)}"
            )
        input_dtype = images.dtype
        with torch.no_grad():
            feature_map = self.feature_encoder(images.to(dtype=torch.float32))
        tokens = self.resize_feature_map(feature_map).to(dtype=input_dtype)
        # One final fnet feature feeds four independent per-layer K/V projectors.
        return (tokens, tokens, tokens, tokens)


__all__ = ["RaftFeatureEncoder"]

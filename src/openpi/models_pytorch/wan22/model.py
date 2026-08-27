"""Frozen single-frame encoder backed by the official Wan2.2 TI2V-5B VAE."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


_LATENT_DIM = 48
_SOURCE_GRID_SIZE = 14
_TARGET_GRID_SIZE = 16
_EXPECTED_IMAGE_SIZE = 224

_LATENT_MEAN = (
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
    -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
    0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
)
_LATENT_STD = (
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
)


def _load_official_vae_module(source_root: str):
    source_path = Path(source_root).expanduser().resolve() / "wan" / "modules" / "vae2_2.py"
    if not source_path.is_file():
        raise FileNotFoundError(f"Official Wan2.2 VAE source not found: {source_path}")
    spec = importlib.util.spec_from_file_location("openpi_official_wan22_vae2_2", source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load official Wan2.2 VAE source: {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Wan22VAEEncoder(nn.Module):
    """Encode the current 224px RGB frame into 256 native 48-d VAE tokens."""

    def __init__(self, source_root: str, checkpoint_path: str, *, dtype: str = "bfloat16") -> None:
        super().__init__()
        source_root = os.environ.get("OPENPI_WAN22_SOURCE_ROOT", source_root)
        checkpoint_path = os.environ.get("OPENPI_WAN22_CHECKPOINT_PATH", checkpoint_path)
        checkpoint_path = os.path.expanduser(checkpoint_path)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Wan2.2 VAE checkpoint not found: {checkpoint_path}")
        if dtype not in ("float32", "bfloat16"):
            raise ValueError("Wan2.2 VAE dtype must be 'float32' or 'bfloat16'")

        official = _load_official_vae_module(source_root)
        vae = official._video_vae(  # noqa: SLF001 - the official factory is the checkpoint contract.
            pretrained_path=checkpoint_path,
            z_dim=_LATENT_DIM,
            dim=160,
            dim_mult=[1, 2, 4, 4],
            temperal_downsample=[False, True, True],
        )
        # Only encoder + posterior-mean head are used. Replacing decoder modules
        # also keeps them out of policy checkpoints and frees their GPU memory.
        vae.decoder = nn.Identity()
        vae.conv2 = nn.Identity()
        self.vae = vae.eval().requires_grad_(False)
        self.compute_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
        self.vae.to(dtype=self.compute_dtype)
        self.register_buffer("latent_mean", torch.tensor(_LATENT_MEAN).view(1, _LATENT_DIM, 1, 1, 1))
        self.register_buffer("latent_inv_std", (1.0 / torch.tensor(_LATENT_STD)).view(1, _LATENT_DIM, 1, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        self.vae.eval()
        return self

    def freeze_unused_weight(self) -> None:
        self.vae.eval().requires_grad_(False)

    @staticmethod
    def resize_latents(latents: torch.Tensor) -> torch.Tensor:
        if latents.shape[1:] != (_LATENT_DIM, 1, _SOURCE_GRID_SIZE, _SOURCE_GRID_SIZE):
            raise RuntimeError(
                "Wan2.2 single-frame latent must have shape "
                f"[B, {_LATENT_DIM}, 1, {_SOURCE_GRID_SIZE}, {_SOURCE_GRID_SIZE}], got {tuple(latents.shape)}"
            )
        feature_map = latents.squeeze(2)
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
                "Wan2.2 encoder expects [B, 3, 224, 224] current-frame RGB, "
                f"got {tuple(images.shape)}"
            )
        input_dtype = images.dtype
        video = images.to(dtype=self.compute_dtype).unsqueeze(2)
        with torch.no_grad():
            latents = self.vae.encode(video, [self.latent_mean, self.latent_inv_std])
        tokens = self.resize_latents(latents).to(dtype=input_dtype)
        # The same final VAE feature is consumed by four independent per-layer
        # K/V projectors; no external width adapter or intermediate-layer split.
        return (tokens, tokens, tokens, tokens)


__all__ = ["Wan22VAEEncoder"]

# GuidedVLA addition: Depth encoder wrapping Depth Anything V3 for geometry-aware action attention (Depth Head).
# Paper: "GuidedVLA: Specifying Task-Relevant Factors via Plug-and-Play Action Attention Specialization" (RSS 2026)
import os
import sys
import types
from typing import Literal

import torch
import torch.nn as nn
from transformers import AutoModel

# depth_anything_3 eagerly imports its 3DGS export utilities, which in turn try to
# import `gsplat`. We don't use 3DGS rendering, so register a lightweight stub before
# depth_anything_3 is loaded to avoid the import-time warning.
if "gsplat" not in sys.modules:
    _stub = types.ModuleType("gsplat")
    _stub.rasterization = None  # placeholder; will raise AttributeError if actually called
    sys.modules["gsplat"] = _stub

from depth_anything_3.api import DepthAnything3

from .token_merging import TokenMerging2D

_FEAT_LAYERS = [5, 7, 9, 11]  # DA3-SMALL out_layers
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class DepthEncoder(nn.Module):
    """
    Wraps a frozen DA3-SMALL or DINOv2-Base encoder to produce patch tokens.

    Input:  [B, 3, H, W] images in [-1, 1]
    Output: tuple of 4 tensors, each [B, N_merged, feature_dim]
    """

    def __init__(
        self,
        depth_model_name,
        feature_dim=1024,
        freeze_depth_model=True,
        depth_encoder_type: Literal["da3", "dinov2_base"] = "da3",
    ):
        super().__init__()

        depth_model_name = os.environ.get("OPENPI_DEPTH_MODEL_PATH", depth_model_name)
        self.depth_encoder_type = depth_encoder_type
        if self.depth_encoder_type == "da3":
            self.da3_model = DepthAnything3.from_pretrained(depth_model_name)
            self.da3_model.eval()
            embed_dim = self.da3_model.model.backbone.pretrained.embed_dim
        elif self.depth_encoder_type == "dinov2_base":
            self.dinov2_model = AutoModel.from_pretrained(
                depth_model_name,
                local_files_only=True,
                trust_remote_code=False,
            )
            if self.dinov2_model.config.model_type != "dinov2":
                raise ValueError("depth_encoder_type='dinov2_base' requires a DINOv2 checkpoint")
            self.dinov2_model.eval()
            embed_dim = self.dinov2_model.config.hidden_size
        else:
            raise ValueError("depth_encoder_type must be 'da3' or 'dinov2_base'")

        if freeze_depth_model:
            self._freeze_depth_model()

        self.token_merging_model = TokenMerging2D(patch_size=4, in_dim=embed_dim, out_dim=feature_dim)
        self.feature_dim = feature_dim
        self.freeze_depth_model = freeze_depth_model

        # ImageNet normalization — kept as buffers so they move with the module
        self.register_buffer("img_mean", torch.tensor(_IMAGENET_MEAN).view(3, 1, 1))
        self.register_buffer("img_std", torch.tensor(_IMAGENET_STD).view(3, 1, 1))

        # Compile sub-modules individually so they benefit from kernel fusion even though
        # the outer forward() is dynamo-disabled (see below).
        #
        # da3_model: use mode="default" (NOT reduce-overhead). DA3's RoPE calls
        # int(tensor.max()), which is a graph break. "default" mode splits the graph
        # around breaks and compiles the compile-friendly segments (ViT attention, FFN,
        # etc.) — the large majority of compute. "reduce-overhead"/CUDA-graphs requires
        # a fully break-free graph and would fall back to eager entirely.
        #
        # token_merging_model: a small learnable adapter with no graph breaks — gets
        # full kernel-fusion benefit from "default" mode.
        # self.da3_model = torch.compile(self.da3_model, mode="default", dynamic=False)
        self.token_merging_model = torch.compile(self.token_merging_model, mode="default", dynamic=False)

        # Disable tracing of this forward() from the outer compiled backbone.
        # DA3's for-loop over a tuple from the (now compiled) da3_model can still
        # confuse dynamo's shape tracking when seen from outside. The boundary here
        # ensures the outer reduce-overhead backbone captures a clean CUDA graph while
        # the sub-modules compile independently.
        self.forward = torch.compiler.disable(self.forward)

    # ------------------------------------------------------------------
    # Weight management helpers
    # ------------------------------------------------------------------

    def _encoder_model(self):
        return self.da3_model if self.depth_encoder_type == "da3" else self.dinov2_model

    def _freeze_depth_model(self):
        for param in self._encoder_model().parameters():
            param.requires_grad = False
        self._encoder_model().eval()

    def freeze_unused_weight(self):
        self._freeze_depth_model()

    def unfreeze_depth_model(self):
        for param in self._encoder_model().parameters():
            param.requires_grad = True
        self._encoder_model().train()

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def _extract_layer_features(self, output, batch_size: int):
        """Extract and reshape per-layer features from DA3 output.

        Args:
            output: DA3 forward output with aux dict.
            batch_size: number of images in the batch.

        Returns:
            Tuple of 4 tensors, each [B, N_patches, embed_dim].
        """
        if self.depth_encoder_type == "dinov2_base":
            if output.hidden_states is None:
                raise RuntimeError("DINOv2 did not return hidden states")
            return tuple(output.hidden_states[layer_idx + 1][:, 1:, :].clone() for layer_idx in _FEAT_LAYERS)

        result = []
        for layer_idx in _FEAT_LAYERS:
            feat = output.aux[f"feat_layer_{layer_idx}"]  # [B, 1, H, W, C]
            feat = feat.squeeze(1)  # [B, H, W, C]
            feat = feat.reshape(batch_size, -1, feat.shape[-1])  # [B, N_patches, C]
            # DA3 uses torch.inference_mode internally; inference tensors cannot be
            # saved for backward by AOT Autograd (needed for token_merging_model
            # parameter gradients). clone() is the documented escape from
            # inference mode — detach() does NOT escape it.
            result.append(feat.clone())
        return tuple(result)

    def forward(self, images):
        """
        images: [B, C, H, W]  (range [-1, 1])
        Returns: tuple of 4 tensors each [B, N_merged, feature_dim].

        Processes the full batch in a single DA3 forward pass rather than
        iterating per-sample, which is faster and avoids Python-loop overhead.
        """
        batch_size = images.shape[0]
        x = (images + 1.0) * 0.5  # [-1, 1] → [0, 1]
        x = (x - self.img_mean) / self.img_std  # ImageNet normalize
        with torch.no_grad():
            if self.depth_encoder_type == "da3":
                output = self.da3_model.forward(x.unsqueeze(1), export_feat_layers=_FEAT_LAYERS)
            else:
                output = self.dinov2_model(pixel_values=x, output_hidden_states=True, return_dict=True)

        layer_features = self._extract_layer_features(output, batch_size)

        merged_features = []
        for features in layer_features:
            merged = self.token_merging_model(features)
            merged_features.append(merged)

        return tuple(merged_features)

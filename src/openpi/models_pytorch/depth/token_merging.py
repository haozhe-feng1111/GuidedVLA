# GuidedVLA addition: strided-conv token merging to downsample depth feature maps before cross-attention.
# Paper: "GuidedVLA: Specifying Task-Relevant Factors via Plug-and-Play Action Attention Specialization" (RSS 2026)
import torch
import torch.nn as nn


class TokenMerging2D(nn.Module):
    """Merge spatial patch tokens with a strided 2-D convolution.

    Accepts a flat sequence of square-grid patch tokens and applies a
    patch_size x patch_size strided convolution to reduce the sequence
    length while projecting to a higher feature dimension.

    Args:
        patch_size: Spatial stride (and kernel size) of the merge convolution.
        in_dim:     Input feature dimension per patch.
        out_dim:    Output feature dimension after merging.
    """

    def __init__(self, patch_size: int = 8, in_dim: int = 384, out_dim: int = 1024):
        super().__init__()
        self.patch_size = patch_size
        self.merge = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, in_dim]  — N must be a perfect square (e.g. 37x37 = 1369).

        Returns:
            [B, N', out_dim]  where N' = (sqrt(N) // patch_size) ** 2.
        """
        # DepthAnything3.forward is wrapped in torch.inference_mode(), so frozen
        # backbone features can arrive here as inference tensors. LayerNorm /
        # Conv2d in this trainable adapter need normal tensors because autograd
        # saves activations for backward when updating adapter weights.
        if x.is_inference():
            x = x.clone()
        x = self.norm(x)
        batch_size, num_tokens, feature_dim = x.shape
        hw = int(num_tokens**0.5)
        x = x.view(batch_size, hw, hw, feature_dim).permute(0, 3, 1, 2)  # [B, D, hw, hw]
        x = self.merge(x)  # [B, out_dim, hw//ps, hw//ps]
        return x.flatten(2).transpose(1, 2)  # [B, N', out_dim]

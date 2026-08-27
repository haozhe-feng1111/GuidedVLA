# GuidedVLA addition: depth cross-attention module that injects geometry features into the action expert.
# Paper: "GuidedVLA: Specifying Task-Relevant Factors via Plug-and-Play Action Attention Specialization" (RSS 2026)
import dataclasses

import torch
import torch.nn as nn


@dataclasses.dataclass
class DepthHeadConfig:
    """Per-layer depth K/V tokens injected into H_depth cross-attention."""

    depth_token_k: torch.Tensor  # [B, num_heads, num_depth_tokens, head_dim]
    depth_token_v: torch.Tensor  # [B, num_heads, num_depth_tokens, head_dim]
    depth_head_indices: tuple[int, ...]


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, cond_dim: int | None = None):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.cond_dim = cond_dim

        # Dense layer for adaptive normalization (if cond_dim is provided)
        if cond_dim is not None:
            # self.dense = nn.Linear(cond_dim, dim * 3, bias=True, dtype=torch.bfloat16)
            self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
            # Initialize with zeros (matches source implementation)
            nn.init.zeros_(self.dense.weight)
        else:
            self.weight = nn.Parameter(torch.zeros(dim, dtype=torch.bfloat16))
            self.dense = None

    def _norm(self, x):
        # Compute variance in float32 (like the source implementation)
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        # Compute normalization in float32
        return x * torch.rsqrt(var + self.eps)

    def forward(self, x, cond=None):
        dtype = x.dtype  # original dtype, could be half-precision
        normed_inputs = self._norm(x)

        if cond is None or self.dense is None:
            # regular RMSNorm
            # scale by learned parameter in float32 (matches source implementation)
            normed_inputs = normed_inputs * (1.0 + self.weight.float())
            return normed_inputs.to(dtype), None  # return in original dtype with None gate

        # adaptive RMSNorm (if cond is provided and dense layer exists)
        if cond.shape[-1] != self.cond_dim:
            raise ValueError(f"Expected cond dimension {self.cond_dim}, got {cond.shape[-1]}")

        # self.dense.to(dtype=torch.bfloat16).to(dtype=torch.float32)
        modulation = self.dense(cond)
        # Reshape modulation to broadcast properly: [batch, 1, features] for [batch, seq, features]
        if len(x.shape) == 3:  # [batch, seq, features]
            modulation = modulation.unsqueeze(1)

        scale, shift, gate = torch.chunk(modulation, 3, dim=-1)

        # Apply adaptive normalization: use model weight dtype to ensure compatibility
        # model_dtype = self.dense.weight.dtype  # Use the model's dtype (bfloat16)
        # scale = scale.to(model_dtype)
        # shift = shift.to(model_dtype)
        # gate = gate.to(model_dtype)
        # normed_inputs = normed_inputs.to(model_dtype)  # Convert normed_inputs to model dtype

        normed_inputs = normed_inputs * (1 + scale.to(torch.float32)) + shift.to(torch.float32)

        return normed_inputs.to(dtype), gate.to(dtype)

    def extra_repr(self):
        repr_str = f"{tuple(self.weight.shape)}, eps={self.eps}"
        if self.dense is not None:
            repr_str += f", adaptive=True, cond_dim={self.cond_dim}"
        return repr_str


class DepthTokenKVProjector(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        num_groups: int,
        depth_head_indices: list[int],
        headwise_xavier_init: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_groups = num_groups
        self.total_kv_dim = num_heads * head_dim

        self.norm = GemmaRMSNorm(dim=hidden_size, eps=1e-6)

        self.k_projectors = nn.ModuleList([nn.Linear(hidden_size, self.total_kv_dim) for _ in range(num_groups)])
        self.v_projectors = nn.ModuleList([nn.Linear(hidden_size, self.total_kv_dim) for _ in range(num_groups)])
        self.depth_head_indices = depth_head_indices
        self.headwise_xavier_init = headwise_xavier_init

        self._init_weights()

    def _init_weights(self):
        for i in range(self.num_groups):
            if self.headwise_xavier_init:
                for head_weight in self.k_projectors[i].weight.view(self.num_heads, self.head_dim, self.hidden_size):
                    nn.init.xavier_uniform_(head_weight)
                for head_weight in self.v_projectors[i].weight.view(self.num_heads, self.head_dim, self.hidden_size):
                    nn.init.xavier_uniform_(head_weight)
            else:
                nn.init.xavier_uniform_(self.k_projectors[i].weight)
                nn.init.xavier_uniform_(self.v_projectors[i].weight)
            nn.init.zeros_(self.k_projectors[i].bias)
            nn.init.zeros_(self.v_projectors[i].bias)

    def forward(self, depth_tokens_tuple: tuple) -> list[DepthHeadConfig]:
        assert (
            len(depth_tokens_tuple) == self.num_groups
        ), f"Expected {self.num_groups} depth token groups, got {len(depth_tokens_tuple)}"

        kv_configs = []

        for i in range(self.num_groups):
            depth_tokens = depth_tokens_tuple[i]
            depth_tokens, _ = self.norm(depth_tokens)

            batch_size = depth_tokens.shape[0]
            num_depth_tokens = depth_tokens.shape[1]

            depth_token_k = self.k_projectors[i](depth_tokens)
            depth_token_v = self.v_projectors[i](depth_tokens)

            depth_token_k = depth_token_k.view(batch_size, num_depth_tokens, self.num_heads, self.head_dim).transpose(
                1, 2
            )

            depth_token_v = depth_token_v.view(batch_size, num_depth_tokens, self.num_heads, self.head_dim).transpose(
                1, 2
            )

            kv_configs.append(
                DepthHeadConfig(
                    depth_token_k=depth_token_k,
                    depth_token_v=depth_token_v,
                    depth_head_indices=tuple(self.depth_head_indices),
                )
            )

        return kv_configs


def prepare_guided_attention_config(
    layer_idx: int,
    guided_layer_indices: list[int],
    depth_kv: list[DepthHeadConfig],
) -> DepthHeadConfig | None:
    if layer_idx not in guided_layer_indices:
        return None
    position = guided_layer_indices.index(layer_idx)
    return depth_kv[position]


__all__ = [
    "DepthHeadConfig",
    "DepthTokenKVProjector",
    "GemmaRMSNorm",
    "prepare_guided_attention_config",
]

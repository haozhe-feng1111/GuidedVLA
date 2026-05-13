# GuidedVLA addition: ControlNet-style dual-branch attention for plug-and-play action head specialization.
# Paper: "GuidedVLA: Specifying Task-Relevant Factors via Plug-and-Play Action Attention Specialization" (RSS 2026)
"""
ControlNet-inspired Attention Module for Pi0 Model.

Architecture:
- Origin branch: pretrained PaliGemma/Gemma attention (optionally frozen)
- Control branch: lightweight branch with num_control_heads heads + optional headwise gate
- Fusion: zero_conv (ControlNet design) — origin + zero_conv(branch_output)

The headwise gate is predicted from the control branch's Q projection,
then applied to the control branch SDPA output before o_proj.
"""

from collections.abc import Sequence
import copy
import logging

import torch
from torch import nn


class ControlAwareAttention(nn.Module):
    """
    ControlNet-style attention wrapper with trainable origin and control branches.

    Fusion mode: zero_conv — output = origin_output + zero_conv(branch_output)
    The zero_conv is zero-initialized so the branch starts with zero contribution,
    matching the standard ControlNet initialization strategy.
    """

    @staticmethod
    def _init_linear(linear: nn.Linear, *, std: float):
        nn.init.normal_(linear.weight, mean=0.0, std=std)
        if linear.bias is not None:
            nn.init.zeros_(linear.bias)

    @classmethod
    def _replace_q_proj_with_headwise_gate(
        cls,
        object_branch,
        *,
        hidden_size: int,
        query_size: int,
        gate_num_heads: int,
        device,
        dtype,
        initializer_std: float,
        copy_query_from: nn.Linear | None = None,
    ):
        old_q_proj = object_branch.q_proj
        gated_q_proj = nn.Linear(
            hidden_size,
            query_size + gate_num_heads,
            bias=old_q_proj.bias is not None,
            device=device,
            dtype=dtype,
        )

        if copy_query_from is None:
            cls._init_linear(gated_q_proj, std=initializer_std)
        else:
            with torch.no_grad():
                gated_q_proj.weight[:query_size].copy_(copy_query_from.weight[:query_size])
                nn.init.normal_(gated_q_proj.weight[query_size:], mean=0.0, std=initializer_std)
                if gated_q_proj.bias is not None:
                    gated_q_proj.bias[:query_size].copy_(copy_query_from.bias[:query_size])
                    nn.init.zeros_(gated_q_proj.bias[query_size:])

        object_branch.q_proj = gated_q_proj

    def __init__(
        self,
        original_attn,
        hidden_size: int,
        *,
        num_control_heads: int = 2,
        copy_weights: bool = False,
        freeze_origin: bool = False,
        use_headwise_gate: bool | None = None,
    ):
        super().__init__()

        # Get device/dtype from original attention layer
        device = next(original_attn.parameters()).device
        dtype = next(original_attn.parameters()).dtype

        # Origin branch — participates in training with main_loss gradients
        self.origin = original_attn

        # Store configuration
        self.config = original_attn.config
        self.layer_idx = original_attn.layer_idx
        self.num_control_heads = num_control_heads
        self.copy_weights = copy_weights
        self.freeze_origin = freeze_origin

        # Get original attention dimensions
        num_heads = self.config.num_attention_heads
        head_dim = original_attn.head_dim
        initializer_std = float(getattr(self.config, "initializer_range", 0.02))
        if use_headwise_gate is None:
            use_headwise_gate = True
        self.use_headwise_gate = bool(use_headwise_gate)
        self.gate_num_heads = num_heads

        # Determine control branch dimensions
        if num_control_heads is None or copy_weights:
            # Full copy mode: copy all weights from original
            self.object_branch = copy.deepcopy(original_attn).to(device=device, dtype=dtype)
            self.num_control_heads = num_heads
            control_hidden_size = self.num_control_heads * head_dim
            if self.use_headwise_gate:
                self._replace_q_proj_with_headwise_gate(
                    self.object_branch,
                    hidden_size=hidden_size,
                    query_size=control_hidden_size,
                    gate_num_heads=self.gate_num_heads,
                    device=device,
                    dtype=dtype,
                    initializer_std=initializer_std,
                    copy_query_from=self.object_branch.q_proj,
                )
        else:
            # Clone config with reduced head count
            control_config = copy.deepcopy(self.config)
            control_config.num_attention_heads = num_control_heads

            # Use the same attention class as the original
            attention_class = type(original_attn)
            self.object_branch = attention_class(control_config, layer_idx=self.layer_idx).to(
                device=device, dtype=dtype
            )

            control_hidden_size = num_control_heads * head_dim  # e.g., 2 * 256 = 512

            # Optional headwise gate on the control branch Q projection.
            # Gate dims equal origin num_heads (not control heads) so each origin
            # head gets its own gate scalar.
            if self.use_headwise_gate:
                self._replace_q_proj_with_headwise_gate(
                    self.object_branch,
                    hidden_size=hidden_size,
                    query_size=control_hidden_size,
                    gate_num_heads=self.gate_num_heads,
                    device=device,
                    dtype=dtype,
                    initializer_std=initializer_std,
                )

            # Re-initialize control branch with small random values;
            # projection biases are zero, so gate logits are centered near 0.
            for name, param in self.object_branch.named_parameters():
                if "proj" in name and param.ndim == 2:
                    nn.init.normal_(param, mean=0.0, std=initializer_std)
                elif "proj" in name and param.ndim == 1:
                    nn.init.zeros_(param)

        # Ensure control branch parameters are trainable
        for param in self.object_branch.parameters():
            param.requires_grad = True
        if self.freeze_origin:
            for param in self.origin.parameters():
                param.requires_grad = False

        # Zero-initialized linear projection for ControlNet-style fusion
        self.zero_conv = nn.Linear(hidden_size, hidden_size, bias=True, device=device, dtype=dtype)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)

        # Q-head expansion layer: expand control branch Q heads to match origin heads
        # Only needed when control has fewer heads than origin (e.g., 2 heads -> 8 heads)
        origin_num_heads = num_heads
        control_q_heads = self.num_control_heads if self.num_control_heads is not None else num_heads

        if control_q_heads != origin_num_heads:
            control_hidden_dim = control_q_heads * self.head_dim
            origin_hidden_dim = origin_num_heads * self.head_dim
            self.q_expand_linear = nn.Linear(
                control_hidden_dim, origin_hidden_dim, bias=True, device=device, dtype=dtype
            )
            nn.init.normal_(self.q_expand_linear.weight, mean=0.0, std=initializer_std)
            nn.init.zeros_(self.q_expand_linear.bias)

            # Re-initialize the o_proj of the control branch to match expanded head count
            self.object_branch.o_proj = nn.Linear(
                origin_hidden_dim,
                self.config.hidden_size,
                bias=self.object_branch.o_proj.bias is not None,
                device=device,
                dtype=dtype,
            )
            nn.init.normal_(self.object_branch.o_proj.weight, mean=0.0, std=initializer_std)
            if self.object_branch.o_proj.bias is not None:
                nn.init.zeros_(self.object_branch.o_proj.bias)

            self.has_q_expansion = True
        else:
            self.q_expand_linear = None
            self.has_q_expansion = False

        mode_str = "copy" if copy_weights or num_control_heads is None else f"{num_control_heads}heads"
        freeze_str = ", frozen_origin" if self.freeze_origin else ""
        logging.info(f"Created ControlAwareAttention for layer {self.layer_idx} [{mode_str}, zero_conv{freeze_str}]")

    # Compatibility properties for transformers library (delegates to control branch)
    @property
    def q_proj(self):
        return self.object_branch.q_proj

    @property
    def k_proj(self):
        return self.object_branch.k_proj

    @property
    def v_proj(self):
        return self.object_branch.v_proj

    @property
    def o_proj(self):
        return self.object_branch.o_proj

    @property
    def head_dim(self):
        return self.origin.head_dim

    @property
    def scaling(self):
        return self.object_branch.scaling

    def get_q_expand_linear(self):
        """Get Q expansion linear layer if available (for expanding 2->8 heads)."""
        return self.q_expand_linear if self.has_q_expansion else None

    # NOTE: forward() is NOT called in dual-path ControlNet mode
    def forward(self, *args, **kwargs):
        """Fallback to the origin attention for compatibility."""
        return self.origin(*args, **kwargs)

    def compute_dual_path_qkv(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor | None,  # q_branch_gate
    ]:
        """
        Compute Q/K/V for both Origin and Branch paths (True ControlNet architecture).

        Both paths can attend to shared context (e.g., PaliGemma KV) independently,
        then their outputs are fused: Final = origin_out + zero_conv(branch_out)

        Args:
            hidden_states: Input hidden states [batch, seq, hidden_dim]

        Returns:
            tuple: (
                (Q_origin, K_origin, V_origin),
                (Q_branch, K_branch, V_branch),
                q_branch_gate,  # [batch, gated_heads, seq, 1] or None
            )
            All Q/K/V tensors in shape [batch, heads, seq, head_dim]
        """
        input_shape = hidden_states.shape[:-1]

        # === Origin Path (pretrained weights, optionally frozen) ===
        q_origin_proj = self.origin.q_proj(hidden_states)
        k_origin_proj = self.origin.k_proj(hidden_states)
        v_origin_proj = self.origin.v_proj(hidden_states)

        q_origin = q_origin_proj.view(*input_shape, -1, self.head_dim).transpose(1, 2)
        k_origin = k_origin_proj.view(*input_shape, -1, self.head_dim).transpose(1, 2)
        v_origin = v_origin_proj.view(*input_shape, -1, self.head_dim).transpose(1, 2)

        # === Control Branch Path (trainable, lightweight) ===
        q_branch_proj = self.object_branch.q_proj(hidden_states)
        k_branch_proj = self.object_branch.k_proj(hidden_states)
        v_branch_proj = self.object_branch.v_proj(hidden_states)

        # Split off headwise gates if present
        has_headwise_gate = hasattr(self, "use_headwise_gate") and self.use_headwise_gate
        expected_q_size = self.num_control_heads * self.head_dim

        q_branch_gate = None
        if has_headwise_gate and q_branch_proj.shape[-1] > expected_q_size:
            # q_branch_proj: [batch, seq, control_heads*head_dim + gate_heads]
            q_branch_proj_query = q_branch_proj[..., :expected_q_size]
            q_branch_proj_gate = q_branch_proj[..., expected_q_size:]

            q_branch = q_branch_proj_query.view(*input_shape, self.num_control_heads, self.head_dim).transpose(1, 2)
            # Gate: [batch, seq, gate_heads] -> [batch, gate_heads, seq, 1]
            q_branch_gate = q_branch_proj_gate.view(*input_shape, self.gate_num_heads, 1).transpose(1, 2)
        else:
            q_branch = q_branch_proj.view(*input_shape, -1, self.head_dim).transpose(1, 2)

        k_branch = k_branch_proj.view(*input_shape, -1, self.head_dim).transpose(1, 2)
        v_branch = v_branch_proj.view(*input_shape, -1, self.head_dim).transpose(1, 2)

        return (q_origin, k_origin, v_origin), (q_branch, k_branch, v_branch), q_branch_gate

    def fuse_outputs(
        self,
        origin_output: torch.Tensor,
        branch_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse outputs from Origin and Branch paths using zero_conv fusion.

        y = origin_output + zero_conv(branch_output)

        zero_conv is zero-initialized, so the branch starts with zero contribution
        and gradually learns to contribute as training progresses.

        Args:
            origin_output: Output from origin attention path [batch, seq, hidden_dim]
            branch_output: Output from branch attention path [batch, seq, hidden_dim]

        Returns:
            Fused output tensor [batch, seq, hidden_dim]
        """
        return origin_output + self.zero_conv(branch_output)


def inject_control_attention(
    model,
    *,
    num_control_heads: int = 2,
    copy_weights: bool = False,
    freeze_origin: bool = False,
    layer_indices: Sequence[int] | None = None,
    use_headwise_gate: bool | None = None,
):
    """
    Replace action expert attention layers with ControlAwareAttention.

    Call this AFTER loading a pretrained checkpoint so that the origin branch
    retains the original pretrained weights unchanged (load-then-inject pattern).

    Args:
        model: PI0Pytorch model instance
        num_control_heads: Number of attention heads for control branch (default=2)
                          Use None for full copy mode (same head count as original)
        copy_weights: If True, copy weights from original; if False, random initialize
        freeze_origin: If True, freeze the original action-expert attention branch
        layer_indices: Optional subset of layer indices to replace (None = all layers)
        use_headwise_gate: Whether to add per-head sigmoid gate to control branch Q proj

    Returns:
        int: Number of layers replaced
    """
    mode_str = "copy" if copy_weights or num_control_heads is None else f"{num_control_heads}heads"
    freeze_str = ", frozen_origin" if freeze_origin else ""
    logging.info(f"Injecting ControlAwareAttention into action expert [{mode_str}, zero_conv{freeze_str}]")

    replaced_count = 0
    layer_indices_set = set(layer_indices) if layer_indices is not None else None

    expert_model = model.paligemma_with_expert.gemma_expert.model
    hidden_size = expert_model.config.hidden_size

    for _layer_idx, layer in enumerate(expert_model.layers):
        if layer_indices_set is not None and _layer_idx not in layer_indices_set:
            continue
        layer.self_attn = ControlAwareAttention(
            original_attn=layer.self_attn,
            hidden_size=hidden_size,
            num_control_heads=num_control_heads,
            copy_weights=copy_weights,
            freeze_origin=freeze_origin,
            use_headwise_gate=use_headwise_gate,
        )
        replaced_count += 1

    logging.info(f"Total {replaced_count} expert layers replaced [{mode_str}, zero_conv{freeze_str}]")
    return replaced_count


def get_trainable_control_params(model):
    """
    Get trainable parameters from ControlAwareAttention modules.

    Returns:
        list: Control branch parameters + zero_conv fusion parameters
    """
    control_params = []

    for _name, module in model.named_modules():
        if isinstance(module, ControlAwareAttention):
            control_params.extend(module.object_branch.parameters())
            if module.q_expand_linear is not None:
                control_params.extend(module.q_expand_linear.parameters())
            control_params.append(module.zero_conv.weight)
            control_params.append(module.zero_conv.bias)

    return control_params

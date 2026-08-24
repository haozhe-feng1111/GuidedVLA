"""TopK + AuxK sparse autoencoder used for offline VLA activation analysis."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SAEConfig:
    input_dim: int
    num_features: int
    topk: int
    auxk: int = 512
    aux_loss_coefficient: float = 1 / 32
    dead_steps: int = 500
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.num_features <= 0:
            raise ValueError("input_dim and num_features must be positive")
        if not 1 <= self.topk <= self.num_features:
            raise ValueError("topk must be in [1, num_features]")
        if self.auxk < 0:
            raise ValueError("auxk must be non-negative")
        if self.dead_steps <= 0:
            raise ValueError("dead_steps must be positive")


@dataclass
class SAEOutput:
    normalized_input: torch.Tensor
    feature_activations: torch.Tensor
    reconstruction: torch.Tensor
    reconstruction_loss: torch.Tensor
    auxiliary_loss: torch.Tensor
    total_loss: torch.Tensor


def geometric_median(samples: torch.Tensor, *, max_iterations: int = 100, eps: float = 1e-6) -> torch.Tensor:
    """Compute a geometric median with Weiszfeld iterations on CPU or GPU."""
    if samples.ndim != 2 or samples.shape[0] == 0:
        raise ValueError("samples must have shape [num_samples, input_dim]")
    estimate = samples.mean(dim=0)
    for _ in range(max_iterations):
        distances = torch.linalg.vector_norm(samples - estimate, dim=1).clamp_min(eps)
        updated = (samples / distances[:, None]).sum(dim=0) / (1 / distances).sum()
        if torch.linalg.vector_norm(updated - estimate) <= eps:
            return updated
        estimate = updated
    return estimate


class TopKAuxKSAE(nn.Module):
    """Bias-free TopK SAE with unit-norm decoder columns and dead-latent AuxK."""

    def __init__(self, config: SAEConfig, *, pre_bias: torch.Tensor | None = None) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.Linear(config.input_dim, config.num_features, bias=False)
        self.decoder = nn.Linear(config.num_features, config.input_dim, bias=False)
        self.register_buffer("pre_bias", torch.zeros(config.input_dim))
        self.register_buffer("steps_since_active", torch.full((config.num_features,), config.dead_steps))
        if pre_bias is not None:
            if pre_bias.shape != (config.input_dim,):
                raise ValueError(f"pre_bias must have shape {(config.input_dim,)}")
            self.pre_bias.copy_(pre_bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.decoder.weight, std=1 / self.config.input_dim**0.5)
        self.normalize_decoder_columns()
        with torch.no_grad():
            scale = (self.config.topk / self.config.num_features) ** 0.5
            self.encoder.weight.copy_(self.decoder.weight.T * scale)

    def normalize_input(self, inputs: torch.Tensor) -> torch.Tensor:
        centered = inputs - self.pre_bias
        centered = centered - centered.mean(dim=-1, keepdim=True)
        return centered / torch.linalg.vector_norm(centered, dim=-1, keepdim=True).clamp_min(self.config.eps)

    def encode(self, normalized_inputs: torch.Tensor, *, update_activity: bool = False) -> torch.Tensor:
        preactivations = self.encoder(normalized_inputs)
        values, indices = torch.topk(preactivations, self.config.topk, dim=-1)
        features = torch.zeros_like(preactivations)
        features.scatter_(-1, indices, F.relu(values))
        if update_activity:
            with torch.no_grad():
                active = features.gt(0).any(dim=0)
                self.steps_since_active.add_(1)
                self.steps_since_active.masked_fill_(active, 0)
        return features

    def decode(self, feature_activations: torch.Tensor) -> torch.Tensor:
        return self.decoder(feature_activations)

    def forward(self, inputs: torch.Tensor, *, update_activity: bool = True) -> SAEOutput:
        normalized = self.normalize_input(inputs)
        features = self.encode(normalized, update_activity=update_activity)
        reconstruction = self.decode(features)
        reconstruction_loss = (normalized - reconstruction).square().sum(dim=-1).mean()
        auxiliary_loss = self._auxiliary_loss(normalized - reconstruction)
        total_loss = reconstruction_loss + self.config.aux_loss_coefficient * auxiliary_loss
        return SAEOutput(
            normalized_input=normalized,
            feature_activations=features,
            reconstruction=reconstruction,
            reconstruction_loss=reconstruction_loss,
            auxiliary_loss=auxiliary_loss,
            total_loss=total_loss,
        )

    def _auxiliary_loss(self, residual: torch.Tensor) -> torch.Tensor:
        dead = self.steps_since_active >= self.config.dead_steps
        num_dead = int(dead.sum())
        auxk = min(self.config.auxk, num_dead)
        if auxk == 0:
            return residual.new_zeros(())
        dead_indices = dead.nonzero(as_tuple=False).flatten()
        preactivations = self.encoder(residual)[:, dead_indices]
        values, local_indices = torch.topk(preactivations, auxk, dim=-1)
        values = F.relu(values)
        selected_indices = dead_indices[local_indices]
        auxiliary_features = torch.zeros(
            residual.shape[0], self.config.num_features, dtype=residual.dtype, device=residual.device
        )
        auxiliary_features.scatter_(1, selected_indices, values)
        aux_reconstruction = self.decode(auxiliary_features)
        return (residual.detach() - aux_reconstruction).square().sum(dim=-1).mean()

    @torch.no_grad()
    def normalize_decoder_columns(self) -> None:
        self.decoder.weight.div_(torch.linalg.vector_norm(self.decoder.weight, dim=0, keepdim=True).clamp_min(1e-8))

    @torch.no_grad()
    def project_decoder_gradients(self) -> None:
        gradient = self.decoder.weight.grad
        if gradient is None:
            return
        columns = self.decoder.weight
        gradient.sub_(columns * (columns * gradient).sum(dim=0, keepdim=True))

    def checkpoint(self) -> dict[str, object]:
        return {"schema_version": "guidedvla-topk-auxk-sae-v1", "config": asdict(self.config), "state_dict": self.state_dict()}


def load_sae_checkpoint(checkpoint: dict[str, object], *, map_location: str | torch.device = "cpu") -> TopKAuxKSAE:
    if checkpoint.get("schema_version") != "guidedvla-topk-auxk-sae-v1":
        raise ValueError("Unsupported SAE checkpoint schema")
    config = SAEConfig(**checkpoint["config"])
    model = TopKAuxKSAE(config).to(map_location)
    model.load_state_dict(checkpoint["state_dict"])
    return model

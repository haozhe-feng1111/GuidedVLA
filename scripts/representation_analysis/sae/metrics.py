"""Episode-level activation statistics from Swann et al. (2026)."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FeatureMetrics:
    episode_coverage: torch.Tensor
    mean_onset_count: torch.Tensor
    mean_activation_magnitude: torch.Tensor
    relative_run_length: torch.Tensor
    active_episode_count: torch.Tensor

    def matrix(self) -> torch.Tensor:
        return torch.stack(
            (
                self.episode_coverage,
                self.mean_onset_count,
                self.mean_activation_magnitude,
                self.relative_run_length,
            ),
            dim=1,
        )


def _active_state(values: torch.Tensor, threshold: float) -> torch.Tensor:
    """Apply the paper's hysteresis: on above threshold, off at exact zero."""
    state = torch.zeros_like(values, dtype=torch.bool)
    current = torch.zeros(values.shape[1], dtype=torch.bool, device=values.device)
    for timestep in range(values.shape[0]):
        current = torch.where(values[timestep] > threshold, True, torch.where(values[timestep] == 0, False, current))
        state[timestep] = current
    return state


def compute_feature_metrics(
    feature_activations: torch.Tensor,
    episode_ids: torch.Tensor,
    *,
    onset_threshold: float = 0.1,
) -> FeatureMetrics:
    """Compute metrics over complete, time-ordered episodes.

    Rows must be sorted by episode and timestep; every episode must occupy one
    contiguous run. Only episodes containing a feature contribute to its onset,
    magnitude, and run-length means.
    """
    if feature_activations.ndim != 2:
        raise ValueError("feature_activations must have shape [timesteps, features]")
    if episode_ids.ndim != 1 or episode_ids.shape[0] != feature_activations.shape[0]:
        raise ValueError("episode_ids must have shape [timesteps]")
    if feature_activations.shape[0] == 0:
        raise ValueError("at least one timestep is required")
    if torch.any(feature_activations < 0):
        raise ValueError("SAE feature activations must be non-negative")
    unique_consecutive = torch.unique_consecutive(episode_ids)
    if unique_consecutive.numel() != torch.unique(episode_ids).numel():
        raise ValueError("each episode must occupy one contiguous run")

    num_features = feature_activations.shape[1]
    active_count = torch.zeros(num_features, dtype=torch.long, device=feature_activations.device)
    onset_sum = torch.zeros(num_features, dtype=feature_activations.dtype, device=feature_activations.device)
    magnitude_sum = torch.zeros_like(onset_sum)
    relative_run_sum = torch.zeros_like(onset_sum)

    boundaries = torch.cat(
        (
            torch.tensor([0], device=episode_ids.device),
            (episode_ids[1:] != episode_ids[:-1]).nonzero(as_tuple=False).flatten() + 1,
            torch.tensor([episode_ids.numel()], device=episode_ids.device),
        )
    )
    for start, end in zip(boundaries[:-1].tolist(), boundaries[1:].tolist(), strict=True):
        values = feature_activations[start:end]
        states = _active_state(values, onset_threshold)
        active = states.any(dim=0)
        previous = torch.cat((torch.zeros_like(states[:1]), states[:-1]), dim=0)
        onsets = (states & ~previous).sum(dim=0)
        active_timesteps = states.sum(dim=0)
        active_count += active.long()
        onset_sum += onsets
        magnitude_sum += values.max(dim=0).values * active
        relative_run_sum += torch.where(
            active,
            active_timesteps / onsets.clamp_min(1) / values.shape[0],
            torch.zeros_like(active_timesteps, dtype=values.dtype),
        )

    denominator = active_count.clamp_min(1).to(feature_activations.dtype)
    return FeatureMetrics(
        episode_coverage=active_count / unique_consecutive.numel(),
        mean_onset_count=onset_sum / denominator,
        mean_activation_magnitude=magnitude_sum / denominator,
        relative_run_length=relative_run_sum / denominator,
        active_episode_count=active_count,
    )

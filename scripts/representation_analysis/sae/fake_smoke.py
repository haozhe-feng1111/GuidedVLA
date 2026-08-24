#!/usr/bin/env python3
"""Train, save, reload, and evaluate an SAE on deterministic fake episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from metrics import compute_feature_metrics
from sae import geometric_median
from sae import load_sae_checkpoint
from sae import SAEConfig
from sae import TopKAuxKSAE


def make_fake_episodes(*, seed: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    num_episodes, episode_length, input_dim, source_features = 12, 40, 16, 8
    dictionary = torch.randn(source_features, input_dim, generator=generator)
    dictionary = torch.nn.functional.normalize(dictionary, dim=1)
    latents = torch.zeros(num_episodes, episode_length, source_features)
    for episode in range(num_episodes):
        latents[episode, 8:11, 0] = 2.0  # General event feature.
        latents[episode, 25:28, 0] = 1.5
        latents[episode, :, 1] = 0.8 if episode < 2 else 0.0  # Memorized context feature.
        latents[episode, :, 2 + episode % 6] += 0.3
    noise = 0.01 * torch.randn(num_episodes, episode_length, input_dim, generator=generator)
    activations = (latents @ dictionary + noise).reshape(-1, input_dim)
    episode_ids = torch.arange(num_episodes).repeat_interleave(episode_length)
    return activations, episode_ids


def run_smoke(output_dir: Path, *, device: str, steps: int) -> dict[str, float | int | str]:
    torch.manual_seed(11)
    activations, episode_ids = make_fake_episodes()
    target = torch.device(device)
    training = activations.to(target)
    pre_bias = geometric_median(training[: min(400, len(training))])
    config = SAEConfig(input_dim=16, num_features=32, topk=4, auxk=8, dead_steps=25)
    model = TopKAuxKSAE(config, pre_bias=pre_bias).to(target)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    with torch.no_grad():
        initial_loss = float(model(training, update_activity=False).reconstruction_loss)
    generator = torch.Generator().manual_seed(13)
    for _ in range(steps):
        indices = torch.randint(len(training), (64,), generator=generator).to(target)
        output = model(training[indices])
        optimizer.zero_grad(set_to_none=True)
        output.total_loss.backward()
        model.project_decoder_gradients()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.normalize_decoder_columns()

    model.eval()
    with torch.no_grad():
        encoded = model(training, update_activity=False)
        final_loss = float(encoded.reconstruction_loss)
    if not final_loss < initial_loss:
        raise RuntimeError(f"Fake SAE did not improve reconstruction: {initial_loss} -> {final_loss}")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "fake_sae.pt"
    torch.save(model.checkpoint(), checkpoint_path)
    restored = load_sae_checkpoint(torch.load(checkpoint_path, map_location=target, weights_only=True), map_location=target)
    restored.eval()
    with torch.no_grad():
        restored_inputs = activations.to(device=restored.pre_bias.device, dtype=restored.pre_bias.dtype)
        restored_features = restored(restored_inputs, update_activity=False).feature_activations.cpu()
    torch.testing.assert_close(restored_features, encoded.feature_activations.cpu())
    metrics = compute_feature_metrics(restored_features, episode_ids)

    report: dict[str, float | int | str] = {
        "schema_version": "guidedvla-sae-fake-smoke-v1",
        "device": str(target),
        "num_timesteps": len(activations),
        "num_episodes": int(torch.unique(episode_ids).numel()),
        "steps": steps,
        "initial_reconstruction_loss": initial_loss,
        "final_reconstruction_loss": final_loss,
        "active_dictionary_features": int((metrics.active_episode_count > 0).sum()),
        "checkpoint": str(checkpoint_path.resolve()),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    run_smoke(args.output_dir, device=args.device, steps=args.steps)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compute standard and episode-excluded CKNNA for PaliGemma and SigLIP streams."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from safetensors.torch import load_file
import torch
import torch.nn.functional as F

from compute_cknna import hsic_unbiased_batched


K_VALUES = (5, 10, 20)
POLICIES = ("nohead", "da3", "dinov2")
REFERENCES = ("dinov2", "da3")
STREAM_LAYERS = {"paligemma": 19, "siglip": 28}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    for shard in manifest["shards"]:
        if sha256_file(Path(shard["path"])) != shard["sha256"]:
            raise ValueError(f"Feature hash mismatch: {shard['path']}")
    return manifest


def allowed_mask(episode_ids: torch.Tensor | None, n: int, device: torch.device) -> torch.Tensor:
    if episode_ids is None:
        allowed = torch.ones(n, n, dtype=torch.bool, device=device)
    else:
        ids = episode_ids.to(device)
        allowed = ids[:, None] != ids[None, :]
    allowed.fill_diagonal_(False)
    return allowed


def topk_mask(kernel: torch.Tensor, topk: int, allowed: torch.Tensor) -> torch.Tensor:
    candidate_counts = allowed.sum(dim=-1)
    if int(candidate_counts.min()) < topk:
        raise ValueError(f"Insufficient allowed neighbours for k={topk}: min={int(candidate_counts.min())}")
    scores = kernel.masked_fill(~allowed, float("-inf"))
    indices = torch.topk(scores, topk, dim=-1).indices
    return torch.zeros_like(kernel).scatter_(-1, indices, 1.0)


def cknna(kernel_a: torch.Tensor, kernel_b: torch.Tensor, topk: int, allowed: torch.Tensor) -> torch.Tensor:
    mask_a = topk_mask(kernel_a, topk, allowed)
    mask_b = topk_mask(kernel_b, topk, allowed)
    intersection = mask_a * mask_b
    ab = hsic_unbiased_batched(intersection * kernel_a, intersection * kernel_b)
    aa = hsic_unbiased_batched(mask_a * kernel_a, mask_a * kernel_a)
    bb = hsic_unbiased_batched(mask_b * kernel_b, mask_b * kernel_b)
    return ab / (torch.sqrt(aa * bb) + 1e-6)


def episode_map(path: Path) -> tuple[dict[int, int], dict[int, int]]:
    sample_to_episode = {}
    sample_to_task = {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            sample_to_episode[int(row["sample_index"])] = int(row["episode_index"])
            sample_to_task[int(row["sample_index"])] = int(row["task_index"])
    if len(sample_to_episode) != 4000:
        raise ValueError("Expected 4,000 frozen sample identities")
    return sample_to_episode, sample_to_task


def feature_stack(data: dict[str, torch.Tensor], stream: str) -> list[torch.Tensor]:
    if stream == "paligemma":
        return [data["paligemma_features"][index] for index in range(19)]
    return [*[data["siglip_block_features"][index] for index in range(27)], data["siglip_projector_features"]]


def block_permutation(episode_ids: torch.Tensor, seed: int) -> torch.Tensor:
    unique = torch.unique_consecutive(episode_ids)
    if unique.numel() != 20:
        unique = torch.unique(episode_ids, sorted=True)
    groups = [torch.where(episode_ids == value)[0] for value in unique]
    if len(groups) != 20 or any(group.numel() != 5 for group in groups):
        raise ValueError("Episode-block null requires 20 episodes x 5 anchors")
    order = torch.randperm(20, generator=torch.Generator().manual_seed(seed))
    return torch.cat([groups[int(index)] for index in order])


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def aggregate(task_rows: list[dict], bootstrap_count: int) -> list[dict]:
    grouped = collections.defaultdict(list)
    for row in task_rows:
        key = (row["policy"], row["reference"], row["stream"], row["layer"], row["k"], row["mode"])
        grouped[key].append((row["task_index"], row["score"], row["null_score"]))
    indices = np.random.default_rng(20260822).integers(0, 40, size=(bootstrap_count, 40))
    output = []
    for key, values in sorted(grouped.items()):
        values = sorted(values)
        if len(values) != 40:
            raise ValueError(f"Expected 40 tasks for {key}, got {len(values)}")
        scores = np.asarray([value[1] for value in values])
        nulls = np.asarray([value[2] for value in values])
        means = scores[indices].mean(axis=1)
        low, high = np.quantile(means, [0.025, 0.975])
        output.append({
            "policy": key[0], "reference": key[1], "stream": key[2], "layer": key[3],
            "k": key[4], "mode": key[5], "mean_score": float(scores.mean()),
            "ci_low": float(low), "ci_high": float(high), "mean_episode_block_null": float(nulls.mean()),
            "num_tasks": 40,
        })
    return output


def paired_differences(task_rows: list[dict], bootstrap_count: int) -> list[dict]:
    lookup = {(r["policy"], r["reference"], r["stream"], r["layer"], r["k"], r["mode"], r["task_index"]): r["score"] for r in task_rows}
    indices = np.random.default_rng(20260822).integers(0, 40, size=(bootstrap_count, 40))
    rows = []
    for reference in REFERENCES:
        for stream, count in STREAM_LAYERS.items():
            for numerator, denominator in (("da3", "nohead"), ("dinov2", "nohead"), ("dinov2", "da3")):
                for layer in range(count):
                    values = np.asarray([lookup[(numerator, reference, stream, layer, 10, "episode_excluded", task)] - lookup[(denominator, reference, stream, layer, 10, "episode_excluded", task)] for task in range(40)])
                    boot = values[indices].mean(axis=1); low, high = np.quantile(boot, [0.025, 0.975])
                    rows.append({"reference": reference, "stream": stream, "comparison": f"{numerator}_minus_{denominator}", "layer": layer, "mean_difference": float(values.mean()), "ci_low": float(low), "ci_high": float(high), "ci_excludes_zero": bool(low > 0 or high < 0), "num_tasks": 40})
    return rows


def render(rows: list[dict], stream: str, output_dir: Path) -> None:
    selected = [row for row in rows if row["stream"] == stream and row["k"] == 10]
    lookup = {(r["policy"], r["reference"], r["mode"], r["layer"]): r for r in selected}
    titles = {"nohead": "No-head 30K", "da3": "DA3/Object/Skill 30K", "dinov2": "DINOv2/Object/Skill 30K"}
    labels = {"dinov2": "DINOv2-B layer 11", "da3": "DA3-SMALL layer 11"}
    count = STREAM_LAYERS[stream]
    x = np.arange(count)
    tick_labels = [str(i - 1) for i in range(count)] if stream == "paligemma" else [str(i) for i in range(27)] + ["P"]
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.6), sharex=True, sharey=True)
    y = [r["mean_score"] for r in selected]
    for ri, reference in enumerate(REFERENCES):
        for ci, policy in enumerate(POLICIES):
            ax = axes[ri, ci]
            for mode, style, name, color in (("episode_excluded", "-", "Episode-excluded", "#2f6df6"), ("standard", "--", "Standard", "#e07a24")):
                points = [lookup[(policy, reference, mode, layer)] for layer in range(count)]
                mean = np.asarray([p["mean_score"] for p in points]); low = np.asarray([p["ci_low"] for p in points]); high = np.asarray([p["ci_high"] for p in points])
                ax.plot(x, mean, style, color=color, linewidth=1.8, marker="o", markersize=2.8, label=name)
                ax.fill_between(x, low, high, color=color, alpha=0.10, linewidth=0)
            if ri == 0: ax.set_title(titles[policy], fontsize=11, fontweight="bold")
            if ci == 0: ax.set_ylabel(f"CKNNA vs. {labels[reference]}\n(task macro, k=10)")
            if ri == 1: ax.set_xlabel("PaliGemma visual-token layer" if stream == "paligemma" else "SigLIP vision block / projector")
            ticks = [0, 3, 6, 9, 12, 15, 18] if stream == "paligemma" else [0, 4, 8, 12, 16, 20, 24, 27]
            ax.set_xticks(ticks, [tick_labels[i] for i in ticks]); ax.grid(axis="y", linestyle=":", alpha=.6)
    axes[0, 2].legend(fontsize=9)
    for ax in axes.flat: ax.set_ylim(min(0.0, min(y) - .03), min(1.0, max(y) + .04))
    fig.suptitle(f"GuidedVLA {stream.capitalize()} Vision Alignment on Frozen Demonstrations", y=.995)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(output_dir / f"guidedvla_{stream}_cknna_2x3.png", dpi=220)
    fig.savefig(output_dir / f"guidedvla_{stream}_cknna_2x3.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    for policy in POLICIES: parser.add_argument(f"--{policy}-manifest", type=Path, required=True)
    for reference in REFERENCES: parser.add_argument(f"--{reference}-encoder-manifest", type=Path, required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_replicates != 2000: raise ValueError("Protocol requires 2,000 bootstrap replicates")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()): raise FileExistsError("Refusing to overwrite non-empty result directory")
    policies = {p: load_manifest(getattr(args, f"{p}_manifest")) for p in POLICIES}
    encoders = {r: load_manifest(getattr(args, f"{r}_encoder_manifest")) for r in REFERENCES}
    cache_hashes = {m["source_cache_manifest_sha256"] for m in [*policies.values(), *encoders.values()]}
    if len(cache_hashes) != 1: raise ValueError("Feature manifests do not share one frozen cache")
    sample_to_episode, sample_to_task = episode_map(args.sample_manifest)
    encoder_shards = {r: {s["task_index"]: s for s in m["shards"]} for r, m in encoders.items()}
    device = torch.device(args.device)
    task_rows = []
    for policy, manifest in policies.items():
        if manifest["extraction_scope"] != "full" or manifest["num_samples"] != 4000: raise ValueError(f"Incomplete {policy}")
        for shard in manifest["shards"]:
            task = shard["task_index"]; data = load_file(shard["path"]); indices = data["sample_index"].to(torch.int64)
            episodes = torch.tensor([sample_to_episode[int(i)] for i in indices], dtype=torch.int64)
            if any(sample_to_task[int(i)] != task for i in indices): raise ValueError("Sample/task mismatch")
            permutation = block_permutation(episodes, 20260821 + task).to(device)
            for reference in REFERENCES:
                ref_data = load_file(encoder_shards[reference][task]["path"])
                if not torch.equal(indices, ref_data["sample_index"]): raise ValueError("Policy/reference sample order mismatch")
                ref = F.normalize(ref_data["features"].to(device=device, dtype=torch.float32), dim=-1); ref_kernel = ref @ ref.T
                null_ref = ref[permutation]; null_kernel = null_ref @ null_ref.T
                for stream in STREAM_LAYERS:
                    for layer, feature in enumerate(feature_stack(data, stream)):
                        feature = F.normalize(feature.to(device=device, dtype=torch.float32), dim=-1); kernel = feature @ feature.T
                        for mode, episode_arg in (("standard", None), ("episode_excluded", episodes)):
                            allowed = allowed_mask(episode_arg, 100, device)
                            for k in K_VALUES:
                                score = float(cknna(kernel, ref_kernel, k, allowed))
                                null_score = float(cknna(kernel, null_kernel, k, allowed))
                                task_rows.append({"policy": policy, "reference": reference, "stream": stream, "task_index": task, "suite": shard["suite"], "layer": layer, "k": k, "mode": mode, "score": score, "null_score": null_score})
    expected = 3*2*40*(19+28)*3*2
    if len(task_rows) != expected: raise RuntimeError(f"Expected {expected} rows, got {len(task_rows)}")
    aggregate_rows = aggregate(task_rows, args.bootstrap_replicates)
    paired_rows = paired_differences(task_rows, args.bootstrap_replicates)
    write_csv(args.output_dir / "task_scores.csv", task_rows); write_csv(args.output_dir / "aggregate_scores.csv", aggregate_rows); write_csv(args.output_dir / "paired_layer_differences.csv", paired_rows)
    render(aggregate_rows, "paligemma", args.output_dir); render(aggregate_rows, "siglip", args.output_dir)
    primary = [r for r in aggregate_rows if r["k"] == 10 and r["mode"] == "episode_excluded"]
    null = [r["mean_episode_block_null"] for r in primary]
    summary = {"schema_version": "guidedvla-vision-cknna-results-v1", "primary_k": 10, "primary_mode": "episode_excluded", "bootstrap_unit": "task", "bootstrap_replicates": 2000, "bootstrap_seed": 20260822, "episode_block_null_seed_base": 20260821, "mean_primary_score": float(np.mean([r["mean_score"] for r in primary])), "mean_episode_block_null": float(np.mean(null)), "num_task_rows": len(task_rows), "num_aggregate_rows": len(aggregate_rows), "num_paired_rows": len(paired_rows)}
    (args.output_dir / "results_manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__": main()

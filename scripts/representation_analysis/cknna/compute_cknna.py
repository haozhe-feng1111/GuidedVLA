#!/usr/bin/env python3
"""Compute task-macro CKNNA and render the preregistered 2x3 figure."""

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


TIMESTEP_LABELS = ("t≈0 (0.001)", "t=0.25", "t=0.5")
K_VALUES = (5, 10, 20)
NUM_TIMESTEPS = 3
NUM_REPLICATES = 3
NUM_LAYERS = 18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nohead-policy-manifest", type=Path, required=True)
    parser.add_argument("--da3-policy-manifest", type=Path, required=True)
    parser.add_argument("--dinov2-policy-manifest", type=Path, required=True)
    parser.add_argument("--da3-encoder-manifest", type=Path, required=True)
    parser.add_argument("--dinov2-encoder-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    for shard in manifest["shards"]:
        if sha256_file(Path(shard["path"])) != shard["sha256"]:
            raise ValueError(f"Feature shard hash mismatch: {shard['path']}")
    return manifest


def validate_input_manifests(policies: dict[str, dict], encoders: dict[str, dict]) -> None:
    expected_tasks = set(range(40))
    for name, manifest in policies.items():
        assert manifest["extraction_scope"] == "full", name
        assert manifest["num_samples"] == 4000 and manifest["num_shards"] == 40, name
        assert manifest["timesteps"] == [0.001, 0.25, 0.5], name
        assert manifest["noise_replicates"] == 3 and manifest["num_layers"] == 18, name
        assert manifest["feature_dim"] == 1024, name
        assert manifest["layer_location"] == "post_block_pre_final_rmsnorm", name
        assert {shard["task_index"] for shard in manifest["shards"]} == expected_tasks, name
    expected_encoder_dims = {"dinov2": 768, "da3": 384}
    for name, manifest in encoders.items():
        assert manifest["extraction_scope"] == "full", name
        assert manifest["num_samples"] == 4000 and manifest["num_shards"] == 40, name
        assert manifest["layer_index"] == 11, name
        assert manifest["feature_dim"] == expected_encoder_dims[name], name
        assert manifest["feature_location"] == "raw_encoder_pre_token_merger", name
        assert {shard["task_index"] for shard in manifest["shards"]} == expected_tasks, name


def topk_mask(kernel: torch.Tensor, topk: int) -> torch.Tensor:
    if topk < 2:
        raise ValueError("CKNNA requires topk >= 2")
    n = kernel.shape[-1]
    if topk >= n:
        raise ValueError(f"topk must be less than N; got topk={topk}, N={n}")
    kernel_for_knn = kernel.clone()
    diagonal = torch.arange(n, device=kernel.device)
    kernel_for_knn[..., diagonal, diagonal] = float("-inf")
    indices = torch.topk(kernel_for_knn, topk, dim=-1).indices
    return torch.zeros_like(kernel).scatter_(-1, indices, 1.0)


def hsic_unbiased_batched(kernel_a: torch.Tensor, kernel_b: torch.Tensor) -> torch.Tensor:
    """Official Song et al. estimator, with sum(A@B) written as column/row sums."""
    if kernel_a.shape[-2:] != kernel_b.shape[-2:]:
        raise ValueError(f"Kernel shape mismatch: {kernel_a.shape}, {kernel_b.shape}")
    m = kernel_a.shape[-1]
    if m <= 3:
        raise ValueError("Unbiased HSIC requires at least four samples")
    diagonal = torch.arange(m, device=kernel_a.device)
    a = kernel_a.clone()
    b = kernel_b.clone()
    a[..., diagonal, diagonal] = 0
    b[..., diagonal, diagonal] = 0
    term_1 = (a * b.transpose(-1, -2)).sum(dim=(-2, -1))
    sum_a = a.sum(dim=(-2, -1))
    sum_b = b.sum(dim=(-2, -1))
    term_2 = sum_a * sum_b / ((m - 1) * (m - 2))
    # sum(A @ B) == dot(column_sums(A), row_sums(B)).
    term_3 = 2.0 * (a.sum(dim=-2) * b.sum(dim=-1)).sum(dim=-1) / (m - 2)
    return (term_1 + term_2 - term_3) / (m * (m - 3))


def cknna_from_kernels(kernel_a: torch.Tensor, kernel_b: torch.Tensor, topk: int) -> torch.Tensor:
    """Vectorized equivalent of AlignmentMetrics.cknna(unbiased=True)."""
    mask_a = topk_mask(kernel_a, topk)
    mask_b = topk_mask(kernel_b, topk)
    intersection = mask_a * mask_b
    sim_ab = hsic_unbiased_batched(intersection * kernel_a, intersection * kernel_b)
    sim_aa = hsic_unbiased_batched(mask_a * kernel_a, mask_a * kernel_a)
    sim_bb = hsic_unbiased_batched(mask_b * kernel_b, mask_b * kernel_b)
    return sim_ab / (torch.sqrt(sim_aa * sim_bb) + 1e-6)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_task_rows(
    policies: dict[str, dict],
    encoders: dict[str, dict],
    device: torch.device,
) -> list[dict]:
    rows: list[dict] = []
    encoder_shards = {
        name: {shard["task_index"]: shard for shard in manifest["shards"]}
        for name, manifest in encoders.items()
    }
    for policy_name, policy_manifest in policies.items():
        for policy_shard in policy_manifest["shards"]:
            task_index = policy_shard["task_index"]
            suite = policy_shard["suite"]
            policy_data = load_file(Path(policy_shard["path"]))
            policy_features = policy_data["features"].to(device=device, dtype=torch.float32)
            if policy_features.shape[:4] != (NUM_TIMESTEPS, NUM_REPLICATES, NUM_LAYERS, 100):
                raise ValueError(f"Unexpected policy feature shape: {policy_features.shape}")
            policy_features = F.normalize(policy_features, dim=-1)
            flat_policy = policy_features.reshape(-1, 100, policy_features.shape[-1])
            policy_kernel = torch.bmm(flat_policy, flat_policy.transpose(1, 2))
            policy_indices = policy_data["sample_index"]

            for encoder_name, encoder_manifest in encoders.items():
                encoder_shard = encoder_shards[encoder_name][task_index]
                if encoder_shard["suite"] != suite:
                    raise ValueError("Policy/encoder suite mismatch")
                encoder_data = load_file(Path(encoder_shard["path"]))
                if not torch.equal(policy_indices, encoder_data["sample_index"]):
                    raise ValueError(f"Policy/encoder sample order mismatch for task {task_index}")
                encoder_features = F.normalize(
                    encoder_data["features"].to(device=device, dtype=torch.float32), dim=-1
                )
                encoder_kernel = encoder_features @ encoder_features.T

                generator = torch.Generator(device="cpu").manual_seed(20260821 + int(task_index))
                permutation = torch.randperm(100, generator=generator).to(device)
                shuffled_features = encoder_features[permutation]
                shuffled_kernel = shuffled_features @ shuffled_features.T

                score_by_k = {
                    k: cknna_from_kernels(policy_kernel, encoder_kernel, k).reshape(
                        NUM_TIMESTEPS, NUM_REPLICATES, NUM_LAYERS
                    )
                    for k in K_VALUES
                }
                null_scores = cknna_from_kernels(policy_kernel, shuffled_kernel, 10).reshape(
                    NUM_TIMESTEPS, NUM_REPLICATES, NUM_LAYERS
                )
                for timestep_index in range(NUM_TIMESTEPS):
                    for replicate in range(NUM_REPLICATES):
                        for layer in range(NUM_LAYERS):
                            for k in K_VALUES:
                                rows.append(
                                    {
                                        "policy": policy_name,
                                        "encoder": encoder_name,
                                        "task_index": task_index,
                                        "suite": suite,
                                        "timestep_index": timestep_index,
                                        "timestep": policy_manifest["timesteps"][timestep_index],
                                        "noise_replicate": replicate,
                                        "layer": layer,
                                        "k": k,
                                        "score": float(score_by_k[k][timestep_index, replicate, layer]),
                                        "shuffle_null_score": (
                                            float(null_scores[timestep_index, replicate, layer]) if k == 10 else ""
                                        ),
                                    }
                                )
                del encoder_data, encoder_features, encoder_kernel, shuffled_features, shuffled_kernel
            del policy_data, policy_features, flat_policy, policy_kernel
    return rows


def aggregate_rows(task_rows: list[dict], bootstrap_replicates: int) -> tuple[list[dict], list[dict], dict]:
    per_task_values: dict[tuple, list[tuple[int, float, float | None]]] = collections.defaultdict(list)
    replicate_groups: dict[tuple, list[dict]] = collections.defaultdict(list)
    for row in task_rows:
        key = (
            row["policy"],
            row["encoder"],
            row["task_index"],
            row["suite"],
            row["timestep_index"],
            row["timestep"],
            row["layer"],
            row["k"],
        )
        replicate_groups[key].append(row)
    for key, rows in replicate_groups.items():
        if len(rows) != NUM_REPLICATES:
            raise ValueError(f"Expected three noise replicates, got {len(rows)} for {key}")
        score = float(np.mean([row["score"] for row in rows]))
        null_values = [row["shuffle_null_score"] for row in rows if row["shuffle_null_score"] != ""]
        null_score = float(np.mean(null_values)) if null_values else None
        policy, encoder, task_index, suite, timestep_index, timestep, layer, k = key
        aggregate_key = (policy, encoder, timestep_index, timestep, layer, k)
        per_task_values[aggregate_key].append((task_index, score, null_score, suite))

    rng = np.random.default_rng(20260822)
    shared_bootstrap_indices = rng.integers(0, 40, size=(bootstrap_replicates, 40))
    aggregate: list[dict] = []
    suite_rows: list[dict] = []
    all_paired = []
    all_null = []

    for key, values in sorted(per_task_values.items()):
        policy, encoder, timestep_index, timestep, layer, k = key
        values = sorted(values, key=lambda item: item[0])
        if len(values) != 40 or len({item[0] for item in values}) != 40:
            raise ValueError(f"Expected 40 unique task scores for {key}, got {len(values)}")
        scores = np.asarray([item[1] for item in values], dtype=np.float64)
        nulls = np.asarray([np.nan if item[2] is None else item[2] for item in values], dtype=np.float64)
        ci_low = ""
        ci_high = ""
        if k == 10:
            bootstrap_means = scores[shared_bootstrap_indices].mean(axis=1)
            ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975]).tolist()
            if not np.isnan(nulls).any():
                all_paired.extend(scores.tolist())
                all_null.extend(nulls.tolist())
        aggregate.append(
            {
                "policy": policy,
                "encoder": encoder,
                "timestep_index": timestep_index,
                "timestep": timestep,
                "layer": layer,
                "k": k,
                "mean_score": float(scores.mean()),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "mean_shuffle_null": "" if np.isnan(nulls).all() else float(np.nanmean(nulls)),
                "num_tasks": 40,
                "num_noise_replicates": NUM_REPLICATES,
            }
        )
        suites = sorted({item[3] for item in values})
        for suite in suites:
            suite_scores = np.asarray([item[1] for item in values if item[3] == suite])
            if suite_scores.shape[0] != 10:
                raise ValueError(f"Expected 10 task scores for suite={suite}, key={key}")
            suite_rows.append(
                {
                    "policy": policy,
                    "encoder": encoder,
                    "suite": suite,
                    "timestep_index": timestep_index,
                    "timestep": timestep,
                    "layer": layer,
                    "k": k,
                    "mean_score": float(suite_scores.mean()),
                    "num_tasks": 10,
                }
            )
    null_summary = {
        "paired_mean_over_all_k10_task_conditions": float(np.mean(all_paired)),
        "shuffle_mean_over_all_k10_task_conditions": float(np.mean(all_null)),
        "mean_difference": float(np.mean(np.asarray(all_paired) - np.asarray(all_null))),
        "num_task_conditions": len(all_paired),
    }
    return aggregate, suite_rows, null_summary


def render_figure(aggregate: list[dict], output_dir: Path) -> None:
    policies = ("nohead", "da3", "dinov2")
    encoders = ("dinov2", "da3")
    policy_titles = {
        "nohead": "No-head 30K",
        "da3": "DA3/Object/Skill 30K",
        "dinov2": "DINOv2/Object/Skill 30K",
    }
    encoder_labels = {"dinov2": "DINOv2-B layer 11", "da3": "DA3-SMALL layer 11"}
    colors = ("#2f6df6", "#78a6ff", "#aac7ff")
    lookup = {
        (row["policy"], row["encoder"], row["timestep_index"], row["layer"]): row
        for row in aggregate
        if row["k"] == 10
    }
    y_values = [row["mean_score"] for row in aggregate if row["k"] == 10]
    y_low = min(0.0, min(y_values) - 0.02)
    y_high = min(1.0, max(y_values) + 0.04)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), sharex=True, sharey=True)
    x = np.arange(NUM_LAYERS)
    for row_index, encoder in enumerate(encoders):
        for column_index, policy in enumerate(policies):
            axis = axes[row_index, column_index]
            for timestep_index, (label, color) in enumerate(zip(TIMESTEP_LABELS, colors, strict=True)):
                points = [lookup[(policy, encoder, timestep_index, layer)] for layer in range(NUM_LAYERS)]
                mean = np.asarray([point["mean_score"] for point in points])
                low = np.asarray([point["ci_low"] for point in points], dtype=float)
                high = np.asarray([point["ci_high"] for point in points], dtype=float)
                axis.plot(x, mean, color=color, marker="o", markersize=3.2, linewidth=1.8, label=label)
                axis.fill_between(x, low, high, color=color, alpha=0.14, linewidth=0)
            axis.set_xlim(0, NUM_LAYERS - 1)
            axis.set_ylim(y_low, y_high)
            axis.set_xticks((0, 3, 6, 9, 12, 15, 17))
            axis.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
            if row_index == 0:
                axis.set_title(policy_titles[policy], fontsize=11, fontweight="bold")
            if column_index == 0:
                axis.set_ylabel(f"CKNNA vs. {encoder_labels[encoder]}\n(task macro, k=10)")
            if row_index == 1:
                axis.set_xlabel("Action Expert Layer Index")
    axes[0, 2].legend(loc="best", frameon=True, fontsize=9)
    fig.suptitle("GuidedVLA Representation Alignment on Frozen LIBERO Demonstrations", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_dir / "guidedvla_cknna_2x3.png", dpi=220)
    fig.savefig(output_dir / "guidedvla_cknna_2x3.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates != 2000:
        raise ValueError("The preregistered protocol requires exactly 2,000 task bootstrap replicates")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for the preregistered CKNNA computation")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError("Refusing to reuse or overwrite a non-empty result directory")

    policy_paths = {
        "nohead": args.nohead_policy_manifest,
        "da3": args.da3_policy_manifest,
        "dinov2": args.dinov2_policy_manifest,
    }
    encoder_paths = {"da3": args.da3_encoder_manifest, "dinov2": args.dinov2_encoder_manifest}
    policies = {name: load_manifest(path) for name, path in policy_paths.items()}
    encoders = {name: load_manifest(path) for name, path in encoder_paths.items()}
    validate_input_manifests(policies, encoders)
    source_cache_hashes = {
        manifest["source_cache_manifest_sha256"] for manifest in (*policies.values(), *encoders.values())
    }
    if len(source_cache_hashes) != 1:
        raise ValueError(f"Feature manifests do not share one frozen cache: {source_cache_hashes}")

    task_rows = compute_task_rows(policies, encoders, torch.device(args.device))
    expected_task_rows = 3 * 2 * 40 * NUM_TIMESTEPS * NUM_REPLICATES * NUM_LAYERS * len(K_VALUES)
    if len(task_rows) != expected_task_rows:
        raise ValueError(f"Expected {expected_task_rows} task/replicate rows, got {len(task_rows)}")
    task_fields = [
        "policy",
        "encoder",
        "task_index",
        "suite",
        "timestep_index",
        "timestep",
        "noise_replicate",
        "layer",
        "k",
        "score",
        "shuffle_null_score",
    ]
    write_csv(args.output_dir / "task_replicate_scores.csv", task_rows, task_fields)

    aggregate, suite_rows, null_summary = aggregate_rows(task_rows, args.bootstrap_replicates)
    expected_aggregate_rows = 3 * 2 * NUM_TIMESTEPS * NUM_LAYERS * len(K_VALUES)
    if len(aggregate) != expected_aggregate_rows:
        raise ValueError(f"Expected {expected_aggregate_rows} aggregate rows, got {len(aggregate)}")
    if len(suite_rows) != expected_aggregate_rows * 4:
        raise ValueError(f"Expected {expected_aggregate_rows * 4} suite rows, got {len(suite_rows)}")
    aggregate_fields = [
        "policy",
        "encoder",
        "timestep_index",
        "timestep",
        "layer",
        "k",
        "mean_score",
        "ci_low",
        "ci_high",
        "mean_shuffle_null",
        "num_tasks",
        "num_noise_replicates",
    ]
    write_csv(args.output_dir / "task_macro_scores.csv", aggregate, aggregate_fields)
    write_csv(
        args.output_dir / "suite_macro_scores.csv",
        suite_rows,
        ["policy", "encoder", "suite", "timestep_index", "timestep", "layer", "k", "mean_score", "num_tasks"],
    )
    render_figure(aggregate, args.output_dir)

    summary = {
        "schema_version": "guidedvla-cknna-results-v1",
        "metric_source_commit": "dcd76ba3c950c1b197a2ae8b1c6713535c94ecf9",
        "metric_source_blob": "92797283031b5201bafe9284033c1330bb420a01",
        "k_values": K_VALUES,
        "primary_k": 10,
        "timesteps": policies["nohead"]["timesteps"],
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": 20260822,
        "source_cache_manifest_sha256": next(iter(source_cache_hashes)),
        "policy_manifests": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in policy_paths.items()},
        "encoder_manifests": {
            name: {"path": str(path), "sha256": sha256_file(path)} for name, path in encoder_paths.items()
        },
        "num_task_replicate_rows": len(task_rows),
        "num_aggregate_rows": len(aggregate),
        "shuffle_null": null_summary,
        "outputs": {
            name: sha256_file(args.output_dir / name)
            for name in (
                "task_replicate_scores.csv",
                "task_macro_scores.csv",
                "suite_macro_scores.csv",
                "guidedvla_cknna_2x3.png",
                "guidedvla_cknna_2x3.pdf",
            )
        },
    }
    summary_path = args.output_dir / "results_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(null_summary, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compute all-reference-layer vision CKNNA matrices and paired max-T inference."""

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
from matplotlib.patches import Rectangle
import numpy as np
from safetensors.torch import load_file
import torch
import torch.nn.functional as F

from compute_cknna import hsic_unbiased_batched


POLICIES = ("nohead", "da3", "dinov2")
REFERENCES = ("dinov2", "da3")
STREAM_LAYERS = {"paligemma": 19, "siglip": 28}
COMPARISONS = (
    ("dinov2", "nohead", "DINO-guided − no-head"),
    ("da3", "nohead", "DA3-guided − no-head"),
    ("dinov2", "da3", "DINO-guided − DA3-guided"),
)
BOOTSTRAP_SEED = 20260822
BOOTSTRAP_REPLICATES = 2000
TOPK = 10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    for shard in manifest["shards"]:
        if sha256_file(Path(shard["path"])) != shard["sha256"]:
            raise ValueError(f"Feature hash mismatch: {shard['path']}")
    manifest["manifest_path"] = str(path)
    manifest["manifest_sha256"] = sha256_file(path)
    return manifest


def episode_map(path: Path) -> tuple[dict[int, int], dict[int, int]]:
    sample_to_episode = {}
    sample_to_task = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            sample_to_episode[int(row["sample_index"])] = int(row["episode_index"])
            sample_to_task[int(row["sample_index"])] = int(row["task_index"])
    if len(sample_to_episode) != 4000:
        raise ValueError("Expected 4,000 frozen sample identities")
    return sample_to_episode, sample_to_task


def feature_stack(data: dict[str, torch.Tensor], stream: str) -> list[torch.Tensor]:
    if stream == "paligemma":
        return [data["paligemma_features"][index] for index in range(19)]
    return [
        *[data["siglip_block_features"][index] for index in range(27)],
        data["siglip_projector_features"],
    ]


def allowed_mask(episode_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
    ids = episode_ids.to(device)
    allowed = ids[:, None] != ids[None, :]
    allowed.fill_diagonal_(False)
    return allowed


def topk_mask(kernel: torch.Tensor, topk: int, allowed: torch.Tensor) -> torch.Tensor:
    if int(allowed.sum(dim=-1).min()) < topk:
        raise ValueError("Insufficient episode-excluded neighbours")
    indices = torch.topk(kernel.masked_fill(~allowed, float("-inf")), topk, dim=-1).indices
    return torch.zeros_like(kernel).scatter_(-1, indices, 1.0)


def cknna_grid(stream_kernel: torch.Tensor, ref_kernel: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """Return [reference_layer, stream_layer] CKNNA scores."""
    stream_mask = topk_mask(stream_kernel, TOPK, allowed)
    ref_mask = topk_mask(ref_kernel, TOPK, allowed)
    a = stream_kernel.unsqueeze(0)
    b = ref_kernel.unsqueeze(1)
    mask_a = stream_mask.unsqueeze(0)
    mask_b = ref_mask.unsqueeze(1)
    intersection = mask_a * mask_b
    ab = hsic_unbiased_batched(intersection * a, intersection * b)
    aa = hsic_unbiased_batched(mask_a * a, mask_a * a)
    bb = hsic_unbiased_batched(mask_b * b, mask_b * b)
    scores = ab / (torch.sqrt(aa * bb) + 1e-6)
    if scores.shape != (ref_kernel.shape[0], stream_kernel.shape[0]) or not torch.isfinite(scores).all():
        raise RuntimeError(f"Invalid CKNNA grid: {tuple(scores.shape)}")
    return scores


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compute_task_rows(policies: dict, references: dict, sample_manifest: Path, device: torch.device) -> list[dict]:
    sample_to_episode, sample_to_task = episode_map(sample_manifest)
    reference_shards = {
        name: {shard["task_index"]: shard for shard in manifest["shards"]}
        for name, manifest in references.items()
    }
    rows = []
    for policy, manifest in policies.items():
        for shard in manifest["shards"]:
            task = int(shard["task_index"])
            data = load_file(shard["path"])
            indices = data["sample_index"].to(torch.int64)
            episodes = torch.tensor([sample_to_episode[int(index)] for index in indices], dtype=torch.int64)
            if any(sample_to_task[int(index)] != task for index in indices):
                raise ValueError("Sample/task mismatch")
            allowed = allowed_mask(episodes, device)
            stream_kernels = {}
            for stream in STREAM_LAYERS:
                kernels = []
                for feature in feature_stack(data, stream):
                    feature = F.normalize(feature.to(device=device, dtype=torch.float32), dim=-1)
                    kernels.append(feature @ feature.T)
                stream_kernels[stream] = torch.stack(kernels, dim=0)
            for reference, reference_manifest in references.items():
                ref_shard = reference_shards[reference][task]
                ref_data = load_file(ref_shard["path"])
                if not torch.equal(indices, ref_data["sample_index"]):
                    raise ValueError("Policy/reference sample order mismatch")
                ref_features = ref_data["features"].to(device=device, dtype=torch.float32)
                if ref_features.shape[:2] != (12, 100):
                    raise ValueError(f"Unexpected reference feature shape: {tuple(ref_features.shape)}")
                ref_features = F.normalize(ref_features, dim=-1)
                # Preserve the original L11-only implementation's per-layer mm path.
                # Batched GEMM can perturb near-tied neighbours under TF32/BF32.
                ref_kernel = torch.stack(
                    [ref_features[layer] @ ref_features[layer].T for layer in range(12)], dim=0
                )
                for stream, stream_kernel in stream_kernels.items():
                    grid = cknna_grid(stream_kernel, ref_kernel, allowed).cpu().numpy()
                    for ref_layer in range(12):
                        for policy_layer in range(STREAM_LAYERS[stream]):
                            rows.append({
                                "policy": policy,
                                "reference": reference,
                                "stream": stream,
                                "task_index": task,
                                "suite": shard["suite"],
                                "reference_layer": ref_layer,
                                "policy_layer": policy_layer,
                                "score": float(grid[ref_layer, policy_layer]),
                            })
            print(json.dumps({"policy": policy, "task": task, "rows": len(rows)}))
    expected = len(POLICIES) * len(REFERENCES) * 40 * 12 * sum(STREAM_LAYERS.values())
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} task rows, got {len(rows)}")
    return rows


def aggregate_scores(task_rows: list[dict], indices: np.ndarray) -> list[dict]:
    grouped = collections.defaultdict(list)
    for row in task_rows:
        key = (row["policy"], row["reference"], row["stream"], row["reference_layer"], row["policy_layer"])
        grouped[key].append((row["task_index"], row["score"]))
    rows = []
    for key, values in sorted(grouped.items()):
        values = sorted(values)
        if len(values) != 40 or [value[0] for value in values] != list(range(40)):
            raise ValueError(f"Expected ordered 40-task family for {key}")
        scores = np.asarray([value[1] for value in values], dtype=np.float64)
        boot = scores[indices].mean(axis=1)
        low, high = np.quantile(boot, [0.025, 0.975])
        rows.append({
            "policy": key[0], "reference": key[1], "stream": key[2],
            "reference_layer": key[3], "policy_layer": key[4],
            "mean_score": float(scores.mean()), "ci_low": float(low), "ci_high": float(high),
            "num_tasks": 40,
        })
    return rows


def paired_inference(task_rows: list[dict], indices: np.ndarray) -> tuple[list[dict], list[dict]]:
    lookup = {
        (row["policy"], row["reference"], row["stream"], row["reference_layer"], row["policy_layer"], row["task_index"]): row["score"]
        for row in task_rows
    }
    output = []
    family_summaries = []
    for reference in REFERENCES:
        for stream, layer_count in STREAM_LAYERS.items():
            for numerator, denominator, label in COMPARISONS:
                cells = []
                task_matrix = []
                for ref_layer in range(12):
                    for policy_layer in range(layer_count):
                        values = np.asarray([
                            lookup[(numerator, reference, stream, ref_layer, policy_layer, task)]
                            - lookup[(denominator, reference, stream, ref_layer, policy_layer, task)]
                            for task in range(40)
                        ], dtype=np.float64)
                        cells.append((ref_layer, policy_layer))
                        task_matrix.append(values)
                task_matrix = np.asarray(task_matrix)
                means = task_matrix.mean(axis=1)
                ses = task_matrix.std(axis=1, ddof=1) / np.sqrt(40)
                if np.any(ses <= 0) or not np.isfinite(ses).all():
                    raise RuntimeError(f"Invalid standard errors for {reference}/{stream}/{label}")
                boot_means = task_matrix[:, indices].mean(axis=2)
                point_low, point_high = np.quantile(boot_means, [0.025, 0.975], axis=1)
                max_t = np.max(np.abs((boot_means - means[:, None]) / ses[:, None]), axis=0)
                critical = float(np.quantile(max_t, 0.95))
                simultaneous_low = means - critical * ses
                simultaneous_high = means + critical * ses
                point_sig = (point_low > 0) | (point_high < 0)
                simultaneous_sig = (simultaneous_low > 0) | (simultaneous_high < 0)
                comparison = f"{numerator}_minus_{denominator}"
                for index, (ref_layer, policy_layer) in enumerate(cells):
                    output.append({
                        "reference": reference, "stream": stream, "comparison": comparison,
                        "reference_layer": ref_layer, "policy_layer": policy_layer,
                        "mean_difference": float(means[index]),
                        "pointwise_ci_low": float(point_low[index]), "pointwise_ci_high": float(point_high[index]),
                        "pointwise_significant": bool(point_sig[index]),
                        "simultaneous_ci_low": float(simultaneous_low[index]),
                        "simultaneous_ci_high": float(simultaneous_high[index]),
                        "simultaneous_significant": bool(simultaneous_sig[index]),
                        "max_t_critical_value": critical, "num_tasks": 40,
                    })
                max_index = int(np.argmax(means))
                min_index = int(np.argmin(means))
                family_summaries.append({
                    "reference": reference, "stream": stream, "comparison": comparison,
                    "num_cells": len(cells), "max_t_critical_value": critical,
                    "num_pointwise_significant": int(point_sig.sum()),
                    "num_simultaneous_significant": int(simultaneous_sig.sum()),
                    "max_difference": float(means[max_index]), "max_reference_layer": cells[max_index][0],
                    "max_policy_layer": cells[max_index][1],
                    "min_difference": float(means[min_index]), "min_reference_layer": cells[min_index][0],
                    "min_policy_layer": cells[min_index][1],
                })
    return output, family_summaries


def render_heatmaps(rows: list[dict], stream: str, output_dir: Path) -> None:
    selected = [row for row in rows if row["stream"] == stream]
    layer_count = STREAM_LAYERS[stream]
    comparisons = [f"{a}_minus_{b}" for a, b, _ in COMPARISONS]
    titles = [label for _, _, label in COMPARISONS]
    vmax = max(abs(row["mean_difference"]) for row in selected)
    vmax = np.ceil((vmax + 0.001) / 0.01) * 0.01
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.4), sharex=True, sharey=True)
    image = None
    for ri, reference in enumerate(REFERENCES):
        for ci, (comparison, title) in enumerate(zip(comparisons, titles)):
            ax = axes[ri, ci]
            panel = [row for row in selected if row["reference"] == reference and row["comparison"] == comparison]
            lookup = {(row["reference_layer"], row["policy_layer"]): row for row in panel}
            matrix = np.asarray([[lookup[(r, p)]["mean_difference"] for p in range(layer_count)] for r in range(12)])
            image = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
            for r in range(12):
                for p in range(layer_count):
                    row = lookup[(r, p)]
                    if row["pointwise_significant"]:
                        ax.plot(p, r, marker=".", color="black", markersize=2.0, zorder=3)
                    if row["simultaneous_significant"]:
                        ax.add_patch(Rectangle((p - 0.48, r - 0.48), 0.96, 0.96, fill=False, edgecolor="black", linewidth=0.75, zorder=4))
            if ri == 0:
                ax.set_title(title, fontsize=10.5, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{reference.upper()} reference layer")
            ax.set_yticks(range(12))
            ax.spines[:].set_linewidth(0.7)
    if stream == "paligemma":
        ticks = [0, 1, 4, 7, 10, 13, 16, 18]
        labels = ["P", "0", "3", "6", "9", "12", "15", "17"]
        xlabel = "PaliGemma visual-token layer"
    else:
        ticks = [0, 4, 8, 12, 16, 20, 24, 27]
        labels = ["0", "4", "8", "12", "16", "20", "24", "P"]
        xlabel = "SigLIP vision block / projector"
    for ax in axes[-1]:
        ax.set_xticks(ticks, labels)
        ax.set_xlabel(xlabel)
    fig.suptitle(f"GuidedVLA {stream.capitalize()} alignment across reference layers", fontsize=13, fontweight="bold", y=0.985)
    fig.text(0.5, 0.012, "Dot: pointwise 95% bootstrap CI excludes 0 · Box: matrix-wide max-T 95% simultaneous CI excludes 0", ha="center", fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.865, bottom=0.10, top=0.91, wspace=0.12, hspace=0.12)
    colorbar_axis = fig.add_axes([0.89, 0.18, 0.014, 0.62])
    cbar = fig.colorbar(image, cax=colorbar_axis)
    cbar.set_label("Task-paired ΔCKNNA (episode-excluded, k=10)")
    stem = output_dir / f"guidedvla_{stream}_reference_layer_sweep"
    fig.savefig(stem.with_suffix(".png"), dpi=230, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def validate_manifests(policies: dict, references: dict) -> None:
    expected_dims = {"dinov2": 768, "da3": 384}
    all_manifests = [*policies.values(), *references.values()]
    cache_hashes = {manifest["source_cache_manifest_sha256"] for manifest in all_manifests}
    if len(cache_hashes) != 1:
        raise ValueError("Inputs do not share one frozen sample cache")
    for name, manifest in policies.items():
        if manifest["extraction_scope"] != "full" or manifest["num_samples"] != 4000:
            raise ValueError(f"Incomplete policy manifest: {name}")
    for name, manifest in references.items():
        if manifest["schema_version"] != "guidedvla-cknna-encoder-all-layers-v1":
            raise ValueError(f"Wrong all-layer schema: {name}")
        if manifest["layers"] != list(range(12)) or manifest["feature_dim"] != expected_dims[name]:
            raise ValueError(f"Wrong reference layout: {name}")
        if manifest["extraction_scope"] != "full" or manifest["num_samples"] != 4000:
            raise ValueError(f"Incomplete reference manifest: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    for policy in POLICIES:
        parser.add_argument(f"--{policy}-manifest", type=Path, required=True)
    for reference in REFERENCES:
        parser.add_argument(f"--{reference}-encoder-manifest", type=Path, required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    args = parser.parse_args()
    if args.bootstrap_replicates != BOOTSTRAP_REPLICATES:
        raise ValueError("Protocol requires 2,000 bootstrap replicates")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite non-empty result directory")
    policies = {name: load_manifest(getattr(args, f"{name}_manifest")) for name in POLICIES}
    references = {name: load_manifest(getattr(args, f"{name}_encoder_manifest")) for name in REFERENCES}
    validate_manifests(policies, references)
    device = torch.device(args.device)
    task_rows = compute_task_rows(policies, references, args.sample_manifest, device)
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 40, size=(BOOTSTRAP_REPLICATES, 40))
    aggregates = aggregate_scores(task_rows, indices)
    paired, families = paired_inference(task_rows, indices)
    write_csv(args.output_dir / "task_scores.csv", task_rows)
    write_csv(args.output_dir / "aggregate_scores.csv", aggregates)
    write_csv(args.output_dir / "paired_matrix_differences.csv", paired)
    write_csv(args.output_dir / "family_summaries.csv", families)
    render_heatmaps(paired, "paligemma", args.output_dir)
    render_heatmaps(paired, "siglip", args.output_dir)
    manifest = {
        "schema_version": "guidedvla-vision-reference-layer-sweep-v1",
        "primary_mode": "episode_excluded", "primary_k": TOPK,
        "reference_layers": list(range(12)), "stream_layers": STREAM_LAYERS,
        "bootstrap_unit": "task", "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "simultaneous_inference": "familywise max absolute centered bootstrap t statistic",
        "family_definition": "one reference x stream x model contrast matrix",
        "num_task_rows": len(task_rows), "num_aggregate_rows": len(aggregates),
        "num_paired_rows": len(paired), "family_summaries": families,
        "input_manifests": {
            "policies": {name: {"path": m["manifest_path"], "sha256": m["manifest_sha256"]} for name, m in policies.items()},
            "references": {name: {"path": m["manifest_path"], "sha256": m["manifest_sha256"]} for name, m in references.items()},
            "sample_manifest": {"path": str(args.sample_manifest), "sha256": sha256_file(args.sample_manifest)},
        },
    }
    (args.output_dir / "results_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"num_task_rows": len(task_rows), "num_paired_rows": len(paired), "families": families}, sort_keys=True))


if __name__ == "__main__":
    main()

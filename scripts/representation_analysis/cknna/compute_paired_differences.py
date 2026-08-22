#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


COMPARISONS = (("da3", "nohead"), ("dinov2", "nohead"), ("dinov2", "da3"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_record(values: np.ndarray, bootstrap_indices: np.ndarray) -> tuple[float, float, float]:
    means = values[bootstrap_indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError("Refusing to reuse or overwrite a non-empty paired-difference directory")

    replicate_groups: dict[tuple, list[float]] = collections.defaultdict(list)
    with args.task_scores.open() as stream:
        for row in csv.DictReader(stream):
            if int(row["k"]) != 10:
                continue
            key = (
                row["policy"],
                row["encoder"],
                int(row["task_index"]),
                int(row["timestep_index"]),
                int(row["layer"]),
            )
            replicate_groups[key].append(float(row["score"]))
    task_scores = {}
    for key, values in replicate_groups.items():
        if len(values) != 3:
            raise ValueError(f"Expected three noise replicates for {key}, got {len(values)}")
        task_scores[key] = float(np.mean(values))

    rng = np.random.default_rng(20260822)
    bootstrap_indices = rng.integers(0, 40, size=(2000, 40))
    layer_rows = []
    global_rows = []
    overall_rows = []
    for encoder in ("dinov2", "da3"):
        for numerator, denominator in COMPARISONS:
            comparison = f"{numerator}_minus_{denominator}"
            per_task_timestep_layers: dict[tuple[int, int], list[float]] = collections.defaultdict(list)
            for timestep_index in range(3):
                for layer in range(18):
                    values = np.asarray(
                        [
                            task_scores[(numerator, encoder, task, timestep_index, layer)]
                            - task_scores[(denominator, encoder, task, timestep_index, layer)]
                            for task in range(40)
                        ],
                        dtype=np.float64,
                    )
                    mean, low, high = bootstrap_record(values, bootstrap_indices)
                    layer_rows.append(
                        {
                            "encoder": encoder,
                            "comparison": comparison,
                            "timestep_index": timestep_index,
                            "layer": layer,
                            "mean_difference": mean,
                            "ci_low": low,
                            "ci_high": high,
                            "ci_excludes_zero": low > 0 or high < 0,
                            "num_tasks": 40,
                        }
                    )
                    for task, value in enumerate(values):
                        per_task_timestep_layers[(task, timestep_index)].append(float(value))
            per_task_all = collections.defaultdict(list)
            for timestep_index in range(3):
                values = np.asarray(
                    [np.mean(per_task_timestep_layers[(task, timestep_index)]) for task in range(40)]
                )
                mean, low, high = bootstrap_record(values, bootstrap_indices)
                global_rows.append(
                    {
                        "encoder": encoder,
                        "comparison": comparison,
                        "timestep_index": timestep_index,
                        "mean_difference_over_layers": mean,
                        "ci_low": low,
                        "ci_high": high,
                        "ci_excludes_zero": low > 0 or high < 0,
                        "num_tasks": 40,
                    }
                )
                for task, value in enumerate(values):
                    per_task_all[task].append(float(value))
            values = np.asarray([np.mean(per_task_all[task]) for task in range(40)])
            mean, low, high = bootstrap_record(values, bootstrap_indices)
            overall_rows.append(
                {
                    "encoder": encoder,
                    "comparison": comparison,
                    "mean_difference_over_timesteps_and_layers": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "ci_excludes_zero": low > 0 or high < 0,
                    "num_tasks": 40,
                }
            )

    write_csv(
        args.output_dir / "paired_layer_differences.csv",
        layer_rows,
        ["encoder", "comparison", "timestep_index", "layer", "mean_difference", "ci_low", "ci_high", "ci_excludes_zero", "num_tasks"],
    )
    write_csv(
        args.output_dir / "paired_timestep_differences.csv",
        global_rows,
        ["encoder", "comparison", "timestep_index", "mean_difference_over_layers", "ci_low", "ci_high", "ci_excludes_zero", "num_tasks"],
    )
    write_csv(
        args.output_dir / "paired_overall_differences.csv",
        overall_rows,
        ["encoder", "comparison", "mean_difference_over_timesteps_and_layers", "ci_low", "ci_high", "ci_excludes_zero", "num_tasks"],
    )
    summary = {
        "schema_version": "guidedvla-cknna-paired-differences-v1",
        "source_task_scores": str(args.task_scores.resolve()),
        "source_task_scores_sha256": sha256_file(args.task_scores),
        "primary_k": 10,
        "noise_replicates_averaged": 3,
        "bootstrap_unit": "task",
        "bootstrap_replicates": 2000,
        "bootstrap_seed": 20260822,
        "num_layer_rows": len(layer_rows),
        "num_timestep_rows": len(global_rows),
        "num_overall_rows": len(overall_rows),
        "layer_cis_excluding_zero": sum(row["ci_excludes_zero"] for row in layer_rows),
        "timestep_cis_excluding_zero": sum(row["ci_excludes_zero"] for row in global_rows),
        "overall": overall_rows,
    }
    (args.output_dir / "paired_results_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

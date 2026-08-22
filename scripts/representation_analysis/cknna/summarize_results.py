#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with (args.results_dir / "task_macro_scores.csv").open() as stream:
        aggregate = list(csv.DictReader(stream))

    def select(encoder: str, policy: str, timestep_index: int | None = None, k: int = 10) -> list[dict]:
        selected = [
            row
            for row in aggregate
            if row["encoder"] == encoder and row["policy"] == policy and int(row["k"]) == k
        ]
        if timestep_index is not None:
            selected = [row for row in selected if int(row["timestep_index"]) == timestep_index]
        return selected

    curves = []
    for encoder in ("dinov2", "da3"):
        for policy in ("nohead", "da3", "dinov2"):
            for timestep_index in range(3):
                rows = select(encoder, policy, timestep_index)
                peak = max(rows, key=lambda row: float(row["mean_score"]))
                layer_scores = {int(row["layer"]): float(row["mean_score"]) for row in rows}
                curves.append(
                    {
                        "encoder": encoder,
                        "policy": policy,
                        "timestep_index": timestep_index,
                        "timestep": float(peak["timestep"]),
                        "peak_layer": int(peak["layer"]),
                        "peak_score": float(peak["mean_score"]),
                        "peak_ci_low": float(peak["ci_low"]),
                        "peak_ci_high": float(peak["ci_high"]),
                        "mean_over_layers": sum(layer_scores.values()) / 18,
                        "layer_17_score": layer_scores[17],
                        "key_layer_scores": {str(layer): layer_scores[layer] for layer in (0, 10, 11, 17)},
                        "layer_10_to_11_jump": layer_scores[11] - layer_scores[10],
                    }
                )

    deltas = []
    for encoder in ("dinov2", "da3"):
        for policy in ("da3", "dinov2"):
            for timestep_index in range(3):
                baseline = {
                    int(row["layer"]): float(row["mean_score"])
                    for row in select(encoder, "nohead", timestep_index)
                }
                guided = {
                    int(row["layer"]): float(row["mean_score"])
                    for row in select(encoder, policy, timestep_index)
                }
                difference = {layer: guided[layer] - baseline[layer] for layer in baseline}
                best_layer = max(difference, key=difference.get)
                worst_layer = min(difference, key=difference.get)
                deltas.append(
                    {
                        "encoder": encoder,
                        "policy": policy,
                        "timestep_index": timestep_index,
                        "mean_delta_over_layers": sum(difference.values()) / 18,
                        "max_delta_layer": best_layer,
                        "max_delta": difference[best_layer],
                        "min_delta_layer": worst_layer,
                        "min_delta": difference[worst_layer],
                    }
                )

    sensitivity = []
    for encoder in ("dinov2", "da3"):
        for policy in ("nohead", "da3", "dinov2"):
            item = {"encoder": encoder, "policy": policy}
            for k in (5, 10, 20):
                rows = select(encoder, policy, k=k)
                item[f"k{k}_mean"] = sum(float(row["mean_score"]) for row in rows) / len(rows)
            sensitivity.append(item)

    summary = {"curves_k10": curves, "guided_minus_nohead_k10": deltas, "sensitivity": sensitivity}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    tmp.replace(args.output)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

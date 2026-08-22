#!/usr/bin/env python3
"""Render paired-delta CKNNA plots from the frozen vision analysis table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REFERENCES = ("dinov2", "da3")
REFERENCE_LABELS = {
    "dinov2": "DINOv2-B L11 reference",
    "da3": "DA3-SMALL L11 reference",
}
COMPARISONS = (
    ("dinov2_minus_nohead", "DINO-guided − no-head", "#2f6df6"),
    ("da3_minus_nohead", "DA3-guided − no-head", "#e07a24"),
    ("dinov2_minus_da3", "DINO-guided − DA3-guided", "#2a9d5b"),
)


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["layer"] = int(row["layer"])
        for key in ("mean_difference", "ci_low", "ci_high"):
            row[key] = float(row[key])
        row["ci_excludes_zero"] = row["ci_excludes_zero"].lower() == "true"
    return rows


def render(rows: list[dict], stream: str, output_dir: Path) -> None:
    selected = [row for row in rows if row["stream"] == stream]
    lookup = {
        (row["reference"], row["comparison"], row["layer"]): row
        for row in selected
    }
    count = 19 if stream == "paligemma" else 28
    layers = np.arange(count)
    if stream == "paligemma":
        tick_positions = [0, 1, 4, 7, 10, 13, 16, 18]
        tick_labels = ["P", "0", "3", "6", "9", "12", "15", "17"]
        xlabel = "PaliGemma visual-token layer (P = SigLIP projector)"
    else:
        tick_positions = [0, 4, 8, 12, 16, 20, 24, 27]
        tick_labels = ["0", "4", "8", "12", "16", "20", "24", "P"]
        xlabel = "SigLIP vision block / projector"

    bound = max(max(abs(row["ci_low"]), abs(row["ci_high"])) for row in selected)
    bound = np.ceil((bound + 0.003) / 0.01) * 0.01
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.2), sharex=True, sharey=True)

    for ax, reference in zip(axes, REFERENCES):
        for comparison, label, color in COMPARISONS:
            points = [lookup[(reference, comparison, layer)] for layer in layers]
            mean = np.asarray([point["mean_difference"] for point in points])
            low = np.asarray([point["ci_low"] for point in points])
            high = np.asarray([point["ci_high"] for point in points])
            significant = np.asarray([point["ci_excludes_zero"] for point in points])
            ax.fill_between(layers, low, high, color=color, alpha=0.11, linewidth=0)
            ax.plot(layers, mean, color=color, linewidth=2.0, label=label, zorder=3)
            ax.scatter(
                layers[~significant], mean[~significant], s=25,
                facecolors="white", edgecolors=color, linewidths=1.25, zorder=4,
            )
            ax.scatter(
                layers[significant], mean[significant], s=28,
                facecolors=color, edgecolors=color, linewidths=1.0, zorder=5,
            )
        ax.axhline(0.0, color="#333333", linewidth=1.0, linestyle="--", zorder=1)
        ax.set_title(REFERENCE_LABELS[reference], loc="left", fontsize=11, fontweight="bold")
        ax.set_ylabel("Paired ΔCKNNA\n(task macro, k=10)")
        ax.set_ylim(-bound, bound)
        ax.grid(axis="y", linestyle=":", alpha=0.45)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlabel(xlabel)
    axes[-1].set_xticks(tick_positions, tick_labels)
    axes[0].legend(loc="upper left", fontsize=9, frameon=False, ncol=3)
    fig.suptitle(
        f"GuidedVLA {stream.capitalize()} alignment: paired model differences",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.99, 0.008,
        "Episode-excluded CKNNA; bands: paired 95% task-bootstrap CI; filled markers: CI excludes 0",
        ha="right", va="bottom", fontsize=8.5, color="#444444",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    stem = output_dir / f"guidedvla_{stream}_paired_delta"
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.input)
    for stream in ("paligemma", "siglip"):
        render(rows, stream, args.output_dir)


if __name__ == "__main__":
    main()

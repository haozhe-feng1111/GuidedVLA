#!/usr/bin/env python3
"""Build the immutable 4,000-sample LIBERO CKNNA analysis manifest."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import pyarrow.parquet as pq


SCHEMA_VERSION = "guidedvla-cknna-libero-v1"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
QUANTILES = (0.1, 0.3, 0.5, 0.7, 0.9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--classification-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--episodes-per-task", type=int, default=20)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--noise-replicates", type=int, default=3)
    return parser.parse_args()


def normalize_slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_suite(task_text: str, classification: dict[str, list[dict]]) -> tuple[str, int]:
    task_slug = normalize_slug(task_text)
    task_token_count = len(task_slug.split("_"))
    candidates: list[tuple[int, str]] = []
    needle = f"_{task_slug}_"
    for suite, entries in classification.items():
        if suite not in SUITES:
            continue
        for entry in entries:
            name_slug = normalize_slug(entry["name"])
            if needle in f"_{name_slug}_":
                candidates.append((len(name_slug.split("_")) - task_token_count, suite))
    if not candidates:
        raise ValueError(f"No suite match for task: {task_text!r}")
    min_extra = min(extra for extra, _ in candidates)
    best_suites = sorted({suite for extra, suite in candidates if extra == min_extra})
    if len(best_suites) != 1:
        raise ValueError(
            f"Ambiguous most-specific suite match for {task_text!r}: "
            f"extra_tokens={min_extra}, suites={best_suites}"
        )
    return best_suites[0], min_extra


def deterministic_noise_seed(base_seed: int, sample_id: str, replicate: int) -> int:
    payload = f"{base_seed}:{sample_id}:noise:{replicate}".encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return value % (2**63 - 1)


def main() -> None:
    args = parse_args()
    if args.output_jsonl.exists() or args.summary_json.exists():
        raise FileExistsError("Refusing to overwrite an existing manifest or summary")
    if args.episodes_per_task != 20:
        raise ValueError("The preregistered protocol requires exactly 20 episodes per task")
    if args.action_horizon != 50:
        raise ValueError("The preregistered protocol requires action_horizon=50")
    if args.noise_replicates != 3:
        raise ValueError("The preregistered protocol requires exactly 3 noise replicates")

    meta_root = args.dataset_root / "meta"
    tasks_path = meta_root / "tasks.parquet"
    episodes_path = meta_root / "episodes" / "chunk-000" / "file-000.parquet"
    classification = json.loads(args.classification_json.read_text())
    task_rows = sorted(pq.read_table(tasks_path).to_pylist(), key=lambda row: row["task_index"])
    episode_rows = pq.read_table(
        episodes_path,
        columns=["episode_index", "tasks", "length", "dataset_from_index", "dataset_to_index"],
    ).to_pylist()

    if len(task_rows) != 40:
        raise ValueError(f"Expected 40 tasks, got {len(task_rows)}")

    episodes_by_task: dict[str, list[dict]] = collections.defaultdict(list)
    for row in episode_rows:
        if len(row["tasks"]) != 1:
            raise ValueError(f"Episode {row['episode_index']} has non-singleton tasks={row['tasks']}")
        episodes_by_task[row["tasks"][0]].append(row)

    manifest: list[dict] = []
    suite_task_counts: collections.Counter[str] = collections.Counter()
    suite_sample_counts: collections.Counter[str] = collections.Counter()
    task_summary: list[dict] = []

    for task_row in task_rows:
        task_index = int(task_row["task_index"])
        task_text = task_row["__index_level_0__"]
        suite, match_extra_tokens = assign_suite(task_text, classification)
        candidates = sorted(episodes_by_task[task_text], key=lambda row: row["episode_index"])
        if len(candidates) < args.episodes_per_task:
            raise ValueError(f"Task {task_index} has only {len(candidates)} episodes")

        rng = np.random.Generator(np.random.PCG64(args.seed + task_index))
        selected_positions = rng.choice(len(candidates), size=args.episodes_per_task, replace=False)
        selected = sorted((candidates[int(pos)] for pos in selected_positions), key=lambda row: row["episode_index"])
        suite_task_counts[suite] += 1

        task_sample_count = 0
        for episode in selected:
            episode_index = int(episode["episode_index"])
            episode_length = int(episode["length"])
            dataset_from = int(episode["dataset_from_index"])
            dataset_to = int(episode["dataset_to_index"])
            if dataset_to - dataset_from != episode_length:
                raise ValueError(f"Episode {episode_index} has inconsistent dataset index bounds")
            max_start = episode_length - args.action_horizon
            if max_start < 0:
                raise ValueError(f"Episode {episode_index} is shorter than the action horizon")
            relative_frames = [int(math.floor(q * max_start)) for q in QUANTILES]
            if len(set(relative_frames)) != len(QUANTILES):
                raise ValueError(f"Episode {episode_index} produces duplicate anchors: {relative_frames}")

            for anchor_index, (quantile, relative_frame) in enumerate(zip(QUANTILES, relative_frames, strict=True)):
                global_frame_index = dataset_from + relative_frame
                if global_frame_index + args.action_horizon > dataset_to:
                    raise ValueError(f"Sample would cross episode boundary: episode={episode_index}, frame={relative_frame}")
                sample_id = (
                    f"{suite}-task{task_index:02d}-episode{episode_index:04d}-frame{relative_frame:04d}"
                )
                noise_seeds = [
                    deterministic_noise_seed(args.seed, sample_id, replicate)
                    for replicate in range(args.noise_replicates)
                ]
                manifest.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "sample_index": len(manifest),
                        "sample_id": sample_id,
                        "suite": suite,
                        "task_index": task_index,
                        "task_text": task_text,
                        "episode_index": episode_index,
                        "episode_length": episode_length,
                        "dataset_from_index": dataset_from,
                        "dataset_to_index": dataset_to,
                        "relative_frame_index": relative_frame,
                        "global_frame_index": global_frame_index,
                        "anchor_index": anchor_index,
                        "anchor_quantile": quantile,
                        "action_horizon": args.action_horizon,
                        "noise_seeds": noise_seeds,
                    }
                )
                task_sample_count += 1
                suite_sample_counts[suite] += 1

        if task_sample_count != 100:
            raise ValueError(f"Task {task_index} produced {task_sample_count} samples instead of 100")
        task_summary.append(
            {
                "task_index": task_index,
                "task_text": task_text,
                "suite": suite,
                "suite_match_extra_tokens": match_extra_tokens,
                "available_episodes": len(candidates),
                "selected_episode_indices": [int(row["episode_index"]) for row in selected],
                "num_samples": task_sample_count,
            }
        )

    if len(manifest) != 4000:
        raise ValueError(f"Expected 4,000 samples, got {len(manifest)}")
    if dict(suite_task_counts) != {suite: 10 for suite in SUITES}:
        raise ValueError(f"Expected 10 tasks per suite, got {dict(suite_task_counts)}")
    if dict(suite_sample_counts) != {suite: 1000 for suite in SUITES}:
        raise ValueError(f"Expected 1,000 samples per suite, got {dict(suite_sample_counts)}")
    if len({row["global_frame_index"] for row in manifest}) != len(manifest):
        raise ValueError("Duplicate global frame indices found in manifest")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    tmp_manifest = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".tmp")
    with tmp_manifest.open("w") as handle:
        for row in manifest:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    tmp_manifest.replace(args.output_jsonl)
    manifest_sha256 = sha256_file(args.output_jsonl)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "seed": args.seed,
        "episodes_per_task": args.episodes_per_task,
        "frames_per_episode": len(QUANTILES),
        "anchor_quantiles": QUANTILES,
        "action_horizon": args.action_horizon,
        "noise_replicates": args.noise_replicates,
        "num_samples": len(manifest),
        "num_tasks": len(task_rows),
        "suite_task_counts": dict(sorted(suite_task_counts.items())),
        "suite_sample_counts": dict(sorted(suite_sample_counts.items())),
        "manifest_path": str(args.output_jsonl),
        "manifest_sha256": manifest_sha256,
        "dataset_root": str(args.dataset_root),
        "tasks_parquet_sha256": sha256_file(tasks_path),
        "episodes_parquet_sha256": sha256_file(episodes_path),
        "classification_json": str(args.classification_json),
        "classification_sha256": sha256_file(args.classification_json),
        "tasks": task_summary,
    }
    tmp_summary = args.summary_json.with_suffix(args.summary_json.suffix + ".tmp")
    tmp_summary.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    tmp_summary.replace(args.summary_json)
    print(json.dumps({"manifest_sha256": manifest_sha256, "num_samples": len(manifest)}, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cache_manifest_path = args.cache_dir / "cache_manifest.json"
    cache_manifest = json.loads(cache_manifest_path.read_text())
    rows = [json.loads(line) for line in args.source_manifest.read_text().splitlines() if line]
    assert cache_manifest["num_samples"] == len(rows) == 4000
    assert cache_manifest["num_shards"] == len(cache_manifest["shards"]) == 40
    assert cache_manifest["source_manifest_sha256"] == sha256_file(args.source_manifest)
    assert cache_manifest["primary_image_key"] == cache_manifest["image_keys"][0] == "base_0_rgb"
    assert cache_manifest["max_image_roundtrip_error"] == 0.0

    observed_indices: list[int] = []
    finite_keys = {"actions", "state", *(f"noise__replicate_{i}" for i in range(3))}
    for shard in cache_manifest["shards"]:
        path = Path(shard["path"])
        assert path.parent.resolve() == args.cache_dir.resolve()
        assert sha256_file(path) == shard["sha256"]
        with safe_open(path, framework="pt", device="cpu") as handle:
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        indices = tensors["sample_index"].tolist()
        observed_indices.extend(indices)
        assert len(indices) == 100
        assert indices[0] == shard["first_sample_index"]
        assert indices[-1] == shard["last_sample_index"]
        for key in finite_keys:
            assert torch.isfinite(tensors[key]).all().item(), (path, key)
        for local_index, sample_index in enumerate(indices):
            row = rows[sample_index]
            assert row["sample_index"] == sample_index
            for replicate, seed in enumerate(row["noise_seeds"]):
                generator = torch.Generator(device="cpu")
                generator.manual_seed(int(seed))
                expected = torch.randn((50, 32), generator=generator, dtype=torch.float32)
                actual = tensors[f"noise__replicate_{replicate}"][local_index]
                assert torch.equal(actual, expected), (sample_index, replicate)

    assert observed_indices == list(range(4000))
    report = {
        "status": "passed",
        "cache_dir": str(args.cache_dir.resolve()),
        "cache_manifest_sha256": sha256_file(cache_manifest_path),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "norm_stats_sha256": sha256_file(args.norm_stats),
        "num_samples": 4000,
        "num_shards": 40,
        "image_keys": cache_manifest["image_keys"],
        "primary_image_key": cache_manifest["primary_image_key"],
        "noise_replicates_exactly_regenerated": 3,
        "sample_indices": [observed_indices[0], observed_indices[-1]],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    tmp.replace(args.output)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

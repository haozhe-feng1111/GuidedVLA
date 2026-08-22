#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--kind", choices=("encoder", "policy"), required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.feature_manifest.read_text())
    assert manifest["extraction_scope"] == "full"
    assert manifest["num_samples"] == 4000
    assert manifest["num_shards"] == len(manifest["shards"]) == 40
    assert manifest["source_cache_manifest_sha256"] == sha256_file(args.cache_manifest)
    assert manifest["saved_dtype"] == "torch.float32"

    feature_dim = int(manifest["feature_dim"])
    indices: list[int] = []
    for shard in manifest["shards"]:
        path = Path(shard["path"])
        assert sha256_file(path) == shard["sha256"]
        tensors = load_file(path)
        sample_index = tensors["sample_index"]
        assert sample_index.shape == (100,)
        assert int(sample_index[0]) == shard["first_sample_index"]
        assert int(sample_index[-1]) == shard["last_sample_index"]
        indices.extend(sample_index.tolist())
        features = tensors["features"]
        if args.kind == "encoder":
            assert features.shape == (100, feature_dim)
        else:
            assert features.shape == (3, 3, 18, 100, feature_dim)
            assert torch.equal(tensors["timesteps"], torch.tensor([0.001, 0.25, 0.5]))
        assert features.dtype == torch.float32
        assert torch.isfinite(features).all().item()
    assert indices == list(range(4000))

    report = {
        "status": "passed",
        "kind": args.kind,
        "feature_manifest": str(args.feature_manifest.resolve()),
        "feature_manifest_sha256": sha256_file(args.feature_manifest),
        "source_cache_manifest_sha256": sha256_file(args.cache_manifest),
        "num_samples": 4000,
        "num_shards": 40,
        "feature_dim": feature_dim,
        "sample_indices": [indices[0], indices[-1]],
    }
    if args.kind == "encoder":
        report.update({"encoder": manifest["encoder"], "layer_index": manifest["layer_index"]})
    else:
        report.update(
            {
                "model_name": manifest["model_name"],
                "timesteps": manifest["timesteps"],
                "noise_replicates": manifest["noise_replicates"],
                "num_layers": manifest["num_layers"],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    tmp.replace(args.output)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

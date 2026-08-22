#!/usr/bin/env python3
"""Extract reusable all-layer DINOv2-B or DA3-SMALL reference features."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import types

from safetensors.torch import load_file, save_file
import torch


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LAYERS = tuple(range(12))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=("dinov2", "da3"), required=True)
    parser.add_argument("--encoder-path", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-shards", type=int, default=40)
    parser.add_argument("--samples-per-shard", type=int, default=100)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_weight_file(path: Path) -> Path:
    if path.is_file():
        return path
    candidate = path / "model.safetensors"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Cannot resolve encoder weight file from {path}")


def load_encoder(kind: str, path: Path, device: torch.device):
    if kind == "dinov2":
        from transformers import AutoModel

        model = AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    else:
        if "gsplat" not in sys.modules:
            stub = types.ModuleType("gsplat")
            stub.rasterization = None
            sys.modules["gsplat"] = stub
        from depth_anything_3.api import DepthAnything3

        model = DepthAnything3.from_pretrained(str(path))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model.to(device)


def extract_batch(kind: str, model, uint8_images: torch.Tensor, device: torch.device) -> torch.Tensor:
    images = uint8_images.to(device=device, dtype=torch.float32, non_blocking=True) / 255.0
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    images = (images - mean) / std
    pooled_layers = []
    with torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        if kind == "dinov2":
            output = model(pixel_values=images, output_hidden_states=True, return_dict=True)
            if output.hidden_states is None or len(output.hidden_states) != 13:
                raise RuntimeError(f"Expected 13 DINOv2 hidden states, got {len(output.hidden_states or [])}")
            for layer in LAYERS:
                pooled_layers.append(output.hidden_states[layer + 1][:, 1:, :].float().mean(dim=1))
        else:
            output = model.forward(images.unsqueeze(1), export_feat_layers=list(LAYERS))
            for layer in LAYERS:
                key = f"feat_layer_{layer}"
                if key not in output.aux:
                    raise RuntimeError(f"DA3 output is missing {key}; keys={sorted(output.aux)}")
                patch = output.aux[key].squeeze(1)
                patch = patch.reshape(patch.shape[0], -1, patch.shape[-1])
                pooled_layers.append(patch.float().mean(dim=1))
    pooled = torch.stack(pooled_layers, dim=0).cpu().contiguous()
    if pooled.ndim != 3 or pooled.shape[0] != len(LAYERS) or not torch.isfinite(pooled).all():
        raise RuntimeError(f"Invalid all-layer encoder features: {tuple(pooled.shape)}")
    return pooled


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for encoder extraction")
    if not 1 <= args.max_shards <= 40 or not 1 <= args.samples_per_shard <= 100:
        raise ValueError("Invalid smoke/full extraction dimensions")
    device = torch.device(args.device)
    cache_manifest = json.loads(args.cache_manifest.read_text())
    if cache_manifest["num_samples"] != 4000 or cache_manifest["num_shards"] != 40:
        raise ValueError("Unexpected reusable cache dimensions")
    primary_key = cache_manifest["primary_image_key"]
    image_tensor_key = f"image__{primary_key}"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "feature_manifest.json"
    if manifest_path.exists() or any(args.output_dir.glob("*.safetensors")):
        raise FileExistsError("Refusing to reuse or overwrite a non-empty output directory")

    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    model = load_encoder(args.encoder, args.encoder_path, device)
    shard_records = []
    feature_dim = None
    total_samples = 0
    for cache_shard in cache_manifest["shards"][: args.max_shards]:
        cache_path = Path(cache_shard["path"])
        if sha256_file(cache_path) != cache_shard["sha256"]:
            raise ValueError(f"Cache shard hash mismatch: {cache_path}")
        cached = load_file(cache_path)
        images = cached[image_tensor_key][: args.samples_per_shard]
        sample_index = cached["sample_index"][: args.samples_per_shard].to(torch.int64)
        parts = [
            extract_batch(args.encoder, model, images[start : start + args.batch_size], device)
            for start in range(0, images.shape[0], args.batch_size)
        ]
        features = torch.cat(parts, dim=1).float().contiguous()
        if features.shape[:2] != (12, args.samples_per_shard):
            raise ValueError(f"Unexpected encoder shard shape: {tuple(features.shape)}")
        feature_dim = feature_dim or features.shape[-1]
        if features.shape[-1] != feature_dim:
            raise ValueError("Feature dimension changed across shards")

        output_path = args.output_dir / Path(cache_path).name
        tmp_path = output_path.with_suffix(".safetensors.tmp")
        save_file({"features": features, "sample_index": sample_index.contiguous()}, tmp_path)
        tmp_path.replace(output_path)
        shard_records.append({
            "task_index": cache_shard["task_index"],
            "suite": cache_shard["suite"],
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "num_samples": features.shape[1],
            "num_layers": features.shape[0],
            "feature_dim": features.shape[2],
            "first_sample_index": int(sample_index[0]),
            "last_sample_index": int(sample_index[-1]),
        })
        total_samples += features.shape[1]
        print(json.dumps({"encoder": args.encoder, "task": cache_shard["task_index"], "samples": total_samples}))

    expected = args.max_shards * args.samples_per_shard
    if total_samples != expected or len(shard_records) != args.max_shards or feature_dim is None:
        raise ValueError("Incomplete all-layer extraction")
    weight_file = resolve_weight_file(args.encoder_path)
    manifest = {
        "schema_version": "guidedvla-cknna-encoder-all-layers-v1",
        "encoder": args.encoder,
        "encoder_path": str(args.encoder_path),
        "encoder_weight_file": str(weight_file),
        "encoder_weight_sha256": sha256_file(weight_file),
        "layers": list(LAYERS),
        "num_layers": 12,
        "feature_dim": feature_dim,
        "feature_location": "raw_encoder_post_block_pre_token_merger",
        "pooling": "mean_over_patch_tokens",
        "autocast_dtype": "torch.bfloat16",
        "saved_dtype": "torch.float32",
        "extraction_scope": "full" if (args.max_shards, args.samples_per_shard) == (40, 100) else "smoke",
        "requested_max_shards": args.max_shards,
        "requested_samples_per_shard": args.samples_per_shard,
        "primary_image_key": primary_key,
        "source_cache_manifest": str(args.cache_manifest),
        "source_cache_manifest_sha256": sha256_file(args.cache_manifest),
        "num_samples": total_samples,
        "num_shards": len(shard_records),
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
        "shards": shard_records,
    }
    tmp_manifest = manifest_path.with_suffix(".json.tmp")
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp_manifest.replace(manifest_path)
    print(json.dumps({"encoder": args.encoder, "num_samples": total_samples, "shape": [12, feature_dim]}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract fixed layer-11 DINOv2-B or DA3-SMALL reference features."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import types

from safetensors.torch import load_file, save_file
import torch
import torch.nn.functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
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
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    images = (images - mean) / std
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        if kind == "dinov2":
            output = model(pixel_values=images, output_hidden_states=True, return_dict=True)
            if output.hidden_states is None or len(output.hidden_states) <= 12:
                raise RuntimeError("DINOv2 did not return hidden state for encoder layer 11")
            patch_features = output.hidden_states[12][:, 1:, :]
        else:
            output = model.forward(images.unsqueeze(1), export_feat_layers=[11])
            patch_features = output.aux["feat_layer_11"].squeeze(1)
            patch_features = patch_features.reshape(patch_features.shape[0], -1, patch_features.shape[-1])
        pooled = patch_features.float().mean(dim=1)
    if not torch.isfinite(pooled).all():
        raise RuntimeError("Non-finite encoder layer-11 features")
    return pooled.cpu().contiguous()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for the preregistered encoder extraction")
    device = torch.device(args.device)
    cache_manifest = json.loads(args.cache_manifest.read_text())
    if cache_manifest["num_samples"] != 4000 or cache_manifest["num_shards"] != 40:
        raise ValueError("Unexpected reusable cache dimensions")
    if not 1 <= args.max_shards <= 40:
        raise ValueError("max-shards must be in [1, 40]")
    if not 1 <= args.samples_per_shard <= 100:
        raise ValueError("samples-per-shard must be in [1, 100]")
    primary_key = cache_manifest["primary_image_key"]
    image_tensor_key = f"image__{primary_key}"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_manifest_path = args.output_dir / "feature_manifest.json"
    if feature_manifest_path.exists() or any(args.output_dir.glob("*.safetensors")):
        raise FileExistsError("Refusing to reuse or overwrite a non-empty encoder output directory")

    model = load_encoder(args.encoder, args.encoder_path, device)
    shard_records: list[dict] = []
    feature_dim: int | None = None
    total_samples = 0

    selected_shards = cache_manifest["shards"][: args.max_shards]
    for cache_shard in selected_shards:
        cache_path = Path(cache_shard["path"])
        if sha256_file(cache_path) != cache_shard["sha256"]:
            raise ValueError(f"Cache shard hash mismatch: {cache_path}")
        cached = load_file(cache_path)
        images = cached[image_tensor_key][: args.samples_per_shard]
        sample_index = cached["sample_index"][: args.samples_per_shard].to(torch.int64)
        feature_parts = []
        for start in range(0, images.shape[0], args.batch_size):
            feature_parts.append(extract_batch(args.encoder, model, images[start : start + args.batch_size], device))
        features = torch.cat(feature_parts, dim=0).float().contiguous()
        if features.shape[0] != args.samples_per_shard:
            raise ValueError(f"Unexpected encoder shard size: {features.shape}")
        if feature_dim is None:
            feature_dim = features.shape[1]
        if features.shape[1] != feature_dim:
            raise ValueError("Encoder feature dimension changed across shards")

        output_path = args.output_dir / Path(cache_path).name
        tmp_path = output_path.with_suffix(".safetensors.tmp")
        save_file({"features": features, "sample_index": sample_index.contiguous()}, tmp_path)
        tmp_path.replace(output_path)
        shard_records.append(
            {
                "task_index": cache_shard["task_index"],
                "suite": cache_shard["suite"],
                "path": str(output_path),
                "sha256": sha256_file(output_path),
                "num_samples": features.shape[0],
                "feature_dim": features.shape[1],
                "first_sample_index": int(sample_index[0]),
                "last_sample_index": int(sample_index[-1]),
            }
        )
        total_samples += features.shape[0]

    expected_samples = args.max_shards * args.samples_per_shard
    if total_samples != expected_samples or len(shard_records) != args.max_shards or feature_dim is None:
        raise ValueError(f"Incomplete encoder extraction: samples={total_samples}, shards={len(shard_records)}")
    weight_file = resolve_weight_file(args.encoder_path)
    feature_manifest = {
        "schema_version": "guidedvla-cknna-encoder-features-v1",
        "encoder": args.encoder,
        "encoder_path": str(args.encoder_path),
        "encoder_weight_file": str(weight_file),
        "encoder_weight_sha256": sha256_file(weight_file),
        "layer_index": 11,
        "feature_location": "raw_encoder_pre_token_merger",
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
        "feature_dim": feature_dim,
        "shards": shard_records,
    }
    tmp_manifest = feature_manifest_path.with_suffix(".json.tmp")
    tmp_manifest.write_text(json.dumps(feature_manifest, indent=2, sort_keys=True) + "\n")
    tmp_manifest.replace(feature_manifest_path)
    print(json.dumps({"encoder": args.encoder, "num_samples": total_samples, "feature_dim": feature_dim}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare policy-internal and standalone raw layer-11 encoder features."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

from safetensors.torch import load_file
import torch
import torch.nn.functional as F

from compute_cknna import cknna_from_kernels
from extract_encoder_features import load_encoder
from extract_policy_features import load_policy, observation_from_cache, sha256_file


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalized(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


def raw_features(kind: str, encoder, images_01: torch.Tensor) -> torch.Tensor:
    x = normalized(images_01)
    if kind == "dinov2":
        output = encoder(pixel_values=x, output_hidden_states=True, return_dict=True)
        return output.hidden_states[12][:, 1:, :].clone().float()
    output = encoder.forward(x.unsqueeze(1), export_feat_layers=[11])
    features = output.aux["feat_layer_11"].squeeze(1)
    return features.reshape(features.shape[0], -1, features.shape[-1]).clone().float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=("dinov2", "da3"), required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-path", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ["OPENPI_DEPTH_MODEL_PATH"] = str(args.encoder_path)
    device = torch.device("cuda:0")
    cache_manifest = json.loads(args.cache_manifest.read_text())
    cached = load_file(cache_manifest["shards"][0]["path"])
    if not 11 <= args.num_samples <= 100:
        raise ValueError("Identity CKNNA requires 11..100 samples")
    sample_indices = cached["sample_index"][: args.num_samples].tolist()

    model_name = "dinov2" if args.encoder == "dinov2" else "da3"
    policy, load_report = load_policy(args.config_name, args.checkpoint, device)
    policy_images = []
    for index in range(args.num_samples):
        observation = observation_from_cache(cached, cache_manifest, index, device)
        images, _, _, _, _ = policy._preprocess_observation(observation, train=False)
        policy_images.append(images[0])
    policy_images = torch.cat(policy_images, dim=0)
    module = policy.depth_module
    internal_encoder = module.dinov2_model if args.encoder == "dinov2" else module.da3_model
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        internal = raw_features(args.encoder, internal_encoder, (policy_images + 1.0) * 0.5).cpu()
    del policy, module, internal_encoder, policy_images
    gc.collect(); torch.cuda.empty_cache()

    standalone = load_encoder(args.encoder, args.encoder_path, device)
    uint8_images = cached[f"image__{cache_manifest['primary_image_key']}"][: args.num_samples]
    images_01 = uint8_images.to(device=device, dtype=torch.float32) / 255.0
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        external = raw_features(args.encoder, standalone, images_01).cpu()
    if internal.shape != external.shape:
        raise RuntimeError(f"Raw feature shape mismatch: {internal.shape} vs {external.shape}")
    difference = internal - external
    pooled_internal = internal.mean(dim=1)
    pooled_external = external.mean(dim=1)
    cosine = F.cosine_similarity(internal.flatten(1), external.flatten(1), dim=-1)
    a = F.normalize(pooled_internal, dim=-1); b = F.normalize(pooled_external, dim=-1)
    self_cknna = float(cknna_from_kernels((a @ a.T).unsqueeze(0), b @ b.T, 10)[0])
    report = {
        "schema_version": "guidedvla-encoder-identity-v1",
        "encoder": args.encoder,
        "model_name": model_name,
        "config_name": args.config_name,
        "checkpoint_sha256": load_report["checkpoint_sha256"],
        "encoder_weight_sha256": sha256_file(args.encoder_path / "model.safetensors"),
        "sample_indices": sample_indices,
        "num_samples": args.num_samples,
        "raw_shape": list(internal.shape),
        "max_abs": float(difference.abs().max()),
        "relative_l2": float(difference.double().norm() / internal.double().norm()),
        "mean_sample_cosine": float(cosine.mean()),
        "min_sample_cosine": float(cosine.min()),
        "pooled_max_abs": float((pooled_internal - pooled_external).abs().max()),
        "pooled_self_cknna_k10": self_cknna,
        "preprocessing": "policy cache uint8 -> [-1,1] -> policy preprocess -> [0,1] -> ImageNet normalization; standalone cache uint8 -> [0,1] -> ImageNet normalization",
        "layer_index": 11,
        "token_pooling": "mean_over_raw_patch_tokens",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

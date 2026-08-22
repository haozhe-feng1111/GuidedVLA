#!/usr/bin/env python3
"""Extract PaliGemma visual-token-stream and SigLIP features from one policy."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import time

from safetensors.torch import load_file, save_file
import torch

from openpi.models_pytorch import pi0_pytorch

from extract_policy_features import load_policy, observation_from_cache, sha256_file


NUM_PALIGEMMA_LAYERS = 18
NUM_SIGLIP_LAYERS = 27
NUM_PATCH_TOKENS = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", choices=("nohead", "da3", "dinov2"), required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--external-encoder-path", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-shards", type=int, default=40)
    parser.add_argument("--samples-per-shard", type=int, default=100)
    return parser.parse_args()


def extract_one(model, observation) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images, image_masks, lang_tokens, lang_masks, _ = model._preprocess_observation(observation, train=False)
    batch_size = images[0].shape[0]
    if batch_size != 1 or len(images) != 3:
        raise ValueError(f"Expected one sample and three views, got batch={batch_size}, views={len(images)}")

    paligemma = model.paligemma_with_expert.paligemma
    all_images = torch.cat(images, dim=0).contiguous(memory_format=torch.contiguous_format)
    vision_output = paligemma.vision_tower(
        pixel_values=all_images,
        output_hidden_states=True,
        return_dict=True,
    )
    if vision_output.hidden_states is None or len(vision_output.hidden_states) != NUM_SIGLIP_LAYERS + 1:
        raise RuntimeError(f"Expected 28 SigLIP hidden states, got {len(vision_output.hidden_states or ())}")
    siglip_blocks = torch.stack(
        [state[:1, :NUM_PATCH_TOKENS].float().mean(dim=1)[0] for state in vision_output.hidden_states[1:]],
        dim=0,
    )

    all_projected = paligemma.multi_modal_projector(vision_output.last_hidden_state)
    if all_projected.shape[1] != NUM_PATCH_TOKENS:
        raise RuntimeError(f"Expected 256 projected image tokens, got {all_projected.shape}")
    siglip_projector = all_projected[:1].float().mean(dim=1)[0]

    projected_views = all_projected.split([image.shape[0] for image in images], dim=0)
    prefix_parts = []
    pad_parts = []
    attention_parts = []
    for projected, mask in zip(projected_views, image_masks, strict=True):
        prefix_parts.append(projected)
        pad_parts.append(mask[:, None].expand(batch_size, projected.shape[1]))
        attention_parts.append(torch.zeros(projected.shape[1], dtype=torch.bool, device=projected.device))
    language = model.paligemma_with_expert.embed_language_tokens(lang_tokens)
    language = language * math.sqrt(language.shape[-1])
    prefix_parts.append(language)
    pad_parts.append(lang_masks)
    attention_parts.append(torch.zeros(language.shape[1], dtype=torch.bool, device=language.device))
    prefix_embs = torch.cat(prefix_parts, dim=1)
    prefix_pad_masks = torch.cat(pad_parts, dim=1)
    prefix_att_masks = torch.cat(attention_parts, dim=0)[None, :].expand(batch_size, -1)
    (prefix_embs,) = model._align_prefix_embeddings_dtype(prefix_embs)

    captures: dict[int, torch.Tensor] = {}
    handles = []
    for layer_index, layer in enumerate(paligemma.language_model.layers):
        def hook(_module, _inputs, output, *, index=layer_index):
            hidden = output[0] if isinstance(output, tuple) else output
            captures[index] = hidden[:, :NUM_PATCH_TOKENS].float().mean(dim=1).detach().cpu()[0]
        handles.append(layer.register_forward_hook(hook))
    try:
        mask_2d = pi0_pytorch.make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        mask_4d = model._prepare_attention_masks_4d(mask_2d, dtype=prefix_embs.dtype)
        model.paligemma_with_expert(
            attention_mask=mask_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
        )
    finally:
        for handle in handles:
            handle.remove()
    if sorted(captures) != list(range(NUM_PALIGEMMA_LAYERS)):
        raise RuntimeError(f"Expected 18 PaliGemma captures, got {sorted(captures)}")
    paligemma_stream = torch.stack(
        [siglip_projector.cpu(), *[captures[index] for index in range(NUM_PALIGEMMA_LAYERS)]], dim=0
    )
    outputs = (paligemma_stream, siglip_blocks.cpu(), siglip_projector.cpu())
    if not all(torch.isfinite(value).all() for value in outputs):
        raise RuntimeError("Non-finite vision feature")
    return outputs


def main() -> None:
    args = parse_args()
    expected_configs = {
        "nohead": "pi0_libero",
        "da3": "pi0_libero_object_depth_skill",
        "dinov2": "pi0_libero_object_dinov2_base_skill",
    }
    if args.config_name != expected_configs[args.model_name]:
        raise ValueError(f"Unexpected config for {args.model_name}: {args.config_name}")
    if args.model_name != "nohead":
        if args.external_encoder_path is None:
            raise ValueError("Guided models require --external-encoder-path")
        import os
        os.environ["OPENPI_DEPTH_MODEL_PATH"] = str(args.external_encoder_path)
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required")

    cache_manifest = json.loads(args.cache_manifest.read_text())
    if cache_manifest["num_samples"] != 4000 or cache_manifest["num_shards"] != 40:
        raise ValueError("Unexpected frozen cache dimensions")
    if not 1 <= args.max_shards <= 40 or not 1 <= args.samples_per_shard <= 100:
        raise ValueError("Invalid extraction subset")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError("Refusing to reuse or overwrite a non-empty vision output directory")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    model, load_report = load_policy(args.config_name, args.checkpoint, device)
    started = time.monotonic()
    records = []
    total = 0
    dimensions = None
    for shard in cache_manifest["shards"][: args.max_shards]:
        cache_path = Path(shard["path"])
        if sha256_file(cache_path) != shard["sha256"]:
            raise ValueError(f"Cache hash mismatch: {cache_path}")
        cached = load_file(cache_path)
        sample_indices = cached["sample_index"][: args.samples_per_shard].to(torch.int64)
        pg_task = siglip_task = projector_task = None
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for local_index in range(args.samples_per_shard):
                observation = observation_from_cache(cached, cache_manifest, local_index, device)
                pg, siglip, projector = extract_one(model, observation)
                if pg_task is None:
                    dimensions = {
                        "paligemma": pg.shape[-1], "siglip": siglip.shape[-1], "projector": projector.shape[-1]
                    }
                    pg_task = torch.empty(19, args.samples_per_shard, pg.shape[-1], dtype=torch.float32)
                    siglip_task = torch.empty(27, args.samples_per_shard, siglip.shape[-1], dtype=torch.float32)
                    projector_task = torch.empty(args.samples_per_shard, projector.shape[-1], dtype=torch.float32)
                pg_task[:, local_index] = pg
                siglip_task[:, local_index] = siglip
                projector_task[local_index] = projector
        if pg_task is None or siglip_task is None or projector_task is None:
            raise RuntimeError("No vision features extracted")
        output_path = args.output_dir / cache_path.name
        temporary = output_path.with_suffix(".safetensors.tmp")
        save_file({
            "paligemma_features": pg_task.contiguous(),
            "siglip_block_features": siglip_task.contiguous(),
            "siglip_projector_features": projector_task.contiguous(),
            "sample_index": sample_indices.contiguous(),
        }, temporary)
        temporary.replace(output_path)
        records.append({
            "task_index": shard["task_index"], "suite": shard["suite"], "path": str(output_path),
            "sha256": sha256_file(output_path), "num_samples": args.samples_per_shard,
        })
        total += args.samples_per_shard
        elapsed = time.monotonic() - started
        print(json.dumps({"model": args.model_name, "tasks": len(records), "samples": total,
                          "samples_per_second": total / elapsed}), flush=True)
    expected = args.max_shards * args.samples_per_shard
    if total != expected or dimensions is None:
        raise RuntimeError(f"Incomplete extraction: {total}/{expected}")
    manifest = {
        "schema_version": "guidedvla-vision-cknna-features-v1",
        "model_name": args.model_name,
        "config_name": args.config_name,
        "load_report": load_report,
        "source_cache_manifest": str(args.cache_manifest),
        "source_cache_manifest_sha256": sha256_file(args.cache_manifest),
        "extraction_scope": "full" if (args.max_shards, args.samples_per_shard) == (40, 100) else "smoke",
        "num_samples": total, "num_shards": len(records),
        "primary_camera": cache_manifest["primary_image_key"], "visual_tokens": NUM_PATCH_TOKENS,
        "pooling": "mean_over_base_camera_patch_tokens",
        "paligemma_x": [-1, *range(18)], "paligemma_location": "projector_output_then_post_block",
        "siglip_x": [*range(27), "projector"], "siglip_location": "post_block_then_final_norm_projector",
        "dimensions": dimensions, "autocast_dtype": "torch.bfloat16", "saved_dtype": "torch.float32",
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
        "elapsed_seconds": time.monotonic() - started,
        "shards": records,
    }
    path = args.output_dir / "feature_manifest.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    print(json.dumps({"complete": True, "model": args.model_name, "samples": total,
                      "peak_cuda_memory_bytes": manifest["peak_cuda_memory_bytes"]}), flush=True)
    del model
    gc.collect()


if __name__ == "__main__":
    main()

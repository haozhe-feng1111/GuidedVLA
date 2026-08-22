#!/usr/bin/env python3
"""Extract post-block action-expert representations for the frozen CKNNA dataset."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

from safetensors.torch import load_file, save_file
import torch

from openpi.models import model as model_lib
from openpi.models import pi0_config
from openpi.models_pytorch import pi0_pytorch
from openpi.training import config as config_lib
from scripts import train_pytorch as train_lib


TIMESTEPS = (0.001, 0.25, 0.5)
NOISE_REPLICATES = 3
NUM_LAYERS = 18


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    candidate = path / "model.safetensors"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Cannot resolve model.safetensors from {path}")


def load_policy(config_name: str, checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    train_config = config_lib.get_config(config_name)
    if not isinstance(train_config.model, pi0_config.Pi0Config):
        raise TypeError(f"Expected Pi0Config, got {type(train_config.model)}")
    runtime_config = train_config.model
    object.__setattr__(runtime_config, "dtype", train_config.pytorch_training_precision)
    model = pi0_pytorch.PI0Pytorch(runtime_config)
    del model.paligemma_with_expert.gemma_expert.lm_head

    control_kwargs = {
        "num_control_heads": getattr(runtime_config, "control_attention_num_heads", None),
        "copy_weights": getattr(runtime_config, "control_attention_copy_weights", None),
        "freeze_origin": getattr(runtime_config, "control_attention_freeze_origin", None),
        "use_headwise_gate": getattr(runtime_config, "control_attention_use_headwise_gate", None),
    }
    if getattr(runtime_config, "control_attention_enabled", False):
        model.enable_control_attention(**control_kwargs)

    checkpoint_file = resolve_checkpoint(checkpoint)
    state_dict = load_file(checkpoint_file, device="cpu")
    state_dict = train_lib.normalize_state_dict_for_loading(state_dict, source_label="CKNNA checkpoint")
    model_lib._validate_depth_encoder_checkpoint_type(  # noqa: SLF001
        state_dict,
        depth_encoder_type=runtime_config.depth_encoder_type,
    )
    with train_lib.temporarily_unwrap_compiled_modules(model, log_prefix="CKNNA checkpoint load") as target:
        missing_keys, unexpected_keys = target.load_state_dict(state_dict, strict=False)
    model_lib._validate_external_adapter_weights_loaded(state_dict, missing_keys, unexpected_keys)  # noqa: SLF001
    expected_missing, unexpected_missing = train_lib.split_missing_keys(missing_keys)
    if unexpected_missing:
        raise RuntimeError(f"Unexpected missing checkpoint keys: {unexpected_missing}")
    if unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected_keys}")
    ca_missing = [key for key in missing_keys if ".origin." in key or "object_branch" in key]
    if ca_missing:
        raise RuntimeError(f"ControlAttention checkpoint keys are missing: {ca_missing[:5]}")

    load_report = {
        "checkpoint_file": str(checkpoint_file),
        "checkpoint_sha256": sha256_file(checkpoint_file),
        "expected_missing_keys": sorted(expected_missing),
        "unexpected_missing_keys": [],
        "unexpected_keys": [],
        "control_attention_enabled": bool(getattr(runtime_config, "control_attention_enabled", False)),
        "depth_encoder_type": runtime_config.depth_encoder_type,
        "guided_layer_indices": list(runtime_config.guided_layer_indices),
        "depth_guided_layer_indices": list(model.depth_guided_layer_indices),
    }
    del state_dict
    gc.collect()
    model = model.to(device)
    model.eval()
    return model, load_report


def observation_from_cache(cached: dict[str, torch.Tensor], cache_manifest: dict, index: int, device: torch.device):
    images = {}
    image_masks = {}
    for key in cache_manifest["image_keys"]:
        uint8_image = cached[f"image__{key}"][index : index + 1]
        images[key] = (uint8_image.to(device=device, dtype=torch.float32) / 255.0 * 2.0 - 1.0).contiguous()
        image_masks[key] = cached[f"image_mask__{key}"][index : index + 1].to(device=device, non_blocking=True)
    return model_lib.Observation(
        images=images,
        image_masks=image_masks,
        state=cached["state"][index : index + 1].to(device=device, non_blocking=True),
        tokenized_prompt=cached["tokenized_prompt"][index : index + 1].to(device=device, non_blocking=True),
        tokenized_prompt_mask=cached["tokenized_prompt_mask"][index : index + 1].to(
            device=device, non_blocking=True
        ),
    )


def build_prefix_cache(model: torch.nn.Module, observation):
    images, image_masks, lang_tokens, lang_masks, state = model._preprocess_observation(observation, train=False)
    depth_kv = model.compute_depth_key_values(images)
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, image_masks, lang_tokens, lang_masks
    )
    (prefix_embs,) = model._align_prefix_embeddings_dtype(prefix_embs)
    prefix_att_2d_masks = pi0_pytorch.make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks, dtype=prefix_embs.dtype)
    _, past_key_values, _ = model.paligemma_with_expert(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    return state, prefix_pad_masks, past_key_values, depth_kv


def capture_one_condition(
    model: torch.nn.Module,
    state: torch.Tensor,
    prefix_pad_masks: torch.Tensor,
    past_key_values,
    depth_kv,
    actions: torch.Tensor,
    noise: torch.Tensor,
    timestep: float,
) -> torch.Tensor:
    captured: dict[int, torch.Tensor] = {}
    action_horizon = model.config.action_horizon

    def capture_hidden(layer_index: int, hidden_state: torch.Tensor) -> None:
        if layer_index in captured:
            raise RuntimeError(f"Layer observer fired twice for layer {layer_index}")
        if hidden_state.shape[1] < action_horizon:
            raise RuntimeError(f"Expert hidden state is shorter than action horizon: {hidden_state.shape}")
        action_hidden = hidden_state[:, -action_horizon:, :]
        captured[layer_index] = action_hidden.float().mean(dim=1).detach().cpu()

    def make_standard_layer_hook(layer_index: int):
        def hook(_module, _inputs, output) -> None:
            hidden_state = output[0] if isinstance(output, tuple) else output
            capture_hidden(layer_index, hidden_state)

        return hook

    expert_layers = model.paligemma_with_expert.gemma_expert.model.layers
    hook_handles = [layer.register_forward_hook(make_standard_layer_hook(index)) for index, layer in enumerate(expert_layers)]
    model.paligemma_with_expert._expert_hidden_state_observer = capture_hidden
    try:
        time_tensor = torch.full((1,), timestep, device=actions.device, dtype=torch.float32)
        x_t = timestep * noise + (1.0 - timestep) * actions
        model.denoise_step(
            state,
            prefix_pad_masks,
            past_key_values,
            x_t,
            time_tensor,
            depth_kv=depth_kv,
        )
    finally:
        model.paligemma_with_expert._expert_hidden_state_observer = None
        for handle in hook_handles:
            handle.remove()
    if sorted(captured) != list(range(NUM_LAYERS)):
        raise RuntimeError(f"Expected exactly 18 captured layers, got {sorted(captured)}")
    stacked = torch.stack([captured[layer][0] for layer in range(NUM_LAYERS)], dim=0).contiguous()
    if not torch.isfinite(stacked).all():
        raise RuntimeError("Non-finite action-expert hidden state")
    return stacked


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for the preregistered policy extraction")
    if args.model_name == "nohead" and args.config_name != "pi0_libero":
        raise ValueError("nohead must use config pi0_libero")
    if args.model_name == "da3" and args.config_name != "pi0_libero_object_depth_skill":
        raise ValueError("da3 must use config pi0_libero_object_depth_skill")
    if args.model_name == "dinov2" and args.config_name != "pi0_libero_object_dinov2_base_skill":
        raise ValueError("dinov2 must use config pi0_libero_object_dinov2_base_skill")
    if args.model_name != "nohead":
        if args.external_encoder_path is None:
            raise ValueError("Guided models require --external-encoder-path")
        os.environ["OPENPI_DEPTH_MODEL_PATH"] = str(args.external_encoder_path)

    device = torch.device(args.device)
    cache_manifest = json.loads(args.cache_manifest.read_text())
    if cache_manifest["num_samples"] != 4000 or cache_manifest["num_shards"] != 40:
        raise ValueError("Unexpected reusable cache dimensions")
    if not 1 <= args.max_shards <= 40:
        raise ValueError("max-shards must be in [1, 40]")
    if not 1 <= args.samples_per_shard <= 100:
        raise ValueError("samples-per-shard must be in [1, 100]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_manifest_path = args.output_dir / "feature_manifest.json"
    if feature_manifest_path.exists() or any(args.output_dir.glob("*.safetensors")):
        raise FileExistsError("Refusing to reuse or overwrite a non-empty policy output directory")

    cuda_device_index: int | None = None
    if device.type == "cuda":
        cuda_device_index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_device(cuda_device_index)
    torch.cuda.reset_peak_memory_stats(cuda_device_index)
    model, load_report = load_policy(args.config_name, args.checkpoint, device)
    extraction_start = time.monotonic()
    shard_records: list[dict] = []
    total_samples = 0
    feature_dim: int | None = None
    expected_samples = args.max_shards * args.samples_per_shard

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        selected_shards = cache_manifest["shards"][: args.max_shards]
        for cache_shard in selected_shards:
            cache_path = Path(cache_shard["path"])
            if sha256_file(cache_path) != cache_shard["sha256"]:
                raise ValueError(f"Cache shard hash mismatch: {cache_path}")
            cached = load_file(cache_path)
            sample_index = cached["sample_index"][: args.samples_per_shard].to(torch.int64)
            if sample_index.shape[0] != args.samples_per_shard:
                raise ValueError(f"Unexpected cache shard length: {sample_index.shape}")

            task_features: torch.Tensor | None = None
            for local_index in range(args.samples_per_shard):
                observation = observation_from_cache(cached, cache_manifest, local_index, device)
                actions = cached["actions"][local_index : local_index + 1].to(device=device, non_blocking=True)
                state, prefix_pad_masks, past_key_values, depth_kv = build_prefix_cache(model, observation)

                for timestep_index, timestep in enumerate(TIMESTEPS):
                    for replicate in range(NOISE_REPLICATES):
                        noise = cached[f"noise__replicate_{replicate}"][local_index : local_index + 1].to(
                            device=device, non_blocking=True
                        )
                        layer_features = capture_one_condition(
                            model,
                            state,
                            prefix_pad_masks,
                            past_key_values,
                            depth_kv,
                            actions,
                            noise,
                            timestep,
                        )
                        if task_features is None:
                            feature_dim = layer_features.shape[-1]
                            task_features = torch.empty(
                                len(TIMESTEPS),
                                NOISE_REPLICATES,
                                NUM_LAYERS,
                                args.samples_per_shard,
                                feature_dim,
                                dtype=torch.float32,
                            )
                        task_features[timestep_index, replicate, :, local_index, :] = layer_features

                del observation, actions, state, prefix_pad_masks, past_key_values, depth_kv

            if task_features is None or feature_dim is None:
                raise RuntimeError("No policy features were captured")
            output_path = args.output_dir / Path(cache_path).name
            tmp_path = output_path.with_suffix(".safetensors.tmp")
            save_file(
                {
                    "features": task_features.contiguous(),
                    "sample_index": sample_index.contiguous(),
                    "timesteps": torch.tensor(TIMESTEPS, dtype=torch.float32),
                },
                tmp_path,
            )
            tmp_path.replace(output_path)
            shard_records.append(
                {
                    "task_index": cache_shard["task_index"],
                    "suite": cache_shard["suite"],
                    "path": str(output_path),
                    "sha256": sha256_file(output_path),
                    "num_samples": args.samples_per_shard,
                    "feature_dim": feature_dim,
                    "first_sample_index": int(sample_index[0]),
                    "last_sample_index": int(sample_index[-1]),
                }
            )
            total_samples += args.samples_per_shard
            elapsed_seconds = time.monotonic() - extraction_start
            samples_per_second = total_samples / elapsed_seconds
            remaining_samples = expected_samples - total_samples
            print(
                json.dumps(
                    {
                        "event": "policy_extraction_progress",
                        "model": args.model_name,
                        "completed_shards": len(shard_records),
                        "total_shards": args.max_shards,
                        "completed_samples": total_samples,
                        "total_samples": expected_samples,
                        "samples_per_second": samples_per_second,
                        "eta_seconds": remaining_samples / samples_per_second,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del cached, task_features
            gc.collect()

    if total_samples != expected_samples or len(shard_records) != args.max_shards or feature_dim is None:
        raise ValueError(f"Incomplete policy extraction: samples={total_samples}, shards={len(shard_records)}")
    feature_manifest = {
        "schema_version": "guidedvla-cknna-policy-features-v1",
        "model_name": args.model_name,
        "config_name": args.config_name,
        "load_report": load_report,
        "external_encoder_path": str(args.external_encoder_path) if args.external_encoder_path else None,
        "timesteps": TIMESTEPS,
        "noise_replicates": NOISE_REPLICATES,
        "num_layers": NUM_LAYERS,
        "layer_location": "post_block_pre_final_rmsnorm",
        "pooling": "mean_over_last_50_action_tokens_excluding_state_token",
        "autocast_dtype": "torch.bfloat16",
        "saved_dtype": "torch.float32",
        "extraction_scope": "full" if (args.max_shards, args.samples_per_shard) == (40, 100) else "smoke",
        "requested_max_shards": args.max_shards,
        "requested_samples_per_shard": args.samples_per_shard,
        "source_cache_manifest": str(args.cache_manifest),
        "source_cache_manifest_sha256": sha256_file(args.cache_manifest),
        "num_samples": total_samples,
        "num_shards": len(shard_records),
        "feature_dim": feature_dim,
        "extraction_wall_seconds": time.monotonic() - extraction_start,
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(cuda_device_index) if cuda_device_index is not None else None
        ),
        "cuda_peak_reserved_bytes": (
            torch.cuda.max_memory_reserved(cuda_device_index) if cuda_device_index is not None else None
        ),
        "shards": shard_records,
    }
    tmp_manifest = feature_manifest_path.with_suffix(".json.tmp")
    tmp_manifest.write_text(json.dumps(feature_manifest, indent=2, sort_keys=True) + "\n")
    tmp_manifest.replace(feature_manifest_path)
    print(json.dumps({"model": args.model_name, "num_samples": total_samples, "feature_dim": feature_dim}))


if __name__ == "__main__":
    main()

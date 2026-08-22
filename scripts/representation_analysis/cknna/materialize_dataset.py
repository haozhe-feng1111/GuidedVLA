#!/usr/bin/env python3
"""Materialize the frozen manifest into reusable per-task model-input shards."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path

from safetensors.torch import save_file
import torch
from torch.utils.data import Subset

from openpi.models import model as model_lib
from openpi.training import config as config_lib
from openpi.training import data_loader as data_loader_lib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--assets-base", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-name", default="pi0_libero_object_depth_skill")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def image_to_uint8_chw(image: torch.Tensor) -> tuple[torch.Tensor, float]:
    image = image.detach().cpu()
    if image.ndim != 4:
        raise ValueError(f"Expected batched image tensor, got shape={tuple(image.shape)}")
    if image.shape[-1] == 3:
        image = image.permute(0, 3, 1, 2)
    if image.shape[1] != 3:
        raise ValueError(f"Expected RGB image tensor, got shape={tuple(image.shape)}")
    if image.dtype == torch.uint8:
        uint8_image = image.contiguous()
        reconstructed = uint8_image.float() / 255.0 * 2.0 - 1.0
        return uint8_image, 0.0 if not torch.is_floating_point(image) else float((image - reconstructed).abs().max())
    float_image = image.float()
    if float_image.min() < -1.00001 or float_image.max() > 1.00001:
        raise ValueError(f"Image is outside [-1, 1]: min={float_image.min()}, max={float_image.max()}")
    uint8_image = torch.round((float_image + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    reconstructed = uint8_image.float() / 255.0 * 2.0 - 1.0
    max_error = float((float_image - reconstructed).abs().max())
    if max_error > 1e-6:
        raise ValueError(f"Image cannot be losslessly represented as uint8; max_error={max_error}")
    return uint8_image.contiguous(), max_error


def flush_task_shard(
    output_dir: Path,
    rows: list[dict],
    tensors_by_key: dict[str, list[torch.Tensor]],
) -> dict:
    if len(rows) != 100:
        raise ValueError(f"Expected 100 rows per task shard, got {len(rows)}")
    task_index = rows[0]["task_index"]
    suite = rows[0]["suite"]
    if any(row["task_index"] != task_index or row["suite"] != suite for row in rows):
        raise ValueError("Task shard contains mixed task/suite rows")
    tensors = {key: torch.cat(values, dim=0).contiguous() for key, values in tensors_by_key.items()}
    if any(tensor.shape[0] != 100 for tensor in tensors.values()):
        raise ValueError({key: tuple(value.shape) for key, value in tensors.items()})
    expected_indices = torch.tensor([row["sample_index"] for row in rows], dtype=torch.int64)
    if not torch.equal(tensors["sample_index"], expected_indices):
        raise ValueError("Cached sample_index order does not match manifest")

    shard_path = output_dir / f"task_{task_index:02d}_{suite}.safetensors"
    if shard_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing shard: {shard_path}")
    tmp_path = shard_path.with_suffix(".safetensors.tmp")
    save_file(tensors, tmp_path)
    tmp_path.replace(shard_path)
    return {
        "task_index": task_index,
        "suite": suite,
        "path": str(shard_path),
        "sha256": sha256_file(shard_path),
        "num_samples": len(rows),
        "first_sample_index": rows[0]["sample_index"],
        "last_sample_index": rows[-1]["sample_index"],
        "tensor_shapes": {key: list(value.shape) for key, value in tensors.items()},
        "tensor_dtypes": {key: str(value.dtype) for key, value in tensors.items()},
    }


def main() -> None:
    args = parse_args()
    manifest = read_jsonl(args.manifest)
    if len(manifest) != 4000:
        raise ValueError(f"Expected 4,000 manifest rows, got {len(manifest)}")
    if [row["sample_index"] for row in manifest] != list(range(4000)):
        raise ValueError("Manifest sample_index must be exactly 0..3999")
    if args.batch_size <= 0 or 100 % args.batch_size:
        raise ValueError("batch-size must be a positive divisor of 100")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_manifest_path = args.output_dir / "cache_manifest.json"
    if cache_manifest_path.exists() or any(args.output_dir.glob("*.safetensors")):
        raise FileExistsError("Refusing to reuse or overwrite a non-empty cache output directory")

    train_config = config_lib.get_config(args.config_name)
    base_config = train_config.data.base_config or config_lib.DataConfig()
    base_config = dataclasses.replace(base_config, local_root_dir=str(args.dataset_root))
    data_factory = dataclasses.replace(train_config.data, base_config=base_config)
    train_config = dataclasses.replace(
        train_config,
        data=data_factory,
        assets_base_dir=str(args.assets_base),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    data_config = dataclasses.replace(data_config, use_object_loss=False)
    if data_config.norm_stats is None:
        raise ValueError("Normalization stats were not loaded")
    if Path(data_config.local_root_dir).resolve() != args.dataset_root.resolve():
        raise ValueError(f"Unexpected dataset root: {data_config.local_root_dir}")

    dataset = data_loader_lib.create_torch_dataset(
        data_config,
        action_horizon=train_config.model.action_horizon,
        model_config=train_config.model,
        split="all",
    )
    if len(dataset) != 277947:
        raise ValueError(f"Unexpected full dataset length: {len(dataset)}")
    transformed = data_loader_lib.transform_dataset(dataset, data_config)
    subset = Subset(transformed, [row["global_frame_index"] for row in manifest])
    torch_loader = data_loader_lib.TorchDataLoader(
        subset,
        local_batch_size=args.batch_size,
        shuffle=False,
        num_batches=math.ceil(len(manifest) / args.batch_size),
        num_workers=args.num_workers,
        seed=20260820,
        framework="pytorch",
    )
    loader = data_loader_lib.DataLoaderImpl(data_config, torch_loader, framework="pytorch")

    shard_records: list[dict] = []
    current_rows: list[dict] = []
    current_tensors: dict[str, list[torch.Tensor]] = {}
    cursor = 0
    max_image_roundtrip_error = 0.0
    image_keys: list[str] | None = None

    for observation, actions, object_targets in loader:
        if object_targets is not None:
            raise ValueError("Object targets should be disabled for the representation cache")
        batch_size = actions.shape[0]
        batch_rows = manifest[cursor : cursor + batch_size]
        if len(batch_rows) != batch_size:
            raise ValueError("Loader produced more samples than the manifest")
        if len({row["task_index"] for row in batch_rows}) != 1:
            raise ValueError("A cache batch crossed task boundaries")

        batch_tensors: dict[str, torch.Tensor] = {
            "sample_index": torch.tensor([row["sample_index"] for row in batch_rows], dtype=torch.int64),
            "state": observation.state.detach().cpu().float().contiguous(),
            "tokenized_prompt": observation.tokenized_prompt.detach().cpu().contiguous(),
            "tokenized_prompt_mask": observation.tokenized_prompt_mask.detach().cpu().bool().contiguous(),
            "actions": actions.detach().cpu().float().contiguous(),
        }
        for replicate in range(3):
            noises = []
            for row in batch_rows:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(int(row["noise_seeds"][replicate]))
                noises.append(torch.randn(actions.shape[1:], generator=generator, dtype=torch.float32))
            batch_tensors[f"noise__replicate_{replicate}"] = torch.stack(noises, dim=0).contiguous()
        if image_keys is None:
            image_keys = list(observation.images)
        if list(observation.images) != image_keys:
            raise ValueError("Image key order changed across batches")
        for key in image_keys:
            uint8_image, error = image_to_uint8_chw(observation.images[key])
            max_image_roundtrip_error = max(max_image_roundtrip_error, error)
            batch_tensors[f"image__{key}"] = uint8_image
            batch_tensors[f"image_mask__{key}"] = observation.image_masks[key].detach().cpu().bool().contiguous()

        for key, value in batch_tensors.items():
            current_tensors.setdefault(key, []).append(value)
        current_rows.extend(batch_rows)
        cursor += batch_size

        if len(current_rows) == 100:
            shard_records.append(flush_task_shard(args.output_dir, current_rows, current_tensors))
            current_rows = []
            current_tensors = {}
        elif len(current_rows) > 100:
            raise ValueError("Task shard accumulated more than 100 rows")

    if cursor != len(manifest) or current_rows:
        raise ValueError(f"Incomplete cache materialization: cursor={cursor}, trailing_rows={len(current_rows)}")
    if len(shard_records) != 40:
        raise ValueError(f"Expected 40 cache shards, got {len(shard_records)}")
    if image_keys is None:
        raise ValueError("No image keys were observed")

    cache_manifest = {
        "schema_version": "guidedvla-cknna-cache-v1",
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": sha256_file(args.manifest),
        "config_name": args.config_name,
        "dataset_root": str(args.dataset_root),
        "assets_base": str(args.assets_base),
        "num_samples": len(manifest),
        "num_shards": len(shard_records),
        "image_keys": image_keys,
        "primary_image_key": image_keys[0],
        "max_image_roundtrip_error": max_image_roundtrip_error,
        "shards": shard_records,
    }
    if cache_manifest["primary_image_key"] not in image_keys:
        raise ValueError(
            f"Primary policy image key {cache_manifest['primary_image_key']!r} is absent from cache keys {image_keys}"
        )
    tmp_manifest = cache_manifest_path.with_suffix(".json.tmp")
    tmp_manifest.write_text(json.dumps(cache_manifest, indent=2, sort_keys=True) + "\n")
    tmp_manifest.replace(cache_manifest_path)
    print(json.dumps({"cache_manifest": str(cache_manifest_path), "num_shards": len(shard_records)}))


if __name__ == "__main__":
    main()

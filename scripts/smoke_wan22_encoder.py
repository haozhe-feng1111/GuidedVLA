#!/usr/bin/env python3
"""Load the official Wan2.2 VAE encoder and verify its single-frame token contract."""

from __future__ import annotations

import argparse
import time

import torch

from openpi.models_pytorch.wan22.model import Wan22VAEEncoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    encoder = Wan22VAEEncoder(args.source_root, args.checkpoint_path, dtype=args.dtype).to(device).eval()
    load_seconds = time.perf_counter() - start

    image = torch.zeros(1, 3, 224, 224, dtype=torch.float32, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    outputs = encoder(image)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - start

    if len(outputs) != 4 or any(output.shape != (1, 256, 48) for output in outputs):
        raise RuntimeError(f"Unexpected Wan2.2 encoder output: {[tuple(output.shape) for output in outputs]}")
    if not all(output is outputs[0] for output in outputs):
        raise RuntimeError("Wan2.2 guided layers must receive the same final feature tensor")
    if not torch.isfinite(outputs[0]).all():
        raise RuntimeError("Wan2.2 encoder produced non-finite tokens")

    print(f"load_seconds={load_seconds:.3f}")
    print(f"forward_seconds={forward_seconds:.3f}")
    print(f"output_shape={tuple(outputs[0].shape)} output_dtype={outputs[0].dtype}")
    print(f"output_mean={outputs[0].float().mean().item():.6f} output_std={outputs[0].float().std().item():.6f}")
    if device.type == "cuda":
        peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
        print(f"peak_cuda_allocated_gib={peak_gib:.3f}")


if __name__ == "__main__":
    main()

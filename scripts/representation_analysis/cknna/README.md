# CKNNA representation analysis

This directory contains the analysis pipeline used to compare GuidedVLA policy
representations with frozen DINOv2 and Depth Anything 3 (DA3) encoders.

The pipeline supports:

- a deterministic, reusable 4,000-frame LIBERO sample manifest;
- frozen model-input caches shared across checkpoints;
- action-expert representations at diffusion timesteps 0.001, 0.25, and 0.5;
- PaliGemma visual-token and SigLIP vision-block representations;
- fixed-layer and all-reference-layer DINOv2/DA3 comparisons;
- task-macro CKNNA, episode-excluded neighborhoods, paired bootstrap confidence
  intervals, and max-T multiple-comparison correction.

## Pipeline

Run scripts from the repository root with `PYTHONPATH=.` so both the GuidedVLA
package and the sibling analysis modules resolve consistently. For example:

```bash
PYTHONPATH=. python scripts/representation_analysis/cknna/compute_cknna.py --help
```

All paths are explicit CLI arguments; datasets, checkpoints, cached features,
and generated results are intentionally not tracked by Git.

1. `build_manifest.py`: construct the deterministic LIBERO sample manifest.
2. `materialize_dataset.py`: save reusable, checkpoint-independent input shards.
3. `extract_policy_features.py`: extract action-expert representations.
4. `extract_vision_features.py`: extract PaliGemma and SigLIP representations.
5. `extract_encoder_features.py`: extract fixed layer-11 reference features.
6. `extract_encoder_all_layers.py`: extract every DINOv2/DA3 reference layer.
7. `compute_cknna.py`: compute the action-side 2-by-3 CKNNA analysis.
8. `compute_vision_cknna.py`: compute the fixed-reference vision analysis.
9. `compute_vision_reference_layer_sweep.py`: sweep policy and reference layers.
10. `compute_vision_stage1_centered_sweep.py`: compare Stage-2 checkpoints with
    the shared Stage-1 initialization.

The `validate_*.py`, `check_encoder_identity.py`,
`compute_paired_differences.py`, `summarize_results.py`, and plotting scripts
provide integrity checks and downstream summaries.

Use `--help` on each entry point for its complete arguments. The extraction
scripts expect the repository environment plus `pyarrow`, `safetensors`,
PyTorch, NumPy, and Matplotlib.

## Tests

From the repository root:

```bash
PYTHONPATH=scripts/representation_analysis/cknna python -m pytest -q \
  scripts/representation_analysis/cknna/test_analysis.py \
  scripts/representation_analysis/cknna/test_reference_layer_sweep.py \
  scripts/representation_analysis/cknna/test_stage1_centered_sweep.py
```

The tests check the unbiased HSIC/CKNNA implementation against a direct
reference calculation and verify the vectorized vision-layer sweeps.

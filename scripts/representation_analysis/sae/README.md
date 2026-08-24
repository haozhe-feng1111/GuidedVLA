# SAE representation analysis

This directory is the data-independent core of the GuidedVLA sparse
autoencoder analysis. It follows the TopK + AuxK setup in Swann et al.,
"Sparse Autoencoders Reveal Interpretable and Steerable Features in VLA
Models" (2026), while keeping activation extraction separate from SAE
training.

Implemented in the initial scaffold:

- bias-free TopK encoding and unit-norm decoder columns;
- dead-latent AuxK reconstruction loss;
- per-sample centering and L2 normalization around a geometric-median bias;
- decoder-gradient projection, checkpoint save, and exact reload;
- episode coverage, mean onset count, mean activation magnitude, and relative
  run length using the paper's activation hysteresis;
- deterministic fake-activation training smoke.

Run the CPU-only checks from this directory:

```bash
PYTHONPATH=scripts/representation_analysis/sae python -m pytest -q \
  scripts/representation_analysis/sae/test_sae.py

PYTHONPATH=scripts/representation_analysis/sae python \
  scripts/representation_analysis/sae/fake_smoke.py \
  --device cpu \
  --steps 300 \
  --output-dir /tmp/guidedvla-sae-fake-smoke
```

The fake smoke is not evidence about a policy checkpoint. It only verifies
optimization, sparse encoding, checkpoint recovery, and metric execution.

The next integration step is an episode-preserving activation-shard manifest.
Every checkpoint must consume the same ordered LIBERO observations, but each
checkpoint trains a separate SAE from its own activations. Model hooks and
real-data extraction are intentionally not part of this first scaffold.

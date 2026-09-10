# Depth-only Stage2, effective batch 24

This candidate tests whether removing object/skill supervision changes the
contribution of the external depth representation. It does not assume encoder
score differences will increase. No experiment is launched by publishing this file.

## Comparison contract

Start from the same `pi0_libero` Stage1 step-30000 weights as the full-guidance
Depth B24 arm, not from an existing Stage2 checkpoint. Use `pi0_libero_depth`:
DA3-SMALL, guided layers 9–12, depth heads 4/5, the existing four intermediate
features and TokenMerging/KV projectors. All eight attention heads remain;
object/skill supervision and their losses are disabled. The other heads continue
ordinary attention. The main vision backbone and control-attention architecture
are unchanged. No encoder sweep or evaluation is included.

Four ranks × six examples = physical global batch 24, accumulation 1 = effective
batch 24. Preserve 30K optimizer steps, float32 training, gradient checkpointing,
DDP unused-parameter handling, two data workers, seed/LR/optimizer defaults,
logging every 10 steps, saving and validation every 10K steps, one validation
batch, and the released `ybwowen/libero` data/delta-action transform.

The existing full-guidance forward disables image augmentation because object
labels use original image coordinates. Simply switching off object loss would
otherwise enable augmentation. The new `disable_image_augmentation` model flag
therefore defaults to false for compatibility, and this launcher sets it true.
Object/skill labels and supervision are no longer requested; this is an intended
consequence of the ablation. No claim of identical random draws is made.

This launcher is based on main `efde8e0450bf239d82df54dd5dbed338eb6b6d2e` and the
existing depth launcher, with B24/GA1 matching the recorded Depth B24 recipe.
Historical reference-arm identity and the full resolved configuration must still
be compared before a formal run. See the bounded smoke validation below.

## Review and launch

From the repository root, print the command without accessing GPUs or writing:

```bash
GUIDEDVLA_PRINT_ONLY=1 bash manifests/train_libero_stage2_depth_only_b24_4gpu.sh
```

Paths default to a sibling asset/runtime layout. Override `GUIDEDVLA_BASE`,
`RUNTIME`, `DATA_ROOT`, `ASSETS_ROOT`, `TOKENIZER_PATH`, `DEPTH_MODEL`,
`STAGE1_CHECKPOINT`, and `NORM_STATS_SOURCE` for the existing target assets.
`NORM_STATS_SOURCE` must be the same normalization assets as the full-guidance
Depth B24 run. `GUIDEDVLA_RUN_ID` selects isolated output/log/cache/W&B directories.
Existing output/log roots are refused; resume and overwrite are not enabled.

`GUIDEDVLA_CHECK_ONLY=1` checks asset paths and four GPU memory readings, then
prints the command without starting training. It is not a model-loading smoke.
Without either review flag, the script starts training and requires separate
experiment approval. W&B remains offline as in the original recipe.

Primary outcome for a later approved comparison is full LIBERO-Plus success rate
with exact episode completeness; suite/category results are secondary. Runtime
budget, retries and numerical success thresholds are not yet approved. First
validate initialization and effective configuration, then obtain approval before
formal training. Preserve launch metadata, resolved config, loss logs and final
checkpoint identity. Stop on configuration mismatch or invalid initialization.

## Bounded smoke validation (2026-09-10)

Four GPUs, B24/GA1, three optimizer steps and one validation batch completed with
exit code 0 after fixing the launcher temporary path. The first attempt failed
before training with `AF_UNIX path too long`: appending the run ID to the asset
root made multiprocessing forkserver socket paths too long. The default now uses
a short per-process `/tmp/gvla-UID-PID` directory; `GUIDEDVLA_TMP_ROOT` can override
it with a short path. Existing results were preserved, and the retry reused only
the completed data cache.

The successful run used the published model implementation and a smoke copy of
the launcher (3 steps, log interval 1, save/validation interval 3, independent run
ID, short TMPDIR). Train losses were approximately 0.0027, 0.0036, 0.0094; validation
loss was 0.00399. Checkpoint metadata confirms global batch 24, accumulation 1,
object/skill disabled, depth enabled, augmentation disabled, heads 4/5 and layers
9–12. This verifies initialization, forward/backward, optimizer updates, validation
and checkpoint writing, not convergence, checkpoint reload or benchmark quality.

CPU regression tests cover the real forward preprocessing policy and creation of
a multiprocessing-style Unix socket with a long asset-root path. Both default
pytest discovery and the push-triggered CPU workflow now include these tests.

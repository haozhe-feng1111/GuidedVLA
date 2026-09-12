# Wan-only Stage2, global batch 24

This arm matches the Depth-only B24 Stage2 experiment's training policy and
replaces the external depth branch with the existing frozen Wan2.2 VAE encoder.
Object and skill supervision are disabled in both model and data configuration.
The eight action-expert attention heads remain; Wan guides heads 4/5 in layers
9–12. "Only Wan" means one external guidance branch, not one attention head.
The primary PaliGemma/SigLIP path remains active.

## Platform startup command

Select **one node with four 80GB-class GPUs**, the existing GuidedVLA runtime
image and the personal dataset mount. Put these two lines in the platform's
startup command field (use the actual prepared checkout path):

```bash
cd <personal-base>/repo-wan-only-b24-company-20260912
bash manifests/train_libero_stage2_wan_only_b24_4gpu_company.sh
```

This is a training entry point, following the repository's company launchers.
It does not create a scheduler job, select an account/image/resource pool, or
run on a development GPU automatically. The shared runtime must be compatible
with the selected platform image; development-host checks cannot prove that.

Default layout under `<personal-base>`: `runtime/.venv`, `models`, `assets`,
`external/Wan2.2`, and `outputs`. The public LIBERO dataset is under
`<owner-root>/datasets/public/ybwowen/libero-477f7959`. The launcher derives these
from its own checkout location, following the existing company layout.
All model/data/runtime path overrides are environment variables in the script.

For CPU-only asset/hash/dependency and resolved-config validation:

```bash
GUIDEDVLA_CHECK_ONLY=1 bash manifests/train_libero_stage2_wan_only_b24_4gpu_company.sh
```

`GUIDEDVLA_PRINT_ONLY=1` only prints the final command and performs no imports,
asset checks, GPU calls or writes. Both modes exit before training. The config
checker invokes the production parser and substitutes only its training callback.

## Fixed experiment contract

- Config: `pi0_libero_wan22_vae_only_b24`.
- Load the Stage1 `pi0_libero` 30000 model weights, with fresh Stage2 optimizer
  and global step. This is not a resume of a previous Wan Stage2 checkpoint.
- 4 DDP ranks × local batch 6 × accumulation 1 = effective batch 24.
- 30000 steps; log every 10; save and validate every 10000 (one validation batch).
- Original float32 training precision, full gradient checkpointing, no compile.
  Depth selective-checkpoint performance probes are not applied to this Wan arm.
- Image augmentation disabled, matching the controlled Depth-only arm and the
  original object-supervised Stage2 preprocessing policy.
- Wan VAE uses BF16, frozen/eval, current base-camera frame, deterministic
  normalized posterior mean. Native 14×14×48 features resize to 16×16×48, shared
  by four independent trainable K/V projectors. No optical flow or video sampling.
- Object/skill model and data flags false; both loss weights zero.
- Default optimizer/LR/seed inherited from the same frozen training code.
- W&B offline; logs and offline files stay under a unique run directory.

The Depth and Wan token interfaces differ; this is an external-vision integration
comparison, not a strict backbone-only replacement. The hypothesis is that the
Wan-only arm changes robustness under the same training policy. Training loss
does not establish this; a later separately authorized, matched evaluation is
required. No evaluation, sweep, or automation is launched by this entry point.

## Evidence and stopping behavior

The default run ID is `guidedvla_libero_stage2_wan_only_b24_ga1_30k_company_v1`.
Existing output/log/cache/W&B directories cause refusal; change
`GUIDEDVLA_RUN_ID` only for an intentional new run. An atomic log-directory claim
prevents simultaneous launches of the same run. No automatic restart/resume.
GPU checks require exactly four visible, largely idle, 80GB-class devices and
preserve the scheduler's CUDA visibility. Multi-node launch is not supported.

Under `logs/<run>`: `resolved-config.log`, `launch_metadata.txt`, `train.log`,
and eventual `exit_code`. Metadata includes code/config/launcher/normalization/
Stage1 weight hashes, pinned Wan provenance and the exact command.
Outputs are `outputs/<run>/pi0_libero_wan22_vae_only_b24/<run>/<step>`.

CPU checks verify resolved configuration and assets, not GPU capacity, actual
weight loading or optimizer behavior. This new Wan-only B24 arm still requires
target-image GPU smoke before claiming runtime validation. No GPU run is part
of code preparation.

# RoboTwin Evaluation

This example evaluates an OpenPI / GuidedVLA policy on [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) without modifying the RoboTwin codebase.

Recommended setup:

```bash
git submodule update --init --recursive third_party/RoboTwin
```

The integration is aligned with RoboTwin's `policy/pi05` branch:

- observation/state order is `[left_arm(6), left_gripper, right_arm(6), right_gripper]`
- actions are **absolute** joint targets in the same 14-D order
- grippers are already normalized to `[0, 1]`
- training still keeps `use_delta_joint_actions=True`, so actions are converted to deltas for the model and converted back to absolute qpos at inference
- `adapt_to_pi=False` is required for RoboTwin and is baked into the RoboTwin configs

## 1. Prepare LeRobot data

Assume you already have a LeRobot-format RoboTwin dataset available locally or on the Hugging Face Hub.

## 2. Compute norm stats

The RoboTwin configs use the fixed asset id `robotwin`, so norm stats should be written there.

```bash
uv run scripts/compute_norm_stats.py pi0_base_aloha_robotwin_full \
  --repo-id robotwin_grab_roller_demo_randomized \
  --local-root-dir /path/to/lerobot/root \
  --asset-id robotwin
```

You can swap `pi0_base_aloha_robotwin_full` for any of the RoboTwin configs below.

## 3. Train

Available RoboTwin configs:

- `pi05_aloha_robotwin_full`
- `pi05_aloha_robotwin_lora`
- `pi0_base_aloha_robotwin_full`
- `pi0_base_aloha_robotwin_lora`
- `pi0_fast_aloha_robotwin_full`
- `pi0_fast_aloha_robotwin_lora`
- `pi0_base_aloha_robotwin_object_depth_skill`

Notes:

- `scripts/train_pytorch.py` uses DDP via `torchrun`; `--nproc_per_node` controls PyTorch parallelism.
- `fsdp_devices` is only consumed by the JAX trainer (`scripts/train.py`) and is ignored by the PyTorch trainer.
- `pi0_base_aloha_robotwin_object_depth_skill` additionally requires `observation.skill_id`
  in the dataset; `skill_soft` will be constructed online during loading.

Backward-compatible aliases from the RoboTwin `policy/pi05` branch are also available:

- `pi05_aloha_full_base`
- `pi05_base_aloha_lora`

Example:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/train_pytorch.py \
  pi0_base_aloha_robotwin_full \
  --exp_name robotwin_grab_roller \
  --repo_id robotwin_grab_roller_demo_randomized \
  --local_root_dir /path/to/lerobot/root
```

## 4. Serve the checkpoint

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_base_aloha_robotwin_full \
  --policy.dir=checkpoints/pi0_base_aloha_robotwin_full/<exp_name>/<step>
```

## 5. Run RoboTwin evaluation

Run this inside an environment that already has RoboTwin installed through the repository submodule at `third_party/RoboTwin`.

```bash
bash examples/robotwin/run.sh \
  --args.task-name adjust_bottle \
  --args.task-config demo_randomized \
  --args.instruction-type unseen \
  --args.action-horizon 50 \
  --args.record-videos
```

Results are written under `data/robotwin/eval/<task>/<task_config>/`.

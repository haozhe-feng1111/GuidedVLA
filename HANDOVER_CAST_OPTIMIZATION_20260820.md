# 交接报告：LIBERO-Plus 正式实验（v3）训练提速探索

> 日期：2026-08-20
> 交接方：本地探索 Agent（已完成的探索、结论与资产）
> 接收方：负责实现「dtype cast 优化」的接手者
> 硬约束：**不得触碰正在运行的正式实验 v3 的任何资产**

---

## 0. 一句话摘要

正式实验 v3（DINOv2-Base 浅层注入消融，4×A100）当前 **eager 训练 13.5s/step（ETA ≈108 小时）**。Profile 证明瓶颈是 **CPU 侧 eager 调度开销（dtype cast + kernel launch），GPU 只忙 ~60%**。torch.compile 能解决 CPU 开销（单卡快 25%），但 4 卡 DDP 下被通信开销抵消（反而慢 9%）。**推荐下一步：① DINOv2 离线 bf16 化（零风险小收益）+ ② 4 卡 compile + NCCL 环境变量调参（纯环境实验）**。原计划的「fp32 master + bf16 shadow 批量 cast」方案因**显存余量不足（75.2/79GB 已占用）**当前不可行，需先降 batch 或等 v3 结束。

---

## 1. 背景与目标

### 1.1 正在跑的正式实验（v3，**不要动**）

| 项 | 值 |
|---|---|
| 实验 | `pi0_libero_object_dinov2_base_skill_shallow_guidance`（2×2 消融：监督层 9-12 + 注入层 5-8，特征层 [5,7,9,11]，DINOv2-Base 编码器） |
| 机器 | `ssh fudan-lab`（10.176.42.24，用户 junke，`~/.ssh/config` 已配） |
| GPU | **GPU 0-3**（各 ~75.2GB/79GB 常驻） |
| 进程 | PID 1920131-1920134（4 rank），torchrun 根 1920117 |
| 代码 | `/data/junke_dont_remove/haozhe/guidedvla_repro_20260730/repo-libero-dinov2-shallow-train-20260820` @ `f794ac2` |
| 日志 | `/data/junke_dont_remove/haozhe/guidedvla_repro_20260730/logs/guidedvla_libero_stage2_object_dinov2_base_skill_depth_shallow_4gpu_30k_20260820_v3/` |
| 输出 | `/data/junke_dont_remove/haozhe/guidedvla_repro_20260730/outputs/guidedvla_libero_stage2_object_dinov2_base_skill_depth_shallow_4gpu_30k_20260820_v3/` |
| 配置 | batch 16/卡 × 4 卡 × GA 4 = effective 256；float32 master + bf16 autocast；gradient checkpointing ON；`TORCH_COMPILE=0`（eager）；30k 步 |
| 速度 | **13.5s/step**，约 13% 进度（step ~1400+），ETA ~108h |
| 运行时 | `/data/junke_dont_remove/haozhe/guidedvla_repro_20260730/runtime/pi0-conversion-env`（**torch 2.7.1+cu126**） |

**禁止事项**：不得 kill 进程、不得改 `repo-libero-dinov2-shallow-train-20260820` 代码、不得动 v3 的 outputs/logs/checkpoints。

### 1.2 目标

在不改变训练语义（fp32 master、数值结果可比）的前提下，找到可落地的提速手段。探索计算资源：**GPU 4-7（空闲）**，可随意使用。

---

## 2. 已完成的探索与结论（含数据）

### 2.1 Profile 结论（决定性数据）

工具：`torch.profiler`（CPU+CUDA，record_shapes），GPU 4 单卡 eager，micro-batch 0/1（batch 16，fwd+bwd）。

**每 micro-batch 稳态（mb1，排除冷启动）：**

| 指标 | 数值 | 说明 |
|---|---|---|
| GPU 计算 | **2.07s** | 硬下限；`aten::mm` 858ms（41%）、attention fwd+bwd ~410ms、copy 268ms、elementwise ~300ms |
| CPU 总时间 | **3.36s** | **瓶颈**；wall ≈ 3.4s/mb × 4 mb = 13.5s/step ✓ 与 v3 吻合 |
| kernel launch | 16,972 次/mb（~0.7s CPU） | eager 固有 |
| dtype cast | `aten::to` 6,150 次 + `copy_` 6,823 次（CPU ~1.2s + GPU ~0.44s） | **最大可优化项** |
| `cudaStreamSynchronize` | 400 次/mb | 主要是早期 step 的 `log_memory_usage`（前 5 步），稳态影响小 |
| `cudaMemcpyAsync` | 909 次 | cast 的 D2D 为主 |

Profile 产物：`/tmp/gv-profile/ka_mb0.txt`、`ka_mb1.txt`（key_averages 表格）、`trace_mb0.json`、`trace_mb1.json`（72MB chrome trace）。

### 2.2 torch.compile 实验（已完成）

**OOM 根因（重要发现）**：`repo.../src/openpi/models_pytorch/gemma_pytorch.py:798`：
```python
use_checkpoint = self.training and self.use_gradient_checkpointing and not compile_active
```
`train_pytorch.py:1100` 会把 `paligemma_with_expert` 整体 compile → `compile_active=True` → **gradient checkpointing 被静默关闭** → activation 全保留 → OOM。**之前"全 compile 就 OOM"的真正机制**。

**已做改动**（仅探索副本）：去掉 `and not compile_active`，让 compile 与 checkpoint 共存（torch 2.7 官方支持）。

**结果（探索副本 `repo-compile-smoke-20260820`，配置与 v3 相同）：**

| 场景 | 每步耗时 | 峰值显存/卡 | 结论 |
|---|---|---|---|
| eager 4 卡（v3 现状） | 13.5s | 常驻 75.2GB | 基线 |
| compile 单卡（1×A100） | 10.3s（64 样本/步） | 68.6GB | ✅ 快 25%，不 OOM |
| **compile 4 卡 DDP** | **14.7s** | 72.8GB | ❌ **比 eager 慢 9%** |

**4 卡 compile 慢的原因**：单卡 compile 10.3s vs 4 卡 compile 14.7s → **DDP 通信/同步吃掉 ~4.4s/步**（GPU 计算仅 2.07s×4mb≈8.3s）。compile 省下的 CPU 开销被 DDP 在编译图上的同步抵消。编译 warmup 另需 ~14.5 分钟。

**实验事故记录（避免重犯）**：首次单卡 smoke 出现**两个进程同时抢 GPU 4**（43.17GB + 35.84GB 双 CUDA context 假 OOM）。根因是启动方式（ssh 后台 job 未完全 detach 留下残留）。**正确启动方式**：
```bash
mkdir -p /tmp/<dir>   # 必须预先存在（重定向目标）
setsid bash <script> > /tmp/<dir>/launcher.log 2>&1 < /dev/null &
```
且脚本内用 `nohup ... > "$LOG/train.log" 2>&1` 嵌套重定向，ssh 命令立即返回。

### 2.3 探索环境速查

- 探索代码副本：`/data/junke_dont_remove/haozhe/guidedvla_repro_20260730/repo-compile-smoke-20260820`
  - 已含补丁：gemma_pytorch.py:798（compile+checkpoint 共存）；train_pytorch.py（profiler 钩子，env `PROFILE_MICROBATCHES`/`PROFILE_OUT_DIR` 控制，`contextlib`/`os`/`torch.profiler` 均已 import 或需确认）
  - 启动脚本：`smoke_single_gpu.sh`（单卡）、`smoke_4gpu.sh`（4 卡）、`profile_run.sh`（单卡 eager profile）——均含 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`、`TORCH_COMPILE`/`GUIDEDVLA_COMPILE_DINO` 开关
- smoke 日志：`/data/junke_dont_remove/haozhe/guidedvla_repro_20260730/logs/compile_smoke_single_gpu_20260820/`、`logs/compile_smoke_4gpu_20260820/`、`logs/profile_eager_single_gpu_20260820/`
- 复用资产（只读）：norm stats `assets/pi0_libero_object_dinov2_base_skill_shallow_guidance/ybwowen/libero/norm_stats.json`；DINOv2 `models/dinov2-base/`（HF safetensors）；Stage-1 checkpoint `outputs/guidedvla_libero_stage1_4gpu_30k/pi0_libero/guidedvla_libero_stage1_4gpu_30k/30000/`；数据集 `datasets/ybwowen-libero-477f7959/`
- 环境变量要点：`PYTHONPATH` 需含 `third_party/depth_anything/src`（指向 `repo-libero-author-ga2-20260731/third_party/depth_anything/src`，探索副本未 checkout 子模块）；`TMPDIR` 用短路径（`/tmp/gv-*`），避免 `AF_UNIX path too long`（v2 失败教训）；`WANDB_MODE=disabled`（探索不打乱正式 wandb）

---

## 3. 推荐的优化方案（按优先级）

### 方案 A：DINOv2 离线 bf16 化（✅ 建议立即做，零风险）

**依据**：整个 forward 在 `torch.autocast(bf16)` 内（`train_pytorch.py:compute_batch_losses`），冻结的 DINOv2（`requires_grad=False` + eval）权重被每 mb 重复 cast，纯浪费。

**做法**：模型加载后对冻结编码器执行 `dinov2_model.to(torch.bfloat16)`（或在 checkpoint 转换阶段存 bf16 副本）。冻结模块权重不变，bf16 输出与原 autocast 行为一致，**不改变任何训练数值**。

**收益**：小（~2-3%），但零风险、改动 1 行。

### 方案 B：4 卡 compile + NCCL 调参（✅ 建议做，纯环境变量实验）

**依据**：compile 单卡快 25%，4 卡被 DDP 通信吃掉（+4.4s/步）。`train_pytorch.py` 注释提到历史上 `reduce-overhead`（CUDA graphs）在 multi-node DDP 崩过；现用 `mode="default"`。

**做法**（GPU 4-7，零代码改动）：4 卡 compile（`TORCH_COMPILE=1` + checkpoint 共存补丁）+ 依次试：
- `TORCH_NCCL_AVOID_RECORD_STREAMS=1`
- `NCCL_BUFFSIZE` / `TORCH_DISTRIBUTED_DETAIL=DEBUG`（观测 allreduce 时长）
- 对比基线 14.7s/step，若能压回 ≤12s 即值得正式采用

### 方案 C：批量 cast（原计划，**当前显存不可行**，需降 batch）

**思路**：fp32 master + bf16 shadow 参数，optimizer 更新 master，每 step 一次 `torch._foreach_` 批量 copy 到 bf16 shadow（6000 次 cast → ~1 次批量）。

**硬约束**：bf16 shadow 需额外 ~7GB/卡（fp32 master 副本则 ~14GB），而 v3 常驻 **75.2/79GB** → **直接 OOM**。仅当 batch 降到 12/卡（或 v3 结束后清空显存）才可实施。若要做，需先量化 shadow 的最小化方案（如只 shadow gemma 主干、不 shadow 冻结/小模块）。

### 方案 D：稳态 sync 清理（低优先）

`cudaStreamSynchronize` 400 次/mb 主要在早期 `log_memory_usage`（前 5 步）；稳态仅剩 `clip_grad_norm_`/`loss.item()` 每 step 一次，影响 <1%。可顺手把 `_EARLY_MEMORY_LOG_STEPS` 调小，但收益忽略。

### 方案 E：GEMM 效率（不推荐现阶段做）

GPU 侧 `aten::mm` 858ms/mb（41%）且 GEMM 效率 ~20%——怀疑 expert 分支 head_dim=256 小 GEMM 形状低效。属模型层改动，风险高、收益不确定，建议 profile 后单独立项。

---

## 4. 验证方法（任何改动必须满足）

1. **数值一致性**：改后单卡跑 2 步（`--num-train-steps 2 --num-workers 2`，从 Stage-1 checkpoint 初始化），对比同 seed 下 eager 基线（profile run 日志 `logs/profile_eager_single_gpu_20260820/train.log` 的 step=1 loss: main=0.0033, object=4.2150, skill=0.5993, grad_norm=0.07 可作参考，但严格对比需同配置同 seed 重跑基线）
2. **显存**：`after_backward` 峰值 + `nvidia-smi` 常驻，必须 < 79GB（预算上限留 5% 余量）
3. **不 OOM + loss 有限 + checkpoint 保存/恢复正常**（走完一次 save 路径）
4. 提速验证：稳态 3 步以上（排除 warmup）取平均 step 时间

---

## 5. 给接手者的操作清单（建议顺序）

1. `ssh fudan-lab` 确认 GPU 4-7 空闲、v3 仍在跑（`nvidia-smi` 前 4 行 = v3，后 4 行应 0 MiB）
2. **方案 A**：在探索副本给 DINOv2 加 `to(bfloat16)`，跑单卡 2 步 smoke（复用 `smoke_single_gpu.sh` 改 `--num-train-steps 2`），确认 loss 与基线一致
3. **方案 B**：GPU 4-7 跑 4 卡 compile smoke（复用 `smoke_4gpu.sh`，确认含 checkpoint 共存补丁），逐个加 NCCL env，对比 14.7s 基线
4. 若 A/B 均无显著收益：接受 v3 现状（eager 13.5s/step 跑完，结果优先），方案 C 留待显存有余量时（batch 12 或 v3 结束后）再评估
5. 所有探索产物放独立目录（`/data/junke_dont_remove/haozhe/guidedvla_repro_20260730/` 下新建 `repo-*`/`logs/*`/`outputs/*`），不污染 v3 资产

## 6. 其他环境备注

- 本机（Mac）访问 GitHub 直连 TLS 被干扰：`git fetch` 用 `https://gh-proxy.com/https://github.com/haozhe-feng1111/GuidedVLA.git` 镜像可达；远程机器 `fudan-lab` 网络正常
- 远程仓库当前 HEAD：`f794ac2`（本地探索分支基于 32d1794 手动移植）；GitHub 远端 `71e56d9` 为同设计的正式解耦版本（代码结构相同，改动可直接对照）
- 正式实验 v3 的历史版本：`...v1`（失败）、`...v2`（`AF_UNIX path too long` 环境问题）、`...v3`（当前运行）

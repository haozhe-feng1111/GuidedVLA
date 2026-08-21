# GuidedVLA 实验记录本

本文件是 GuidedVLA 的正式实验台账。它记录已经得到用户认可、并且实际提交运行
过的实验；`WORKLOG.md` 继续记录准备、排障、迁移和工程过程。

## 收录规则

- 只有用户明确认可且已经实际运行的实验才收录。预注册、待提交命令、dry-run、
  未提交方案和仅由协作者自行执行的检查不自动进入本文件。
- 成功、失败和负结果都可以收录，但必须保留真实状态，不以预期结果替代日志。
- 每条记录至少包含：实验目的、执行环境、代码、数据、完整命令、训练 setting、
  关键结果、结论及结论边界、日志和 checkpoint 路径。
- 代码、数据和命令必须来自该 job 的 preflight/manifest 或实际日志；跨公司卡和
  实验室卡时分别核验，不能沿用另一环境的结论。
- smoke、容量测试、消融和正式复现实验必须明确分类。smoke 的 loss 只用于判断
  数值与链路是否正常，不能写成模型质量或论文指标。
- 已收录记录不静默改写。若后续发现字段错误，在原记录下追加带日期的“更正”，
  并说明证据来源。

## 实验索引

| 实验编号 | 日期 | Job | 类型 | 状态 | 一句话结论 |
| --- | --- | --- | --- | --- | --- |
| EXP-20260729-001 | 2026-07-29 | `job-fljsvx2lltdq` | 单卡两阶段链路 smoke | 成功 | 原作 release 代码在单张 A800、batch 1 下完成 LIBERO 两阶段训练、保存和完整恢复 |
| EXP-20260729-002 | 2026-07-29 | `job-z6lo6w3ee2io` | 单卡 batch-2 两阶段 1k smoke | 成功 | 两阶段各 1k steps 稳定完成；Stage 2 峰值 66,816 MiB，单卡 batch 2 容量成立 |
| EXP-20260729-003 | 2026-07-29 | `job-5aflpkopl3hq` | 8 卡 local-batch-4 + GC 容量 smoke | 成功 | 两阶段各完成 2 steps 并保存；global batch 32 容量通过，但逐卡采样最坏仅余 1,220 MiB |
| EXP-20260821-004 | 2026-08-21 | `fudan-lab:w4yox80c` | DA3 shallow-guidance 30K 正式训练 | 运行中 | DA3 注入层改为 5–8，object/skill 层保持 9–12；加速 DDP 路径通过 smoke 后已完成正式 step 10 |

---

## EXP-20260729-001：LIBERO 原作 release 两阶段单 A800 链路 smoke

### 认可与定位

- 用户认可状态：已认可，并于 2026-07-29 明确要求作为正式实验记录。
- 实际任务：`job-fljsvx2lltdq`
- 最终状态：`SUCCESS`
- 执行时间：2026-07-29 17:25–17:41（Asia/Shanghai）
- 实验类型：engineering smoke，不是完整训练或 benchmark 复现结果。
- 目的：在不修改作者 release 源码的前提下，验证 `π0 base → Stage 1 →
  完整 GuidedVLA Stage 2` 的真实 LIBERO 数据训练、checkpoint 保存和完整恢复
  链路，并测量单张 A800、physical batch 1 的显存。

### 代码、数据和命令

| 项目 | 实际值 |
| --- | --- |
| 执行环境 | 公司任务环境，`1 × NVIDIA A800-SXM4-80GB` |
| 作者代码 checkout | `/mnt/dataset/haozhe_feng/personal/GuidedVLA/repo-libero-author-release-04be059` |
| Git commit | `04be059e0d6bd448be5cb45fdbafc775f7eb5e38` |
| 工作区状态 | clean；无 source diff |
| Depth Anything 3 子模块 | `2c21ea849ceec7b469a3e62ea0c0e270afc3281a` |
| launcher | `/mnt/dataset/haozhe_feng/personal/GuidedVLA/manifests/a800_1gpu_libero_author_release_smoke_20260729.sh` |
| launcher SHA-256 | `93f267c0ed4be3e195eb459c45d9e853bdd396c5a0c76ab4ce48b3d7aedbe2e0` |
| 实际提交命令 | `bash /mnt/dataset/haozhe_feng/personal/GuidedVLA/manifests/a800_1gpu_libero_author_release_smoke_20260729.sh` |
| 数据 | `ybwowen/libero`，revision `477f79595e4bc55829706fc655419523ae1da3b9`，LeRobot v3 |
| 数据规模 | 1,722 episodes，277,947 frames，40 tasks，10 FPS |
| 数据划分 | frame 级 93:7，seed 42 |
| `info.json` SHA-256 | `b94a2f0abe4a75b13935869f5d8aca7e746e7473c6f8f60e35fb245d74c067e6` |
| `stats.json` SHA-256 | `7a188908edbf8b8465b6c663c66a863fbc85cc75482a012b91a5c957ac73e926` |
| `tasks.parquet` SHA-256 | `3166b8b5a12b19b5ae39c3300ffef15c676c3f49a5e73bf4400b8baf7b316748` |
| π0 base 权重 SHA-256 | `8f7c9e1b48b86d5b55629b6b95a181cf4907411da1563df90c8d4d7c4080f641` |
| DA3-SMALL 权重 SHA-256 | `364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf` |
| 两阶段 norm stats | 内容完全一致，SHA-256 `9393d032f6caf10b6433b50a0c23b09a211f773fd42ee14142cb691149a93855` |

### 实际 setting

- world size 1，physical/global batch 1，data workers 2，action horizon 50。
- 作者配置的 `pytorch_training_precision="float32"`；FP32 master parameters，
  保留作者 trainer 的 CUDA autocast 行为；未启用 gradient checkpointing。
- optimizer 为作者配置的 AdamW；学习率计划保持原配置：warmup 1,000 steps，
  peak LR `2.5e-5`，end LR `2.5e-6`。本次只做链路 smoke，不代表执行了完整
  30,000-step 训练。
- `WANDB_MODE=disabled`，allocator 为
  `max_split_size_mb:256,expandable_segments:True`。
- Stage 1 使用作者 `pi0_libero` 配置，从公开 π0 base 初始化，不启用 object、
  depth 或 skill 辅助 loss。
- Stage 2 使用作者 `pi0_libero_object_depth_skill` 完整配置，从 Stage 1 step 2
  初始化：
  - guided action-expert layers：`[9, 10, 11, 12]`
  - ControlAttention：8 heads
  - object heads：`[0, 1]`，loss weight `0.001`
  - depth heads：`[4, 5]`，使用 frozen DA3-SMALL 与 control attention
  - skill heads：`[6, 7]`，4 classes，loss weight `0.001`
- 每个 stage 先运行 optimizer step 1 并保存 checkpoint，再以 `--resume` 完整
  恢复 model、optimizer 和 metadata，运行 optimizer step 2 并再次保存。

### 结果

| 阶段 | step | loss | 分项 loss | compute time | 训练器 CUDA 峰值 |
| --- | ---: | ---: | --- | ---: | --- |
| Stage 1 | 1 | 0.0775 | main 0.0775 | 111.99 s（冷编译） | allocated 26.04 GB；reserved 26.68 GB |
| Stage 1 resume | 2 | 0.0769 | main 0.0769 | 32.44 s | allocated 51.96 GB；reserved 53.23 GB |
| Stage 2 | 1 | 0.0714 | main 0.0667；object 4.3391；skill 0.3735 | 103.18 s（冷编译） | allocated 27.19 GB；reserved 32.55 GB |
| Stage 2 resume | 2 | 0.0700 | main 0.0653；object 4.3394；skill 0.3737 | 50.70 s | allocated 54.12 GB；reserved 59.46 GB |

补充测量：

- `nvidia-smi` 每秒采样的整场物理显存峰值为 **52,890 MiB
  （51.65 GiB）**；Stage 2 resume 区间采样峰值为 52,888 MiB。
- 训练器数值均为有限值，Stage 2 的 object 与 skill 分项均为非零。
- Stage 1 和 Stage 2 的 step 1、step 2 均存在完整
  `model.safetensors + optimizer.pt + metadata.pt`。
- checkpoint metadata 核验的 global step 分别为 1 和 2，说明两次 resume
  都不是仅重新加载模型后从头运行。
- Stage 1 checkpoint 单份约为 model 12.95 GB、optimizer 25.91 GB；
  Stage 2 单份约为 model 13.60 GB、optimizer 26.92 GB。

### 结论

1. **链路通过。** 在作者 release commit、零源码修改、真实 `ybwowen/libero`
   数据和单张 A800 80GB 上，Stage 1 与完整 Stage 2 均能 forward、backward、
   optimizer step、保存并完整恢复后继续训练。
2. **单卡 batch 1 容量通过。** 恢复 AdamW optimizer state 后的第二步仍成功；
   本次物理显存采样峰值为 52,890 MiB，训练器内部峰值 reserved 为
   59.46 decimal GB。
3. **辅助链路被实际执行。** Stage 2 的 object/skill loss 为有限非零值，depth
   分支也按完整作者配置启用；这证明计算链路存在，不证明辅助监督已经收敛。
4. **不能据此宣称完成论文复现。** 本实验没有完整 30k-step 两阶段训练、没有
   global batch 64、没有多卡训练，也没有 LIBERO/LIBERO-Plus closed-loop
   success rate。
5. **不能直接外推 A100。** 该结果只证明 A800 80GB、batch 1；A100 是否可跑
   仍需在确认具体显存规格后，用同一代码、数据、命令语义做独立 capacity smoke。
6. step time 受首次编译、resume 和 checkpoint 保存影响，样本数过少，不把本次
   时间写成稳定吞吐量或完整训练工时。

### 证据与产物

- 日志：
  `/mnt/dataset/haozhe_feng/personal/GuidedVLA/logs/a800_1gpu_libero_author_release_smoke_20260729`
- preflight manifest：
  `/mnt/dataset/haozhe_feng/personal/GuidedVLA/logs/a800_1gpu_libero_author_release_smoke_20260729/preflight/manifest.json`
- checkpoint 根目录：
  `/mnt/dataset/haozhe_feng/personal/GuidedVLA/outputs/a800_1gpu_libero_author_release_smoke_20260729`
- Stage 1：
  `pi0_libero/pi0_libero_author_stage1_batch1_run01/{1,2}`
- Stage 2：
  `pi0_libero_object_depth_skill/pi0_libero_author_full_guided_batch1_run01/{1,2}`

---

## EXP-20260729-002：LIBERO 原作 release 单 A800、batch 2、两阶段 1k smoke

### 认可与定位

- 用户认可状态：已运行，并于 2026-07-29 明确报告完成，纳入正式实验记录。
- 实际任务：`job-z6lo6w3ee2io`
- 日志 hostname：`job-z6lo6w3ee2io-master-0`
- 最终状态：`SUCCESS`
- 执行时间：2026-07-29 18:48–19:11（Asia/Shanghai）
- 实验类型：engineering throughput/capacity smoke，不是完整训练或 benchmark
  复现结果。
- 目的：验证作者完整两阶段配方在单张 A800 80GB、physical/global batch 2
  下能否持续运行，并测量 Stage 1 与 Stage 2 的稳定 step time 和显存，用于
  估算两阶段各 30k optimizer steps 的工程成本。

### 代码、数据和命令

| 项目 | 实际值 |
| --- | --- |
| 执行环境 | 公司任务环境，`1 × NVIDIA A800-SXM4-80GB`，总显存 85,174,583,296 bytes |
| 作者代码 checkout | `/mnt/dataset/haozhe_feng/personal/GuidedVLA/repo-libero-author-release-04be059` |
| Git commit | `04be059e0d6bd448be5cb45fdbafc775f7eb5e38` |
| 工作区状态 | clean；source diff 为空 |
| launcher | `/mnt/dataset/haozhe_feng/personal/GuidedVLA/manifests/a800_1gpu_libero_author_release_batch2_1k_smoke_20260729.sh` |
| launcher SHA-256 | `559b044bae45555da58dd91f91263f224b07b634512791302664a10df9c02473` |
| 实际提交命令 | `bash /mnt/dataset/haozhe_feng/personal/GuidedVLA/manifests/a800_1gpu_libero_author_release_batch2_1k_smoke_20260729.sh` |
| 数据 | `ybwowen/libero`，revision `477f79595e4bc55829706fc655419523ae1da3b9`，LeRobot v3 |
| 数据规模 | 1,722 episodes，277,947 frames，40 tasks，10 FPS |
| 数据划分 | frame 级 93:7，seed 42 |
| π0 base 权重 SHA-256 | `8f7c9e1b48b86d5b55629b6b95a181cf4907411da1563df90c8d4d7c4080f641` |
| DA3-SMALL 权重 SHA-256 | `364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf` |
| 两阶段 norm stats | 字节完全相同，SHA-256 `9393d032f6caf10b6433b50a0c23b09a211f773fd42ee14142cb691149a93855` |

任务内 preflight 重新读取了两个作者 config 的真实 batch 2：

- 三路 RGB 均为 `[2, 224, 224, 3]`；
- state `[2, 32]`，actions `[2, 50, 32]`，prompt `[2, 48]`；
- Stage 1 不含 object/skill supervision；
- Stage 2 含 skill-soft `[2, 4]`、object maps `[2, 3, 256]` 和 object masks
  `[2, 3]`。

### 实际 setting

- world size 1，physical/global batch 2，data workers 2，action horizon 50。
- Stage 1 与 Stage 2 各运行 1,000 optimizer steps，共 2,000 steps。
- 作者配置 `pytorch_training_precision="float32"`，FP32 master parameters，
  保留作者 trainer 的 CUDA autocast；未启用 gradient checkpointing 或
  gradient accumulation。
- optimizer 为 AdamW：betas `(0.9, 0.95)`、weight decay `1e-10`、
  gradient clip norm 1.0。
- scheduler：1,000-step warmup、peak LR `2.5e-5`、30,000-step cosine
  decay、end LR `2.5e-6`。本次 1k 结束时刚到 peak LR。
- `COMPILE_WARMUP_STEPS=0`，首次真实 step 承担编译；`WANDB_MODE=disabled`。
- 每 10 steps 聚合一次 loss/timing；steady timing 定义为结束 step 不小于 20
  的 98 个窗口。
- 每阶段在 step 1,000 做最多 1 个 validation batch。
- Stage 1 使用 `pi0_libero`，从公开 π0 base 初始化，不启用 guidance。
- Stage 2 使用 `pi0_libero_object_depth_skill`，从本次 Stage 1 step-1000
  初始化：
  - guided layers `[9, 10, 11, 12]`
  - object heads `[0, 1]`，weight `0.001`
  - depth heads `[4, 5]`，冻结 DA3-SMALL
  - skill heads `[6, 7]`，4 classes，weight `0.001`
- 作者 release 的 final-save off-by-one 未修改。`num_train_steps=1000`、
  `save_interval=1000` 实际保存了每阶段 step 999 和 step 1,000。

### 结果

| 指标 | Stage 1 | Stage 2 |
| --- | ---: | ---: |
| 完成 optimizer steps | 1,000 | 1,000 |
| 1k wall time（含模型加载、编译、两次保存、一次 validation） | 647 s（10.78 min） | 756 s（12.60 min） |
| steady compute mean | 276.40 ms/step | 336.09 ms/step |
| steady compute median | 276.50 ms/step | 334.75 ms/step |
| steady compute p90 | 277.00 ms/step | 336.00 ms/step |
| steady data mean | 9.30 ms/step | 9.72 ms/step |
| median compute throughput | 7.23 samples/s | 5.97 samples/s |
| `nvidia-smi` 物理峰值 | 63,832 MiB（62.34 GiB） | 66,816 MiB（65.25 GiB） |
| 相对 80 GiB 的采样余量 | 18,088 MiB（17.66 GiB） | 15,104 MiB（14.75 GiB） |
| trainer peak allocated | 52.82 decimal GB | 54.99 decimal GB |
| trainer peak reserved | 66.33 decimal GB | 69.45 decimal GB |
| 最后 10-step train loss | main/total 0.0889 | total 0.0695；main 0.0674；object 1.6161；skill 0.5557 |
| 单个 validation batch | total/main 0.0766 | total 0.0660；main 0.0646；object 0.8640；skill 0.5208；skill acc 0.0 |

补充事实：

- Stage 1 首个 10-step 编译窗口平均 compute 11.81 s/step；之后稳定在约
  276–277 ms/step。
- Stage 2 首个 10-step 编译窗口平均 compute 11.13 s/step；之后稳定在约
  334–336 ms/step。
- 两阶段全部 loss 和 gradient norm 均为有限值。
- Stage 2 的前 10-step object loss 为 4.6410，最后 10-step 为 1.6161；
  这只是短程训练观察，不作为收敛或模型质量结论。
- Stage 1 step-1000 checkpoint 为 model 12,952,323,536 bytes、optimizer
  25,905,064,999 bytes；Stage 2 为 model 13,597,856,692 bytes、optimizer
  26,921,690,396 bytes。step 999 与 step 1,000 的完整 metadata 均通过
  global-step 核验。

### 时间估算

以 steady median compute 线性外推相同单 A800、batch-2 setting：

- Stage 1 30k：`0.2765 s × 30,000 = 2.30 h`
- Stage 2 30k：`0.33475 s × 30,000 = 2.79 h`
- 两阶段纯 compute 合计：约 **5.09 h**

实际长训练还要计入 data、每阶段一次模型加载与首次编译、checkpoint 和
validation。结合本次测量，单 A800、batch 2、两阶段各 30k 的工程预算约为
**5.5–6 小时**。这是容量规划估计，不是承诺工时。

不能把本次 1k wall time直接乘 30：这样会得到 Stage 1 5.39 h、Stage 2
6.30 h、合计 11.69 h，但相当于把模型加载、首次编译、额外 step-999 保存和
首次 validation 编译重复计算 30 次，属于明显偏高的朴素上界。

### 结论

1. **单 A800、batch 2 容量通过。** 两阶段都在 AdamW state 已建立后连续运行
   到 1,000 steps；完整 Stage 2 峰值更高，但仍有约 14.75 GiB 的物理采样余量。
2. **短程吞吐稳定。** 排除首次编译窗口后，98 个 timing 窗口波动很小：
   Stage 1 median/p90 为 276.5/277.0 ms，Stage 2 为 334.75/336.0 ms。
3. **两阶段 30k 的单卡 batch-2 时间可初步按 5.5–6 小时规划。** 该估计只
   对相同代码、硬件、batch 和 I/O 环境有效。
4. **这仍不是论文复现结果。** 没有完成 30k+30k、global batch 64、多卡或
   LIBERO/LIBERO-Plus closed-loop evaluation。
5. **不能据此估算 global-batch-64 的多卡速度。** batch 2 每个 optimizer step
   只处理 2 个样本；论文 global batch 64 每步处理 32 倍样本。直接在 8 张
   A800 上使用 global batch 64 意味着 local batch 8，该容量尚未验证，也不能
   从本实验宣称可行。
6. validation 只有 1 个 batch，`skill_acc=0.0` 既不能证明 skill head 无效，也
   不能证明模型质量；完整训练和固定闭环评测前不做质量归因。

### 证据与产物

- 日志根：
  `/mnt/dataset/haozhe_feng/personal/GuidedVLA/logs/a800_1gpu_libero_author_release_batch2_1k_smoke_20260729`
- 结果 manifest SHA-256：
  `0a51f18154728f924976cfe3523925d799b6a9d7503014443cec159c1ae5490d`
- preflight manifest SHA-256：
  `43527be548e0423b7801ef44da79861b277a9f25553caa9040bb707d76c95b4f`
- Stage 1 日志 SHA-256：
  `758f21c4b707052caabf790ee9106de7a8e68c231f3e04a6a76779022bca995d`
- Stage 2 日志 SHA-256：
  `ef0cde6ec8ede8356534e928495f62b0a623d0bf9bef8f3ae35d715743141a99`
- checkpoint 根：
  `/mnt/dataset/haozhe_feng/personal/GuidedVLA/outputs/a800_1gpu_libero_author_release_batch2_1k_smoke_20260729`
- Stage 1：
  `pi0_libero/pi0_libero_author_stage1_batch2_steps1000_run01/{999,1000}`
- Stage 2：
  `pi0_libero_object_depth_skill/pi0_libero_author_full_guided_batch2_steps1000_run01/{999,1000}`

---

## EXP-20260729-003：LIBERO 原作 release 8×A800、local batch 4、GC 容量 smoke

### 认可与定位

- 用户认可状态：已运行，并于 2026-07-29 明确报告 8 卡 smoke 成功，纳入正式
  实验记录。
- 实际任务：`job-5aflpkopl3hq`
- 日志 hostname：`job-5aflpkopl3hq-master-0`
- 最终状态：`SUCCESS`
- 执行时间：2026-07-29 19:44–19:57（Asia/Shanghai）
- 实验类型：engineering distributed/capacity smoke，不是完整训练、吞吐 benchmark
  或论文复现结果。
- 目的：紧急验证作者完整两阶段配方在 8 张 A800 80GB、每卡 physical batch 4、
  开启 gradient checkpointing 时能否完成 DDP forward/backward、optimizer step
  和 checkpoint 保存。

### 代码、数据和命令

| 项目 | 实际值 |
| --- | --- |
| 执行环境 | 公司任务环境，`8 × NVIDIA A800-SXM4-80GB`，每卡 85,174,583,296 bytes |
| 作者代码 checkout | `/mnt/dataset/haozhe_feng/personal/GuidedVLA/repo-libero-author-release-04be059` |
| Git commit | `04be059e0d6bd448be5cb45fdbafc775f7eb5e38` |
| 工作区状态 | clean；working-tree diff 为空 |
| launcher | `/mnt/dataset/haozhe_feng/personal/GuidedVLA/manifests/a800_8gpu_libero_author_release_localbatch4_gc_capacity_smoke_20260729.sh` |
| launcher SHA-256 | `8e47ef47a63321e1764049e629bfb5e40e1b44c753ea122692b51c32cb57ea53` |
| 实际提交命令 | `bash /mnt/dataset/haozhe_feng/personal/GuidedVLA/manifests/a800_8gpu_libero_author_release_localbatch4_gc_capacity_smoke_20260729.sh` |
| 数据 | `ybwowen/libero`，revision `477f79595e4bc55829706fc655419523ae1da3b9`，LeRobot v3 |
| 数据规模 | 1,722 episodes，277,947 frames，40 tasks，10 FPS |
| 数据划分 | frame 级 93:7，seed 42 |
| 数据 metadata | 与 EXP-001/002 相同，三项 SHA-256 均由 job preflight 重新核验 |
| 两阶段 norm stats | 字节完全相同，SHA-256 `9393d032f6caf10b6433b50a0c23b09a211f773fd42ee14142cb691149a93855` |

### 实际 setting

- world size 8，per-GPU physical batch 4，physical/effective global batch 32。
- gradient checkpointing 开启；gradient accumulation 为 1。
- Stage 1 与 Stage 2 各运行 2 个 optimizer steps。
- 作者配置 `pytorch_training_precision="float32"`，FP32 master parameters，
  保留作者 trainer 的 CUDA autocast；allocator 为
  `max_split_size_mb:256,expandable_segments:True`。
- optimizer 为 AdamW，weight decay `1e-10`，gradient clip norm 1.0；
  scheduler 为 warmup 1,000、peak LR `2.5e-5`、30,000-step cosine decay至
  `2.5e-6`。
- Stage 1 使用原作 `pi0_libero`，从公开 π0 base 初始化。
- Stage 2 使用原作 `pi0_libero_object_depth_skill`，从本次 Stage 1 step 2
  checkpoint 初始化；object、depth、skill 三类 guidance 保持作者 release
  配置。
- 每阶段保存 step 1 和 step 2 的完整 model、optimizer 与 metadata。本任务没有
  执行 `--resume`，因此它验证 checkpoint 可写和 metadata 可读，不验证 8 卡
  optimizer-state 恢复后继续训练。
- `WANDB_MODE=disabled`，data workers 2，action horizon 50。

### 结果

| 指标 | Stage 1 | Stage 2 |
| --- | ---: | ---: |
| 完成 optimizer steps | 2 | 2 |
| step 1 loss | 0.0630 | total 0.1981；main 0.1940；object 3.3692；skill 0.7627 |
| step 2 loss | 0.1648 | total 0.0835；main 0.0787；object 3.7167；skill 1.0938 |
| step 2 compute | 2,171.7 ms | 1,971.9 ms |
| rank-0 trainer peak allocated | 68.04 decimal GB | 71.76 decimal GB |
| rank-0 trainer peak reserved | 84.16 decimal GB | 84.15 decimal GB |
| 逐秒 `nvidia-smi` 阶段最高 | GPU 5：80,700 MiB | GPU 0：79,516 MiB |

逐卡整场 `nvidia-smi` 采样峰值为：

`[79,516, 78,974, 78,522, 78,914, 78,672, 80,700, 78,954, 78,042] MiB`。

最坏设备为 GPU 5，距离 81,920 MiB 总显存仅余 **1,220 MiB（1.19 GiB）**。
Stage 2 八卡峰值范围为 78,042–79,516 MiB，对应最小采样余量 2,404 MiB。
两阶段全部记录 loss 与 gradient norm 均为有限值，Stage 2 object/skill 分项
为非零。Stage 1 和 Stage 2 的 step 1/2 checkpoint metadata 均准确记录对应
global step。

### 结论

1. **8 卡 DDP 容量链路通过。** 作者 release 零源码修改下，8×A800、
   local batch 4、global batch 32、gradient checkpointing 能完成两阶段各
   2 个 optimizer steps，并保存完整 checkpoint。
2. **容量极其贴边。** 最坏逐卡采样只余 1.19 GiB，rank-0 allocator peak
   reserved 也达到约 84.16 decimal GB。该结果不足以把相同 setting 直接视为
   稳健长训配置；更长运行、validation、allocator 碎片或不同 rank 数据都可能
   触发 OOM。
3. **没有达到论文 global batch 64。** 当前是 global batch 32，未使用梯度累积。
   若保持 local batch 4，要达到 effective global batch 64，需要 accumulation
   2；这属于下一项执行变量，不能把本次结果记成 batch-64 复现。
4. **不能据两步估算正式吞吐。** step 1 包含冷编译；step 2 只有单个样本点，
   且每步后都保存超大 checkpoint。约 1.97–2.17 s 的单步 compute 只作链路
   观察，不作为 30k 工时预测。
5. **没有验证 8 卡 resume。** checkpoint 文件和 global-step metadata 已验证，
   但本 job 未从保存后的 optimizer state 恢复并继续训练。
6. 本实验没有完整 30k+30k、global batch 64 或 closed-loop evaluation，不能
   宣称完成论文复现，也不能直接外推到实验室 A100。

### 证据与产物

- 日志根：
  `/mnt/dataset/haozhe_feng/personal/GuidedVLA/logs/a800_8gpu_libero_author_release_localbatch4_gc_capacity_smoke_20260729`
- 结果 manifest SHA-256：
  `5f8296931a853a054a6425f4fd2962e884a4cfac1b27f416837fc86547f5a7d3`
- preflight manifest SHA-256：
  `217a8efdd7c374b5476a4c6a635901d199743ea852e83e7df9786337248c8775`
- Stage 1 日志 SHA-256：
  `145f0dea47a740ff76075f5249cf63e85f812c83e633eabfa9f5cc24d7aae1be`
- Stage 2 日志 SHA-256：
  `71b9959fb1fa0ccebcbbd8529abb2f7da2531483ce1c32cad2518933275aff0d`
- checkpoint 根：
  `/mnt/dataset/haozhe_feng/personal/GuidedVLA/outputs/a800_8gpu_libero_author_release_localbatch4_gc_capacity_smoke_20260729`
- Stage 1：
  `pi0_libero/pi0_libero_author_stage1_8gpu_localbatch4_gc_steps2_run01/{1,2}`
- Stage 2：
  `pi0_libero_object_depth_skill/pi0_libero_author_full_guided_8gpu_localbatch4_gc_steps2_run01/{1,2}`

---

## EXP-20260821-004：fudan-lab DA3 shallow-guidance `[5,6,7,8]` 30K 正式训练

### 认可与定位

- 用户于 2026-08-21 明确要求，在 DINO `[5,6,7,8]` 对照之外启动 Depth/DA3
  `[5,6,7,8]`，使用已验证加速路径，且不改变 object/skill layer。
- 类型：LIBERO Stage-2 30K 正式训练；当前状态为运行中，不是完成结果。
- 环境：fudan-lab，`4 × NVIDIA A100-SXM4-80GB`，physical GPU index `4–7`。
- W&B run：`w4yox80c`，名称
  `guidedvla_libero_stage2_object_da3_skill_shallow_guidance_fast_4gpu_30k_20260821_v1`。

### 代码、数据和命令

- worktree：
  `/data/junke_dont_remove/haozhe/guidedvla_repro_20260730/repo-libero-da3-shallow-fast-train-20260821`
- 基线 commit：`f794ac2522c136c31572f72dbddb364a3c2c9fcf`；detached 专用 worktree，未改
  正在运行的 DINO v3。
- 源码改动只有三项：新增 DA3 shallow 配置；默认关闭、按精确名称 fail-fast 的 6 个
  结构性死 Pali 参数冻结；默认关闭的首步 unused-gradient 审计。patch SHA-256：
  `07ad4e711024c71c3c9acdd250257f3904b0cc5d4fab4746ada8ffeee37d029d`、
  `a4e2b52af93c34d3c1178f1de74ab7ea65d09181744fbb691a39951e4279c3e3`、
  `ee96b0cc73045b168240632e2c167a02b82668801c4fb9e8a465d7d30fa6dec1`。
- launcher：
  `/data/junke_dont_remove/haozhe/guidedvla_repro_20260730/manifests/train_da3_shallow_fast_layers5_8_4gpu_30k_20260821_v1.sh`，
  SHA-256 `00233488d20a89c0869950f23d326f080635348625b9f050b37b87d919a8d8c0`。
- 入口：`torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4`，配置
  `pi0_libero_object_depth_skill_shallow_guidance`；完整参数与环境见 launcher/preflight。
- 数据：本地冻结的 `ybwowen/libero@477f7959`；norm stats 显式复用
  `assets/pi0_libero_object_depth_skill/ybwowen/libero/norm_stats.json`，SHA-256
  `9393d032f6caf10b6433b50a0c23b09a211f773fd42ee14142cb691149a93855`。
- Stage-1 初始化：
  `outputs/guidedvla_libero_stage1_4gpu_30k/pi0_libero/guidedvla_libero_stage1_4gpu_30k/30000`。
- Depth encoder：`models/da3-small-e08cab65`。

### 实际 setting

- CLI global batch 16，world size 4，实际 local batch 4；GA4 后 effective global batch 64。
- FP32 training precision、gradient checkpointing、workers 2、30,000 optimizer steps；
  log/validation/checkpoint interval 为 `10/10000/10000`。
- `TORCH_COMPILE=0`；`GUIDEDVLA_FREEZE_DEAD_PALI_TAIL=1`；不传
  `--ddp-find-unused-parameters`。
- 运行时 config contract（`preflight/config_contract.json`，SHA-256
  `cd54e1a8e2bf43d5136cee5f22ae9b46e050f9759c392709d6affafe1250a2f8`）：DA3 depth
  injection `[5,6,7,8]`；object/skill supervision `[9,10,11,12]`；object/depth/skill
  heads `[0,1]/[4,5]/[6,7]`；object/skill loss weights 均为 `0.001`。

### 启动门禁与当前结果

- 首次 smoke 因新 config 默认查找同名 norm-stats 目录而在 data preflight fail-fast，未加载
  模型；显式指向内容完全相同的既有 Depth norm stats 后，retry1 通过 2 steps。
- smoke retry1：4 ranks 均冻结 6 tensors / 104,861,696 parameters；完整 GA4 后
  `901 trainable / 0 unused`。step 2 compute `13.1839s`；object loss
  `3.8486/4.4247`、skill loss `0.7688/0.5701`；无 OOM/NCCL/非法访存/traceback。
- 正式 run 已完成 step 10：total `0.0096`、main `0.0047`、object `4.1969`、skill
  `0.6530`、grad norm `0.08`，窗口 compute `14.3648s/step`。首个完整 GA4 backward
  再次确认 `901 trainable / 0 unused`。
- GPU 4–7 各约 `71.6–71.7GB` 并持续运行；正式 DINO v3 同时在 GPU 0–3 运行且未修改。
  `/data` 启动后可用约 513GB。

### 结论边界与产物

- 当前只证明配置、数据、DA3、object/skill、加速 DDP 与正式训练链路成立；30K 未完成，
  尚无 10K checkpoint 或 closed-loop 评测，不能报告最终质量结论。
- 加速路径已通过独立语义等价对照，但 optimizer param-group/checkpoint resume 仍是工程兼容
  边界；不得把 running 状态记成 resume 门禁已通过。
- 日志根：
  `/data/junke_dont_remove/haozhe/guidedvla_repro_20260730/logs/guidedvla_libero_stage2_object_da3_skill_shallow_guidance_fast_4gpu_30k_20260821_v1`
- 输出根：
  `/data/junke_dont_remove/haozhe/guidedvla_repro_20260730/outputs/guidedvla_libero_stage2_object_da3_skill_shallow_guidance_fast_4gpu_30k_20260821_v1`
- PID 文件：
  `/data/junke_dont_remove/haozhe/guidedvla_repro_20260730/manifests/train_da3_shallow_fast_layers5_8_4gpu_30k_20260821_v1.pid`

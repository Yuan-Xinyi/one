# seed_selection

SMM-aware 的起始关节 q0 选择：给定线任务 `c = (p0, line_dir, n_target)`，从扩散模型采样
多个候选 q0，再用 Newton IK 精化到任务起始流形。10 模块流水线，覆盖
"任务扰动 → 锥-IK 枚举 → SMM 1D walk → 鲁棒性过滤 → 数据集落盘 → 扩散训练 → 评估"。

## 目录结构

```
seed_selection/
├── smm/                 SMM 标签生成流水线（数据制作端）
├── diffusion/           c → q0 扩散模型（定义、数据集、训练、采样）
├── eval/                训练后评估、操作端评估、可视化
└── runs/                pilot_20k 等数据集 + ckpt
```

## smm/ — 数据生成流水线

按模块号组织：

| 文件 | 模块 | 功能 |
|------|------|------|
| [smm/perturb.py](smm/perturb.py) | 4 | `perturb_task`：在保持 `line_dir ⊥ n_target` 的前提下扰动 `c`（n 先转、d 后转再投影 + p0 球扰动）。 |
| [smm/cone_ik.py](smm/cone_ik.py) | 3a | 锥约束 IK 枚举：在 5-DOF 锥内随机采朝向 × 多次 DLS 重启，拿到尽量多分支的可行 q 候选。 |
| [smm/rollout.py](smm/rollout.py) | 2 | `rollout_one`：单 (q0, c) 在 classical 控制下跑一回，返回归一化 L = progress / 1.5m。覆盖 `env.p_start` 以匹配扰动后的 p0。 |
| [smm/rollout_batched.py](smm/rollout_batched.py) | 2-batched | `batched_rollout_many`：把 B 个 (q, c) pack 进同一个 `NSRLBatchedEnv` 一次 step 完，n_envs=64 时每个 rollout 近乎零成本。 |
| [smm/robustness.py](smm/robustness.py) | 5, 6 | `evaluate_robustness`（在 n_perturb 个扰动 c 下评 L）+ `filter_robust_candidates`（按 `L_robust ≥ τ × L_clean` top-K' 过滤）。 |
| [smm/label_builder.py](smm/label_builder.py) | 7 | `build_labels_for_one_task`：单任务端到端 — 锥-IK 枚举 → 强制注入 q0_seed → 6-DOF 严格投影 → SMM 1D 分支枚举 → 弧长均匀采样 → clean L 评 → 鲁棒过滤 → top-k 标签。 |
| [smm/dataset_builder.py](smm/dataset_builder.py) | 8 | `build_dataset`：批量调用 `build_labels_for_one_task` 落 NPZ。partial 原子写、SIGINT 安全、bit-exact resume。 |
| [smm/build_worker.py](smm/build_worker.py) | 子进程 | 单进程数据构建入口，按 `--n-tasks/--seed/--cache-name` 切片，独立缓存。 |
| [smm/build_parallel.py](smm/build_parallel.py) | 启动器 | K 进程切片 + 合并到最终 NPZ，支持断点续跑、SIGINT 转发。 |

## diffusion/ — c → q0 扩散模型

| 文件 | 功能 |
|------|------|
| [diffusion/dataset.py](diffusion/dataset.py) | `SeedSelectionDataset`：每次 `__getitem__` 从 `labels_q0[:n_labels]` 中均匀采样一个 q0，多模态目标在 SGD 下保持多模态。支持 xz 平面 mirror aug（y 翻转 + 特定关节符号翻转，已物理验证保 L 不变）。默认保留 status ∈ {kept, edge, edge_seed_fallback}。 |
| [diffusion/model.py](diffusion/model.py) | `SeedQ0DiT`：极简 MLP 扩散网络（c_dim=9, q_dim=7, d_model=256, n_layers=4），v-prediction，复用 fr3_dit 的 DDPM 余弦 schedule、关节限位归一化、sinusoidal time embedding。 |
| [diffusion/sampling.py](diffusion/sampling.py) | `ddim_sample_q0`：DDIM 采样器，v-prediction + CFG，返回归一化空间的 q0。`load_ckpt` 复用 EMA 权重。 |
| [diffusion/train.py](diffusion/train.py) | Pilot 训练入口：DataLoader + v-prediction loss + EMA。Train/val split 写到 `split.json` 复用于 eval。 |

## eval/ — 评估与可视化

| 文件 | 功能 |
|------|------|
| [eval/eval_joint_distance.py](eval/eval_joint_distance.py) | **模型内 sanity check**：每任务从 DiT 采 M 个 q0，最近邻指派到 labels，统计 fidelity（最近距离）、mode-coverage（哪几个 label 被覆盖到 `match_rad` 内）、assignment entropy。 |
| [eval/eval_rollout.py](eval/eval_rollout.py) | **操作端 headline 评估**：DP 采样 → Newton IK 精化到严格 (p0, R_target) → 从精化 q0 实跑 rollout 得真实 L。报 IK 收敛率、best-of-N L、相对 label 上限的 ratio、相对 L_seed 的提升。 |
| [eval/plane_collision.py](eval/plane_collision.py) | 检查每任务 q0_seed 与所有 label 的 FR3 球碰撞模型是否穿过任务平面，按 status / bucket 交叉统计，给 system_eval 提供 "safe" 子集筛选。 |
| [eval/viz_dataset.py](eval/viz_dataset.py) | 数据可视化：NPZ 模式（已生成）或 RAW 模式（实时跑标签流水线），把 label ghost robot + L 柱状叠加到场景。 |
| [eval/viz_inference.py](eval/viz_inference.py) | 模型推理可视化：DiT 采样 → Newton IK → rollout，把精化样本（橙色 ghost + L 柱）和 GT label（蓝色 ghost + L 柱）画在同一场景里。 |

## 关键设计

- **L 归一化**：`L = progress_m / 1.5m`，方便绝对值/相对比；system_eval 改回 meters 报告。
- **q0_seed 强制入选**：标签流水线 Step 2 一定把 random feasible seed 投到候选池，避免锥-IK 漏分支。
- **鲁棒性**：`L_robust_mean / L_robust_min` 在多个扰动 c 上评的鲁棒 L，是过滤候选的关键阈值。
- **Mirror aug**：xz 平面对称（FR3 y-flip + joint sign-flip）在 classical 下已物理验证 0 mm 差（10 任务，2026-05-28），可放心免费 ×2 数据量。

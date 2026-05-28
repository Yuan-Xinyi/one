# RL_controller

FR3 笔尖沿无穷射线行走任务的 PPO 控制器：在 3-DOF 位置零空间内学习关节避位/避奇异/避锥的动作。
训练目标只有 episode 寿命（progress-only reward），不存在 success terminate。

## 目录结构

```
RL_controller/
├── config.yaml           训练 / env / 任务分布 / eval 的统一配置（env=20Hz, a_max=0.5, cone=30°, 30M steps）
├── algorithms/           PPO 算法与训练入口
├── env/                  Torch-batched 环境、任务分布、经典 nullspace 基线、单 episode rollout 工具
├── eval/                 训练后评估：RL vs 经典对照、步级 RL↔Cls 混合控制
└── runs/                 训练日志、ckpt、eval 缓存（如 p0_progress_only_30M_0520）
```

## 主要文件

### algorithms/

| 文件 | 功能 |
|------|------|
| [algorithms/ppo.py](algorithms/ppo.py) | 从 cleanrl 改写的连续动作 PPO（`Agent`, `PPOConfig`, `train`）。差异：torch-batched env、truncation 用 `info["terminal_obs"]` 而非 auto-reset 的 next obs。 |
| [algorithms/train.py](algorithms/train.py) | 训练入口。读 `config.yaml`，构建 `NSRLBatchedEnv` + `LineDistribution`，调 `ppo.train`，周期性跑 eval rollout，落 ckpt。 |

### env/

| 文件 | 功能 |
|------|------|
| [env/env.py](env/env.py) | `NSRLBatchedEnv`：Torch-batched Gymnasium 风格环境。状态 31 维 (q, line_dir, n_target, t, a_prev, done)；动作 4 维（前 3 维零空间方向 + 第 4 维 q_ref 微调）；P0 progress-only reward = `clip(Δp·u_hat / (v·dt), 0, 1)`。包含 DLS pseudo-inverse、Nakamura-Hanafusa adaptive λ、横向 PD term（Framing B）、6 种 termination 编码。 |
| [env/classical_nullspace.py](env/classical_nullspace.py) | `ClassicalNullspaceController`：经典零空间控制器（Yoshikawa 可操作度 + Liegeois JL avoid + 锥吸引子 + q_ref 吸引）。作为 baseline、混合控制 fallback、SMM 标签生成器共用同一份。 |
| [env/line_distribution.py](env/line_distribution.py) | `LineDistribution`：MC 可达性采样 + 可行性预筛（丢弃 classical 都走不到 10 cm 的任务），磁盘缓存。`ScriptedLineDistribution` 用于 eval 时回放固定 spec 列表。 |
| [env/rollout.py](env/rollout.py) | `rollout_first_episode`：单 episode 通用工具，`auto_reset=False` 下跑到所有 env 都终止，记录 `episode_len`、`term_reason`、`progress`。被 train 周期 eval、可行性筛选、所有 eval 脚本复用。 |

### eval/

| 文件 | 功能 |
|------|------|
| [eval/rl_vs_classical.py](eval/rl_vs_classical.py) | 对比评估：RL 与 classical 在同一组 N 个任务上各跑一次，落 `rollouts.npz`（含整段 q_traj，供后续切片分析无需 re-run）+ `per_task.csv`。 |
| [eval/hybrid.py](eval/hybrid.py) | 步级 state-conditional 混合控制：每步基于 `max\|q_norm(q_t)\|` 与滞回阈值 `(tau_enter, tau_exit)` 在 RL ↔ Classical 间切换。把所有 (tau_enter, tau_exit) 组合 tile 到同一个大 env 一次并行评。 |

## 关键设计

- **任务**：无穷射线 `(p_0, u_hat, n_target)`，无 success terminate，控制器目标是最大寿命。
- **奖励**：P0 progress-only（投影到 u_hat 的 EE 平移），terminal penalty 全 0，由 reward shaping history 选定（详见 memory）。
- **动作 scale**：`a_max=0.5 rad/s`（0520 之后从 1.0 降到 0.5）。
- **评估指标**：报 `L_policy / L_oracle` 等 ratio，**不**报绝对 `success_rate` 或 `mean_len`（部分任务本质不可行）。

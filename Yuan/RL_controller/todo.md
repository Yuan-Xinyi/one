# 待裁决项 & 不确定清单（工作笔记，不喂 LLM）

## 修改条款（按优先级降序，编号即为执行顺序）

### P2 — 训练前/训练中扫参验证，不影响代码正确性

按优先级排序：

**[P2-1] `w_alive` 缩放**
- 当前 1.0，所有终止 penalty 已统一为 0（P1-2 已落地）
- 决策点变成"是否归一化让 typical return ~ O(1)"
- 扫 `w_alive ∈ {0.01, 0.1, 1.0}`

**[P2-2] PPO 样本预算**
- 跑完 1e6 看 eval_mean_len 曲线是否还在上升
- 若是 → `total_timesteps` 提到 3e6–5e6
- 不是先验决定，是训练中观察

**[P2-3] `init_log_std`**
- 当前 −0.5 (std ≈ 0.6)
- **不要先动**。先看前 50 update learning curve
- 若长时间平的：先试 `ent_coef = 1e-3` 而不是改 init std（理由：4-DOF nullspace 多数方向 reward 平坦，大 init std 会在平坦方向瞎走）

**[P2-4] `a_max` 扫参**
- 当前 0.5 rad/s，比例 a_max:v = 10:1
- 扫 {0.3, 0.5, 1.0}，看 mean ratio

**[P2-5] `k_jl` baseline 调参**
- 当前 1.0，smoke test 中 baseline 中位数 ~50–100 步（主要因 JL/cone 越界）
- 手工调一组使 baseline 中位数 ~几百步（"像样"的对手）
- 若调到 k_jl = 5 仍 cone 越界占主导 → 确认 baseline 本来就该弱，写进 paper 限制条款

**[P2-6] MC reachability pool 规模**
- 当前 100k，未量化覆盖均匀度
- 待验：eval term_reason 按空间分 bin，看是否有"某区域失败率显著偏高"
- 有偏置 → 提升到 1e6

**[P2-7] 30° 锥软 penalty 辅助 signal** — ✅ DONE (B 路线落地)
- runs1–5 (alive-only) 跑完证明 PPO 学不到比初始随机更好的策略
- env.py + config.yaml 加入三项 soft penalty（cone / JL / σ-min），都在 q_new 上计算，对动作可微
- 默认权重：`w_cone_soft=0.05` (deg²), `w_jl_soft=1.0`, `w_sigma_soft=5.0`
- 随机 policy 下 per-step 平均 penalty ≈ 0.030（alive 的 3%），如训练后觉得信号弱可 2-3× 调大权重
- ablation：把三个 w 设 0 即退回 alive-only

### P3 — 已识别但可以暂时忽略

**[P3-1] PPO rollout-boundary truncation 处理**
- cleanrl 简化实现：t == n_steps−1 时 next_obs 已是新 episode 起点
- 训练初期影响大（episode 长度与 n_steps 同量级），中后期可忽略
- **诊断**：log `rollout_boundary_truncation_rate`。若训练初期 > 20% → 修；< 5% → 忽略
- 严格修复：在 rollout 循环里维护 `pre_reset_obs` 副本，抄 sb3 PPO 实现

**[P3-2] SELIKSolver position-only 模式**
- 当前 workaround：6-DOF 目标 + z 列设 n_target + multi-twist seed (10)
- 风险：q₀ 分布可能系统性偏向某些 SMM 分支
- 触发条件：eval 中某些 line direction 上 ratio 异常低
- 不触发不动；触发了再去改 `NumIKSolver._backward` 丢弃 delta_theta

**[P3-3] `B_prev_valid` auto-reset 后单步不连续**
- 对当前 reward 无影响（无 smoothness 项）
- 引入 smoothness reward 时再修

**[P3-4] damped_pinv `eigvalsh` 数值退化**
- FR3 reachable 区域内未触发，已 clamp(min=0)
- 不动

**[P3-5] `update_epochs` 内 mini-batch break-on-KL 估计有偏**
- `target_kl=None` 不触发
- 若启用 target_kl，改为 epoch 平均 KL

**[P3-6] mini-batch shuffle seed 不可复现**
- 改用专属 numpy RNG 实例，但 PPO 顺序复现非必需
- 不动

**[P3-7] tcp_offset = 0.0 决策**
- 任务语义决策，非 RL 问题
- 默认 bare flange（更干净、paper 故事更清楚）
- 已在 `README.md` Known Issues #5 中 flag

### P4 — 基础设施 / 文档

**[P4-1] libstdc++ ABI 不兼容**
- `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` 临时绕过
- 已在 README "Run" 段顶部写明 export 命令
- cluster 部署时同样需要 — 可考虑写 `scripts/launch.sh` 自动 export

**[P4-2] logger**
- 当前 stdout + 文本 log
- 加 tensorboard：cleanrl PPO 自带 `writer`，已存在的话直接启用 `--track` flag
- 加 wandb：可选

---

## 执行顺序

1. **启动 1e6 training**，跑完看验收门槛
2. 未达标 → **P2 按顺序扫**（每轮约 0.5–1 day）
3. **P3 按需触发**
4. **P4 闲时补**

## 验收门槛

1. `python -m Yuan.RL_controller.train ...` 跑完 1e6 不崩
2. `eval_mean_len` 曲线前 50 个 update 单调上升
3. 200-line holdout 上 mean ratio > 2.0 且 median ratio > 1.5
4. RL truncated（活到 max_steps）占比 > 50%；baseline truncated 占比 < 10%

---

## README "Open Questions"（转载追踪，不变）
1. MC pool 1e5 够不够
2. RL/baseline T-ratio 预期（pure guess: 2–5×）
3. 1e6 steps 是否够 PPO 收敛
4. 4-DOF nullspace exploration 难度
5. Baseline 是否过弱

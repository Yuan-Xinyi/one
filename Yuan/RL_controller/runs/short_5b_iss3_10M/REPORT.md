# FR3 NSRL Plateau Resolution

**Branch**: `connectivity-flow` | **完成**: 2026-05-18 | **状态**: A 路径（accept plateau）

---

## 1. 问题描述

旧 RL controller 在 9.2M 步训练后 plateau 于 `L_rl / L_zero ≈ 0.93`（即 RL policy 平均比 `a≡0` 还短）。结论：null-space 动作 net-harmful，policy 学到的策略不如不动。

> 注：本报告全程以 `a≡0` 为 denominator（"no null-space action" baseline）。原始诊断中亦报告过 `L_rl / L_oracle = 0.274` vs `L_zero / L_oracle = 0.295` 的口径，比值与 vs-zero 一致（0.929），但报告中其他章节均不引用 oracle 口径，以避免混淆。

---

## 2. 诊断路径

**Step A —— sign-seed diagnostic** ([Yuan/RL_controller/diagnose_sign_seed.py](../../diagnose_sign_seed.py))

2048 个 reset 上对当前 SVD basis B(q) 应用 sign 约定后，统计 3 个物理量 (∇w_u, ∇cos(z,n), ∇−qn²) 与 B 各列的内积符号分布。12 个 cell 的 sign +% 全部落 48–52% → basis 列与任何物理量都无结构性关联，**随 q 各向同性旋转**。

**Step B —— 3-way baseline 排序**：a≡0 > RL > GPM-JL。Confirm policy 的 null-space 动作整体在伤害任务。

---

## 3. 两个 Root Cause

**(5b) Null-space basis 的 SO(4) gauge ambiguity**
- B(q) 来自 `torch.linalg.svd(J_p)` 的 V[..., -4:]，跨 episode 任意旋转 4×4 unitary。
- 同一 q 的 a=[1, 0, 0, 0] 在不同 episode 触发不同物理方向 → policy 学不到 `a → q̇_null` 的稳定映射。
- 修符号约定不充分，必须从构造上消除。

**(iss3) State-dependent log_std head 饱和**
- 原 `_logstd_head` 是 `Linear(hidden_dim, act_dim)`，被 entropy bonus 反推到 `LOG_STD_MAX = 0` (σ=1.0) 后贴顶。
- 50M 调试 run 在 step 3M 起 σ_mean 顶到 0.999–1.000 直到 15M 仍不动 → policy 永远在 max-explore，无法 exploit。

---

## 4. Fix 设计

### 5b: Task-aligned Gram-Schmidt Basis

新函数 [`build_task_aligned_basis`](../../env/env.py)：

| 步骤 | 实现 |
|---|---|
| Gradient 计算 | autograd ∇w_u + ∇cos, 解析 ∇(−qn²) |
| Null-space 识别 | **fp64** SVD on `J_p_d.double()`（fp32 SVD null-space 漂移 ~1e-6 × ‖g‖ → 残差 1e-5；fp64 压到 ~1e-12）|
| 锚定 e_0..e_2 | modified Gram-Schmidt, **twice-is-enough**（残差 4e-6 → 1.2e-7）|
| Fallback | 在 `‖v‖ < max(εabs, εrel·‖g‖)` 时取最小未用 SVD 列，**对残余 g_raw 做 sign-anchor 后**正交化 |
| e_3 | cleanup direction：剩余未用 SVD 列对 (e_0,e_1,e_2) 双遍正交化 |
| Output | `B_basis ∈ R^{B×7×4}` (fp32 cast back), `fb_mask ∈ R^{B×3} bool` |

物理语义：a_0>0 ⇔ 提升 dirmanip；a_1>0 ⇔ 在 dirmanip 等高线上提升 cone；a_2>0 ⇔ 在前两者等高线上提升 JL margin；a_3 = cleanup direction（对 ∇w_u/∇cos/∇qn² 数值零投影）。

### iss3: State-independent log_std

[`ppo.py:Agent`](../../ppo.py)：

```diff
- self._logstd_head = nn.Linear(hidden_dim, act_dim)   # state-dep, saturated
+ self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))   # state-indep
- LOG_STD_MIN, LOG_STD_MAX = -5.0, 0.0
+ LOG_STD_MIN, LOG_STD_MAX = -2.0, 0.5
```

### ent_coef schedule

[`ppo.py:PPOConfig`](../../ppo.py) 新增 2 字段：
- `ent_coef`: 0.01 → **0.003**（PPO continuous-control 标准；旧值把 σ 推上 clamp）
- `ent_coef_floor`: 0.0 → **1.0e-4**（防止 deterministic collapse）
- `ent_coef_anneal_frac`: 1.0 → **0.3**（30% 训练步内完成 anneal，之后 hold floor）

---

## 5. 验证（5b basis）

`diagnose_sign_seed.py --n-total 2048` 输出关键指标：

| 检查 | 期望 | 实测 |
|---|---|---|
| Anchor cells +% (k=0/g_1, k=1/g_2, k=2/g_3) | ≥ 95% | **100% / 100% / 100%** |
| 下三角 off-anchor `\|mean\|` | ≤ 1e-6 | 最大 1.5e-7 ✓ |
| `‖B^T B − I‖` max over 2048 | < 1e-6 (fp32 floor) | **1.19e-7** |
| Fallback rate (e_0, e_1, e_2) | < 5% | **0% / 0% / 0%** |

---

## 6. 训练结果

### 主结果

| Run | Step | mean per-line `L_rl/L_zero` | median per-line `L_rl/L_zero` | σ_final | 备注 |
|---|---|---|---|---|---|
| 旧 plateau | 9.2M | 0.93 | — | — | a≡0 worse |
| 5b only | 1M | 1.624 | 1.125 | 0.80 | ent_coef anneal 巧合让 σ 没饱和 |
| 50M debug | 15M (killed) | — | — | 0.999 (顶) | iss3 暴露：σ 饱和 LOG_STD_MAX |
| **5b + iss3** | **10M** | **1.770** | **1.217** | **0.152** | 主结果 |
| Cheap check (resume) | 13M | 1.989 | 1.227 | 0.140 | median +0.8% → plateau 确认 |

> **Ratio 口径**：表中所有 ratio 是 _per-line_ 比值的均值/中位数 — 即对每条 line `i` 算 `L_rl,i / L_zero,i` 后取 mean / median；不是 `mean(L_rl) / mean(L_zero)`。两个口径在高方差分布下差距可达 10%+。eval.py 的实现见 `ratio_rl_zero = rl_len / zero_len.clip(min=1.0)` 后 `.mean() / np.median()`.

### Cross-baseline 对照（200-line holdout @ 10M ckpt）

| Denominator | mean per-line ratio | median per-line ratio | 来源 |
|---|---|---|---|
| a≡0 (no null-space motion) | 1.770 | 1.212 | eval.py 主路径 |
| **Classical hand-tuned (multi-term)** | **1.484** | **1.183** | eval.py 补充评估 (2026-05-18) |
| GPM-JL k_jl=1.0 (broken baseline) | 4.191 | 2.000 | eval.py 主路径 |
| Uniform U(−1,1)⁴ random | 4.651 | 2.129 | 补充评估 (2026-05-18，one-off script) |
| N(0, 0.61) → tanh random | 4.775 | 1.809 | 补充评估 (2026-05-18，one-off script) |

> 补充评估的 random baselines 不在 eval.py 默认输出中，是会话期间为回答 "policy 比随机噪声好多少" 的 one-off 实验，使用同一 200-line holdout 和同一 deterministic policy (actor_mean clamped)，random seed 42。代码片段未归档；若日后重做，复用 `rollout_first_episode` + `zero_nullspace_action_fn` 模式即可。

### Term reason 演化（200-line eval @ 10M）

| | cone | jl | coll |
|---|---|---|---|
| RL | 64.5% | 32% | 3.5% |
| Classical | 86.5% | 12.5% | 1% |
| a≡0 | 84% | 16% | 1% |

RL 把 cone-failure 从 84% 压到 65%（−19pp），代价是 jl 从 16% 涨到 32% —— policy 学到了**正确的 trade-off priority**（cone 是更紧的约束）。

---

## 7. Future work（已识别未修）

**Issue 2 — 单边 clipped delta reward**: cone shaping reward `clamp(min=0.0)` 让 policy 在 cone 上只能学"避免恶化"，无法学"主动改善"。cone 仍占 65% RL 失败模式，是潜在下一轮优化 +10–20% 的最大杠杆。**不建议改 w_cone 权重**（user-flagged：那是放大不对称信号，不解决根因）。修需重新设计 telescoping reward shape。

**σ lower clamp -2.0 现象 attribution 不明**: 10M ckpt log_std = `[-2.0001, -2.0001, -2.0001, -1.5995]`，3/4 维贴下 clamp。三个候选解释：
1. PG advantage 仍在推 σ 下降（"clamp 真的太紧"假设）；
2. advantage 在低 σ 附近趋零后，`nn.Parameter` 因数值漂移碰到 clamp 边（"无害贴边"假设）；
3. ent_coef floor=1e-4 太小，对 σ 的 entropy 反向推力不足，单方向 PG 把 σ 推到底（"floor 调参"假设）。

**放宽到 LOG_STD_MIN=-3（σ_min=0.05）是 cheap diagnostic test**：若放宽后 σ 进一步下降且 ratio 不变 → 假设 (1)；若 σ 不变 → 假设 (2)；若 ratio 改善 → 假设 (3) 部分成立。**plateau 是否会被改善未知**，本轮不做。

**GPM-JL baseline k_jl=1.0 调坏**: L_GPM/L_zero ≈ 0.65 (worse than no-op)。独立于 RL 主线，但 eval.py 仍输出对其的 ratio，可能在 figure 中误导。建议调到 0.1–0.3 区间或换成 "strong baseline" (`k_dm > 0`) 才有意义对比。

---

## 8. 算力对比

| Train | Step | Wall-clock | Final mean per-line `L_rl/L_zero` | 备注 |
|---|---|---|---|---|
| 旧 plateau | 9.2M | — | **0.93** | net-harmful |
| 新 plateau | 10M | ~46 min | **1.77** | net +90% lift |
| Marginal cheap check (resume) | +3M | ~14 min | 1.99 (mean) / 1.22 (median) | median +0.8% confirms plateau |

**相同 compute 量级**（9.2M vs 10M, < 10% 差异），结果 **0.93 → 1.77 = +90% 绝对提升**。

---

## 9. 改动文件清单

```
env/env.py                          +/- 核心：build_task_aligned_basis；删 B_prev 状态
env/baseline_controller.py          callers migrate
env/classical_nullspace.py          callers migrate
tests/test_reward.py                callers migrate
diagnose_sign_seed.py               swap basis call + (d) fb rate + (e) ortho_err
ppo.py                              Agent state-indep log_std；ent_coef floor+anneal_frac；
                                    resume_from_ckpt；σ + fb_rate logging
train.py                            --resume-from-ckpt CLI
```

**Ckpts 留底**：
- `runs/short_5b_iss3_10M/agent.pt` — main 5b+iss3 plateau model
- `runs/short_5b_iss3_13M/agent.pt` — cheap-check fine-tune（+0.8% median over 10M）
- `runs/short_5b/agent.pt` — 5b-only 1M baseline（无 iss3 fix）
- `runs/big_silu_smoke/agent.pt` — lever-2 dead-end 3M smoke（见 §10）
- `runs/big_silu_10M/agent.pt` — lever-2 dead-end 7M partial（已中止，见 §10）

---

## 10. Negative result: Lever-2 capacity scaling 死胡同

**测试设计**（2026-05-19）: 在 5b+iss3 plateau 之外尝试架构扩容 —— width 512→1024, ReLU→SiLU, init gain sqrt(2)→1.0（参数量 1.09M → 4.27M, ~3.93×）。其他 hyperparam 不动（同 ent_coef / LR / clamp）。

**3M smoke 结果**:

| | 新 arch 3M | 旧 arch 1M (5b only) | 旧 arch 10M (5b+iss3) |
|---|---|---|---|
| L_rl/L_zero mean | **1.393** | 1.624 | 1.770 |
| L_rl/L_zero median | 1.152 | 1.125 | 1.217 |
| σ_mean final | 0.286 | 0.80 | 0.152 |
| log_std max-dim | −1.10 | — | −1.60 |

新 arch 3M 时 ratio 已**低于旧 arch 1M**（1.393 < 1.624），但 σ 收缩**更快**（0.286 < 旧 arch 同期 ~0.32）—— 即 policy 更早 commit 到一组更差的策略。

**7M 续训（partial, 中止于 cum 4.75M）**: σ trajectory:

| cum step | σ_mean |
|---:|---:|
| 3.00M | 0.286 |
| 4.00M | 0.288 (no movement first 1M of continuation) |
| 4.75M | 0.267 (rate ≈ −0.011 per 1M) |
| 10.0M (projected) | ~0.21 |

按照 σ-aware 判定矩阵：projected ratio < 1.77 + projected σ < 0.20 → **直接 revert α**，不做 SiLU vs width 拆解，不调 lr/schedule。

**结论 + revert**: 1024 SiLU 架构在当前 PPO + 5b+iss3 setup 上的 plateau 严格低于 512 ReLU。原因不深入诊断（用户决定不进 hyperparam 搜索；后续 push 应该是 BO/PBT 路线而非手调）。ppo.py 已回退到 512 ReLU + orthogonal sqrt(2) 默认。

**留底 artifacts**:
- `runs/big_silu_smoke/` — 3M smoke ckpt + train.log
- `runs/big_silu_10M/` — 7M partial continuation ckpt + train.log（中止时 cum step ≈ 4.75M）

**对未来如想再试 capacity scaling 的建议**: arch 扩容是 binary choice，单独跑短 smoke 没有信息量（≥10M 才能区分"训练不够"与"plateau 更低"）。下次直接配 BO sweep，不要手调单一变量。

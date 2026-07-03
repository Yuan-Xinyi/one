# 实验设计：Seed Ranking —— DP 推理端 K 候选 + 学习排序

（2026-07-03；动机与证据来自 `RL_controller/runs/REPORT_0703.md` 灾难尾归因章节）

> **最终结果（v3，同日深夜）**：**论文口径 97.77%**（起点 first-valid 91.41，capture 57.5%，上限 best-of-17 = 102.46%）。配方：17 路候选（16 DP + pilot）× ens10-pairwise 排序器（obs31+logμ 标准化）× 40k 任务训练集。Easy 93.2 / Medium 98.5 / Difficult 101.2；灾难尾 75% 脱离 <50%。迭代链与 scaling 规律见 `RL_controller/runs/REPORT_0703.md` v3 节。工件：`runs/rank_train/ranker_v3.pt`。
>
> **v1 执行记录（全四阶段完成）**：Phase 0 gate 大幅通过（best-of-8 − first-valid = **+9.18pp**，best-of-2/4/8 = 94.2/98.4/100.6%，候选可交换 mean≈first）；Phase 2 排序器 rank-point held-out capture 26.3%（> min-qn 启发式 16.2%，V 网基线 ≈0，pairwise 实现有 bug 弃用待修）；**Phase 3 端到端：论文口径 91.41 → 93.24%（+1.83pp ± 0.21，显著），Easy 桶 84.0 → 87.7%，灾难尾 385 任务 mean pct 30 → 65.5%（58% 脱离 <50%）**。达成设计目标区间（+1~2pp）。剩余头寸 ~7.4pp（capture 仅 20%）——后续杠杆：锥余量特征、修复 pairwise 损失、K=16、按 L̂ 不确定度混采。工件：`runs/rank_phase0/`（10k×8 候选+全槽 L）、`runs/rank_train/`（20k×8 训练集 + ranker.pt）。

---

## 1. 背景与动机

**控制器战役已收敛**（系统 1.0669 / 单网 1.0510，τ 面、标签质量、soup、两种切换学习化全部摸到平台或被否决）。论文口径（DP 种子 × pct oracle′，现 Hybrid 行 91.4%）的剩余失血定位如下：

- Easy 桶灾难尾（405 任务，pct<50%）占总失血 62% 的主体；
- 死因 65% 是 cone violation 早死（93% 在 50 步内，中位 31 步），**不是 joint-limit**；
- 归因实验：换 classical 最优种子救回 46%，再换 oracle 逐任务最优种子又救回其余的 67%——**约 85-90% 的灾难尾是种子选择问题**。同一控制器从对的种子出发就能解。

而当前 DP 部署策略（`system_eval/sweep_cfg_only.py: dp_lazy_newton`）是 **lazy Newton：DDIM 采样后取第一个 IK 有效样本**——推理端完全没有选择环节。候选之间的质量方差就是白白丢掉的性能。

**核心命题**：在推理端生成 K 个 IK 有效候选，用一个学习的打分器 L̂(s₀) 排序取 argmax。部署成本 = K 次 DDIM 采样（本来就是批量的）+ K 次 MLP 前向，**零额外 rollout**。

预期收益：粗算灾难尾修复价值 +2pp 论文口径（91.4→93+），Medium 桶同类修复或再 +1pp；按排序器通常能兑现 best-of-K 头寸的 50-70% 估计，**现实目标 +1~2pp**。

## 2. 可复用资产（全部已存在）

| 资产 | 位置 | 用途 |
|---|---|---|
| K 候选生成器 | `system_eval/seed_sources.py: diffusion_seeds(n_samples, ddim_steps, cfg_w, ...)` → (n_tasks, N, 7) + ik_ok | Phase 0/1/3 的候选来源 |
| DiT + Newton 投影 | `seed_selection/diffusion`, `seed_selection/smm: newton_project` | 同上（diffusion_seeds 内部已封装） |
| 批量精确评估 | `system_eval/rollout_controllers.py: rollout_seeds_batched`（确定性环境 → 标签精确） | 标签生成、端到端评估 |
| 采用控制器 | `RL_controller/runs/distill_r12m_b0.965_soup2` + τ=0.985/0.96 | 标签用部署控制器打，避免 train/deploy 失配 |
| 零训练基线打分器 | `distill_r11_belt0.965/vswitch/{vcls_sym,vpi_sym}.pt`（obs→V 的回归网） | 免费 baseline：按 V̂ 排序 |
| 训练任务池 | `LineDistribution.load_or_build`（训练分布，与 10k 评估集不相交） | 排序器训练数据 |
| oracle′ 归一化 | `cell_oracle_hyb_results.npz: L_best` | 论文口径分母（不变，保证可比） |

## 3. 分阶段设计（每阶段带 gate）

### Phase 0 — 头寸验证（gate，~1 GPU 小时）

灾难尾归因用的"最优种子"来自 SMM 标签池（oracle_hyb 由 `run_oracle_prime` 从 top-K′ 池构建），**不是 DP 采样器自己的样本**。必须先确认真实采样器的候选内部有足够方差：

1. 在 10k 评估集上（或先抽 2000 任务）用 `diffusion_seeds` 生成 **K=8** 个 IK 有效候选（w=1.5，与现部署一致）；
2. 全部候选 × 采用控制器 rollout（8 × 10k = 8 万 rollout ≈ 20 chunk，~30-60 min）；
3. 报告三条线：**first-valid**（现状）、**best-of-8**（排序上限）、mean-of-8（随机基线），均按 pct oracle′。

**Gate：best-of-8 − first-valid ≥ +1pp 才继续**；不足说明 DP 采样太集中，改进方向应转向扩散模型本身（提高采样多样性/条件质量），本设计中止。
副产品：这批 8 万 rollout 直接量化"选择"这个动作在真实采样器上的价值，可写进论文（cell：diff_hyb_firstvalid vs diff_hyb_bestof8）。

### Phase 1 — 训练数据生成（~2-3 GPU 小时）

- **任务**：训练池采样 20,480 任务（seed 与一切既有实验不相交，建议 9200）；
- **候选**：每任务 K=8 个 IK 有效 DP 样本（与部署同 w、同 DDIM steps；IK 无效样本照 lazy 逻辑重采补足，记录重采次数）；
- **标签**：每候选用采用控制器精确 rollout → L（确定性环境，标签零噪声）；
- **特征**（逐候选存齐，训练时做消融）：
  - 初始 obs（31 维，env reset + p_start 约定与评估一致——复用 `mode_selector.py: initial_obs` 的实现）；
  - 派生量：init_max_qn、FK 位姿、锥边界余量（death 主因是 cone，这个特征大概率关键）、Newton 残差、manipulability；
- **产出**：`seed_selection/runs/rank_train_20k_K8.npz`（obs0/features/L/task_id），一次生成永久缓存。

规模账：163,840 rollouts ≈ 40 chunk × ~1 min ≈ 1 小时级；加 DDIM 采样与 Newton 若干。

### Phase 2 — 排序器训练与离线选型（CPU/轻 GPU，~1 小时）

- **模型**：小 MLP（256×3，对齐 VClsNet 规格足够）；
- **两种损失对比**：
  1. pointwise：直接回归 L（标签精确，最简）；
  2. pairwise：同任务内候选两两 margin ranking loss（选择问题的标准做法，通常 top-1 regret 更低）；
- **选型指标**（held-out 任务上，绝不用 MSE 选型）：
  - **top-1 regret** = mean(L_best − L_picked)；
  - **capture rate** = (L_picked − L_firstvalid) / (L_best − L_firstvalid)；
- **必跑 baseline**（防"排序器只是学了个简单启发式"）：
  - first-valid（现状）；随机候选；
  - min init_max_qn（离关节限位最远）；
  - max 锥余量（针对 cone 死因的手工启发式）；
  - **V̂_cls / V̂_π 排序**（vswitch 现成网，零训练）——若它们已接近学习排序器，说明不需要新训练，直接部署旧网。
- **消融**：K ∈ {2,4,8,16}（收益-成本曲线，定部署 K）；特征组（obs0-only vs +锥余量 vs 全量）。

**Gate：capture rate ≥ 50% 才进 Phase 3**；30-50% 之间看特征消融还有没有救；<30% 记录负结果（与 value-switch/mode-selector 同一档案：头寸存在但不可从 s₀ 兑现——不过此处先验强得多：种子质量在 s₀ 完全可观测，不依赖轨迹后段，这正是它和前两个被否决方案的本质区别）。

### Phase 3 — 端到端与论文口径（~1 GPU 小时）

1. 10k 评估集逐任务：K=8 候选（Phase 0 已缓存）→ 排序器 argmax → 该种子的 rollout 结果（Phase 0 已缓存，直接切片，与 mode_selector 评估同法，零新 rollout）；
2. 报告：paper 口径 All/Easy/Medium/Difficult 四桶 + 内部口径，对照 first-valid（91.4%）、best-of-8（上限）、各手工基线；
3. 专项验证：灾难尾 405 任务的 pct 分布前后对比（设计动机的直接闭环）；
4. 部署成本表：t/task（DDIM K 样本 + K 次 Newton + K 次 MLP），对照 `sweep_cfg_only` 现有的 t_seed_per_task_ms 口径。

### Phase 4 —（可选）集成与加固

- 新 cell 进 `system_eval`（diff_hyb_ranked），全表刷新时一并出；
- 排序器跨控制器转移测试（hybrid 标签训练的排序器用于 pure 单网——若转移好，一个排序器服务两条产品线）；
- cfg_w × K 联合小扫（w 影响样本多样性，可能与 K 有交互）；
- q0 扰动鲁棒性复验（排序器挑的种子是否更脆）。

## 4. 风险与对策

| 风险 | 信号 | 对策 |
|---|---|---|
| DP 采样多样性不足 | Phase 0 best-of-8 ≈ first-valid | 中止本设计；转向扩散模型多样性（温度/DDIM steps/多 w 混采）或 SMM 池混合候选 |
| 排序器只学到 qn 启发式 | 与 min-qn baseline 打平 | 特征消融定位；若手工启发式已够，直接部署启发式（更简单也是赢） |
| 训练/评估任务分布偏移 | held-out capture 高、10k capture 低 | 训练池与评估集同生成族，先验风险低；必要时训练任务量翻倍 |
| 标签-控制器耦合 | 换控制器后 capture 下降 | 标签用采用控制器打（已定版 1.0669）；控制器再变时用 Phase 1 管线重标（缓存化，成本 1 次） |
| IK 有效候选不足 K 个 | 重采次数暴涨 | 记录重采统计；对 IK 难任务允许 K' < K，排序器照常 |

## 5. 成本与时间线汇总

| 阶段 | GPU | 人力 |
|---|---|---|
| Phase 0 gate | ~1 h | 脚本半天（大部分复用 diffusion_seeds + rollout_seeds_batched） |
| Phase 1 数据 | ~2-3 h | 与 Phase 0 共用管线 |
| Phase 2 训练选型 | <1 h | 半天（模型/损失/基线全是现成模式） |
| Phase 3 端到端 | ~1 h（大部分切缓存） | 半天 |
| 合计 | **~5 GPU 小时** | **2-3 天** |

## 6. 成功标准

- 最低：论文口径 All +1pp（91.4 → 92.4+），灾难尾 pct<50% 任务数减半；
- 目标：+2pp（→93+），Easy 桶 83 → 87+；
- 上限参考：Phase 0 的 best-of-8 线。

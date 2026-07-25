# IK 候选池实验手册（E1–E7）

**版本：** 2026-07-23
**目标：** 把种子动作空间从 diffusion 候选池（8+1）扩大为锥约束 IK 枚举池（K=32+1），用 controller 完整轨迹回报后向监督重训 selector，在"一次 selector 前向、一个种子、一次 controller rollout"的部署约束下显著超过原系统。
**前置证据（500 任务 pilot，已缓存于 `runs/_ikpool_pilot_v1/`）：**

| 量 | diffusion 池 | IK 池 |
|---|---:|---:|
| oracle 天花板 | 0.567 m | **0.618 m（+51.4mm）** |
| 池内 spread 中位 | 145 mm | 472 mm |
| 迁移旧 selector capture | — | −25%（失败） |
| **重训 listwise MLP capture（仅 400 任务）** | — | **+33.5%，Spearman 0.53** |

**总决策树：** E1 → E2 出 capture–训练量 scaling 曲线。15k 任务时 held-out capture ≥45% → 全线执行 E3–E7；35–45% → 先做 E3 看绝对进度能否赢旧系统再决定；<35% → 本线降级终止，回到弱结论叙事。

---

## 0. 环境与通用纪律

### 0.1 运行环境

```bash
PY=/home/lqin/miniconda3/envs/one/bin/python     # torch 2.10.0+cu126
cd /home/lqin/one                                 # 一切命令的 cwd
PYTHONPATH=/home/lqin/one                         # 独立脚本需要；-m 方式不需要
# GPU: RTX 4090 24GB。E1 峰值 ~6GB，ensemble 训练 ~2GB，可与轻负载共存
```

已知问题：

- **matplotlib CXXABI 报错**只出现在部分 ad-hoc `python -c` 片段（`one` 包链会 import matplotlib）；正式脚本从未受影响。遇到就把逻辑写进 .py 文件跑。
- **本机 CUDA 1-D`torch.dot` SIGFPE**：新代码一律用 `(a*b).sum()`。
- **GPU SVD 有 batch 依赖的浮点差异**：rollout 的 chunk 大小是协议的一部分，必须固定并写入产物元数据（本手册统一 chunk=512）。

### 0.2 六条纪律（每条都有本项目的事故背景）

1. **复用历史配方前先核对 provenance**：不要按文件名猜。已验证的口径：从 checkpoint 里读 `args`/`offline_seed_ensemble_provenance.settings` 提取真实 argv，只改 `--seed`/`--out-dir`，先跑一个种子验证与原 artifact 逐位一致（gate 阈值等）再扩量。
2. **先小样本再过夜**：任何多小时任务先用 1/30 规模冒烟（E1 已由 500 任务 pilot 承担）。
3. **per-task 结果一律缓存成 npz** 放在 run 目录里；后续分析只做切片，绝不重跑 rollout。oracle-vs-K 曲线就是零成本切片得到的。
4. **驱动脚本必须 `set -e` 且每个阶段结束检查输出文件存在**。事故：多种子 sweep 脚本里一个路径笔误导致 eval 阶段静默空转，训练照常、日志看似正常。
5. **基线产物不可变**：新实验全部写新目录（`runs/ikpool_*`），不触碰 `r2_*`、`final_holdout_*`、`joint_*`。
6. **门槛 fail-closed**：promotion 失败就回滚，不为叙事保留只在单一开发集偶然变好的模型。

### 0.3 关键资产清单

| 资产 | 路径 | 说明 |
|---|---|---|
| C0 controller + 源 checkpoint | `runs/r2_grouped_best/`（unified.pt / agent.pt / config.yaml） | phase=round_complete，35-D，18,432 train 行，split_mode=task-geometry-grouped-v1 |
| diffusion 候选缓存 | `Yuan/seed_selection/runs/rank_train/candidates_K8.npz` | 20,480×8+pilot；外部池 rank_train_b / rank_train_c |
| diffusion 回报表 | `runs/r2_full_returns_v1/train_returns.npz` | C0 下全候选 progress_m，与 train 行对齐 |
| 45-D ensemble 训练器 | `offline_seed_ensemble_train.py` | 已验证命令见 §E2.2；70 s/种子 |
| 锥约束 IK 枚举器 | `Yuan/seed_selection/smm/cone_ik.py::cone_constrained_ik_enumerate` | 注意默认 cone=5°，必须显式传 29.5 |
| pilot 三脚本（已入库） | `ikpool_pilot.py` / `ikpool_retrain.py` / `eval_actor_q_gain.py` | gen/roll/analyze、重训信号测试、actor–Q 增益复现 |
| pilot 缓存 | `runs/_ikpool_pilot_v1/` | 500 任务候选+回报+分析 json |
| 多种子 ensemble | `runs/_multiseed/` | seed 31000 复现件 + 32000–35000 |

### 0.4 本次会话踩过的 provenance 门（写新代码前必读）

- `materialize_actor_q_selector.py` 要求 controller-source 是 **45-D、phase=round_complete** 的 selector checkpoint；ensemble 产物（phase=offline_seed_ensemble_complete）不满足 → 不要试图硬灌，走 §E2 的绑定方案。
- `load_return_cache` 校验回报缓存与**生成它的 source checkpoint** 的 phase/sha 一致；给它传 ensemble checkpoint 会报 `source phase differs`。正确做法：source 参数永远传 `r2_grouped_best/unified.pt`（或 E1 新建的 source）。
- ensemble checkpoint 顶层 `candidate_cache` 是**字符串路径**；带 size/sha 的 dict 在 `offline_seed_ensemble_provenance.source_candidate_cache`。
- 用目录当 `--base/--updated-checkpoint` 时，目录里必须同时有 `unified.pt`、`agent.pt`、`config.yaml`（hardlink 即可）。
- `CachedSeedCandidateDataset.from_npz` 要求 mask 名为 `ik_ok` 或 `ok`；有 `q0_pilot` 时自动追加为 fallback 槽（n_candidates 变 K+1，`fallback_index=K`）。无效槽的 NaN seeds 是历史合法格式，下游自动清洗。
- `validate_cached_dataset` 的子集保留**原始 task_indices 值**——它们是进 r2 回报表的行号，不是位置下标。pilot analyze 阶段曾因混淆两者出过 IndexError。

---

## E1：15k 任务 IK 池生成 + C0 全候选回报缓存（过夜）

### E1.1 任务集

用 `r2_grouped_best` 的**全部 18,432 train 行**（已 geometry-grouped，与 validation 2,048 / external-dev 2,048 天然不重叠）。不复用 pilot 的 500 任务产物：E1 的 per-task RNG 以**全局行号**播种（`default_rng(GEN_SEED*1000 + global_row)`），与 pilot 的局部编号不同，池子不一致，混用会破坏可复现性。pilot 目录退役为只读参考。

### E1.2 生成参数（锁定，全部进产物元数据）

```
cone_angle_deg = 29.5      # 校验门是 30°(vs n_target)，留 0.5° 余量。默认值 5.0 是错的
n_orientations = 16, n_ik_restarts = 8    # 128 次 DLS IK/任务，中位产出 49 解
joint_margin   = 0.02
dedup_rad      = 0.08
K_POOL         = 32        # oracle-vs-K 在 16–24 饱和(K=28→32 边际 +0.2mm)，不上 64
FPS 最远点采样保序截取      # 多样性是曲线形状的来源，随机截取无效
q0_pilot       = 沿用 diffusion 缓存的 classical fallback 槽
GEN_SEED       = 另取新值并记录(不要沿用 pilot 的 20260723)
```

### E1.3 执行

改 `ikpool_pilot.py`（或复制为 `ikpool_build_full.py`）：任务源从 calibration 前缀改为全部 train 行；RNG 改全局行号播种；加 `--shard i/n` 支持按行区间分片（分片安全的前提就是全局行号播种）。然后：

```bash
# 冒烟：一个 shard 的前 200 任务，确认产出统计与 pilot 同量级(中位~49 解、frac_valid~0.87)
$PY ikpool_build_full.py gen --shard 0/8 --n-tasks 200 --out-dir runs/ikpool_full_v1_smoke
# 正式（后台、分片、驱动脚本 set -e）
for s in 0 1 2 3 4 5 6 7; do $PY ikpool_build_full.py gen --shard $s/8 --out-dir runs/ikpool_full_v1; done
$PY ikpool_build_full.py roll --out-dir runs/ikpool_full_v1     # chunk=512 固定
```

预算（按 pilot 线性外推）：gen ≈ 5–8 h，roll 18,432×~31 有效候选 ≈ 570k 条完整 episode ≈ 6–9 h。两者可流水（gen 完一个 shard 就 roll 一个）。峰值显存 ~6GB。

### E1.4 验收标准（不满足即停）

- [ ] 每 shard 的 `n_solutions_raw` 中位 ∈ [40, 60]，`frac_valid` ∈ [0.80, 0.95]；
- [ ] 无零有效候选任务（`validate_cached_dataset` 会拒绝，出现说明生成参数错了）；
- [ ] 抽 100 任务核对 oracle-vs-K 前缀曲线形状与 pilot 一致（K=16 处 ≥95% 饱和）；
- [ ] 产物含 SHA-256、生成参数、shard 边界、chunk 大小；
- [ ] rollout 表 NaN 只出现在 invalid 槽。

---

## E2：selector 重训 + capture–训练量 scaling 曲线（决策实验）

### E2.1 split

对 18,432 任务做 **exact float32 (p0, line_dir, n_target) 签名的 geometry 三段 split**（fit / model-select / calibration ≈ 70/15/15），复用 `_three_way_geometry_split`。历史教训：v3 曾因 row-heldout 泄漏 18.6% 几何，结论作废。

### E2.2 两条轨道

**轨道 A（决策用，先跑）**：扩展 `ikpool_retrain.py` —— listwise MLP + 与生产一致的 5-member geometry-bootstrap mean-set encoder（45-D 特征、range-normalized listwise target、feasibility Huber）。在 fit 子集的 {1k, 2k, 5k, 10k, 全部} 任务上各训 3 个种子，held-out（calibration 段）报 capture 和绝对进度。**注意特征归一化必须用 IK 池 fit 段的 valid-only 矩（迁移失败的根源就是 diffusion 分布的旧归一化）。**

**轨道 B（论文用，A 通过后做）**：把 IK 池接入认证管线 `offline_seed_ensemble_train.py`。它的 provenance 门要求 source checkpoint 绑定候选缓存 sha，因此需要先写 `make_ikpool_source.py`：以 r2_grouped_best 为模板构造绑定 IK 缓存的 round_complete 源（format_version=4、provenance format=unified-bidirectional-v4、train/validation 索引与 valid mask、controller 三件套 sha）。之后训练命令沿用已验证配方，只换缓存路径：

```bash
$PY -m Yuan.unified_rl.offline_seed_ensemble_train \
  --source-checkpoint runs/ikpool_source_v1/unified.pt \
  --return-cache runs/ikpool_full_v1/train_returns.npz \
  --controller-ckpt runs/r2_grouped_best \
  --out-dir runs/ikpool_ens_v1_seed<S> \
  --learning-rate 5e-4 --temperature 0.05 --headroom-weight 2.0 --seed <S>
# 非默认参数就这三个；成本 ~70 s/种子；训完先验证一个种子的 gate 阈值可复现
```

### E2.3 决策阈值

- capture(15k) ≥45% → 继续全线；
- 35–45% → 用绝对进度做 E3 预判（新系统估算 = ik_first + capture×(ik_oracle−ik_first)，须 > 旧系统在同任务上的 ≈0.532 m 至少 5mm）；
- <35% 或曲线在 5k 后走平 → 终止本线。

同时记录：3 种子间 capture 标准差（>5pp 说明仍不稳，加种子再判）；MLP train-capture（pilot 时 99%，过拟合程度应随数据量下降）。

---

## E3：系统对决（新 vs 旧，同 C0、同部署预算）

### E3.1 协议

- 任务集：validation 2,048 + external-dev 2,048（`build_external_holdout.py` 产物，已审计零重叠）。需要先为这两个集合生成 IK 池并全候选 rollout（E1 同参数、新 GEN_SEED 记录）≈ 2×2,048×31 rollout ≈ 1.5 h。
- 旧系统臂：diffusion 池 + 生产 S1Q/C0（或 S0/C0，两个都报）；直接用现有回报表切片，零 rollout。
- 新系统臂：IK 池 + 重训 selector + **margin gate fail-closed 到 classical fallback 槽**（阈值只在 model-select 段定，calibration 只做通过/回滚复核——完整照抄 `materialize_actor_q_selector.py` 的固定网格纪律，w 网格、scale=0.01 不变）。
- 两臂 paired：同任务、同 controller、各自一次 rollout。

### E3.2 指标（主表全报，不许只报 mean）

mean / 5% trimmed / ±50mm clipped / >1mm harm & win rate / geometry-bootstrap 20k 次 95% CI。多种子叙述：报"n 个训练种子中 m 个通过 promotion 且 CI>0"。历史教训：diffusion 线的 +1.6mm 曾是 trimmed≈0 的重尾均值、5 种子 2 个回滚——同样的检查这里必须先自己做。

### E3.3 通过标准

新系统相对旧系统：paired mean CI 下界 >0 且 trimmed mean >0（这次天花板 +51mm，若仍是重尾均值故事，说明 selector 只学会了少数任务，回 E2 加数据或改特征）。

---

## E4：多训练种子稳健性

- ≥5 个种子完整重复 E2 轨道 B + E3 评测（每种子 train 70s + eval 切片分钟级）。
- 纪律：第一个种子先做**逐位复现验证**（同 argv 同 seed 重跑，比对 gate 阈值/deployment 字典完全一致），确认命令忠实后再扩其余种子。
- 预注册规则（写进 runs/ 下的 json，sealed 前不改）：报所有种子的 mean±std；发布 checkpoint 取 model-select 段 paired mean 中位数的那个种子，**不取最好的**。

---

## E5：基线与消融（大部分零成本切片）

| 消融 | 方法 | 成本 |
|---|---|---|
| 旧 selector 迁移失败 | pilot 已有（capture −25%），在 15k 版复算一遍 | 切片 |
| 几何单特征失败 | pilot 已有（最优单特征 −23%，Spearman 0.28）；补 log-manip / directional-manip / joint_horizon 三个命名特征各自的 capture | 切片 |
| 池大小 K∈{8,16,24,32} | FPS 前缀天然嵌套：截取回报表前 K 列，selector 各训一次 | 4×70s+切片 |
| union 池（diffusion∪IK） | 两表按任务拼接（9+33 槽），oracle 直接取 max；selector 在 union 上重训一次 | 1 次训练 |
| 特征消融 | `--no-log-manip`、去 directional 10 维，各训一次 | 2×70s |
| ridge vs MLP vs ensemble | 轨道 A 已有 ridge/MLP，补生产 ensemble 同表对比 | 切片 |

45-D 特征布局备查（0 起）：0–6 q_norm，7–13 q_norm²，14–16 line_dir，17–19 z_tool，20–22 n_target，23 cos 锥角，24–26 z×n，27–30 零占位，31–33 lateral ray error，34 log 位置可操作度，35–41 沿线关节速度，42 其范数，43 关节限位 horizon，44 方向可操作度。

---

## E6：sealed final holdout（一切规则冻结后才做）

照抄 `final_holdout_joint_v2` 的纪律，checklist：

- [ ] 冻结：selector 权重、w、gate 阈值、K、生成参数、chunk、发布种子选择规则（E4 的预注册 json）；
- [ ] 新 LineDistribution pool seed + task seed + IK GEN_SEED（三个都是新值，记录进 meta）；
- [ ] 10,000 任务、10,000 唯一几何、内部重复 0；
- [ ] 与**全部**排除缓存做 exact float32 geometry overlap 审计 = 0：rank_train A/B/C、10k systematic、external-dev、历史 fresh、sealed v1/v2、**ikpool_full_v1 及 E3 的两个 IK 池**；
- [ ] 候选缓存 SHA-256 记录；生成后主模型零改动；
- [ ] 每臂一次读取：旧系统、新系统、first-valid、complete-candidate oracle（oracle 是诊断不是方法）；
- [ ] 部署元数据：`selector_forwards=1, controller_probes=0, model_rollouts=0`；
- [ ] 报告全套稳健统计（同 E3.2）+ 2×2 若 E7 做了 controller 更新。

预算：候选生成 10k×(128 IK) ≈ 3–5 h，四臂 rollout ≈ 每臂 5–10 min（静态）+ oracle 全跑 ≈ 3 h。

---

## E7（可选）：controller 前向适配

这次值得再试的理由：IK 种子跨运动学分支，selector 诱导的 reset 分布真的变了（先量化：新旧 selector 选中种子的关节空间距离分布、换分支比例——diffusion 线上只有 14.78% 任务换种子且同分支，这是前向无效的根源）。

- 方法：C0 在新 selector Top-1 种子分布上继续训练（reset 混合沿用 70/20/10），或 `joint_controller_search_distill.py` 的局部搜索蒸馏；
- 门槛：沿用既有 promotion gate（geometry CI 下界>0、harm≤10%、validation+external 双集不回退），失败自动回滚；
- 通过则补 2×2 分解进 E6；失败则如实写"前向边际仍不显著"，不影响主线——主结论由 selector 承载。

---

## 时间预算汇总

| 阶段 | 墙钟 | 可否过夜 |
|---|---:|---|
| E1 gen+roll | 11–17 h | 是（分片流水） |
| E2 轨道 A + 决策 | 2–3 h | 否（要人看曲线做决定） |
| E2 轨道 B + E4 五种子 | 1–2 h | — |
| E3（含两个 dev 集 IK 池生成） | 3–4 h | — |
| E5 全部 | 2–3 h | — |
| E6 sealed | 8–12 h | 是 |
| E7 | 1–2 天 | 部分 |

**关键路径：E1（过夜）→ E2 决策点 → 其余并行度高。第一个自然停点是 E2 的 scaling 曲线。**

# system_eval

端到端系统评估：在分层抽样的 8953 任务 held-out 子集上做 **seed × controller 5-cell 消融**，
量化扩散种子选择 vs 经典 seed、与 hybrid 控制器 vs 经典控制器各自的贡献。

Cell 命名 = `<seed_source>_<controller>`：

| Cell | Seed 来源 | Controller | 含义 |
|------|-----------|-----------|------|
| `cls_cls`     | pilot q0_seed                       | classical          | baseline                |
| `diff_cls`    | diffusion best-of-N + Newton IK     | classical          | seed-only 消融          |
| `cls_hyb`     | pilot q0_seed                       | hybrid (variant B) | controller-only 消融    |
| `diff_hyb`    | diffusion best-of-N + Newton IK     | hybrid (variant B) | 完整方法                |
| `oracle_cls`  | `labels_q0[argmax(labels_L_clean)]` | hybrid (variant B) | 经典-标签 oracle（控制器错配） |
| `oracle_hyb`  | argmax over SMM top-K' under hybrid | hybrid (variant B) | **控制器感知 oracle**，真实上界 |

## 目录结构

```
system_eval/
├── config.yaml             冻结配置：数据/扩散/RL/env/runner/output/metrics
├── build_eval_set.py       构建分层 safe 子集 eval_set_*.npz
├── seed_sources.py         每个 cell 的 seed 构造器（baseline/oracle/diffusion+IK）
├── rollout_controllers.py  classical + hybrid_variantB 两种控制器的批量 rollout
├── run_cell.py             单 cell 跑批入口，落 cell_<name>_results.npz
├── run_oracle_prime.py     控制器感知 oracle (`oracle_hyb`) 专用入口
├── aggregate.py            CSV + markdown report + figures
├── run_all_cells.sh        6 cell 按顺序跑（diff_cls 写扩散种子 cache，diff_hyb 复用）
└── runs/
    ├── pilot_2k/           Pilot 验证（2k 任务，已通过 sanity check）
    └── eval_10k_systematic/ 主结果目录
```

## 主要文件

| 文件 | 功能 |
|------|------|
| [config.yaml](config.yaml) | 数据源（pilot_20k.npz + plane_collision.npz）、bucket 阈值与配额（weak 2500 / medium-weak 2500 / medium 3000 / strong 2000）、扩散超参（n_samples=8, ddim_steps=50, cfg_w=1.5）、hybrid 阈值（tau_enter=tau_exit=0.98 即 variant B）、env 引用 `RL_controller/config.yaml`、metrics（progress 阈值、catastrophic 阈值、recovery_fraction=0.9）。 |
| [build_eval_set.py](build_eval_set.py) | 从 `pilot_20k.npz` 中按 status ∈ {kept, edge, edge_seed_fallback} + `any_label_collides == False` 过滤，再按 L_seed bucket 分层抽样到目标配额。保存对齐字段 `src_idx, bucket, cs_*, q0_seed, L_seed, max_label_L, max_label_q, n_labels`。 |
| [seed_sources.py](seed_sources.py) | `baseline_seeds`(pilot q0_seed) / `oracle_seeds`(label-argmax) / `diffusion_seeds`(DiT 采样 N 个 + Newton IK 精化，返回 ik_ok 掩码)。`oracle_hyb` 不走这里，run_oracle_prime 直接从 SMM top-K' 池取。 |
| [rollout_controllers.py](rollout_controllers.py) | `rollout_seeds_batched`：扁平 (B, 7) seed 通过共享 `NSRLBatchedEnv` 跑批，支持 `controller='classical'` 与 `controller='hybrid_variantB'`。`env.p_start` 覆盖到任务 p0，与 SMM 数据生成对齐。 |
| [run_cell.py](run_cell.py) | 单 cell 主循环：加载 eval set + 构造 seed + 调对应控制器 rollout，写 `cell_<name>_results.npz`（含每样本 L、IK 掩码、term reason、best-of-N reduction）。支持每 N 任务 checkpoint，断点续跑。 |
| [run_oracle_prime.py](run_oracle_prime.py) | `oracle_hyb` 专用：对每任务遍历 `top_Kprime_q[t]` 所有 SMM 候选，逐个在 hybrid 控制下 rollout 取 max L。输出 schema 与 run_cell 一致，aggregate 自动识别为新 cell。 |
| [aggregate.py](aggregate.py) | 跨 cell 聚合 → `summary_table.csv` + `summary_report.md` + 4 张 figures（deployment_gain_by_bucket / recovery_distribution / ablation_decomposition / oracle_gap）。扩散 cell 同时报 `progress_best_m`（IK 失败计 0）与 `progress_realistic_m`（全失败 fallback 到 cls_cls）。 |
| [run_all_cells.sh](run_all_cells.sh) | 6 cell + aggregate 一键执行；`diff_cls --write-diffusion-cache` → `diff_hyb --diffusion-cache` 共享扩散+IK 种子池避免重算。 |

## 关键设计

- **报告单位用 meters**：`progress_m = L_best * target_distance_m`（1.5m 归一化只是中间约定，部署关心绝对位移）。
- **Realistic 列**：扩散 cell 的现实部署数 = best-of-N，但 N 个 IK 全失败时 fallback 到 cls_cls 的结果。
- **Oracle 的两个层级**：
  - `oracle_cls` 是 **经典控制器下** 的标签最优 seed，搬到 hybrid 控制器下不再保证最优（`diff_hyb` 在 ~84% 任务上反而胜过它）。
  - `oracle_hyb` 是 **hybrid 控制器下** 在 SMM 候选池里取最优，是真正的可达上界。
- **配置冻结**：`config.yaml` 是 10k 系统评的不动产；调参开新目录，不要原地改。

# Seed ranking pipeline (2026-07-03/04)

DP 推理端 K 候选 + 学习排序，论文口径 91.41 → 98.38%。设计与结果见
`../DESIGN_seed_ranking_0703.md`，完整证据链见 `../../RL_controller/runs/REPORT_0703.md`。

执行顺序（工件缓存在 `../runs/rank_phase0|rank_train*/`，重跑自动跳过已有缓存）：

1. `rank_phase0.py` — 10k 评估集 × K=8 候选（w=1.5）+ 全槽 rollout；gate 实验。
2. `rank_phase1.py` — 训练分布 20480 任务 × 8 候选精确标签 + 初始 obs。
   `rank_train_b/`、`rank_train_c/` 由同脚本改 `manual_seed(9250/9350)`、
   `sample_seed(9500/9700)`、`OUT` 生成（当时用 sed 派生，未存副本）。
3. `rank_phase23.py` — v1：pointwise 排序器 + 10k 应用（93.24%）。
4. `rank_v2.py` — v2：+pilot 第 9 候选、obs31+logμ 标准化、修复 pairwise、ens5（95.77%）。
5. `rank_k16_ext.py` — eval 侧 K=16 扩展（槽 8-15，w=1.5）。
6. `rank_v3.py` — v3：40k 数据 + ens10-pair + 17 路（97.77%）。
7. `rank_extw1.py` — eval 侧 w=1.0 多样性候选（槽 16-23）。
8. `rank_loss_iter.py` — 损失对决：pair+list 复合胜出。
9. `rank_v4.py` — v4 终版：60k 数据 + pair+list ens10 + 25 路混 w（98.38%）。

部署配方：`rank_train/ranker_v4.pt`（含标准化参数），25 候选一次前向选择，
零 rollout，~0.42 s/任务（含 DDIM+Newton；打分本身 <0.1 ms）。

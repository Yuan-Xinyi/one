# Qling 已验证可完成任务的策略存档（2026-08-31）

| 文件 | 任务 | 验证成绩（真起点/修正口径） | 环境 kind / obs / act |
|---|---|---|---|
| v0_30M_agent.pt | 点物体桌面拖拽（10k 任务分布） | 92.2% placed / 89.1% success（可行性天花板 ~94.5%） | drag / 27 / 7 |
| cont_v2_10M_agent.pt | 30×20 容器 180° 调头（0 换抓） | **100%**（256/256，13 抓取全绿） | container / 35 / 7 |
| rg50_ctd_agent.pt | 50×40 原始尺寸调头（1 换抓，slot_y=0.40） | **100%**（512/512，全部经计划换抓） | regrasp / 37 / 8 |
| rg45_8k_v2_agent.pt | 45×35 调头（剧本换抓版） | 63.3%（成功全经换抓；残余失败=出工作区） | regrasp / 37 / 8 |
| bottle_v3_10M_agent.pt | compare_exp 躺瓶 SE(2) 全位姿（20 抓取候选） | **100%**（critic 选抓取 6/8；~4M 步中期快照，训练仍在续） | bottle / 35 / 7 |

- 配套 `*_config.yaml` 为各自训练时的完整解析配置（复现入口：`python -m drag.train_drag --config <yaml>`）。
- 加载评测：见 drag/eval_container.py、drag/autopsy_regrasp.py、drag/eval_final.py 的 Agent 构造方式（hidden 512，squashed_entropy）。
- 注意：bottle/regrasp_free 类环境评测须用 2026-08-31 修复后的代码（倾角-方位保持、候选主序选择器、start_mode 生效）。
- 待补：rg45f（自由 regrasp）训练中，收官后追加。

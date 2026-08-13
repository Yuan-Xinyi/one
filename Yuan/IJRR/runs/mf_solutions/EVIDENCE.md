# 任务27 "RL解决方案" 系统尝试 — 证据记录
基准: 硬墙0.7321 / 学习纪录0.8405(BC) / 搜索1.0614 / 深段1.698 / 上界1.73
评测口径: 从q0确定性策略, 真实env
## 方法1: expert iteration
- BC on search seq (reachtree 1.0614): policy from q0 = **1.0716 m** (agent_bc.pt) — 首个破1.0的策略
- anchored PPO fine-tune 2M running...
- anchored PPO fine-tune 2M (coef 1.0, norm-returns 0, ent 0.003): 20个评测点中16个 = **1.0716 稳定**;
  两次被冲破(300k→0.27, 500k→0.60)后锚均自愈拉回; 后半程(600k-2M)完全稳定于 1.0716, term=lateral(线出工作空间侧界)
- **方法1结论: 成功。expert iteration = 首个从q0确定性≥1.0的策略(1.0716), 锚定解决了裸微调的自毁**
- 证据: runs/single_task_ppo_v2_ei/{agent_bc.pt, agent.pt, golden_dataset.npz, finetune.log}

## 方法2: 能力门控倒序课程 (无黄金数据, 纯RL+复位+门控)
- 窗口[95,106]: 一次通过(存活66.6, 达标6.6), q0=0.632 — 深段可学
- 窗口[85,106]: 3次尝试(共1.8M步)存活恒定1.4步(需12.6), q0爬到0.683
- 窗口[75,106]: 首次尝试存活6.2(需18.6), 同模式
- **结论: 失败(掐点不可学)。门控正确诊断出0.75-0.85窗口无法被掌握——
  复位状态多为死变体, 回合活不过几步, V^π死循环在课程内复现;
  在模式统计充分后提前终止(节省GPU), q0最好0.683**
- 证据: runs/single_task_ppo_v2_cgc/driver.log, agent.pt

## 方法3: 档案数据上的 max 备份 (离线FQI, 无策略梯度无模仿)
[fqi] dataset 400000 transitions, mean r 0.994, done frac 0.143
[fqi] greedy policy from q0: 0.6524 m -> /home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05/Yuan/IJRR/runs/single_task_ppo_v2_fqi/fqi_qnet.pt

## 方法4: 自我模仿学习 (SIL, 10轮×200k, 锚=自己最好回合)
- 收获曲线: 0.22→0.28→0.29→0.30→0.31, 第5轮起冻结于0.3111
- 最终 q0 确定性 = **0.3111** (甚至低于香草PPO 0.70: 强锚把策略锁死在早期平庸回合上)
- **结论: 失败。SIL 的天花板=自身运气上限, 无法自举过墙; 自锚还会造成早期锁死**
- 证据: runs/single_task_ppo_v2_sil/{sil_driver.log, sil_anchor.npz, agent.pt}

## 方法5: OFU/乐观探索 (RND内在奖励, beta=0.5, 3M)
- eval峰值 0.6814, 末段均值 ~0.63 — 与香草PPO/novelty同一平台, 未破墙
- **结论: 失败。认知不确定性红利在细丝上被真实死亡数据迅速压灭, 与count-novelty同病**
- 证据: runs/single_task_ppo_v2_rnd/{train_stdout.log, agent.pt}

# 总表 (从q0确定性, 单位m)
| 方法 | 结果 | 判定 |
|---|---|---|
| 1 expert iteration (搜索+BC+锚定PPO) | **1.0716 稳定** | 成功 |
| 2 能力门控倒序课程 | 0.683 (掐点窗口不可学) | 失败 |
| 3 离线max备份FQI | 0.652 | 失败 |
| 4 SIL自我模仿 | 0.311 (自锚锁死) | 失败 |
| 5 RND乐观探索 | 0.681 | 失败 |

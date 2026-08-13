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

## 附: 方法1 vs 纯BC 的差距审计 (用户追问)
- 名义: 两者同为1.0716 (锚coef1.0+ent0.003使微调≈BC恒等)
- 鲁棒性: 无差距 — 0.5°起始噪声下两者均崩至~0.36, 动作噪声10%下均~0.26
- 判定: 方法1的实际增量 = "防破坏保险"(已兑现, 对照裸微调两次自毁);
  教科书的"RL鲁棒化"未兑现 — 部分因锚太强, 更因走廊物理容差近零
  (可能不存在鲁棒的1.07策略); 数据: scratchpad/bc_vs_ft.py 输出

## 附2: Q-续接实验 (用户设计) — "协同策略跳跃"的直接证据
同一瓶颈状态s(获胜路线上), 同一入口动作a*, 不同续接:
arc0.40: PPO把a*排16/16; Q_PPO(s,a*)=0.433 vs Q_PPO(s,aS)=0.413 (adv≈0.02, 盲视)
         Q_BC(s,a*)=1.263 (3x); Q_BC(s,aS)=0.413 (一步错专家也救不回)
结论: 动作价值属于(动作,续接)联合体; PG是固定续接的局部算子, 其改进方向不含此解;
需要coordinated policy jump = 把V*信息注入advantage (势函数塑形/搜索值targets)

## 附3: V̂_search-guided improvement (vguide, 用户设计) — 三个V̂变体
机制: 状态来自q0 on-policy rollout; 每个状态模型枚举16后继;
J = E_s[Σ_a π(a|s)·Q̂(s,a)], Q̂=alive·(1+γV̂(s')); 无BC/专家标签/reset
- v1 (探针40步标签, R²=0.87): 0.33-0.49震荡(峰0.49) — Q̂在易区饱和拉平, 无梯度
- v2 (拟合价值迭代去饱和, max值165): 自信地死于0.302 — max备份过估计气泡
- v3 (孪生clipped VI, max值124): 同样0.302 — clip压泡不足以支撑全程贪心
诊断: V̂的**局部排序精度**(audit中gate/safe全对)≠**全程贪心一致性**;
贪心沿途放大最坏误差, 与锚定/BC只需局部纠错本质不同。
下一步(未跑): value-DAgger循环 — 用贪心策略自己的访问状态做探针重标注迭代

## 附4: 经典nullspace接管task 27
classical 0.1698 m/17步/cone (与myopic逐位一致的死点); sgnclassical 0.0991 m/10步/cone。
关节图: classical把j4定在-50°不动, 而所有>0.7m的存活轨迹都要求j4深潜到-170°;
局部梯度律连PPO的0.73都到不了, 更谈不上1.06分支。
数据 classical_task27.npz, 图 classical_vs_bundle.png, 脚本 classical_task27.py

## 附5: ISRR式混合 — 经典nullspace从RL轨迹上接管 (全切换点扫描)
PPO确定性策略走到第k步交给classical续走, k=0..70全扫:
- 每个切换点都死于cone; classical能续的步数随k单调缩短 17→2步
  (RL轨迹越走越贴约束边界, 局部律的余地越来越小)
- 最优混合: k=69 (s=0.69, RL死前一步) 终点0.7109, 比纯PPO 0.7007只多+0.0102 (≈1步)
- 没有任何切换点能进入长分支: 进分支需要几十步前的j4/j2大重排,
  s≈0.69时RL已在注定封死的扇形下缘, 局部律无法回头
数据 hybrid_takeover.npz, 图 hybrid_takeover.png, 脚本 hybrid_takeover.py

## 附6: 精确的ISRR variant B切换 (max|q_norm|滞回, 双向可多切) — task 27
附5是"单次移交上界"; 本节照搬system_eval/rollout_controllers.py的真实规则:
tau 0.98/0.94: 0.4811 m (48步, 1次切换, cone) — **比纯PPO 0.7007更差**
tau放宽扫描: 0.95/0.90→0.36, 0.90/0.85→0.27, 0.85/0.80→0.18, 0.80/0.75→0.17(=classical)
机制: PPO局部最优在s≈0.44起就要求max|q_norm|≥0.98(j6贴限位骑行到0.70),
而1.06分支要求更深的挤压(j4→-170°); variant B的触发信号"任一关节贴近限位→classical"
与长分支的必要条件**恰好反相关** — 它把"贴限位骑行"当危险信号, 而这正是过墙的代价
数据 hybrid_variantB.npz, 图 hybrid_variantB.png, 脚本 hybrid_variantB_task27.py

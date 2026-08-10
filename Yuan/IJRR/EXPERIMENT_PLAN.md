# 实验重设计计划（2026-08-12 定稿讨论版）

四个 claim、四组实验、三个执行阶段。所有比较逐任务配对；比值与分数为主指标，
绝对行程只在同一机械臂内报告。

---

## 0. Claim 与实验的对应

| Claim | 内容 | 证据来源 |
|---|---|---|
| C1 | 这是真正需要长视界冗余分配的问题 | Exp I：myopic ≈ classical < MPC(H↑) < RL 的阶梯 + γ 扫描 |
| C2 | RL 比瞬时冗余分解走得更远 | Exp I 主表 + 终止原因表 + survival 曲线 |
| C3 | 顶点参数化无损且更好学（命题 1 的实验面） | Exp II：matched continuous/projected/vertex + 优化诊断 |
| C4 | 收益不局限于单一直线设定 | Exp III 泛化 + Exp IV 有限速率/平滑性 |

二维定位表（放实验节开头，一小表）：

| 方法 | 视界 | 动作处理 |
|---|---|---|
| No-nullspace (a=0) | 无 | 零 |
| Classical law | 瞬时 | 连续启发式 |
| Myopic one-step | 1 步 | 优化 |
| Receding CEM | 有限 H | 优化序列 |
| Continuous PPO | 长 | 学习·连续 |
| Vertex PPO | 长 | 学习·结构化 |

---

## Exp I — 主表：能力阶梯（固定初始构型）

行（替换现有 6 行中的 WLN、manip-gradient）：

1. **No-nullspace**：a = 0。现有 DLS 行改名即是。
2. **Classical law**：现有。
3. **Myopic one-step optimizer**（新）：
   `a* = argmax_{a∈{±1}^m} M(q + (f+Ga)Δt)`，M 为下一步约束裕度
   （softmin 或加权：关节限位 / 锥角 / 碰撞 / 横向）。
   实现直接复用 switching_signal.py 的 16 顶点一步前瞻，critic 换成 M。
   附带一个连续版（在箱内梯度上升 M）作对照。
   **正文彩蛋**：一步裕度对 a 线性 → myopic 最优本身就在顶点上，
   与命题 1 同构；这不是巧合而是控制仿射结构。
4. **Receding-horizon CEM**，H ∈ {4, 8, 16}（新）：
   每步重规划、只执行首动作；**积分固定在已收敛的 25 ms**
   （开环 CEM 在 50 ms 上被判无效的教训——闭环 + 收敛积分是它回归的前提）；
   模型即环境本身（运动学一致，公平）；只在 512–1024 任务子集上跑；
   同时报告单步计算时间——RL 的 amortization 论点靠这一列。
5. **Continuous PPO**：现有。
6. **Vertex PPO**：现有（rl_vertex_line_30M）。
7. **完整系统**（vertex + hybrid 监督 + selector）单独一行，仅出现一次。

指标（替换单一均值）：mean / median / p10–p90 / 配对 bootstrap 95% CI /
**逐任务胜率** / 单步计算时间。
**终止原因表升为主结果**（锥角/限位/碰撞/横向），它回答"为什么更远"。
**survival 曲线**：S(x) = 实现分数 ≥ x 的任务占比（x = L/ℓ^pw ∈ [0,1]，
有界、跨方法可比、正合退出时间问题的本性）。展示右移是整体的而非长尾拉动。

## Exp II — 顶点机制（严格 matched）

- continuous / projected-at-eval（诊断，不进主表）/ vertex；
  同观测、同奖励、同网络、同步数、同任务分布、同评测集。
- 优化诊断图（双面板）：熵曲线（连续冻结 7.68 vs 顶点 2.77→0.94）+
  饱和统计（|a| 分布、tanh 斜率中位 0.0000、μ 分布）。
- 种子方差：双种子散布 ±0.03 作为全文噪声底线，写进协议段。
- **γ 扫描**（新训练，排 CEM 之后）：γ ∈ {0.9, 0.95, 0.99}，
  物理视界 0.5/1/5 s。预测：γ↓ → 更短视 → 行程降。C1 的直接证据。
- critic-greedy 诊断段落已在文中，归入此组。

## Exp III — Selector（与控制器彻底解耦）

- 控制器固定为 vertex hybrid；现有 tab:ablate_seed 结构保留
  （task-generating / 两个启发式 initialization / first-feasible /
  selector / within-pool oracle + capture 列），**新增 random feasible
  candidate 行**（池内随机，capture 定义的天然零点）。
- oracle = best-of-K 全 rollout，已有；capture 报告
  (selector − random)/(oracle − random)。

## Exp IV — 泛化与部署稳健性

**IV-a 曲线**：
- 主实验维持**常曲率弧线零样本**设计（单参数 κ，支撑贡献三的
  "总转角支配"主张）——该主张目前无数据，弧线扫描必须跑；
  若不成立，贡献三该句改写为蛇形数据支持的版本（"随摆角衰减缓慢"）。
- 蛇形迁移矩阵（2 训练族 × 2 评测族 + 曲率盲 + 双种子）压缩为一段 +
  小表：它证明"学到的是通用冗余策略而非轨迹模板"，
  也是弧线零样本评测合法性的依据。

**IV-b 有限速率一致性**（新小节，兑现 §IV Remark 2 的悬空承诺）：
- dt_sim 收敛（50/25/12.5 ms，排名不变，已有）
- dt_ctrl 扫描（25/50/100/200 ms，投影差 50–100 ms 变号，已有）
- 切换率 16.3 Hz 对 20 Hz（已有）
- 两个 dt 必须分开陈述。

**IV-c 平滑性**：
- smoothness audit 表（已跑：三个 learned 臂逐位相同 → 粗糙不是顶点特有；
  加速度 p95 ≈ 20 rad/s² 是超限项，jerk 在厂商预算内）
- **rate-limiter Pareto**（待跑）：Δa_max ∈ {2,1,0.5,0.25,0.125}，
  行程保持率 vs 加速度 p95，applied 回喂观测。
- tab:continuity（TOPPRA/ruckig 时间参数化）保留作部署面。

**IV-d 难度分桶**：ℓ^pw 三分位之外，加解释性维度：初始限位裕度、
σ_min(J)、初始锥角裕度。预测：收益集中在接近未来可行性瓶颈的任务。

---

## 分期执行

**Phase A — 纯写作 + 已有数据（半天）**
主表行名与列改造、二维定位表、IV-b 小节成文、tab:action_space 升级
（matches→exceeds + CI 列）、统计协议段、终止原因表、
tab:curvature 行名残留（"Degradation, motion"）、Exp II 诊断图占位。

**Phase B — FR3 协议新实验（1–2 晚，互相独立可并行）**
1. a=0 评测（分钟级）
2. myopic one-step（半天实现 + 分钟级评测）
3. receding CEM H 扫描（一晚，512–1024 任务）
4. rate-limiter Pareto（半小时）
5. 弧线 κ 扫描（一晚；决定贡献三措辞）
6. γ 扫描 3 次训练（一晚）
7. survival 曲线 + 难度分桶图（从缓存 npz 出，零计算）

**Phase C — C0 替换与三机械臂（最大，须先过风险点）**
1. **Cobotta m=3（8 顶点）试训**——链条里唯一未知数，先行
2. xArm7 / Cobotta 批式运动学移植（现 env 仅 FR3）
3. 三臂 vertex 训练 → stage-1 重打标签 → selector 重训 → 3 × 10k 重评
4. 填 tab:mainresult / tab:cost / attribution

**决策点**：
- D1（Phase B-5 后）：贡献三保留或改写
- D2（Phase C-1 后）：C0 替换链全量启动与否
- D3（Phase B-3 后）：CEM 若接近 RL，叙事强调 amortization；
  若差距大，强调 horizon 阶梯本身

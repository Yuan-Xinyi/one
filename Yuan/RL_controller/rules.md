# FR3 直线 path-following NSRL 控制器训练规约

## 角色与读者
- 实现者：机器人学 + 强化学习工程师，为 Franka Research 3（FR3,7-DOF 非 SRS 臂）实现 RL 训练管线
- 读者：熟悉冗余机械臂运动学和现代 RL（PPO、Gymnasium API）的研究者
- 不解释基础概念，写代码 + 必要设计注释；不设计过多的安全保障代码

## 总指令
- 一切代码编辑严格按本文件执行；代码与本文件一一对应，代码更新后及时回写本文件
- 在训练框架加入任何"奇妙设计"前先征求用户同意
- 优先复用 `one` 库现有代码；需重新造轮子时征求用户同意

## 任务定义
"沿任务方向 $\hat u$ 尽可能久不被终止"。**line 没有长度概念**，是一条无限方向射线 $(p_0, \hat u, n_{target})$。没有"走完"、"还剩多远"这种概念；agent 唯一的优化目标是最大化 episode 时长。

## 输出文件清单
1. `env/env.py` — 复用 `one` 库的 FK/Jacobian/碰撞实现，定义 torch-batched 并行 Gymnasium 环境
2. `env/ik_init.py` — 给定起点位置 $p_0$，求一个可行 q₀
3. `env/baseline_controller.py` — 手工 GPM nullspace baseline，eval 时作 ratio 分母
4. `ppo.py` — 基于 `cleanrl/ppo_continuous_action.py`，适配 torch-batched env
5. `train.py` — PPO 训练入口
6. `eval.py` — 评估脚本，输出 episode 时长 ratio (RL / baseline)
7. `config.yaml` — 所有超参集中管理
8. `README.md` — 使用说明 + 关键设计决策 rationale + "Known Issues / 待验证" + "Open Questions"

约束：
- 不伪造未使用过的 API；不确定的接口标 TODO
- 不硬编码路径（FR3 在 `one` 库内是过程式构建，不走 URDF）

---

## 代码复用映射（`one` 库现有实现，优先复用）

| 需求 | `one` 库实现 |
|---|---|
| 批量 FK + 6×7 Jacobian | `one/robots/manipulators/franka/fr3_pen/batched_fr3_kin.py:BatchedFR3Kinematics` — `fk_jac(q)` 接 (B,7) 返 `(p_tcp, R_last, J:(B,6,7), T_last)`；`tcp_fk_jac(q)` 已含 TCP offset |
| 批量阻尼伪逆 | `Yuan/fr3_dit/data_generation/generate_fr3_plane_dataset.py:damped_pseudoinverse_batch` — `J^T(JJ^T+λ²I)^{-1}`；Nakamura-Hanafusa 的 λ 自适应在外层包一层 |
| 关节限位常数 | `BatchedFR3Kinematics.__init__` 的 `lmt_lo / lmt_up / jnt_ranges` |
| 自碰撞 | `one/robots/manipulators/franka/fr3/sphere_collision.py:FR3SphereCollision` — `is_collided(link_tfs)` 批量 |
| 数值 IK（scalar） | `SELIKSolver`（位置/姿态选择性 IK）+ `one/robots/manipulators/manipulator_base.py:ManipulatorBase.ik_tcp_nearest` — LM-style 单解 |
| 批量 IK + null-space swivel（参考） | `Yuan/flow_connectivity/batched_rollout.py:_batched_ik_project / _branch_seed_bank` |

注意：`one` 库无 pinocchio 依赖；IK 用 `SELIKSolver`（position-only 模式）。

---

## 数学规约（Position-only NSRL）

### 控制律
$$
\dot{q} = J_p^+ \dot{x}_{task} + B(q)\, a
$$

- 任务 3-DOF：$J_p(q) \in \mathbb{R}^{3\times 7}$ = 完整几何雅可比 $J_g \in \mathbb{R}^{6\times 7}$ 的前 3 行（线速度部分）。姿态不进任务侧，全部由 nullspace action + reward + 终止条件调节。
- 阻尼最小二乘：$J_p^+ = J_p^T(J_p J_p^T + \lambda^2 I)^{-1}$
  - λ 自适应（Nakamura-Hanafusa）：$\sigma_{\min}(J_p) > \sigma_{thr}$ 时 $\lambda = \lambda_0$，否则 $\lambda = \lambda_0 \sqrt{1 - (\sigma_{\min}/\sigma_{thr})^2}$
  - 默认 $\lambda_0 = 0.05$，$\sigma_{thr} = 0.05$
- 零空间 4-DOF：$B(q) \in \mathbb{R}^{7\times 4}$ = SVD $J_p = U \Sigma V^T$ 取 $V$ 的最后 4 列；Procrustes 对齐保证时间连续性（见 §关键陷阱 1）
- 动作：$a \in \mathbb{R}^4$，策略输出在 $[-1, 1]^4$，缩放到 $[-a_{\max}, a_{\max}]$

### 任务速度
$$
\dot{x}_{task} = v \hat{u} \in \mathbb{R}^3
$$
- $\hat u$：任务方向（无限射线）；$v$：恒定线速度（超参，默认 $v = 0.05$ m/s）
- 无角速度任务项——姿态完全交给 reward + nullspace action + 30° 锥终止

### 离散时间
- $dt = 0.01$ s（即每步 EE 期望推进 $v \cdot dt = 0.5$ mm）
- $q_{t+1} = q_t + \dot q_t \cdot dt$（前向欧拉）

---

## 奖励（每步）

设计原则：position-only 任务下 EE 位置由 $J_p^+ \dot x_{task}$ 闭式决定，与 $a$ 无关。RL 唯一的优化目标是最大化 episode 时长，单项 alive reward 即可表达。

| 项 | 公式 |
|---|---|
| 存活 | $r_{alive} = w_{alive}$（per-step 常量，默认 1.0） |

终止条件见 §终止条件，惩罚一律 0（V function 与 episode 长度对齐，无需区分死法）。

权重写进 `config.yaml`。README 给出 $r_{alive}$ 在典型 rollout 中累积量级的估计。

---

## 终止条件

| 条件 | reward | terminated/truncated |
|---|---|---|
| 自碰撞 | 0 | terminated=True |
| 旋转锥越界 $\angle(z_t, n_{target}) > 30°$ | 0 | terminated=True |
| 任一关节极限触发 | 0 | terminated=True |
| `step_count ≥ max_steps` | 0 | truncated=True |

- $n_{target}$ 是 line spec 提供的目标法向（episode 内恒定）。agent 需用 4-DOF nullspace 让 $z_t$ 不超出 30° 锥
- **无 success 终止**——任务定义是"沿 $\hat u$ 方向活到底"，没有"走完"的概念
- `max_steps` 默认 **10000**（= 100 秒物理时间）。设置目标是让绝大多数 episode 因 terminated（JL / 锥 / 碰撞）自然结束，truncated 是少见兜底情况。若实际训练中 truncated 频率 > 5%，提示 agent 学会了某种"永久存活策略"，需要调大 max_steps 或重新审视终止条件

**Gymnasium 语义**：`terminated` / `truncated` 严格区分。**PPO 下 truncated 必须 bootstrap**（$V(s_T)$ 加入 advantage 计算）；terminated 不 bootstrap。

---

## 关键实现陷阱

### 1. 零空间基 $B(q)$ 的时间连续性
SVD 右奇异向量有符号和顺序歧义 → naive 实现导致 $B(q_t) \leftrightarrow B(q_{t-1})$ 跳变 → $a$ 语义不连续 → 学不出来。

实现：
- $J_p = U \Sigma V^T$，取 $V$ 最后 4 列得 $B_{raw}(q) \in \mathbb{R}^{7\times 4}$
- Procrustes 对齐：$M = B_{prev}^T B_{raw} \in \mathbb{R}^{4\times 4}$ 的 SVD $= U_M \Sigma_M V_M^T$，对齐矩阵 $R = U_M V_M^T$，输出 $B = B_{raw} R^T$
- Episode 第一步用 deterministic 初始化（如固定 sign 约定：每列首个非零分量为正）

单独函数 `align_nullspace_basis()` + 单元测试：构造平滑 q(t) 轨迹，验证相邻 $B$ 的 Frobenius 距离 < 阈值，且 $J_p B \approx 0$。

### 2. q₀ 初始化（IK，position-only）
给定起点位置 $p_0$（姿态自由），调用 `SELIKSolver`（position-only 模式）解可行 $q_0$：
- IK 只约束 position（3-DOF），姿态由 seed + 解过程自由选
- 解后必须 post-check 才接受：
  1. $q_0$ 在 JL 内
  2. $\sigma_{\min}(J_p(q_0)) > \sigma_{thr}$
  3. 不自碰撞
  4. 初始姿态在 30° 锥内：$\angle(z_t(q_0), n_{target}) < 30°$
- 重试上限 **10 次**（IK seed 在 JL 内均匀采样）；全部失败则跳过该 line（采样新 line 重试）
- 不考虑 SMM 分支选择，IK 给哪个用哪个

### 3. 观测空间（20 维）
```
obs = concat(
    q_normalized,               # 7,  (q - q_mid) / q_half_range ∈ [-1,1]^7
    u_hat,                      # 3,  任务方向单位向量,episode 内恒定
    z_tool,                     # 3,  当前工具轴单位向量
    n_target,                   # 3,  目标法向（旋转锥参考）,episode 内恒定
    a_prev,                     # 4,  上一步策略输出 ∈ [-1,1]^4
)  # total = 20
```

设计说明：
- **不含 line 长度 / progress / remaining_distance 任何信息**：任务定义里 line 是无限射线，没有"走完"的概念。把"还剩多远"塞进 obs 会让 policy 学到与训练 setup 耦合的伪策略（"接近终点时改变行为"），损害泛化
- 移除 `sin(q), cos(q)`：FR3 所有 JL 远离 ±π，无需周期性编码；`q_normalized` 已覆盖 JL 距离信息
- 移除 `sigma_min_normalized`：原是 soft 信号，reward 中已不使用，policy 必要时可从 $q$ 隐式建模
- 移除独立 `jl_distances`：等价于 `|q_normalized|`，冗余
- 保留 `a_prev`：对 MLP policy 学短时一致性有微弱帮助，几乎无害

### 4. 动作尺度
$a \in [-1, 1]^4$ 是 policy 输出，乘 $a_{\max}$（rad/s）得 $\dot q_{null}$ 幅值。默认 $a_{\max} = 0.5$ rad/s；扫参建议范围 `a_max ≈ 0.5–1.0 · v / L_{臂}` 量级。

---

## Baseline Nullspace Controller（eval ratio 分母）

`env/baseline_controller.py` 实现一个无 RL 的手工 nullspace 控制器，与 RL 控制器在 200 条 holdout line 上做 1:1 对比。

控制律：
$$
\dot{q} = J_p^+ v\hat{u} + B(q) \cdot k_{JL} \cdot B(q)^T \nabla_q H(q)
$$

其中：
- $H(q) = \frac{1}{2} \sum_{i=1}^{7} \left( \frac{q_i - q_{i, mid}}{q_{i, range}/2} \right)^2$：远离 JL 中心的二次势函数
- $\nabla_q H(q) = \left( \frac{q_i - q_{i, mid}}{(q_{i, range}/2)^2} \right)_i \in \mathbb{R}^7$
- $B(q)^T \nabla_q H$：把 GPM 梯度投影到 nullspace 4D 坐标
- $k_{JL}$：手工增益（默认 1.0，可调）
- 终止条件与 RL 环境完全一致（自碰撞 / 30° 锥 / JL / max_steps）

baseline 不优化姿态——只靠"远离 JL 中心"作为副产物维持姿态。这是有意的弱 baseline；RL 应该明显超过。

---

## RL 训练设置

### 算法 & PPO 包
- 算法：PPO（基于 `cleanrl/ppo_continuous_action.py`，单文件改造）
- $\gamma = 0.99$，$\lambda_{GAE} = 0.95$，clip_range = 0.2
- on-policy rollout：`n_envs = 128`，`n_steps = 32`（共 4096 transitions/update）
- mini-batch：`n_minibatches = 32` → `batch_size = 128`，`n_epochs = 10`
- 学习率：actor + critic 共用 3e-4（可加 linear schedule）
- `ent_coef = 0.0`；`vf_coef = 0.5`；`max_grad_norm = 0.5`
- 网络：三层 MLP `[256, 256, 256]`，ReLU；actor 输出 $\mu$，$\log\sigma$ 作 state-independent 可学参数（cleanrl 默认）
- device：`cuda`（torch-batched env 要求）

### 并行环境形态
- torch-batched 单进程：B = 128 个 env 状态张量在一个 step 内推进，最大化 `BatchedFR3Kinematics` 的 GPU 利用
- 不用 sb3 `SubprocVecEnv`
- cleanrl PPO 需自写薄 wrapper 把 batched torch env 接入（不是 sb3 `VecEnv` API）；wrapper 要正确处理 per-env reset 和 terminated/truncated mask

### 训练步数
- 共 $10^6$ env step（$\approx 250$ updates @ 4096 transitions/update），每 $10^4$ 步评估一次

### 训练 line 采样
"Line" 在本规约中表示无限方向射线 $(p_0, \hat u, n_{target})$，无长度。
- 起点 $p_0$：Monte-Carlo reachability 采样——预先 sample 一大组 q ∈ JL（建议 $10^5$ 个），FK 得 reachable point cloud；运行时从中 rejection sample
- 方向 $\hat u$：单位球均匀
- 目标法向 $n_{target}$：与 $\hat u$ 垂直 + 噪声（line spec 的一部分，episode 内恒定）
- 单独写 `LineDistribution` 类（命名沿用历史，语义为"任务方向 + 起点采样器"），方便替换分布

---

## 评估
- 固定 holdout 集：200 条 lines，seed 固定，可重现
- 不报 success_rate / 绝对 mean lifetime——只看 ratio
- 与 baseline 对比：手工 GPM nullspace controller（见 §Baseline）在同一 200 条 lines 上的 episode 时长作分母
- `T_rl / T_baseline` 中分子分母都是 **episode 实际步数 × dt**（按物理时间计），不区分终止原因（terminated 各类 / truncated 都算"被结束"）
- `eval.py` 输出 CSV：`line_id, T_rl, T_baseline, ratio, term_reason_rl, term_reason_baseline, mean_sigma_min_rl`
- stdout 汇总：mean ratio、median ratio、按 term_reason 分组频次
- 若任一方法 truncated 频率 > 5%（撞 max_steps 上限），需在 README 中标注并考虑提高 max_steps

---

## 不要做的事
- 不用 `gym`（旧 API），用 `gymnasium`
- 不在 reward 加未要求的"巧妙"项（内在好奇心、势函数等），先把上述跑通
- 不硬编码路径
- 不伪造未核实的库 API；不确定的写 TODO 并说明
- 不写"教学性"注释解释 PPO / 雅可比——读者是专家
- 不报 success_rate / 绝对时长均值，只报 ratio
- **不让任何"line 长度 / 进度 / 剩余距离"信息进入 agent obs 或 reward**——任务是无限射线

---

## README 末尾 "Known Issues / 待验证"（实现者负责填）
- IK reset 性能：scalar `SELIKSolver` 在并行 env reset 时是否成为瓶颈？128 envs 同时 reset 会 sequential 调用 128 次 IK。若 wall-clock 占比过高，需要 (a) 批量化 IK，或 (b) 预先离线生成 $10^4$ 个有效 $q_0$ 池，运行时直接采样
- Procrustes 对齐在 batched 维度上的实现（是否 vmap 友好）
- truncated 处理在 cleanrl PPO 的具体改动点
- truncated 频率：训练和 eval 中分别多少？若 > 5%，max_steps 不够大

## README 末尾 "Open Questions"（实现者给初步判断 + 不确定度，不是问用户）
1. $w_{alive}$ 量级如何选？默认 1.0 配合终止惩罚 −50 / −100 是否合理？典型 episode 累积 reward 与终止惩罚的比例如何？
2. MC reachability 点云规模 $10^5$ 够用？起点采样的覆盖均匀度如何？
3. 预期 episode 时长 ratio (RL / baseline) 能达到多少？
4. `n_envs = 128, n_steps = 32` 是否够覆盖 $10^6$ env step 训练？on-policy sample efficiency 是否瓶颈？
5. 4-DOF nullspace action 相对 2-DOF（旧 5-DOF 任务时）增加了 agent 的自由度，learning curve 是否会更慢？
6. baseline 用纯 GPM-JL 是否过弱（30° 锥几乎必失败）？是否应该再给一个"GPM-JL + 姿态对齐"的强 baseline 作上界？
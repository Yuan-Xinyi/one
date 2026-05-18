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
- $\hat u$：任务方向（无限射线）；$v$：恒定线速度（超参，默认 $v = 0.2$ m/s）
- 无角速度任务项——姿态完全交给 reward + nullspace action + 30° 锥终止

### 离散时间
- $dt = 0.05$ s（20 Hz 控制；每步 EE 期望推进 $v \cdot dt = 10$ mm = 1 cm）
- $q_{t+1} = q_t + \dot q_t \cdot dt$（前向欧拉）

---

## 奖励（每步）

设计原则：alive reward 提供"活久就好"的总目标；其他三项是 **telescoping delta**——只有 metric 改善才给正 reward，维持不变给 0，恶化给负。这样 reward 只奖励"行为有效性"，不奖励"已经处在好状态"，避免 always-on 形式下的"待在 saturated 区域刷分"问题。

| 项 | 公式 | 默认权重 |
|---|---|---|
| 存活 | $r_{alive} = w_{alive}$ | 0.25 |
| JL Δ | $r_{jl} = w_{jl} \cdot K \cdot (\overline{q_{norm}^2}\big|_{prev} - \overline{q_{norm}^2}\big|_{now})$ | 0.25 |
| Cone Δ | $r_{cone} = w_{cone} \cdot K \cdot (\cos\angle(z_t, n_{target})\big|_{now} - \cdot\big|_{prev})$ | 0.25 |
| Dirmanip Δ | $r_{dm} = w_{dm} \cdot K \cdot (w_{\hat u}(q)\big|_{now} - \cdot\big|_{prev})$，$w_{\hat u}(q) = 1/\sqrt{\hat u^T (J_p J_p^T + \lambda^2 I)^{-1} \hat u}$ | 0.25 |

**权重归一化**：runtime 自动除以 $\sum w_i$ 让 $w_{alive} + w_{jl} + w_{cone} + w_{dm} = 1$，每项是"对总信号的贡献比例"。

**delta_scale $K = 100$**：把每步 delta（典型 ~0.01）放大到 ~1 magnitude，让每项 contribution per step 与 $w_i$ 同阶。

**Reset 处理**：每个 prev 缓存（`q_norm_sq_prev`, `cos_angle_prev`, `w_u_prev`）episode 起始置 NaN 哨兵，第一步 delta 强制为 0，避免 reset 跳变污染信号。

**Episode-sum telescoping**：每项 episode 总和 = $w_i K (\text{final} - \text{initial})$。agent 无法靠"维持高 metric"刷分，只有沿 trajectory 累计改善才贡献。

终止条件见 §终止条件，惩罚一律 0（V function 与 episode 长度对齐，无需区分死法）。

**alive-only ablation**：把三个 $w$ 都置 0 即可退回 pure-alive reward（runs1–5 设定）。runs1–5 证明 alive-only PPO 无法显著超过初始随机 policy，故引入 dense penalty。

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
- `max_steps` 默认 **500**（= 25 秒物理时间 @ dt=0.05）。几何天花板约 100 步（1 m EE 推进）。设置目标是让绝大多数 episode 因 terminated（JL / 锥 / 碰撞）自然结束，truncated 是少见兜底情况。若 truncated 频率 > 5%，需调大 max_steps

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

### 3. 观测空间（31 维）
```
obs = concat(
    q_normalized,               # 7,  (q - q_mid) / q_half_range ∈ [-1,1]^7
    q_normalized**2,            # 7,  element-wise; 直接暴露 |q_i_norm| 接近 1 的二次信号
    u_hat,                      # 3,  任务方向单位向量,episode 内恒定
    z_tool,                     # 3,  当前工具轴单位向量
    n_target,                   # 3,  目标法向（旋转锥参考）,episode 内恒定
    cos_angle,                  # 1,  z_tool · n_target; cone-relevant 标量
    z_cross_n,                  # 3,  z_tool × n_target; 对齐误差的旋转轴方向
    a_prev,                     # 4,  上一步策略输出 ∈ [-1,1]^4
)  # total = 31
```

设计说明：
- **不含 line 长度 / progress / remaining_distance 任何信息**：任务定义里 line 是无限射线，没有"走完"的概念。把"还剩多远"塞进 obs 会让 policy 学到与训练 setup 耦合的伪策略（"接近终点时改变行为"），损害泛化
- **显式 cone-relevant 特征**（`cos_angle`, `z_cross_n`）：ReLU MLP 难以从原始 `(z_tool, n_target)` 学到 dot/cross 这种乘性交互；显式给出避免 actor 浪费 capacity 重学
- **`q_normalized**2`**：JL soft penalty 是 `max(0, |q_i_norm| - 0.8)²`，actor 通过 `q_norm` + 隐式 abs() 学起来低效；平方项直接给出二次接近信号
- 移除 `sin(q), cos(q)`：FR3 所有 JL 远离 ±π，无需周期性编码
- 移除 `sigma_min_normalized`：原是 soft 信号，reward 中已不使用
- 保留 `a_prev`：对 MLP policy 学短时一致性有微弱帮助

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
- 网络：三层 MLP `[512, 512, 512]`，ReLU；actor 共享 trunk，分两个 head 输出 $\mu(s)$ 和 $\log\sigma(s)$（state-dependent log_std；clamp 到 [-5, 2]）。critic 独立同结构
- **Tanh-squashed action**: $a = \tanh(z)$，$z \sim \mathcal N(\mu(s), \sigma(s))$；log_prob 含 Jacobian 修正 $\log\pi(a) = \log \mathcal N(z) - \sum_i \log(1 - \tanh^2(z_i))$。PPO buffer 存 $z$（unsquashed），env 见 $a$。**不加 squash → actor μ 会被 clip 后的有偏梯度推到无界**（runs7/10 观察到 μ 涨到 3.5+，deterministic action 永远饱和 ±1，behavior ≈ random Gaussian baseline）
- state-dependent log_std 是 "σ 爆炸覆盖 state-dependent best mean" 这个失败模式的修复手段
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
- 方向 $\hat u$：单位球均匀（采样时 $\perp n_{target}$）
- 目标法向 $n_{target}$：$z_{tool}(q_0)$ + 小角度噪声（line spec 的一部分，episode 内恒定）
- `LineDistribution` 在 init 时把每条 spec 全部预生成（q0, line_dir, n_target 各自 deterministic per-index），`sample(n)` 只是从池里 index——这让"按 spec 过滤"成为可能
- **Feasibility 过滤**（`feasibility_filter: true`）：init 时对池里每条 line 跑一次 `ClassicalNullspaceController`，丢弃寿命 < `feasibility_threshold_m` (默认 10 cm) 对应步数的 line。意图：classical 都活不过 10 cm 的 line **几何上注定失败**（line 方向把 EE 推出可达空间或直奔 JL），RL 在这种 line 上的梯度是纯噪声，**训练带毒**。过滤后只在"可学的" line 上训练 / 评估

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
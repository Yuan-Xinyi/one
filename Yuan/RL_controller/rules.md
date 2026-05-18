# FR3 NSRL 训练规约

## 任务
"沿任务方向 $\hat u$ 尽可能久不被终止"。line = 无限方向射线 $(p_0, \hat u, n_{target})$，无长度。agent 唯一目标是最大化 episode 时长。

## 控制律
$$\dot q = J_p^+ v\hat u + B(q)\,a$$
- $J_p \in \mathbb R^{3\times 7}$ 位置雅可比；$J_p^+ = J_p^T(J_p J_p^T + \lambda^2 I)^{-1}$，λ Nakamura-Hanafusa 自适应
- $B(q) = V[:,-4:]$ + Procrustes 对齐 `B_prev`（首步 sign-convention seed）
- $a \in [-1,1]^4$ tanh-squashed，缩放 $a_{max}=2.5$；$q_{t+1} = q_t + \dot q\,dt$（$dt = 0.05$ s，$v = 0.2$ m/s）

## 观测（31 维）
`q_norm(7) + q_norm²(7) + u_hat(3) + z_tool(3) + n_target(3) + cos(z,n)(1) + (z×n)(3) + a_prev(4)`

显式 cone-relevant 特征避免 ReLU MLP 重学 dot/cross。**不含 line 长度 / progress / 剩余距离**。

## 奖励（每步，权重 runtime 归一化使 $\sum w = 1$）

| 项 | 公式 | 默认 $w$ |
|---|---|---|
| Progress | $w_{prog}\cdot\mathrm{clip}(\Delta p\cdot\hat u/(v\,dt), 0, 1)$ | 0.6 |
| JL Δ | $w_{jl}\cdot K\cdot(\overline{q_{norm}^2}\big|_{t-N} - \cdot\big|_{now})_+$ | 0.2 |
| Cone Δ | $w_{cone}\cdot K\cdot(\cos\angle(z,n)\big|_{now} - \cdot\big|_{t-N})_+$ | 0.1 |
| Dirmanip Δ | $w_{dm}\cdot K\cdot(w_{\hat u}\big|_{now} - \cdot\big|_{t-N})_+$ | 0.1 |

$w_{\hat u}(q) = 1/\sqrt{\hat u^T (J_p J_p^T + \lambda^2 I)^{-1}\hat u}$；$K = 100$，lookback $N = 10$ 步；ring buffer NaN 哨兵让前 $N$ 步 delta = 0。episode 结束补 telescoping bonus $K\sum w_i(\text{final} - \text{init})$ 捕捉整体改善。终止惩罚一律 0（V 与 episode 长度对齐）。

## 终止
- terminated：自碰撞 / 30° 锥越界 / 任一 JL 触发
- truncated：`step ≥ max_steps`（500 = 25 s）
- **PPO 下 truncated 必须 bootstrap $V(s_T)$；terminated 不 bootstrap**

## 训练
PPO（cleanrl 改），$\gamma = 0.99$，$\lambda_{GAE} = 0.95$，clip 0.2；$n_{envs} = 128$，$n_{steps} = 32$（4096 trans/update），10 epochs × 32 mini-batches；actor/critic 各 `[512,512,512]` ReLU，actor 双 head 输出 $\mu, \log\sigma$（state-dep，clamp $[-5,0]$）；running-std return scaling 让 V 学 z-score 量级；line pool $10^5$ 预生成 + 可选 feasibility filter（丢弃 classical 跑不过 10 cm 的 line）。

## 评估
固定 200-line holdout（seed 固定），`auto_reset=False`，输出 $T_{rl}/T_{baseline}$ ratio CSV。**只报 ratio**，不报 success_rate 或绝对均值（任务本身存在 intrinsically-infeasible line）。

## 不要做
- 不把 line 长度 / progress / 剩余距离塞进 obs 或 reward
- 不加未要求的"巧妙"奖励项；不硬编码路径；不伪造未核实 API
- 不写教学性注释；只报 ratio

# 面向长时域操作的种子选择—连续控制统一强化学习框架

## 单种子、零额外模型规划预算下的设计、优化与封存评测

**版本：** 2026 年 7 月 22 日  
**项目：** Yuan RL Controller  
**评测口径：** 一次静态 selector 前向、选择一个种子、执行一次下游 controller

## 摘要

本工作研究一个长时域机器人操作中的关键耦合：初始关节构型（seed）不仅决定逆运动学是否可行，还会显著影响后续连续控制能够稳定推进多远。原系统分别训练 seed selector 和 controller，导致 selector 看不到真实下游回报，controller 也无法适应 selector 实际诱导的初始状态分布。

受 *Sequential Dexterity: Chaining Dexterous Policies for Long-Horizon Manipulation* 的 transition feasibility 与前后向策略适配思想启发，本工作把 seed selection 视为 episode 的第一个离散宏动作，把后续 controller 视为连续低层策略，在同一个 SMDP 目标下进行交替前向和后向优化。与原论文的多技能链不同，本项目只有“选择初始构型—连续控制”两个阶段；借鉴的是跨阶段信用分配思想，而不是直接复现其系统。

最终部署严格遵守以下约束：每个任务只运行一次静态 selector，输出一个 seed，随后只运行一次 controller；不执行 Top-K probe、world-model rollout、MPC 或多 seed 真实试错。训练阶段可以使用完整候选回报和局部动作搜索生成监督，但这些计算不进入部署路径。

在模型和阈值全部冻结后生成的 10,000-task sealed final holdout 上，最终联合模型达到 **0.547301 m** 平均进度，相对 original decoupled 基线的 **0.545694 m** 提升 **1.606 mm**，配对 geometry-bootstrap 95% CI 为 **[+0.418,+2.817] mm**。相对 first-valid，最终系统提升 **53.392 mm**，捕获有限候选 oracle headroom 的 **60.388%**；原基线为 58.501%，即 capture 增加 **1.887 个百分点**。

严格 2×2 分解显示，selector 在冻结旧 controller 下贡献 **+1.357 mm**，controller 在旧 selector 下只贡献 **+0.178 mm**，交互项为 **+0.072 mm**。后两项置信区间均跨零。因此，sealed 结果支持“统一训练结果优于原始解耦系统”，但改进主要来自 seed selection，而不是已经证明 controller 更新本身显著有效。

为避免不公平比较，本报告还补充了同等 actor–Q 能力的 controller-frozen 基线。该匹配基线达到 **0.547316 m**；联合模型相对它为 **−0.015 mm**，95% CI **[−0.740,+0.716] mm**，二者统计上持平。因而，当前证据**不支持**“联合训练优于最强匹配 frozen-controller 方法”的更强结论。论文可以可靠主张统一框架、同预算提升以及跨阶段联合目标的有效性，但不能把 frozen 基线描述成已经被联合训练显著超越。

## 1. 研究动机

### 1.1 原解耦系统的问题

对任务上下文 (c)，候选生成器给出至多 (K) 个可行关节构型：

\[
Q(c)=\{q_1,\ldots,q_K\}.
\]

原流程先由独立 selector 选择 $q_0$，再由连续 controller 执行动作：

\[
i\sim\pi_{\mathrm{seed}}(i\mid c,Q),\qquad
q_0=Q_i(c),\qquad
a_t\sim\pi_{\mathrm{ctrl}}(a_t\mid o_t).
\]

如果 selector 只学习 IK 残差、关节限位距离或 manipulability，它优化的是静态几何代理，而不是最终控制结果。由此产生三个问题：

1. 几何上同样可行的 seed，可能具有完全不同的闭环可持续运动距离；
2. controller 的训练 reset 分布与部署时 selector 诱导的分布不一致；
3. 任一模块更新后，另一模块先前学习的最优性都可能失效。

### 1.2 与 Sequential Dexterity 的关系

Chen 等人在 CoRL 2023 的 *Sequential Dexterity* 中使用 transition feasibility function 逐步微调相邻子策略，以提高长技能链的衔接成功率。本文采用相同的核心 insight：前一阶段的输出必须由后一阶段的成功性来评价，并把后续回报反向传递到前一阶段；同时，前一阶段产生的新状态分布应继续用于训练后一阶段。

本项目的具体映射是：

- 前一阶段：从可行候选集合中选择初始关节角；
- 后一阶段：连续 controller 沿目标射线执行；
- transition feasibility：候选 seed 在当前 controller 下的完整轨迹进度；
- backward update：用完整 controller return 更新 seed actor 与 feasibility head；
- forward update：在 selector 相关状态上更新 controller，再重新标注 seed。

参考文献：Y. Chen, C. Wang, L. Fei-Fei, and C. K. Liu, [Sequential Dexterity: Chaining Dexterous Policies for Long-Horizon Manipulation](https://proceedings.mlr.press/v229/chen23e.html), CoRL 2023, PMLR 229:3809–3829。

## 2. 统一问题建模

### 2.1 两阶段 SMDP

我们将 seed selection 建模为 episode 的第一个离散宏动作，将 controller 建模为后续连续动作。完整轨迹的部署目标为最终净射线进度：

\[
G(c,q_i;\theta)=\operatorname{progress}
\left(\tau\left(c,q_i,\pi_{\mathrm{ctrl},\theta}\right)\right).
\]

统一优化目标为

\[
\max_{\phi,\theta}
\;\mathbb{E}_{c,Q}
\left[
G\left(c,Q_{i};\theta\right),
\quad i\sim\pi_{\mathrm{seed},\phi}(\cdot\mid c,Q)
\right].
\]

其中 $\phi$ 是 selector 参数，$\theta$ 是 controller 参数。两者共享最终进度目标，但为了控制 non-stationarity，实际训练采用 immutable snapshot 上的粗粒度交替更新，而不是每个 minibatch 同时更新两方。

### 2.2 双向优化循环

一次完整循环包含：

1. **冻结 controller，后向标注。** 对训练任务中的所有物理有效候选运行完整 controller rollout，得到候选级进度表；
2. **更新 selector。** 用候选级回报训练 actor、value 与 feasibility head，使 seed 决策直接对下游控制结果负责；
3. **冻结 selector，前向更新 controller。** 在 selector 相关状态和局部 tail state 上生成受支持的改善动作标签，并用 retention 约束更新 controller；
4. **冻结新 controller，重新标注。** 重新计算 seed 候选回报，更新 selector 输出头；
5. **独立开发集 promotion。** 只有通过固定门槛的 snapshot 才能进入下一阶段；sealed final set 在所有决策冻结后才生成和读取。

这种实现是一个统一训练框架，但不是无约束的 simultaneous end-to-end gradient。交替冻结使每次信用分配都有明确的行为策略和回报定义，也便于做严格的 2×2 因果式消融。

## 3. 最终方法

### 3.1 约束感知候选层

本阶段保留 diffusion proposal 与 Newton IK 作为候选生成器。每个任务包含 8 个 proposal 加 1 个 fallback，共 9 个候选槽。RL 只在物理有效候选中选择，部署前再次检查：

- 数值有限性与关节限位；
- 初始末端位置误差；
- 工具方向锥约束；
- 自碰撞和环境碰撞；
- 每行至少存在一个有效候选。

无效候选在网络 logits 和最终 argmax 中均被硬 mask，不能通过 fallback 逻辑被意外选中。

### 3.2 五成员 seed actor–critic

每个 ensemble member 使用 permutation-equivariant mean-set encoder。输入包含 45 维 controller-aligned 特征，包括任务几何、候选关节角、初始 controller 观测、IK 残差、log positional manipulability，以及沿目标方向的阻尼关节速度、速度范数、到关节限位的可运动距离和 directional manipulability。

每个成员输出：

- 候选 actor log probability；
- 集合级 value；
- 候选 feasibility 值 (F(c,q_i))，目标为相对 first-valid 的完整进度增量。

网络宽度为 512，ensemble size 为 5。成员通过 geometry bootstrap 训练，部署时聚合成员平均 log probability 和平均 feasibility。

### 3.3 后向候选监督

候选回报不是稀疏 sampled action 标签，而是冻结 controller 下的完整候选 Monte-Carlo 表。actor 使用任务内 range-normalized listwise target；feasibility head 使用以 metre 表示的候选相对进度。该设计让一个任务的多个可行动作同时提供相对排序信息，并减少任务难度尺度对损失的支配。

训练集按 task geometry 分组后拆分为：

| 子集 | 行数 | 唯一几何数 | 用途 |
|---|---:|---:|---|
| Fit | 12,902 | 11,557 | 拟合网络参数 |
| Model selection | 2,765 | 2,499 | 选择固定超参数和 gate |
| Calibration | 2,765 | 2,511 | 固定规则的一次性复核 |

### 3.4 单次静态 actor–Q 决策

最终 selector 不执行任何 controller probe。对物理有效候选 (i)，定义

\[
s_i=\overline{\log\pi_i}
+w\frac{\overline{F_i}}{\sigma_F},
\qquad w=0.20,\quad \sigma_F=0.01\ \mathrm{m}.
\]

actor–Q proposal 为

\[
i^*=\arg\max_{i\in\mathcal V}s_i.
\]

为降低错误改写 first-valid 的风险，再使用 feasibility margin gate：

\[
\hat i=
\begin{cases}
i^*, & \overline F_{i^*}-\overline F_{i_{first}}
\ge 4.86150384\times10^{-6},\\
i_{first}, & \text{otherwise}.
\end{cases}
\]

$w$ 从预先固定的

\[
\{0,.05,.10,.15,.20,.25,.30,.40,.50,.75,1\}
\]

中只在 model-selection split 上选择；完整规则冻结后，calibration split 只允许通过或整体回滚，不再调权重。最终 calibration 上相对原 S0/C1 selector 的平均提升为 +1.200 mm，$>1$ mm harm rate 为 5.62%。

部署调用链只有：

\[
(c,Q)\xrightarrow{\text{一次 ensemble forward}}\hat i
\xrightarrow{\text{一次 controller rollout}}G.
\]

### 3.5 Controller 前向更新

controller 使用现有 deterministic-mean tanh-Gaussian actor。训练阶段在 fit tasks 的轨迹 tail state 上，对当前动作附近的 16 个局部扰动做完整续跑，保留相对当前 controller 至少提升 1 mm 的动作，再以 retention loss 约束更新幅度。这里的局部搜索只用于产生训练监督，部署时搜索步数和模型 rollout 数均为 0。

最终 C1 是预先物化的 epoch-2 controller candidate。它在 selector 固定时没有达到独立显著 promotion 门槛，因此本报告不把 C1 描述为单独优于 C0；它的价值必须与后续重新标注并训练的 S1 一起评价。

随后在 C1 上重跑完整 seed-return cache，并对五成员 selector 的输出 heads 进行 20 epochs 小学习率更新，encoder 保持稳定。最终 actor–Q 融合使用更新后的 actor 和 feasibility，形成 S1Q/C1 联合部署。

## 4. 实验协议

### 4.1 比较对象

定义两个 selector 和两个 controller：

- **S0：** 原 45-D 五成员 selector，使用 actor proposal；
- **S1Q：** 在 C1 回报上后向更新，并使用静态 actor–Q proposal；
- **C0：** 原 controller；
- **C1：** 前向局部动作蒸馏得到的 controller candidate。

四个单元为 S0C0、S0C1、S1QC0、S1QC1。所有单元使用同一任务顺序、同一候选集合、同一物理 valid mask。S0 在 C0/C1 下选择相同候选，S1Q 亦然，因此 selector 决策没有偷看 controller rollout。

### 4.2 开发集与 sealed final set

Validation 和 external-dev 各含 2,048 tasks，只用于模型工程和冻结决策，不作为最终泛化结论。

所有超参数冻结后，使用独立随机种子生成 sealed final holdout v2：

- 10,000 tasks，10,000 个唯一 task geometries；
- 每个任务 8 个 diffusion proposals 加 1 个 fallback；
- 内部重复数为 0；
- 与 rank-train A/B/C、external-dev、systematic-10k、历史 fresh sets 和 sealed-v1 等 9 个排除 cache 的精确 float32 geometry overlap 均为 0；
- candidate cache SHA-256 为 `baae4cc4...507c5`；
- pool/task/diffusion seeds 分别为 2026072211、2026072212、2026072213。

sealed set 生成后，主模型没有再发生变化。后续匹配 frozen 消融复用已冻结的开发规则，并只在 sealed 完整候选表上做一次 lookup；该消融不参与主模型选择。

### 4.3 指标

主指标为最终净射线进度 $P$。first-valid 是物理 valid mask 中的第一个候选。complete-candidate oracle 离线执行所有有效候选并选进度最大者，只用于估计当前有限候选集合的上界，不是部署方法。

定义 oracle headroom capture：

\[
\mathrm{Capture}=
\frac{P_{method}-P_{first}}
{P_{oracle}-P_{first}}.
\]

另外报告：

- oracle hit：静态 selector 是否直接选中离线 oracle 候选；
- $>1$ mm harm/win rate：相对配对基线下降或上升超过 1 mm 的任务比例；
- 5% 双侧 trimmed mean；
- 将每任务差值截断到 $\pm50$ mm 后的 clipped mean；
- 20,000 次 paired unique-geometry percentile bootstrap 95% CI。

每个 sealed task 都是唯一几何，因此 row mean 与 geometry-macro mean 完全一致。

## 5. Sealed final 结果

### 5.1 严格 2×2 主结果

| 单元 | Selector | Controller | 平均进度 |
|---|---|---|---:|
| S0C0 | 原 actor | 原 C0 | 0.545694 m |
| S0C1 | 原 actor | 更新 C1 | 0.545872 m |
| S1QC0 | actor–Q | 原 C0 | 0.547051 m |
| S1QC1 | actor–Q | 更新 C1 | **0.547301 m** |

对应的配对效应为：

| 效应 | 平均差值 | 95% CI | 判断 |
|---|---:|---:|---|
| Controller at S0 | +0.178 mm | [−0.159,+0.549] mm | 不显著 |
| Selector at C0 | **+1.357 mm** | **[+0.211,+2.507] mm** | 显著 |
| Interaction | +0.072 mm | [−0.086,+0.246] mm | 不显著 |
| Selector at C1 | **+1.429 mm** | **[+0.292,+2.578] mm** | 显著 |
| Controller at S1Q | +0.250 mm | [−0.092,+0.633] mm | 不显著 |
| Joint: S1QC1−S0C0 | **+1.606 mm** | **[+0.418,+2.817] mm** | 显著 |

严格分解满足

\[
\Delta_{joint}
=\Delta_{controller@S0}
+\Delta_{selector@C0}
+\Delta_{interaction}.
\]

S1Q 与 S0 在 14.78% 的 sealed tasks 上选择不同 seed。主要均值提升来自 selector；controller 边际贡献和 interaction 都很小。

### 5.2 单种子、单 controller rollout 指标

| 方法 | First-valid | Policy | Oracle |
|---|---:|---:|---:|
| Original S0C0 | 0.494048 m | 0.545694 m | 0.582330 m |
| Final S1QC1 | 0.493909 m | **0.547301 m** | 0.582324 m |

| 方法 | 相对 first 提升 | Oracle capture | Oracle hit |
|---|---:|---:|---:|
| Original S0C0 | +51.646 mm | 58.501% | 29.96% |
| Final S1QC1 | **+53.392 mm** | **60.388%** | **30.30%** |

最终模型相对原系统：

- 平均进度增加 1.606 mm；
- oracle capture 增加 1.887 个百分点；
- policy episode length 从 54.878 增至 55.040 steps；
- 相对 first-valid 的 $>1$ mm harm rate 从 15.34% 增至 17.67%，win rate 从 42.00% 增至 44.49%。

最后一项说明均值提高并不等于逐任务单调提高。S1Q 更积极地改变 seed，一部分任务获得较大收益，另一部分任务出现退化。

### 5.3 稳健统计解释

S1QC1−S0C0 的普通配对均值为 +1.606 mm，但 5% 双侧 trimmed mean 只有 +0.031 mm，$\pm50$ mm clipped mean 为 +0.193 mm；$>1$ mm harm 和 win rate 分别为 9.88% 和 10.22%，79.90% 的任务变化不超过 1 mm。

因此主提升具有明显 heavy-tail 特征：大多数任务几乎不变，少数任务的大幅改善拉高平均值。普通均值的 bootstrap CI 已显著大于零，但如果论文强调“典型任务”而不是“期望进度”，就必须同时报告 trimmed/clipped 结果，不能只展示 mean。

## 6. 公平的 frozen-controller 消融

### 6.1 为什么需要重新做 frozen 基线

原 S0 使用 actor-only proposal，而最终 S1Q 使用 actor–Q 融合。如果只比较 S1QC1 与原 S0C0，无法区分收益来自联合训练，还是来自更强的静态部署规则。为此，我们给 frozen C0 的 S0 selector 配置完全相同的 actor–Q 网格、尺度、model split 选择和 calibration 复核。

该 frozen 规则在 model split 上选择 $w=0.10$，gate 为 $1.53668225\times10^{-5}$。它在 model split 和 calibration 上相对原 S0C0 分别提高 +1.547 mm 和 +1.462 mm，$>1$ mm harm rate 分别为 3.21% 和 3.02%。规则在读取 sealed 表之前已经冻结。

### 6.2 Sealed 比较

| 方法 | Controller 是否更新 | 平均进度 |
|---|---|---:|
| Original S0C0 | 否 | 0.545694 m |
| Matched frozen actor–Q/C0 | 否 | **0.547316 m** |
| Joint S1Q/C1 | 是 | 0.547301 m |

Matched frozen 相对原 S0C0 为 +1.621 mm，95% CI `[+0.608,+2.649]` mm。Joint 相对 matched frozen 为 **−0.015 mm**，95% CI `[−0.740,+0.716]` mm；trimmed 和 clipped 差值分别为 −0.001 mm 和 −0.010 mm。

结论非常明确：当前联合模型与强 frozen 基线在统计上持平。联合模型已经达到同等性能，同时提供了统一训练和未来继续前后向迭代的框架价值；但现有实验不能证明 controller 更新带来了超过 frozen 方法的额外性能。

## 7. 时间与推理成本

### 7.1 部署计算图

| 项目 | Final S1QC1 |
|---|---:|
| Selector forward | 1 次 |
| 输出 seed 数 | 1 个 |
| 正式 controller rollout | 1 次 |
| Controller probe | 0 次 |
| World/model rollout | 0 次 |
| MPC/replanning | 0 次 |

actor–Q 只复用一次 ensemble forward 已经产生的 actor 与 feasibility 输出，再做逐候选线性融合和 argmax；它没有引入第二次网络前向。

### 7.2 Sealed 静态评测墙钟时间

在 RTX 4090、10,000-task batched sealed static evaluation 上：

| 方法 | 总墙钟 | 摊销时间/task | 峰值内存 |
|---|---:|---:|---:|
| Original S0C0 | 71.41 s | 7.141 ms | 1.592 GB |
| Final S1QC1 | 71.03 s | 7.103 ms | 1.578 GB |

两者差异属于测量噪声，说明最终提升没有增加 rollout 次数或可测的端到端批量推理成本。该时间包含物理有效性检查、特征计算、一次 selector 选择和一次正式 controller rollout；不包含候选 diffusion 生成。

同架构的早期 profiling 中，2,048-task batched selector-only 成本约 0.050 ms/task；32-task batch=1 工程测量中 selector 平均约 4.33 ms，而一次静态端到端 controller 执行平均约 652 ms。batch=1 数字受 episode length 和小样本影响，只用于量级判断：主要在线成本来自正式 controller，而不是 actor–Q 融合。

### 7.3 不属于部署的离线成本

- sealed 10,000-task candidate cache 生成：16 min 09.89 s，峰值内存约 1.55 GB；
- 为计算 oracle 而完整运行所有有效候选：每个主单元约 5 min 11 s；
- 第二轮 robust controller 搜索：99.85 s，因 validation 退化自动回滚；
- 闭式 ridge-Q 最小实验：约 4.7 s，因 calibration/validation 退化停止。

这些成本只用于训练、诊断或离线评测，不应计入 final policy 的在线 inference latency。

## 8. 继续优化后的消融结果

### 8.1 Actor–Q 是当前有效提升点

固定 C1 时，S1Q 相对 S0 在 validation 上提升约 +1.388 mm，external-dev 上提升 +2.307 mm；后者 95% CI 下界略高于零。sealed 上 selector effect 在 C0/C1 下均显著，说明当前可靠增益集中在单次静态种子选择。

### 8.2 Controller 第二轮更新未通过

在当前 S1Q 诱导分布上进行第二轮 robust local search：356 个动作通过预筛，31 个通过 exact interpolated-label verification。候选 C2 相对 C1：

- validation：−0.082 mm；
- external-dev：+0.140 mm；
- development minimax：−0.082 mm。

由于 validation gate 失败，系统自动回滚到 C1，发布 controller hash 未变化。这避免了为了叙事保留只在一个开发集偶然变好的 controller。

### 8.3 闭式 ridge feasibility head 未通过

我们尝试冻结 encoder/actor，只对 feasibility 的最终表示拟合 ridge head，且仍保持一次静态选择。固定 actor–Q $w=0.2$ 后，model split 选择 clip=0.1 m、$\lambda=10^{-5}$。相对 S0C1：

| 数据集 | Ridge 提升 | 现有 S1Q 提升 |
|---|---:|---:|
| Calibration | +0.535 mm | +1.200 mm |
| Validation | +0.134 mm | 约 +1.38 mm |
| External-dev | +2.486 mm | 约 +2.3 mm |

ridge 只在 external-dev 略好，却在 calibration 和 validation 明显退化，因此没有发布，也没有读取 sealed set。

### 8.4 不采用 model-based probe

早期实验曾证明，多候选 controller prefix 能显著接近有限候选 oracle，但它会增加 controller/model rollout，违反本版部署约束。因此 final checkpoint、正式主表和论文主方法都关闭 probe。它最多可以作为训练 teacher 或 oracle 诊断，不能与本报告的单次静态结果混为同成本方法。

## 9. 对论文结论的建议

### 9.1 当前证据支持的表述

建议使用以下主结论：

> 我们把 seed selection 与连续控制建模为统一的两阶段 SMDP，并通过前向 controller 适配和后向完整轨迹信用传递进行交替训练。在不增加部署 rollout 数的条件下，最终静态联合策略在 10,000 个零重叠 sealed tasks 上相对原始解耦系统显著提高 1.606 mm，并把有限候选 oracle capture 从 58.50% 提高到 60.39%。

还可以主张：

- 部署严格为一个 seed 和一次 controller rollout；
- selector improvement 在 C0 和 C1 下均显著；
- 统一框架能够安全地尝试 controller 更新、重新标注 seed，并在失败时自动回滚；
- 最终联合模型达到强 frozen actor–Q 方法的同等性能。

### 9.2 当前证据不支持的表述

不建议写成：

- “联合训练显著优于 controller frozen”；
- “controller 更新是主要收益来源”；
- “大多数任务都得到稳定改善”；
- “接近 oracle”。

原因分别是：matched frozen 与 joint 持平；controller marginal CI 跨零；trimmed mean 接近零；final capture 仍为 60.39%，约 39.61% 的有限候选 headroom 尚未捕获。

### 9.3 如何真正得到更强的联合训练结论

下一阶段应把资源集中在 controller，而不是继续扩大静态 selector 搜索：

1. **Selector-conditioned controller occupancy。** 直接从当前 selector 的 Top-1 seed 分布采样长轨迹训练 controller，减少当前局部 tail-search 标签与真实部署 occupancy 的错配；
2. **风险敏感的 controller promotion。** 同时优化 mean、5% trimmed mean 和 harm rate，避免只靠少数大收益任务；
3. **Cross-fitted backward labels。** 用不同 geometry fold 的 controller snapshot 生成 seed labels，减少 selector 对单一 controller 噪声的过拟合；
4. **多训练随机种子。** 至少训练 3–5 个 C1/S1 pair，在 sealed 前预注册 snapshot 选择规则，报告跨训练种子的方差；
5. **静态闭环价值蒸馏。** 可在训练时使用完整轨迹 teacher，但 student 仍只做一次前向；目标是提升 median/trimmed gain，而不引入在线 probe；
6. **保持匹配 frozen 对照。** 每次新增 selector head 或部署打分规则，都必须给 frozen controller 同等容量和同等调参预算。

只有当 S1C1 显著超过这种 matched frozen 对照，且 controller marginal 或 interaction 至少一项在独立 sealed set 上显著为正，才能把论文主结论升级为“联合训练优于 frozen-controller”。

## 10. 可复现性

### 10.1 主要产物

最终部署 checkpoint：

```text
Yuan/unified_rl/runs/joint_actor_q_wgrid_v1_seed31000/unified.pt
```

- checkpoint SHA-256：`aaa150e7...5dc0`；
- controller state SHA-256：`98d431f7...3f99`；
- controller agent SHA-256：`d5e09fc8...c49b`；
- 部署元数据明确记录 `selector_forwards=1`、`controller_probes=0`、`model_rollouts=0`。

Sealed 主结果：

```text
Yuan/unified_rl/runs/final_holdout_joint_v2/
├── candidates_K8.meta.json
├── eval_s0_c0_full.npz
├── eval_s0_c1_static.npz
├── eval_s1q_c0_static.npz
├── eval_s1q_c1_full.npz
├── analysis_2x2.json
└── matched_frozen_actor_q_ablation.json
```

2×2 分析 JSON 的 SHA-256 为 `ece6b056...`，记录四单元输入文件 hash、controller/selector identity、valid-mask hash、bootstrap seed 和完整效应分解。目录中的 matched-frozen 消融 JSON 则记录公平 frozen 对照的开发集冻结规则和 sealed lookup 结果。

### 10.2 软件验证

当前实现包含：

- backward-compatible actor–Q deployment；
- 单次静态 evaluate 路径与成本元数据；
- fixed-grid model selection、一次性 calibration 和 fail-closed rollback；
- controller local-label exact verification 与 robust promotion；
- 严格 paired (2\times2) geometry-bootstrap analysis；
- candidate/task/mask/controller/checkpoint SHA-256 审计。

完整 standalone test suite 共 **82 tests，82/82 通过**；同时完成 Python compile 检查。测试覆盖旧 checkpoint 兼容性、actor–Q 数值路径、无效候选 mask、静态单 seed 语义、split 防泄漏、artifact 身份、promotion rollback 和 (2\times2) 分解。

## 11. 最终结论

本工作已经把原先分离的 seed selection 和连续 controller 纳入一个统一的两阶段 RL 训练框架，并在部署端严格保留“一次选择、一个 seed、一次 controller rollout”。最终系统在 10,000-task sealed set 上相对 original decoupled baseline 获得统计显著的 +1.606 mm 平均进度和 +1.887 个百分点 oracle-capture 提升，且没有增加在线 model-based 计算。

不过，严格分析也定位了真实瓶颈：收益主要由更强的静态 seed selector 驱动；controller 更新的边际效应尚未显著，联合模型与 matched frozen actor–Q 基线完全持平。因此，这一版最准确的科研结论是：**统一目标和后向信用传递已经有效，统一框架优于原始解耦实现；但“联合训练优于最强 frozen-controller”仍是下一阶段需要用更强 controller 学习和独立 sealed 实验去证明的命题。**

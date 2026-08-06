# 统一学习机器人初始姿态与连续控制

## Direct Seed、PPO 与 FlashSAC 的自包含中文技术报告

**版本：** 2026-07-28

**面向读者：** 了解基本机器人或机器学习概念，但不需要有本项目代码背景

**实验状态：** development 研究报告，而非最终 sealed benchmark

**核心问题：** 能否让一个强化学习框架同时学会“从什么关节姿态开始”和“开始后
如何持续运动”，并且在部署时仍然只选择一个初始姿态，不增加多候选试跑成本？

这里的 `development` 表示数据已经用于观察、比较或调参；`sealed` 则像一份
尚未拆封的新试卷，只有模型和规则完全冻结后才能读取。因此，development 结果
可以指导下一步研究，但不能冒充最终独立测试结论。

本文有意分成两层：

1. **主报告**从具体机器人任务出发，用普通技术语言解释动机、方法、结果和限制；
2. **技术附录**完整保留训练协议、配置、统计口径、逐阶段数字、文件指纹和复现
   约束，供实现者和审稿人核查。

读者只阅读主报告，也应当能够回答以下问题：

- 机器人在完成什么任务？
- 为什么一个“看起来正确”的 IK 解，可能让后续控制失败？
- 旧的“候选池加选择器”和新的“直接生成一个 seed”有什么本质区别？
- 所谓 backward/forward 统一训练究竟在优化什么？
- 当前 Direct Seed 比内部旧基线提高多少、与旧候选池系统还差多少？代价是什么？
- 哪些结果已经有证据，哪些仍然不能作为论文结论？

---

# 第一部分：面向非项目开发者的主报告

## 1. 执行摘要与关键结论

### 1.1 研究对象不是抽象的“机械臂控制”，而是一个具体的长程任务

本项目使用 **Franka FR3 七自由度机械臂**。机械臂末端安装一支笔，任务给定：

- 笔尖的起始位置 \(p_0\)；
- 一条从 \(p_0\) 出发、无限延伸的目标射线，其方向为 `line_dir`；
- 一个工具朝向参考方向 `n_target`。

机器人需要让笔尖沿这条射线尽可能向前运动。它没有一个走到即结束的固定终点；
只要没有违反安全或几何条件，就可以继续前进。因此，它是典型的
**长 horizon（长时域）连续控制任务**。

一次执行会在碰撞、关节限位、工具轴超出允许锥角、横向偏离过大或达到时间上限时
终止。核心成绩不是分类准确率，也不是单步位置误差，而是：

> **终止前，笔尖沿目标射线实际前进了多少米。**

这就是全文表格中的 `progress`。例如，0.469842 m 表示平均沿射线前进约
46.98 cm；提高 0.003846 m 等于提高 3.846 mm。

### 1.2 为什么初始关节角如此重要

FR3 有七个关节。对同一个笔尖位置和近似相同的工具朝向，通常存在多个不同的
七维关节解。机器人可以“肘部朝上”“肘部朝下”，也可以用不同的肩部、腕部组合
到达同一笔尖位置。这种多解性来自机械臂的冗余自由度，也可以理解为末端约束之下
仍然存在可移动的零空间。

这些姿态在当前一刻可能都满足 IK，但它们对未来并不等价：

- 有的姿态很快会碰到关节限位；
- 有的姿态在前进方向上保留了更大的零空间余量；
- 有的姿态容易自碰撞；
- 有的姿态使工具轴更容易保持在 `n_target` 周围的 30°锥内；
- 有的姿态会把后续 controller 带进难以恢复的局部区域。

因此，**IK 正确只说明“现在能到达”，不说明“从这里出发能走得远”**。这正是本
项目最重要的观察：前一阶段输出的初始关节姿态，会显著改变后一个长时域控制阶段
的成功率。

### 1.3 本轮工作的核心改变

旧系统先离线生成很多 IK 候选，再用一个 seed selector 选一个，最后交给连续
controller：

```text
任务几何
   |
   v
IKPool：大量 IK 尝试，得到候选姿态
   |
   v
Seed Selector：从候选中选择一个
   |
   v
连续 Controller：沿射线运动
   |
   v
最终 progress
```

这种系统可以工作，但“生成候选”“选择 seed”和“后续控制”是分开的。selector
只能在已经存在的候选中挑选，不能自己创造新的关节姿态；seed 学习也没有天然地
与后续 controller 的学习闭合在同一个优化循环中。

新的 Direct Seed 路径让网络根据任务几何**直接生成一个七维近似关节角**，然后
至多进行一次确定性 IK 修正，最后只执行一次 controller：

```text
任务几何
   |
   v
Direct Seed 网络：直接生成一个 7-D 初始关节角
   |
   v
安全路由：直接接受 / 至多一次确定性 IK 修正 / 安全 fallback
   |
   v
连续 Controller：只执行一次真实轨迹
   |
   v
最终 progress
   |
   +---- 训练期反馈给 seed learner
```

部署时不尝试多个 seed，不让 controller 试跑后再反选，也不调用一个模型预测多条
轨迹。用户要求的“只选择一个最好的种子进行下游规划”被保留下来。

### 1.4 当前最佳 Direct Seed development 候选

先给出全文最容易混淆的四组比较。它们回答不同问题，不能互相替代：

| 比较对象 | 当前结果 | 证据等级 | 回答的问题 |
|---|---:|---|---|
| 上一代候选式统一框架 vs original decoupled | +1.606 mm | 10,000-task sealed | 统一回报目标是否改善候选式系统？ |
| P12 vs P4 | validation/external/历史集分别 +3.846/+3.744/+5.088 mm | development；前两域 CI 跨 0 | Direct Seed 家族内部是否继续进步？ |
| P12 vs 旧 IKPool+SetSel | validation/external 分别 -94.888/-104.418 mm | 同口径 development | 单生成 seed 是否已追平旧候选池质量？答案是否定的。 |
| FlashSAC vs fresh PPO | 单训练重复 +40.779 mm | 2M-transition pilot；cold wall 更长 | 新 controller learner 是否值得继续正式比较？ |

当前相对稳健的 Direct Seed 候选是 **P12-pruned-q15**：`P12` 表示本研究第
12 轮方法迭代，`pruned` 表示开发阶段禁用了 expert 2，`q15` 表示训练 OOF 中
约 15% 的任务被分配给 specialist，并不表示生成 15 个 seed。它包含一个与旧 P4
逐位一致的 baseline 分支和两个可用 specialist 分支；一个很小的 task gate 只看
任务几何，先硬选择一个分支，然后仅由这个分支生成最终 seed。

表中的 **exact P4** 是“单分支 P4 Direct Seed 基线的逐位一致副本”。这里
`exact` 只描述软件数值复现，**不表示 P4 或 P12 已经直接生成精确 IK 解**。

还要区分“框架使用 RL 回报”和“当前 P12 每一层都由 RL 端到端更新”。P4 是
contextual RL seed 基线；P12 新增的 specialist 先用真实 return 筛出的
projected-q 标签做 winner-take-all 监督拟合，gate 再用冻结分支的真实 outcome
做监督学习，评估时 controller 始终冻结为 C0。因此更准确的名称是
**return-informed specialist**。统一 runner 支持 seed/controller 交替更新，
但当前最佳 P12 checkpoint 本身并不是“已经证明优于 frozen 的端到端联合 RL”。

它在三个已经查看过的数据域上的结果如下：

| 数据域 | 任务数 | exact P4 | P11 q05 | P12-pruned-q15 | P12 相对 P4 |
|---|---:|---:|---:|---:|---:|
| grouped validation | 1,956 | 0.465996 m | 0.468609 m | 0.469842 m | +3.846 mm，95% CI `[-0.477, +8.177]` |
| external-dev | 1,961 | 0.474197 m | 0.475404 m | 0.477941 m | +3.744 mm，95% CI `[-1.461, +8.988]` |
| 历史 holdout v2 | 9,560 | 0.464441 m | 0.465755 m | 0.469530 m | +5.088 mm，95% CI `[+3.128, +7.079]` |

这三个点估计都高于 P4，说明多分支中的互补姿态确实能被部分利用。但目前最严格、
最准确的结论仍然是：

> **P12 是当前最好的 Direct Seed development 候选，不是所有 seed 方法中的
> 最好系统，也不是已经证明优于基线的最终模型。**

原因有三：

1. validation 和 external-dev 上的 95% 置信区间都跨过 0；
2. P12 的分支裁剪与 q15 选择已经看过开发集结果；
3. 历史 holdout v2 在本工作之前已经被读取，不能重新称为全新的 sealed test。

还必须前置说明当前生成精度。P12-pruned-q15 在 grouped validation 上：

- `DIRECT = 0%`；
- 67.638% 的任务经一次 IK refinement 得到可执行解；
- 32.362% 的任务 refinement 未形成严格可执行解，转而使用共同安全 fallback；
- 原始生成 seed 的笔尖位置误差 mean 为 0.410168 m，p50 为 0.363077 m。

external-dev 同样为 `DIRECT = 0%`，66.344% 经 refinement，33.656% 走
fallback。这里的“近似 seed”不是“距离 exact IK 只有几毫米”：validation 的
平均 raw position error 仍超过 41 cm。现阶段更准确的定位是：

> **网络学习的是数值 IK 求解器的初始猜测和对后续控制有利的姿态偏好，而不是
> 已经学会 solver-free 地直接输出严格 exact IK。**

### 1.5 与旧 IKPool+SetSel 的同口径对照：计算更轻，但质量尚未追平

P12 是 Direct Seed 家族内的当前最好候选，**但不是所有 seed 方法中的最好系统**。
在 P12 strict-safe 的共同任务子集上，已经完成与旧
`IKPool + SetSel (S0)` 的逐任务对齐比较。两侧使用同一个
`C0 = r2_grouped_best` controller，task index 100% 对齐且唯一；差别只在 seed
如何得到：

- `strict-safe` 指共同 fallback 经过严格安全检查后保留下来的同一组任务；
- `C0` 是两侧共同冻结使用的 controller checkpoint 名称；
- `SetSel` 是 set selector，即根据整个候选集合的特征静态挑选一个 action 的
  集合选择器；`S0` 表示它在 C0 return cache 上训练得到的版本。

- 旧系统：128 次阻尼最小二乘（DLS）IK 尝试，去重并经最远点采样（FPS）后
  最多保留 32 个生成式 IK 候选，再追加一个共同安全 fallback，由 SetSel 在最多
  33 个 action 中静态选择一个；
- P12：task gate 选择一个生成分支，直接产生一个近似 seed，至多做一次确定性
  IK refinement。

| 数据域 | P12 Direct Seed | 旧 IKPool+SetSel | P12 - 旧系统 | paired 95% CI |
|---|---:|---:|---:|---:|
| grouped validation，n=1,956 | 0.469842 m | 0.564730 m | **-94.888 mm** | `[-105.610,-84.463] mm` |
| external-dev，n=1,961 | 0.477941 m | 0.582358 m | **-104.418 mm** | `[-115.626,-92.799] mm` |

差距不仅来自极少数异常任务：

| 数据域 | 5% trimmed delta | paired median | P12 harm `>1 mm` | P12 win `>1 mm` |
|---|---:|---:|---:|---:|
| grouped validation | -76.038 mm | -14.191 mm | 71.217% | 21.524% |
| external-dev | -80.420 mm | -11.937 mm | 70.066% | 22.947% |

这里的 CI 统一使用 `direct_seed_eval._paired_summary` 默认口径：逐任务配对
bootstrap 5,000 次，bootstrap seed 为 20260728。

validation 的 1,956 行中有 1,759 个唯一任务几何，另有 197 行属于重复几何；
external-dev 的 1,961 行几何全部唯一。为检查重复几何是否让逐行 CI 过于乐观，
又做了“每个唯一几何等权”的聚类敏感性分析：validation 的差值为
-95.743 mm，95% CI `[-106.663,-84.548] mm`；结论不变。上述旧系统均值只适用
于 P12 strict-safe 共同子集，不是旧系统在完整 2,048 个任务上的全量成绩。

因此，本轮结果必须被描述为一个清楚的质量—计算权衡：

> **Direct Seed 把部署前的规划结构从 128 次 IK 尝试、最多 32 个生成式候选、
> 一个安全 fallback 和一次 selector，压缩为一个近似 seed 和至多一次 IK；
> 但当前 P12 的下游 progress
> 仍显著低于旧候选池选择器。**

这两个开发集已经被查看，所以这仍是 development 对照；但差值很大、两个 CI 都
完全低于 0，不能为了强调计算优势而省略。

### 1.6 当前离 oracle 仍然很远

grouped validation 上，IKPool reference oracle 为 0.621990 m。这个 oracle
是事后查看候选池中每个完整候选的真实结果，再取最好值；它是分析上限，不是可部署
方法。

P12-pruned-q15 为 0.469842 m，仍低 **152.148 mm**。即使事后知道 P12 四个
分支各自的真实执行结果，再逐任务选择最佳分支，branch oracle 也只有
0.532720 m，仍低于 IKPool reference oracle 89.270 mm。

以 P4 为起点，四分支事后 oracle 提供 66.724 mm 的潜在分支收益，而 P12 的
可部署 gate 实际提高 3.846 mm，只兑现约 **5.8%**。这不是说剩余 94.2% 一定能
由现有 gate 学到；它只是直观显示“已有分支互补性”和“部署前能否辨认正确分支”
之间仍有很大差距。

所以，当前工作证明了“分支存在互补性”和“task gate 能回收少量互补收益”，还不
支持“已经接近 oracle”。

### 1.7 Controller 算法线的初步结果

另一条研究线比较 PPO 和用户指定的官方 FlashSAC。当前只有一个
2M-transition pilot：

| 指标 | fresh PPO | FlashSAC official-density |
|---|---:|---:|
| 实际 transitions | 2,002,944 | 2,000,000 |
| 最终平均进度 | 0.375553 m | 0.416333 m |
| transition AUC | 0.371921 m | 0.389479 m |
| core train | 431.532 s | 447.351 s |
| cold end-to-end | 443.908 s | 468.359 s |

在这个单一训练随机种子上，FlashSAC 最终高 40.779 mm，transition AUC 高
17.558 mm；但包含首次编译后，它的完整墙钟更长。共享 RTX 4090 上的稳态片段
显示 FlashSAC 约有 1.236× transitions/s，但这不是独占 GPU、多次重复的正式
速度结论。

它足以支持“继续做六训练种子的正式比较”，不足以支持“FlashSAC 总体优于 PPO”。

### 1.8 最重要的负结果

本项目的负结果对后续设计同样关键：

- 只让 seed 更接近几何 IK，并不保证 controller 走得更远；
- 把多个好姿态压进一个 deterministic mean，容易把不同模式平均掉；
- specialist 单独看通常比 baseline 差，真正的价值来自任务相关互补；
- 增大 gate 容量可以提高训练期 OOF，却不一定提高跨域实际结果；
- 当前 matched 实验尚未证明更新 controller 比冻结 controller 更好。
- 在完全相同的 C0 和 strict-safe 任务上，当前 Direct Seed 仍比旧
  IKPool+SetSel 低约 95--104 mm。

换句话说，瓶颈已经不是“能不能生成另一个姿态”，而是：

> **在不试跑候选的前提下，仅根据任务几何，能否可靠判断哪个姿态会让后续
> controller 走得更远。**

### 1.9 两代“统一框架”必须分开理解

本项目在 Direct Seed 之前，已经完成过一代**仍依赖候选池**的统一训练：

- 旧系统从候选集合中做一次静态选择；
- backward 用完整候选 rollout 回报改善 selector；
- forward 尝试更新 controller；
- 部署仍只执行一个被选 seed 和一次 controller rollout。

那一代系统在自己的 10,000-task sealed final holdout 上，从 original decoupled
基线的 0.545694 m 提高到 0.547301 m，即 +1.606 mm，geometry-bootstrap
95% CI 为 `[+0.418,+2.817] mm`。但与同等 actor--Q 能力的
frozen-controller 基线 0.547316 m 相比，联合模型反而低 0.015 mm，CI
`[-0.740,+0.716] mm`。所以它证明的是“统一的下游回报目标改善了静态
selector”，不是“更新 controller 已经显著更好”。

本报告研究的是下一代问题：**连候选池本身也不希望在部署时依赖，而由网络直接生成
近似 seed。** Direct Seed 的 grouped validation、strict fallback subset、模型和
路由口径与上一代 10,000-task sealed 主表不同。P12 的 0.469842 m 不能和上一代
sealed 表的 0.545694 m 直接相减，P12 相对 P4 的 +3.846 mm 也不是“相对
original decoupled selector”的提升。

不过，在本报告自己的 validation/external strict-safe subset 上，现在已有
同一 C0 的直接 head-to-head：P12 分别比旧 IKPool+SetSel 低 94.888 mm 和
104.418 mm。它回答的是当前直接生成方法与旧候选池系统的任务质量差距；上一段
+1.606 mm 则回答上一代候选式统一训练相对其 original decoupled 版本的收益。

因此，本报告会把两种证据分开：

1. 候选式统一框架已经有一次 sealed 的 +1.606 mm 证据；
2. 直接生成式框架当前最好的内部证据，是 P12 相对 Direct Seed exact P4 的
   development 点估计提升；与旧 IKPool selector 的同口径 development
   head-to-head 已完成且仍明显落后，尚缺的是完全冻结后的全新 sealed 比较。

---

## 2. 任务、约束和评价指标

### 2.1 一次任务包含什么

可以把一次任务想象成下面的场景：

1. FR3 的笔尖需要出现在给定起点 \(p_0\)；
2. 从这个点出发有一条方向固定、没有终点的射线；
3. 笔尖要持续沿射线正方向运动；
4. 运动期间工具轴要留在目标方向周围 30°的锥内；
5. 机械臂不能自碰撞，不能越过关节限位，也不能横向偏离过大；
6. 任一终止条件触发后，本次 progress 停止累计。

```text
                        工具轴允许的 30°方向锥
                              /\
                             /  \
                            /    \
                           笔尖
                            o=================================>
                           p0          给定的无限射线

评分 = 从 p0 开始，终止前沿射线方向走出的距离
```

“无限射线”使这个任务与普通点到点 IK 不同。点到点 IK 只关心当前能否到达目标；
这里更关心到达后是否还有足够的关节余量和合适的姿态，支持很长一段后续运动。

### 2.2 什么是七自由度冗余和零空间

一个关节配置写成：

\[
q = (q_1,q_2,\ldots,q_7).
\]

它是本文所说的七维关节角。末端笔尖位置只有三个坐标，工具方向也只施加部分约束，
因此不同的 \(q\) 可能对应相同或近似相同的笔尖状态。那些在不破坏末端主要约束的
情况下仍可改变的关节组合，通常称为零空间运动。

连续 controller 的工作，正是不断在“让笔尖沿射线前进”和“利用零空间调整姿态、
保持安全余量”之间做权衡。如果初始 \(q\) 已经把某些关节推到极限附近，后续策略
即使局部动作正确，也可能没有足够空间挽救。

### 2.3 progress 为什么比 IK error 更接近最终目标

本项目曾经观察到：

- P1 的 raw position mean 为 0.4290 m，最终 progress 为 0.454747 m；
- P2 把 raw position mean 降到 0.2871 m，几何上明显更准确；
- 但 P2 的最终 progress 反而只有 0.451134 m，比 P1 低 3.613 mm，
  95% CI 为 `[-11.585, +4.922] mm`。

这不是矛盾。raw position error 衡量网络输出离当前 IK 目标多远；progress 衡量
经过修正并执行完整长轨迹以后走了多远。一个关节解可以在当前时刻更精确，却让
机械臂更快碰到未来约束。

因此，几何误差仍是必要的安全与可解性信号，但不能代替下游 return。

### 2.4 终止、fallback 与 intent-to-treat

网络输出不会未经检查就直接送入 controller。router 依次判断：

1. 输出是否为有限数；
2. 是否在关节限制内，并保留至少 0.02 rad 的 joint margin；
3. 笔尖位置误差是否不超过 5 mm；
4. 工具轴是否在 30°锥内；
5. 是否无自碰撞。

若原始输出直接通过，则使用 `DIRECT` 路径。若没有直接通过，但任务输入有效、
网络输出为有限数且关节角已满足保守限位，它才是“可投影”样本，并至多接受一次
固定、无随机 restart 的 DLS IK refinement。任务输入非法、输出含 NaN/Inf 或
关节限位不合格的样本不尝试 IK，直接进入安全 fallback。修正后再次接受同样
检查；仍不通过，则使用共同的安全 `q0_pilot`，即 `FALLBACK`。

当前 P12 三个已报告域的网络输出都满足上述可投影前提，同时 `DIRECT=0`，所以
实际记录为平均每任务一次 IK attempt；这是一项当前模型结果，不是 router 对未来
任意模型都强制执行一次 IK。

最终主结果按 intent-to-treat 统计：无论网络输出走 DIRECT、REFINE 还是
FALLBACK，都计入总体成绩。这样不会因为只展示成功样本而掩盖 generator 失败。

当前主结果的 `DIRECT=0`。也就是说，现阶段网络生成的是对下游有帮助的
**近似 IK seed**，但还没有直接产生满足全部严格条件的 exact IK 解。收益主要来自
“更好的单次 IK 初值”以及保守 fallback，而不是完全消除了 IK。

---

## 3. 关键术语：不需要编程背景的解释

| 术语 | 本报告中的含义 |
|---|---|
| IK | 逆运动学：已知希望笔尖在哪里、朝向如何，求七个关节应取什么角度。 |
| FK | 正运动学：已知七个关节角，计算笔尖实际在哪里、朝向如何。 |
| Seed | IK 或后续控制开始时使用的一组七维关节角。不同 seed 可能落入不同姿态分支。 |
| DLS | Damped Least Squares，阻尼最小二乘：一种数值 IK 方法，从一个初始猜测迭代修正关节角；阻尼用于减轻奇异位形附近的不稳定。 |
| FPS | Farthest Point Sampling，最远点采样：从大量相似解中保留彼此尽量不同的姿态，使小候选池仍有多样性。 |
| IKPool | 对同一任务做 16 orientations × 8 restarts，即 128 次 DLS IK 尝试；去重并经 FPS 后最多保留 32 个生成式候选，再追加一个共同安全 fallback。 |
| Selector | 从已经生成的候选中挑一个 seed 的模型；它不能跳出候选池创造新解。 |
| SetSel | 本项目的集合选择器：同时查看候选集合的特征，静态选出一个 action。 |
| C0 | 本报告用于公平 seed 对比的固定 controller checkpoint，即 `r2_grouped_best`。 |
| strict-safe subset | 共同 fallback 通过严格关节、几何和碰撞检查后保留下来的共同任务子集。 |
| Direct Seed | 不先枚举 IK 候选，而是由网络根据任务几何直接输出一个七维近似 seed。 |
| Controller | seed 确定以后，在每一个时间步输出关节零空间动作、让笔尖持续沿射线运动的策略。 |
| Rollout | 让 controller 从一个 seed 开始完整执行一次，直到终止或达到时间上限。 |
| Return / outcome | 一次 rollout 的实际结果；本任务最核心的 return 是最终 progress。Monte-Carlo outcome 表示先完整执行，再用观察到的总结果学习。 |
| Macro critic | 对“选这个初始 seed 后，整段任务预计能走多远”的价值估计器；它评价的是 seed 级决策，不是单个控制时间步。 |
| Replay | 保存过去交互样本并在训练中再次使用。Controller 改变后，旧 seed 回报不再是真值，因此相应 macro replay 必须清空。 |
| Candidate | 在做最终决定前可供比较的备选 seed。主部署协议禁止多候选试跑。 |
| Oracle | 事后知道多个候选各自真实结果后逐任务取最好值，是诊断上限，不是可部署方法。 |
| Branch / expert | 同一个 seed 网络中的一个输出分支，能表达一种不同的关节姿态模式。 |
| Gate | 只根据任务几何硬选择一个 branch 的小模型。选完以后只执行该 branch。 |
| Deterministic mean | 对同一任务总输出同一个平均关节向量，而不是随机采样多个可能姿态。 |
| MLP | Multi-Layer Perceptron，多层感知机；这里指小型全连接神经网络。 |
| WTA | Winner-Take-All，赢家全得训练：每个样本只更新最匹配的一个分支，避免所有分支被同一目标拉成平均值。 |
| OOF | Out-of-fold。把训练任务按几何分组轮流留出，用未见过该组的模型作预测，以减少训练内自我拟合造成的乐观偏差。 |
| CI | 置信区间。本报告主要用逐任务 paired bootstrap 的 95% CI 表示估计的不确定性。 |
| Frozen | 参数冻结。训练另一个模块时不更新该模块，用于隔离收益究竟来自哪里。 |
| Sealed set | 在方法完全冻结以后才生成、只允许一次正式读取的全新测试集。 |

### 3.1 “只选择一个 seed”不等于“模型只能有一个分支”

这是理解 P12 的关键。

训练期可以让多个冻结 branch 分别在训练任务上执行，了解它们各自擅长什么；但在
部署期，gate 只能看任务几何，然后一次性做出决定：

```text
                    +--> baseline branch --+
任务 --> hard gate +--> specialist 1 ------+--> 只输出被选中的一个 seed
                    +--> specialist 3 ------+
```

部署时不会：

- 同时生成三个 seed 后比较；
- 查询三个分支的 predicted return；
- 分别运行 IK 后选择；
- 让 controller 分别试跑后选择。

因此，多分支表示能力与多候选部署成本是两件不同的事。P12 增加的是少量神经网络
表示和一个 hard gate，不增加 IK attempt 或 controller rollout 的数量。

实现上还有一个容易混淆的细节：为了保证 baseline 行与原 P4 逐位一致，部署代码会
在整个 batch 上计算一次完整 baseline 线性层；若某行被路由到 specialist，该行
还计算一次被选 specialist 的线性层。它不会计算所有 specialist 的关节向量，
也只有一个七维关节向量离开网络并进入 IK 和 controller。因此这里的“单 seed”
约束指后续物理候选、IK 和 rollout 数量，而不是声称神经网络内部只有一次矩阵乘法。

### 3.2 “reference oracle”为什么不能作为实际算法

假设一个任务有四个 branch。为了知道哪个 branch 真正最好，最可靠的方法是让
四个 branch 都完成 IK 和 controller rollout，再查看四个 progress。这就是
branch oracle。

问题在于，这相当于实际执行了四次。它违反了用户要求的单次选择，也把推理和真实
控制成本扩大了。Oracle 的作用是回答“这些分支之间究竟有多少互补潜力”，不是
提供可部署方案。

IKPool reference oracle 同理。它是在现有完整候选中事后取最好值，不是全局所有
可能关节姿态的真正最优解，因此全文始终称它为 **reference oracle**。

### 3.3 95% CI 应该如何阅读

例如，P12 在 grouped validation 上相对 P4 的平均提升是 +3.846 mm，95% CI 为
`[-0.477,+8.177] mm`。可以直观理解为：当前有限任务样本支持的合理差值范围，
仍包含轻微退化和明显提升两种可能。

因此：

- 点估计为正：值得继续验证；
- CI 完全大于 0：在当前统计口径下有更强的正收益证据；
- CI 跨 0：不能写成已经确认 superior。

历史 holdout v2 上 P12 相对 P4 的 CI 是 `[+3.128,+7.079] mm`，虽然完全为正，
但该数据过去已经被读取。统计显著性不能消除数据被反复查看带来的选择偏差，所以
它仍然只能叫历史诊断。

### 3.4 grouped OOF 为什么按几何而不是按数据行划分

同一种任务几何可能在数据中出现多行。如果随机按行切分，同一几何的近似重复可能
同时进入训练折和测试折，gate 只需“认出见过的题”就能得到虚高分数。

本项目用 `(p0, line_dir, n_target)` 的 exact float32 byte signature 分组；
同一几何的所有行必须在同一 fold。这样得到的 training OOF 仍不是外部测试，
但比普通 row-level split 更接近“预测新几何”的真实难度。

---

## 4. 从解耦框架到统一 backward/forward 学习

### 4.1 旧框架为什么不够优雅

旧框架把问题分成两个相对独立的模块：

1. IKPool 先生成候选，selector 根据离线数据学习选哪个 seed；
2. controller 在给定 seed 分布上学习后续连续动作。

这种解耦带来三个限制：

- **搜索空间受限。** selector 只能在已有候选中选，无法生成候选池之外的姿态；
- **目标错位。** IK 几何质量好，不一定等价于 downstream progress 高；
- **分布错位。** seed 生成器改变初始姿态分布后，controller 面对的状态也改变；
  controller 更新后，旧 seed 的 return 排序又可能失效。

这与长时域灵巧策略链中的核心问题相同：前一阶段留下的状态，不只是“当前成功或
失败”，还决定下一阶段是否容易成功。本项目受到 Chen 等人的
[*Sequential Dexterity: Chaining Dexterous Policies for Long-Horizon
Manipulation*](https://proceedings.mlr.press/v229/chen23e.html) 启发，借鉴的是
跨阶段 transition feasibility 和前后向适配的动机，不是机械复刻其多技能链
实现。

### 4.2 新框架中的 backward phase

backward phase 固定当前 controller，训练 seed 侧。每个任务只采一个 seed，
完成一次真实 rollout，记录最终 progress。训练信号同时包含：

- 下游真实 progress，告诉 seed learner 哪种初始姿态真正走得远；
- 可微 FK、安全锥和关节余量约束，防止只追求回报而忽视物理可行性；
- 成功 refinement 的投影结果，用作 self-distillation，帮助近似 seed 更容易被
  单次 IK 修正。

这里的 “backward” 是从下游结果反过来改善上游 seed 决策。它并不意味着对整个
MuJoCo rollout 逐时间步做解析反向传播；真实 rollout 先提供 Monte-Carlo outcome，
seed actor 再通过 macro critic 和可微约束获得梯度。

### 4.3 新框架中的 forward phase

forward phase 固定 seed actor，在它当前产生的初始关节分布上更新连续
controller。这样 controller 不再只适应历史 IKPool 的起点，而可以适应 seed
generator 实际留下的姿态。

controller 一旦更新，同一个 seed 的 downstream return 可能改变，所以旧的
macro replay 不能继续当作新 controller 的真值。实现会在 controller 更新后清空
旧 macro-return replay，并重新收集。

### 4.4 交替优化的完整闭环

```text
                 backward：改善“从哪里开始”
        +------------------------------------------------+
        |                                                |
        v                                                |
任务 -> Seed Actor -> Router / IK1 -> Controller -> progress
                                      |
                                      +---- forward：改善“开始后怎么走”

一次循环：
  1. 冻结 controller，用真实长程结果更新 seed actor；
  2. 冻结 seed actor，在新初始姿态分布上更新 controller；
  3. controller 变化后清空旧回报；
  4. 再回到 seed actor，重新评价并继续训练。
```

这就是“统一框架”的含义：seed 生成和 controller 训练属于同一个交替优化过程，
而不是两个互不知情的独立产品。

### 4.5 当前证据没有证明 forward update 已经带来收益

框架能运行，不等于每个更新方向已经有效。当前 matched 结果是：

- 第一次 1M-transition PPO forward 后，joint-vs-frozen 为 -0.312 mm，
  95% CI `[-3.707,+3.069] mm`；
- 随后 backward 让 joint 分支从 0.449237 m 提升到 0.463992 m，即
  +14.755 mm，CI `[+5.333,+24.369] mm`；
- 但严格 matched frozen-feedback 对照达到 0.465996 m，比 joint-feedback
  高 2.078 mm，CI `[-3.224,+7.322] mm`；
- 在最佳 actor 上再以 `3e-5` 训练 controller 250k transitions，仅提高
  0.290 mm，CI `[-2.069,+2.412] mm`。

所以现有证据支持 **return-aware backward 有效**，还不支持
**joint controller update 优于 frozen controller**。论文若要把联合训练作为
正结论，仍需更好的 forward learner、matched frozen 对照和多训练随机种子。

---

## 5. 为什么最终采用多分支 Direct Seed

### 5.1 单头 deterministic actor 的“平均姿态”问题

早期方法试图让一个网络对每个任务只输出一个 deterministic mean。问题在于，同一
任务可能存在多个相距很远、但都合理的关节姿态模式。如果训练数据里同时出现
“肘上”和“肘下”两个好解，普通均方误差或 imitation 可能把它们平均成一个两边都
不像的姿态。

P6--P8 的实验反复出现同一现象：

- online RL 能找到一些回报很好的局部投影；
- 更高 target coverage 和更长 imitation 会继续降低 raw IK error；
- 但 progress 不再提高，甚至下降。

这说明问题不是完全没有好 seed，而是单头模型难以同时表达和正确选择多种姿态。

### 5.2 为什么先训练 specialist，再用 outcome-matched gate

P9 曾让不同 expert 按 joint-space 最近目标分工。它能把几何 target 分开，却不能
判断哪个 expert 的实际 controller return 更高。P10 用 explorer 的旧标签训练
safe gate，也出现大量 false positive。

P11/P12 的 specialist 训练本质上仍是监督拟合：先用真实 controller return 判断
baseline 或 explorer 的 projected-q 哪个更好，再用 winner-take-all imitation
让不同 specialist 拟合这些关节标签。随后，gate 又以冻结 branch 的真实 rollout
outcome 为监督信号。这比纯几何 imitation 更“懂下游结果”，但不等于 rollout
本身可微，也不等于 P12 的 controller 与 specialist 在同一步端到端反向传播。

P11 改为 outcome-matched：

1. 保持 P4 baseline 完全冻结且逐位一致；
2. 训练一个 specialist；
3. 在训练任务上强制执行这个**最终冻结后的 specialist 输出**；
4. 收集它经过同一个 router、一次 IK 和同一个 controller 后的真实 progress；
5. gate 学习“这个已部署 specialist 是否真的比 baseline 好”。

这避免了一个重要错配：训练期 explorer 找到的好 target，经过网络拟合以后未必
仍然输出同一个关节角。Gate 必须学习最终输出的 outcome，而不是学习一个已经发生
漂移的旧 target outcome。

### 5.3 为什么 specialist 总体更差仍然有价值

P11 的单个 specialist 在 17,588 个训练任务上的平均 progress 为
0.461701 m，低于 exact P4 的 0.469692 m。但它在 3,854 个任务上比 baseline
至少高 10 mm，占 21.91%；这些优势任务平均高 +161.211 mm。

如果能完美判断何时切换，双分支 training branch oracle 可达到 0.505610 m，
比 baseline 高 35.918 mm。也就是说，specialist 的价值不是“到处都更好”，而是
“在少数任务上提供 baseline 没有的姿态”。

这也解释了为什么 gate 必须保守。错误切换一次的平均损失可能很大，不能为了追求
更高 specialist 使用率而牺牲大量普通任务。

### 5.4 P12 如何扩大互补性

P12 保留 exact P4 为 expert 0，并训练三个不同 specialist。三个 specialist
在 validation 上单独运行都明显低于 P4：

| branch | train progress | validation progress |
|---|---:|---:|
| expert 0：exact P4 | 0.469692 m | 0.465996 m |
| expert 1 | 0.451572 m | 0.439693 m |
| expert 2 | 0.450804 m | 0.444120 m |
| expert 3 | 0.454467 m | 0.446512 m |

但是，三个 specialist 各自在约 24% 的 validation 任务上比 baseline 高
10 mm 以上，而且彼此 advantage 的相关性只有 0.34--0.39。它们失败的任务和
成功的任务并不完全相同。四分支事后 oracle 因而达到 0.532720 m，比 P4 高
66.724 mm。

训练 OOF 显示 expert 2 对线性 gate 的边际贡献为负，所以当前候选只保留
expert 1 和 expert 3。Gate 的 specialist quota 固定为训练任务的 15%，故名
`pruned-q15`。

### 5.5 为什么 P13 更复杂却没有更好

P13 把线性 gate 换成一个小型非线性 MLP，直接回归三个 specialist 相对
baseline 的 advantage。它在 training grouped OOF 上得到：

- q10：+5.468747 mm；
- q15：+6.249040 mm。

两者都高于 P12 线性 gate 的训练 OOF，但实际 validation 只有：

- q10：0.467319 m，相对 P4 +1.323 mm，CI `[-0.813,+3.494]`；
- q15：0.468452 m，相对 P4 +2.456 mm，CI `[-0.304,+5.230]`。

它们都低于 P12-pruned-q15 的 0.469842 m。因此 P13 没有继续运行
external-dev 或历史 holdout。

这次失败说明，更强的函数拟合能力可能只是更充分地拟合训练域中的 branch pattern，
并没有解决跨几何、跨数据域的判断误差。拒绝 P13 的原因是实际结果没有提升，
不是推理太慢；其 gate 微基准只比线性版本多 0.047 us/task。

---

## 6. 方法演进：每一步解决了什么，又暴露了什么

下面的表格给出从早期单头模型到当前候选的主线。详细数字、CI 和所有失败配置都在
技术附录中完整保留。

| 阶段 | 核心尝试 | grouped validation progress | 得到的认识 |
|---|---|---:|---|
| Random | 未训练 seed actor | 0.420400 m | 提供最低参考。 |
| P1 | 让 actor mean 更贴近可修正目标 | 0.454747 m | 学到近似 seed 有明显价值。 |
| P2 | 更强调 FK/IK 几何精度 | 0.451134 m | 几何更准不等于下游更好。 |
| P4 | 用真实 downstream return 做 frozen feedback | 0.465996 m | return-aware backward 是当前最可靠收益来源。 |
| P5 | 在 P4 上继续更新 controller | 0.466286 m | 仅 +0.290 mm，不能证明 joint 更好。 |
| P6 | 全局 top-25% elite replay | 0.464152 m | 全局 elite 会偏向易任务。 |
| P7 online | 每任务保存最好投影并在线更新 | 0.463605 m | memory、RL 和 imitation 同时更新会干扰。 |
| P7 offline-2k | 冻结 elite 后离线拟合 2,000 次 | 0.469605 m | 出现早停峰值；CI 跨 0，且继续拟合会退化。 |
| P8 | 提高覆盖、paired target、positive-only | 最高 0.464482 m | 普通 anchor 不能解决多模态平均。 |
| P9 | 四头、按 joint target 做 WTA | 最高 0.459558 m | 几何 winner 不等于 outcome winner。 |
| P10 | 冻结 baseline 的 advantage gate | 最高 0.466059 m | 类别加权会增加 false positive。 |
| P11 q05 | 冻结 specialist 后按实际 outcome 学 gate | 0.468609 m | 标签终于与部署输出一致，得到保守小收益。 |
| P12 q15 | 四分支 outcome bank、裁掉 expert 2 | **0.469842 m** | 当前最稳健 development 候选。 |
| P13 | 非线性 advantage regressor | 最高 0.468452 m | OOF 更好但 validation 更差，拒绝晋级。 |

### 6.1 P4：真正把长程结果带回 seed 学习

P4 相对 random actor 提高 45.596 mm，95% CI
`[+34.963,+56.332] mm`；相对 P1 再提高 11.249 mm，CI
`[+2.116,+20.190] mm`。这一步建立了最重要的正证据：

> seed 不能只用 IK 几何标签训练；真实 controller return 能提供额外且有用的
> 学习信号。

P4 也是后续所有 gate 的 exact baseline。为避免浮点 kernel 变化让所谓
“baseline 分支”在 IK basin 边界产生不同结果，P11 以后保留原始完整输出层，并
用单元测试要求 expert 0 与 P4 的关节输出逐位相等。

### 6.2 P7：证明存在局部好姿态，也证明过度拟合会伤害 progress

P7-offline 的训练步数曲线很有解释力：

| offline updates | progress | raw position mean |
|---:|---:|---:|
| 100 | 0.457748 m | 0.3546 m |
| 500 | 0.459568 m | 0.3531 m |
| 2,000 | **0.469605 m** | 0.3188 m |
| 3,000 | 0.467544 m | 0.3037 m |
| 5,000 | 0.463271 m | 0.2800 m |
| 10,000 | 0.461913 m | 0.2581 m |

2,000 次以后，raw position error 继续变小，progress 却持续下降。这是本项目最
清楚的证据之一：继续把网络推向某个几何平均目标，会损伤对未来控制真正重要的
关节模式。

### 6.3 P11：从“拟合好 target”转向“选择真实有效的最终输出”

P11 q075 在 grouped validation 上相对 P4 提高 +3.591 mm，CI
`[+0.857,+6.379]`，但 external-dev 变成 -1.143 mm，CI
`[-4.473,+2.161]`。这说明只看一个开发域会得到过于乐观的结论。

更保守的 q05 在三个域上分别为：

- grouped validation：+2.613 mm，CI `[+0.373,+4.888]`；
- external-dev：+1.206 mm，CI `[-1.575,+4.103]`；
- 历史 holdout v2：+1.314 mm，CI `[+0.034,+2.597]`。

q05 的特点不是平均提升很大，而是约 95% 训练任务继续走 exact baseline，只对一小
部分任务切换。

### 6.4 P12：更多互补分支带来更高点估计，但仍未跨过确认门槛

P12-pruned-q15 相对 P11 q05：

- validation：+1.232 mm，CI `[-3.070,+5.650]`；
- external-dev：+2.537 mm，CI `[-2.522,+7.547]`；
- 历史 holdout v2：+3.774 mm，CI `[+1.696,+5.830]`。

历史 holdout 上，P12 相对 P4 的平均提升为 +5.088 mm，但 5% trimmed
提升只有 +0.635 mm；改善超过 1 mm 的任务占 8.159%，退化超过 1 mm 的任务占
4.854%。相对 P11 的平均提升为 +3.774 mm，5% trimmed 提升为 +0.598 mm，
`>1 mm` harm 为 5.795%，win 为 8.546%。两组统计都说明，均值收益主要来自
一部分任务的较大改善，不是所有任务均匀提高。

---

## 7. 公平评价：为什么不能只看一个平均数

### 7.1 三类数据域的角色

| 数据域 | 角色 | 本报告能否据此作最终结论 |
|---|---|---|
| 训练集 | 拟合 seed actor、specialist 和 gate | 不能 |
| grouped validation | 开发和模型选择 | 不能；已经重复查看 |
| external-dev | 开发阶段的外部确认 | 当前也不能；已参与 q15/q20 判断 |
| 历史 holdout v2 | 过去保留的数据 | 不能；在本研究前已被读取 |
| 新 sealed set | 方法完全冻结后新生成、一次性读取 | 可以，但目前尚未执行 |

“external”这个名字本身不保证独立。如果研究者看完它再修改方法，它就参与了开发。
同理，历史 holdout 即使任务数较多，也不能因为结果漂亮而重新命名为 sealed。

### 7.2 为什么同时报告 mean、trimmed、harm 和 win

长程控制的任务差异可能很大。一个 gate 可能在少数任务提高 150 mm，却在更多任务
各退化几毫米，最后均值仍然为正。因此至少要同时看：

- mean progress；
- 去掉两端极端样本后的 5% trimmed mean；
- 相对基线退化超过 1 mm 的 harm rate；
- 相对基线提高超过 1 mm 的 win rate；
- paired delta 的 bootstrap CI；
- 碰撞、锥角、关节限位等终止比例。

P12 历史 holdout 的 mean improvement 比 trimmed improvement 大，正说明收益分布
并不均匀。这不是否定结果，而是限定它的实际风险形态。

### 7.3 训练随机性和任务随机性是两层不同的不确定性

在固定 checkpoint 上对很多任务做 bootstrap，只能估计“换一批类似任务”时结果
如何变化。它不能回答“重新训练一次网络”会不会得到同样结果。

FlashSAC/PPO 当前只有一个 training seed；Direct Seed 主线也没有完成协议要求的
全部训练种子。因此，现有任务级 CI 不能替代多次独立训练。

### 7.4 promotion gate 的作用

每个阶段都应先规定什么情况下继续扩大实验。P13 的 training OOF 虽然更高，但
validation 没超过 P12，所以停止，不再去 external-dev 寻找有利配额。这种
fail-fast 规则可以减少“看过越多数据，总能找到一个看起来更好配置”的风险。

---

## 8. 推理成本、训练成本和“单次选择”

### 8.1 部署协议

P12-pruned-q15 对每个任务严格执行：

```text
task gate 次数                 = 1
最终 seed 数                   = 1
候选枚举                       = 0
return / critic 查询           = 0
controller probe               = 0
最大 IK refinement 次数        = 1
真实 controller rollout 次数   = 1
```

因此，P12 没有用 model-based planning 或多次执行换取结果。和 P4 相比，新增的
部署成本只是一个很小的 hard gate 和被选中的 branch head。

### 8.2 P12 的训练期额外成本

训练多个 specialist 后，需要在训练任务上执行冻结分支，建立 outcome matrix。
三次 17,588-task frozen-specialist collection 的核心分段计时为：

| 训练期阶段 | 时间 |
|---|---:|
| 三个 branch generator | 0.230 s |
| 三个 branch router + 至多一次 IK | 7.292 s |
| 三个 branch C0 controller rollout | 163.627 s |
| 合计 | 171.149 s |

specialist bank 本身 2,000 次 GPU update 用时 6.035 s。这些成本发生在训练期，
而不是每个部署任务上。171.149 s 也不包括加载、初始化和首次构建 fallback filter
manifest，所以不能称为完整 end-to-end 总时间。

### 8.3 P12 的部署均摊时间

在 9,560-task 历史 holdout v2 上，batched evaluation 的均摊值为：

| 部署区段 | ms/task |
|---|---:|
| generator + hard task gate | 0.00360 |
| strict gate + 至多一次 IK | 0.23183 |
| controller rollout | 4.32999 |
| 合计 | 4.56543 |

这组数字回答了“只单次选择时计算指标和成本是什么”。其中初始化部分
`generator + gate + router/IK` 合计约 0.23543 ms/task。它们都是 GPU 批量仿真与
评估的**计算墙钟均摊**，不等于在线 `B=1` latency，也绝不表示机器人在 4.6 ms
内完成了约 0.47 m 的物理运动。真实 controller 按 20 Hz 运行；表中的
controller 时间来自大量并行仿真的均摊吞吐。实验又运行在有其他进程的共享
RTX 4090 上，因此这些数字既不是单台真实机器人 latency 保证，也不能用来把 P11
和 P12 不同时刻的细小差别解释成稳定加速或减速。

### 8.4 旧 IKPool 与 Direct Seed 的计算差别

完整旧 IKPool 对每个任务做 16 orientations × 8 restarts，即 128 次 DLS IK
尝试；去重并经 FPS 后最多保留 32 个生成式候选，再追加一个共同安全 fallback，
由 selector 在最多 33 个 action 中选一个。这里的“128”是上游 IK attempt 数，
“32”是生成式候选池容量，“33”是包含安全 fallback 后的最大 selector action
数，三者不能混称。Direct Seed 是：

```text
一次神经网络生成 + 至多一次确定性 IK
```

两者最终都只把一个 seed 交给 controller，但到达这个决定的计算过程不同。正式
论文仍需要在独占 GPU、online `B=1` 下报告 Direct 和 IKPool 的 p50/p95/p99，
才能给出可靠的实际加速倍数。结构成本降低也不能替代质量比较：同一 C0 和共同
strict-safe 任务上，P12 在 validation/external 的 progress 仍分别低
94.888 mm 和 104.418 mm。因此当前 Direct Seed 是一个明显更轻、但质量尚未
追平的 Pareto 候选，不是旧 IKPool 的无损替换。

---

## 9. PPO 与 FlashSAC：比较的究竟是什么

### 9.1 为什么要尝试 FlashSAC

PPO 是 on-policy 方法。一批轨迹用于若干轮更新后，通常不会长期保存在 replay
buffer 中反复使用。FlashSAC 是 off-policy 方法，可以从 replay 中复用历史
transition，并利用 GPU 并行和编译提高 learner 吞吐。理论上，它可能：

- 用更少的新环境交互达到同一成绩，即样本效率更高；
- 用更少墙钟达到同一成绩，即时间效率更高；
- 在相同 transition 预算下达到更高最终 progress。

这三件事必须分别验证。训练更快不自动等于样本更省；早期学习更快也不自动等于
最终性能更好。

### 9.2 当前 2M pilot 可以说明什么

FlashSAC 在单一训练种子上的最终 progress 为 0.416333 m，PPO 为
0.375553 m，差 +40.779 mm。200 个共享任务的 paired bootstrap 95% CI 为
`[+18.308,+65.265] mm`。

但提升分布不均匀：win rate 只有 52%，paired median 只有 +0.198 mm。FlashSAC
的 0--2M transition AUC 为 0.389479 m，PPO 为 0.371921 m，差
17.558 mm（+4.72%）。

墙钟方面：

- FlashSAC core train 447.351 s，PPO 431.532 s；
- FlashSAC cold end-to-end 468.359 s，PPO 443.908 s；
- FlashSAC 首次编译占 108.180 s；
- 仅作反事实扣除编译后，FlashSAC core 为 339.170 s，相对 PPO 少
  21.40%，约为 1.272× throughput；
- 500k--2M 稳态区间约为 5.87k vs 4.75k transitions/s，即约 1.236×。

因此当前结果更像“FlashSAC 有潜力在较长运行中摊薄编译成本”，还不能说已经获得
端到端训练加速。

### 9.3 为什么必须 fresh-vs-fresh

历史 C0 controller 包含蒸馏、历史搜索和多轮训练，不是一次从头训练的纯 PPO。
若拿它直接对 fresh FlashSAC，会把算法、初始化和历史训练预算混在一起。

正式主比较必须固定：

- 同一个环境与 34-D observation；
- 同一个 4-D action；
- 同一训练任务池和 reset 随机流；
- 同一 transition 预算；
- paired training seeds；
- 相同评测任务；
- 同时报告含编译的 cold wall 和排除评测/存盘的 core wall。

### 9.4 为什么 FlashSAC 更新频率容易比较错

官方 GPU 配方在 1,024 个并行环境下，每个 vector step 收集 1,024
transitions，每步做 2 次 learner update，等价于每 512 transitions 做一次
update。

本项目固定 128 个并行环境。要保持同样 update density，应当每 4 个 vector
steps 做 1 次 update：

\[
4 \times 128 = 512\ \text{transitions/update}.
\]

若误设为每个 vector step 更新一次，就变成每 128 transitions 更新一次，是官方
密度的 4 倍。这个版本只能叫 `FlashSAC-highUTD` 消融，不能与 official-density
主结果混称。

---

## 10. 对研究目标的回答

### 10.1 是否已经统一 seed selection 和 controller

**在框架层面，已经统一。** Seed actor、真实 router/IK、controller rollout、
macro return、可选 controller forward update 已经位于同一可交替训练的 runner
中。它不依赖 diffusion，主训练数据入口也不会把 IKPool 候选 joint 交给 actor。

**在效果层面，只完成了一半。** Return-aware seed backward 已有明确正证据；
controller forward update 尚未在 matched 对照中证明额外收益。

### 10.2 是否已经摆脱候选池

部署主路径已经摆脱 IKPool 枚举和 selector。网络直接输出一个近似 seed，至多一次
IK 修正后执行。

但需要诚实区分两点：

- safe fallback 仍来自预先验证的共同 `q0_pilot`；
- P12 的 outcome gate 在训练期用多个固定 branch outcome 建立监督。

前者是安全兜底，后者是训练期监督成本；二者都不会在部署时枚举多个候选。但若要
声称“训练全过程也完全不依赖任何外部 IK 解”，还需要继续提高 raw DIRECT rate，
并从新任务生成流程中构造安全 fallback。

### 10.3 是否已经让网络直接求出 exact IK

没有。当前主结果 `DIRECT=0`，所以还不能写成“网络已经直接求出精确 IK”。

当前更准确的贡献是：

> 网络根据下游长程回报学习一个近似关节姿态，它能作为一次确定性 IK 的初始化；
> 若修正失败，系统安全回退；整个部署仍然只选择并执行一个 seed。

### 10.4 相对旧框架提升多少

这个问题必须按两代实验回答，不能只给一个混合数字。

**上一代候选式统一框架：** 在它自己的 10,000-task sealed 统一口径下，相对
original decoupled selector/controller 从 0.545694 m 提高到 0.547301 m，
提升 +1.606 mm，95% CI `[+0.418,+2.817] mm`。但提升主要来自 selector；
matched frozen-controller 为 0.547316 m，joint 相对它是 -0.015 mm，CI
`[-0.740,+0.716] mm`。

**本报告的直接生成式框架：** 已经在 validation 和 external-dev 的
P12 strict-safe 共同任务上完成与旧 IKPool+SetSel 的同 controller
head-to-head。P12 分别为 0.469842 m 和 0.477941 m，旧系统分别为
0.564730 m 和 0.582358 m，所以 P12 分别低 94.888 mm 和 104.418 mm；
paired 95% CI 为 `[-105.610,-84.463] mm` 和
`[-115.626,-92.799] mm`。这表明当前直接生成的任务质量还没有追平候选池。

在 Direct Seed 家族内部，把 exact P4 看作旧单分支基线，
P12-pruned-q15 相对 P4 的点估计为：

- grouped validation：+3.846 mm；
- external-dev：+3.744 mm；
- 历史 holdout v2：+5.088 mm。

P4 相对未训练 random actor 已经提高 45.596 mm；P12 在 P4 之上进一步取得上述
小幅增益。它说明主要收益首先来自 return-aware seed 学习，多分支 gate 是在强
Direct Seed 基线上再挖掘少量互补性。不能把这组三个数字改写成“Direct Seed 已经
比 original decoupled IKPool 系统高 3.846--5.088 mm”；同口径结果恰好说明
它目前仍明显更低。

### 10.5 当前最值得投入的下一步

按科学优先级排序：

1. **先正视并缩小对旧 IKPool+SetSel 的 95--104 mm 质量差距。** 新设计只能在
   training/model-select 范围继续优化，不能再用已查看的 validation/external
   追逐配置；若短期无法缩小差距，应把 Direct Seed 定位为低规划成本 Pareto 点，
   而不是全面替换。
2. **改进 task-only gate 的跨域可辨识性，而不是盲目增加容量。** 优先研究只由
   任务几何一次前向即可得到的表示、cross-fitting、保守校准和不确定性约束；
   不能为选择 branch 生成多个 seed、运行多个 IK 或执行 controller probe。
3. **提高 raw DIRECT exact rate。** 让 seed 网络更接近直接产生严格合法 IK，
   同时保留下游 return，避免退回纯几何优化。
4. **改进 forward learner。** FlashSAC 是候选之一；必须用 matched frozen
   feedback 证明 controller 更新确实让后续 seed 学习受益。
5. **做多个独立训练种子。** 当前任务 bootstrap 不能覆盖重新训练的随机性。
6. **方法真正冻结后建立全新 sealed set。** branch、quota、router 和所有
   阈值冻结后，不再查看既有开发域调参，并在新 sealed 上同时比较 P4、P12 和
   旧 IKPool+SetSel。
7. **独占 GPU 做正式计时。** 分别测 online `B=1` 和 batched throughput，
   不能用共享 GPU 的单次结果作速度结论。

### 10.6 适合对外汇报的严谨表述

可以这样总结：

> 我们研究 Franka FR3 笔尖沿无限射线持续运动的长时域任务。由于七自由度冗余，
> 同一初始笔尖位姿对应多个关节姿态，而不同姿态会显著改变后续 controller 的
> 可操作空间。我们把原先“先枚举 IK 候选、再选 seed”的流程改成一个 Direct
> Seed 强化学习框架：网络根据任务直接生成一个近似七维关节角，至多经过一次
> 确定性 IK 修正，并只执行一次 controller rollout。Return-aware backward
> training 已经显著优于随机初始化；当前 P12 hard-gated specialist 在不增加
> 候选枚举、IK 次数或 controller probe 的条件下，相对 exact P4 在三个已查看
> 数据域取得 +3.846、+3.744 和 +5.088 mm 的点估计提升。另一方面，在相同 C0
> 与相同 strict-safe 任务上，它仍比旧 IKPool+SetSel 低 94.888 mm 和
> 104.418 mm。Direct Seed 因而显著降低了规划结构成本，但当前不是旧候选池系统的
> 质量等价替代。所有 P12 结果仍属于 development evidence；下一步应在不继续
> 使用已查看开发域调参的前提下缩小质量差距，之后再做全新 sealed、多训练种子和
> 独占 GPU 验证。

不应当这样总结：

> “已经直接生成精确 IK”“已经接近 oracle”“联合训练已经优于冻结 controller”
> 或“FlashSAC 已经总体优于 PPO”。

## 11. 外部参考与实现来源

1. Yuanpei Chen、Chen Wang、Li Fei-Fei、Karen Liu，
   [*Sequential Dexterity: Chaining Dexterous Policies for Long-Horizon
   Manipulation*](https://proceedings.mlr.press/v229/chen23e.html)，CoRL 2023。
   本项目借鉴其“用后一阶段的可行性反向改善前一阶段、并让相邻阶段逐步适配”的
   动机，不声称复现了论文的灵巧手技能链。
2. Holiday Robotics，
   [FlashSAC 官方实现](https://github.com/Holiday-Robot/FlashSAC)。本项目接入时
   固定以下 upstream commit，保留 MIT license，并把本地环境适配与算法核心
   区分记录：

   ```text
   87edc9061150ae9e962dd84e6544e27a1554b3ab
   ```

---

# 第二部分：技术协议、完整结果与复现附录

以下内容保留完整实验协议和开发过程。任何未在协议或对应 freeze manifest 中预先
声明的超参数、筛选规则和指标，不能在读取新 sealed 结果后补充。本文记录的
grouped validation、external-dev 和历史 holdout 均已被查看，不能重新包装成
sealed evaluation。

---

## 附录 A：不能改变的部署与科学口径

### A.1 共同部署约束

所有主方法均满足：

```text
一个任务
  -> 一次 seed 模型前向
  -> 最终只提交一个 seed
  -> 一次真实 controller rollout
```

主结果中禁止：

- controller probe；
- model-based trajectory rollout；
- 试运行多个 seed 后择优；
- 根据真实执行结果重新选择 seed。

Direct Seed 允许一次确定性数值 IK refinement。它属于初始化求解成本，
必须单独计时，不属于 controller rollout。

### A.2 两条线先独立，最后才组合

- FlashSAC 对 PPO 时固定相同的初始 seed，避免 seed 方法变化混入 controller
  算法差异。
- Direct Seed 对 IKPool 时固定同一个 C0 controller，避免 controller
  变化混入 seed 方法差异。
- 只有两条线各自在 development protocol 中冻结以后，才允许补充
  `Direct Seed + FlashSAC` 的组合结果。
- 组合结果不能反过来用于选择 FlashSAC 超参数或 Direct Seed 模型。

### A.3 历史 C0 不能冒充 fresh PPO

`runs/r2_grouped_best` 是包含蒸馏、历史搜索和多轮训练的生产 controller，
不是一次从头训练的纯 PPO 结果。

- controller 算法主对比必须是 **fresh PPO vs fresh FlashSAC**；
- 两者使用同一环境、训练任务分布、transition 预算和训练随机种子；
- 历史 C0 只用于 Direct Seed 主线及可选 warm-start 实用性实验。

---

## 附录 B：数据、随机种子与污染纪律

### B.1 已有数据的角色

| 数据 | 数量 | 允许用途 |
|---|---:|---|
| IKPool full train | 18,432 | 训练、内部 model selection、内部 calibration |
| grouped validation | 2,048 | development；可重复查看，不是 final |
| external-dev | 2,048 | development confirmation；不是 final |
| `ikpool_sealed_v1` | 10,000 | 历史结果；新方法禁止用它做选择或新 final claim |

`ikpool_sealed_v1` 已经被读取。即使新方法现在才在其上运行，也只能标为
post-hoc diagnostic，不能称为 sealed evaluation。

### B.2 Direct Seed 内部三段 split

对 18,432 个训练任务使用 `(p0, line_dir, n_target)` 的 **exact float32
byte signature** 做 geometry group，不允许 row-level split，也不允许先转
float64 或四舍五入后再分组。

固定分为：

- fit：约 70%，只用于拟合权重和归一化统计；
- model-select：约 15%，只用于架构、损失权重、refinement 预算和 fallback
  gate 的选择；
- calibration：约 15%，只对完全固定的方法做一次 pass/fail 审计。

建议在 freeze manifest 中固定：

```text
split_seed = 2026072801
generator_train_seeds = [61000, 62000, 63000, 64000, 65000, 66000]
```

若 calibration 失败，该候选回滚。继续在同一个 calibration split 上反复
修改并尝试的方法，必须把该 split 降级为 development，并另建 confirmation
split。

### B.3 Controller 训练与随机流

FlashSAC 和 PPO 使用六个 paired training seeds：

```text
controller_train_seeds = [71000, 72000, 73000, 74000, 75000, 76000]
```

每个 run 至少拆分并记录以下 RNG stream：

- task/reset sampling；
- actor exploration；
- replay sampling 或 PPO minibatch permutation；
- network initialization；
- evaluation（固定且不得进入 replay buffer）。

同一 paired seed 下，PPO 和 FlashSAC 使用相同的训练任务 pool 和 reset
sampling seed。算法性能不同会导致 reset 消耗时间不同，这是环境交互的一部分，
不应通过复制 trajectory 强行消除。

### B.4 新 sealed set

两条方法完全冻结后，生成一个新的 10,000-task sealed set。它必须满足：

- 10,000 个 exact-float32 唯一 geometry，内部重复为 0；
- pool seed、task seed、diffusion seed、IK generation seed 全部为新值；
- 与所有 train、model-select、calibration、validation、external-dev、
  pilot、smoke、systematic 10k、历史 fresh holdout、sealed v1/v2、
  Direct Seed debug 和 FlashSAC eval cache 的 exact geometry overlap 为 0；
- 生成前已经保存两个方法的 freeze manifest、代码 SHA、模型 SHA、配置 SHA
  和发布训练种子选择规则；
- sealed 生成后不得修改主模型、gate、IK iteration budget、评价指标或统计方法。

同一个新 sealed geometry set 可以支持两项正交评价：

- controller 对比：PPO 与 FlashSAC 从相同的预生成 `q0_pilot` 开始；
- seed 对比：Direct Seed 与 IKPool 都由冻结 C0 执行。

这样可以节省几何生成成本，但两个比较仍保持单变量控制。

---

## 附录 C：Track A——FlashSAC vs PPO 完整协议

### C.1 主问题与三种结论

分别检验，不得混称：

1. **墙钟效率更高：** 达到同一预注册进度阈值所需训练秒数更少；
2. **样本效率更高：** 达到同一阈值所需真实 environment transitions 更少，
   或固定 transition 区间的 learning-curve AUC 更高；
3. **最终结果更好：** 相同 30M transitions 的最后 checkpoint 表现更好。

只观察到训练更快，不能写成 sample-efficient；只观察到较早学习更快，
不能写成 asymptotic performance 更好。

### C.2 官方实现边界与版本固定

使用用户指定的官方实现：

```text
https://github.com/Holiday-Robot/FlashSAC
```

首次接入时必须记录：

- 完整 commit SHA；
- upstream license；
- vendored 文件列表；
- 本项目为适配 `NSRLBatchedEnv` 所做的 patch；
- Python、PyTorch、CUDA、driver 和 `torch.compile` mode。

agent、critic、target update、distributional critic、temperature、reward
normalizer 和 replay 逻辑尽量保持官方代码。环境 adapter 不得重写算法核心。

### C.3 相同环境条件

两臂固定：

```text
n_envs = 128
obs = 相同 34-D observation（observe_ray_error=true）
action = 相同 4-D tanh-bounded action
gamma = 0.99
max_steps = 500
train pool / q0 / reward / termination = 完全一致
device = 同一张独占 GPU
```

GPU SVD 存在 batch-dependent 浮点差异，所以 `n_envs=128` 是协议的一部分。
若另做 256/512/1024 throughput sweep，只能作为吞吐消融，不能混入主质量表。

环境 adapter 必须正确处理 auto-reset：

- replay 中的 `next_obs` 对 episode 结束那一步使用 `terminal_obs`，不能使用
  reset 后的新任务 observation；
- `terminated` 的 TD target 不 bootstrap；
- `truncated` 使用 terminal observation bootstrap；
- evaluation transitions 永不进入 replay buffer 或 normalization statistics。

### C.4 PPO 主臂

fresh PPO 采用当前强配方，至少锁定：

```text
total_timesteps = 30_000_000
n_steps = 32
n_minibatches = 32
update_epochs = 10
learning_rate = 3e-4
gamma = 0.99
gae_lambda = 0.95
clip_coef = 0.2
target_kl = 0.02
hidden_dim = 512
normalize_returns = true
```

任何从历史 C0 加载的 run 都必须另标为 `warm-start`，不能放入 fresh
algorithm 主表。

### C.5 FlashSAC 主臂：保持官方 GPU update density

官方 GPU 脚本使用：

```text
1024 envs/vector step
2 learner updates/vector step
sample_batch_size = 2048
CUDA replay buffer
AMP on
```

官方更新密度为：

```text
rho_update = 2 / 1024 = 1 / 512 learner update / environment transition
```

因此在本项目固定 `n_envs=128` 时，FlashSAC **主臂**必须使用：

```text
每 4 个 vector steps 做 1 次 learner update
```

即：

```text
128 transitions/vector step
1 update / (4 * 128 transitions) = 1/512 update/transition
```

不要把 `1 update/vector step` 当成官方等密度配置。它在 128 envs 下是
官方 GPU update density 的 4 倍，只能作为 `FlashSAC-highUTD` 消融。

FlashSAC 主配置：

```text
sample_batch_size = 2048
buffer_device = cuda
buffer_min_length = 100_000
buffer_max_length = 10_000_000
AMP = on
actor_update_period = 2
target_tau = 0.01
gamma = 0.99
n_step = 3
compile_mode = auto（实际值写入日志）
update_interval_vector_steps = 4
updates_on_tick = 1
```

这里的 `n_step=3` 跟随官方 `scripts/run_isaaclab.sh` GPU 配方。配置文件 base
值或部分其他模拟器脚本使用的 `n_step=1` 只能作为消融，不能替代正式主臂而仍
标为本协议的 official-density 配置。

`FlashSAC-highUTD`：

```text
update_interval_vector_steps = 1
updates_on_tick = 1
```

它等价于 `1/128 update/transition`，并使 replay samples drawn per new
transition 从主臂的 4 增至 16。该臂用于判断质量差异是否仅来自更多 learner
compute，不作为默认 FlashSAC。

### C.6 训练预算和 checkpoint

所有正式 run 训练到 **30M environment transitions**。固定评价点：

```text
[0, 0.1M, 0.25M, 0.5M, 1M, 2M, 5M, 10M, 20M, 30M]
```

environment transitions 的定义为：

```text
env_transitions = vector_steps * n_envs
```

不能把 vector step 写成 environment step。FlashSAC replay 重复采样也不能
增加 environment transitions。

最终结果使用 30M 的 **last checkpoint**，禁止在 final/sealed 上选择 best
checkpoint。

### C.7 分阶段执行和 promotion gate

### A0：adapter 正确性

必须通过：

- action shape/range 和 observation shape 测试；
- 10,000 个 scripted transitions 的 env adapter 与原 env step 输出一致；
- terminal/truncation target 单元测试；
- replay 不含 reset 后 observation 作为 terminal transition 的 next state；
- 100k-transition smoke 无 NaN/Inf，checkpoint 可保存和恢复；
- 固定 checkpoint 的 deterministic evaluation 可重复。

任一失败则停止算法比较。

### A1：单种子 2M pilot

运行：

- fresh PPO；
- FlashSAC 官方等密度主臂；
- FlashSAC-highUTD。

为缩短首次接入的等待，A1 smoke/pilot 可临时使用
`buffer_min_length=10_000`。该 run 只能验证 adapter、数值稳定性和吞吐；
所有 A2/A3 正式比较必须恢复官方 GPU 口径
`buffer_min_length=100_000, buffer_max_length=10_000_000`。

pilot 只用于发现实现错误和确定吞吐，不能作为论文效果结论。进入多种子阶段的
最低条件：

- 无数值崩溃；
- deterministic policy 明显优于随机初始化；
- replay size、update count、samples drawn 与理论值一致；
- 计时字段完整，三次短 throughput replay 的偏差不超过 10%。

### A2：六种子 5M gate

三臂运行六个 paired seeds 到 5M。FlashSAC 主臂若满足下列任一条件则继续
到 30M：

1. median progress 不低于 PPO 5M 超过 5 mm，且 core wall 不高于 PPO；
2. 达到任一预注册阈值所需墙钟显著少于 PPO；
3. 0--5M transition-normalized AUC 的 paired seed bootstrap CI 下界大于 0。

即使不通过，也保留至少 5M 的负结果；不能通过改为 highUTD 后仍称“官方主配置”。

### A3：六种子 30M 正式 development

主表报告所有六个种子。发布 checkpoint 的规则在 sealed 前固定为：

```text
按 validation 的 0--30M AUC 排序，选六个 run 的中位 run；
并列时选择较小 train seed。
```

算法级显著性使用 paired training-seed bootstrap；单个发布模型的任务差异使用
geometry bootstrap。两类不确定性不能混成一个普通 row bootstrap。

### C.8 学习曲线和效率指标

预先在固定 eval set 上计算：

```text
P_init = 0-transition deterministic actor performance
P_ref  = 历史 C0 在同一固定 seed/task set 上的 performance
```

阈值固定为：

```text
T50 = P_init + 0.50 * (P_ref - P_init)
T75 = P_init + 0.75 * (P_ref - P_init)
T90 = P_init + 0.90 * (P_ref - P_init)
```

主指标：

- transitions-to-T50/T75/T90；
- core-wall-seconds-to-T50/T75/T90；
- normalized AUC over 0--5M transitions；
- normalized AUC over 0--30M transitions；
- wall-clock AUC 到 paired run 中较快方法到达 30M 的时间；
- final 30M mean progress。

若某方法未达到阈值，按 right-censored 报告，不得用最后一个点伪造到达时间。

### C.9 最终 controller 结果

每个 evaluation checkpoint 保存逐任务结果，并至少报告：

- mean / median progress；
- 5% two-sided trimmed mean；
- ±50 mm clipped paired delta；
- mean episode length；
- `collision/cone/joint-limit/lateral/truncated` termination rate；
- 相对 PPO 的 `>1 mm` harm 和 win rate；
- geometry-bootstrap 20k 次 95% CI。

结论门槛：

- **sample-efficient：** 0--5M AUC paired-seed CI 下界大于 0，或至少两个
  threshold 的 transitions-to-target ratio CI 上界小于 1；
- **wall-clock-efficient：** 至少两个 threshold 的 core-time ratio CI
  上界小于 1；
- **final superior：** 30M progress delta 的 hierarchical CI 下界大于 0；
- **final non-inferior：** 预注册 margin 为 5 mm，CI 下界大于 −5 mm。

---

## 附录 D：Track B——Direct Seed Generator 完整协议

### D.1 方法契约

Direct Seed 模型接收一个任务：

```text
c = (p0, line_dir, n_target)
```

主方法是一步 contextual RL，而不是 diffusion 或 IKPool 标签回归。连续
tanh-Gaussian seed actor 输出一个 7-DoF 关节向量：

```text
q_hat ~ pi_seed(q | c)                  # 训练时单样本探索
q_hat = mean(pi_seed(. | c))            # 部署时确定性单解
```

`q_hat` 经固定 router 执行一次完整 controller episode。该真实 progress 是
twin macro-Q 的 Monte-Carlo target；没有 seed-stage Bellman bootstrap。
actor 通过 conservative `min(Q1,Q2)` 的梯度、可微 FK 约束以及成功
refinement 的 self-distillation 更新。Q、FK 和 self-distillation 都作用于
部署时同一个 deterministic mean；随机性只用于收集，每 50k 个真实 macro
rollouts 从 1.0 线性退火到 0.05。每个 backward phase 更新 seed actor，
每个 forward phase 冻结 seed actor、在其诱导的 reset 分布上更新 controller。

部署时固定：

```text
generator_forwards = 1
generated_candidates = 1
resampling = 0
controller_probes = 0
model_rollouts = 0
```

训练时 stochastic actor 每个任务也只能取一个样本并产生一次真实 rollout。
不能生成多个样本再用 return model 选择，否则又退化为候选池方法。部署一律
使用 actor deterministic mean。

IKPool return-weighted soft-support 训练只允许作为显式 bootstrap/ablation，
不得作为 Direct Seed 主方法，也不得据此声称“RL 自己学会生成”。

### D.2 严格接受、refinement 与 fallback

先对 `q_hat` 做 FK 和物理验证。直接接受的条件为：

- finite；
- 关节限制内，最小 joint margin ≥ 0.02 rad；
- TCP position error ≤ 5 mm；
- tool-axis cone angle ≤ 30°；
- 无 self-collision。

若直接通过，则不运行 IK，使用 `q_hat`。

若未通过且同时满足 `input_valid & finite & joint_limits`，才允许一次
deterministic DLS refinement；不满足这些可投影前提的样本直接走共同 fallback：

```text
initial q = q_hat
preserve_seed = true
damping = 1e-4
max_iters = 50
random restart = 0
```

现有 DLS 的 z-axis 收敛容差是 5°。为保证最坏情况下仍满足 30°硬约束，
refinement target tool direction 是 `FK(q_hat).z` 在 **24.5° cone** 内的
确定性投影：

- 已在 24.5° 内则保持该方向；
- 超出时，沿 `n_target` 与 `FK(q_hat).z` 的最短球面弧投影到 24.5°边界；
- 当球面切向量退化时，当前代码依次使用固定 world-x/world-y 参考轴构造确定性
  fallback。`line_dir` 只参与补全旋转矩阵的 x 轴，而 z-axis IK 实际使用的目标
  是上述 z 列。当前 `projection_config` 只保存数值参数；正式 freeze manifest
  还应显式记录 tie-break 规则或对应代码 SHA。

refinement 输出重新经过同一严格验证。失败时使用共同的、预先验证的
`q0_pilot` fallback。

当前 IKPool train 的 `q0_pilot` 按项目实际保守限位和 0.02 rad margin
重新验证后，固定共同集为 17,588/18,432；844 条仅因 joint margin 不足被
排除。kept/excluded geometry-list SHA256 为：

```text
kept     9069774f3fee8f940f2c661e025e22499819b08b1a9b4251c5e32314907fe85e
excluded d99aa3b9b253ee0ae5a76ef369dd432ff049a535eca09931d01f829b639ebb4e
```

正式程序仍必须用实际 kinematics 动态重算并写 manifest，不能硬编码数量。
新 sealed builder 必须从源头保证 fallback 通过相同门槛。

**主方法禁止直接执行未通过严格验证的 approximate q。** 这样避免：

- 初始 position residual 被错误计为“免费进度”；
- lateral safety-net 的任务定义变化；
- 不同方法实际执行了不同的 Cartesian 起点。

“近似 IK 促进下游 IK”的证据由 refinement convergence、迭代次数和最终进度
证明，而不是放宽执行安全约束。

### D.3 比较臂

### B0：严格 direct

`Direct-0IK`：只接受原始 `q_hat`；否则走共同 fallback。它测量 generator
自身直接输出精确 IK 的比例。

### B1：等数值 IK 预算

对每个任务先由 generator 确定同一个 projected tool target，然后比较：

- `Random-IK1-shared-target`：一个固定随机 q 初始化，一次 50-iter DLS；
- `Direct-IK1`：`q_hat` 初始化，同一个 target，同一次 50-iter DLS。

两者使用相同 solver、target、iteration cap 和 fallback。这是判断 learned
approximate seed 是否确实促进 IK 收敛的最干净因果对比。

### B2：完整系统替换

比较：

- `Direct-IK1 + C0`；
- `IKPool128 + SetSel + C0`。

两者都只执行一个 seed 和一次 C0 rollout。差别是初始化计算：

- Direct：一次网络前向，至多一次 50-iter IK；
- IKPool：16 orientations × 8 restarts，即 128 个 IK attempts，去重/FPS 到
  最多 K=32 个生成式候选，再追加共同安全 fallback，最后做一次 selector 前向。

### B3：初始化计算 Pareto

IKPool 按固定 attempt 顺序评测：

```text
N_attempts in [1, 4, 8, 16, 32, 64, 128]
```

每个预算重新锁定有效 mask，并只在 model-select split 训练对应 selector。
Direct 点与整条 `latency--progress` Pareto 曲线比较，而不只与最昂贵的 128
attempt 点比较。

### D.4 Contextual RL 训练公平性

主方法从 task cache 只读取：

- `(p0, line_dir, n_target)`；
- 一个预先锁定并通过 strict gate 的 `q0_pilot` fallback；
- task index 与 geometry fingerprint。

actor 和 critic不得读取 IKPool 的其它候选 `q`、selector score 或 complete-
candidate return table。每个真实 backward 样本固定为：

```text
c -> one stochastic q_raw -> fixed router -> one controller rollout
  -> (c, q_raw, q_projected, fallback, route, progress)
```

macro replay 可以多次复用这些真实样本；必须分别报告真实 controller rollouts、
replay samples、critic updates 和 actor updates，不能把 replay 重采样计为新
环境样本。critic 使用训练期 shaped macro target：

```text
progress - 0.002m * I[REFINE] - 0.010m * I[FALLBACK/INVALID]
```

原始 progress 始终单独记录并作为最终质量指标；以上小代价只防止 REFINE/FALLBACK
掩盖 raw seed 不准确。actor 在 deterministic mean 上最大化 conservative
macro-Q，并使用：

- differentiable FK position/cone/joint-margin/self-collision precision；
- successful REFINE `q_projected` 到 actor mean 的 self-distillation；
- 所有 context 的 deterministic-mean FK precision，以及
  FALLBACK/INVALID context 的额外 precision correction；
- 不模仿外部 fallback；behavior anchor 默认关闭；
- controller forward phase 改变下游策略后清空旧 macro replay，避免旧回报
  与新 controller 的 target 非平稳性。

IKPool soft-support、best-return teacher 或 diffusion 初始化只能作为独立消融。
使用它们时必须报告 offline IK attempts、teacher controller labels 和 GPU-hours，
且方法名标注 `bootstrap` 或 `amortized`。

### D.5 Direct Seed 分阶段 gate

### B0：500-task smoke

必须满足：

- 0 个非法 seed 被送入 controller；
- fallback 路径逐位复现；
- raw/refined position、cone、collision、joint-margin 统计完整；
- 同一个模型和任务 seed 可逐位复现 q；
- timing 计数与实际 solver iteration 一致。

### B1：contextual-RL 学习信号 gate，三个训练种子

先在固定 common-set 上检查随真实 macro rollouts 增加：

- raw DIRECT rate 上升；
- REFINE/FALLBACK rate 下降；
- raw position/cone/collision/joint-margin 分位数改善；
- frozen-controller progress 不退化。

同时做去除 macro-Q、去除 FK precision、去除 projection self-distillation
三个消融，确认“下游回报”和“IK 精度自提升”各自承担真实作用。

随后在 3,000-task internal held-out 上，`Direct-IK1` 相对
`Random-IK1-shared-target` 至少满足一项：

- refinement convergence-rate paired CI 下界大于 0；
- mean IK iterations paired CI 上界小于 0；
- downstream progress delta CI 下界大于 0 且 trimmed delta > 0。

全部失败则“learned approximate seed helps IK”假设不成立，停止 full-scale
promotion。

### B2：训练量 scaling

固定模型族，在**真实 macro controller rollouts**数量：

```text
[1k, 2k, 5k, 10k, 20k, full budget]
```

各跑三个 seeds，报告 replay ratio、raw exact rate、post-refinement exact
rate、fallback rate和 downstream progress。若指标走平，优先检查 replay
利用率、critic calibration和 actor extrapolation，而不是读取更多 development
数据调 gate。

### B3：validation + external-dev

完整六种子模型相对 `IKPool128 + SetSel` 的替换 gate：

- mean progress delta 的 geometry-bootstrap CI 下界 > −5 mm；
- 5% trimmed delta > −5 mm；
- ±50 mm clipped delta > −5 mm；
- 两个 development set 都满足，而不是平均后掩盖一个 split 的退化；
- 执行 seed physical-invalid 数为 0；
- online B=1 seed-stage median latency 至少快 4×，或不超过 IKPool128 的 25%。

通过该 gate 可称为“低成本、结果 non-inferior 的直接生成替代”。只有 mean
progress delta CI 下界 > 0 时才称为“结果 superior”。

### B4：六种子与发布规则

报告全部六个 generator seeds。发布模型在 sealed 前按：

```text
validation 与 external-dev 的 mean paired delta 取平均，
选择六个 run 的中位 run；并列时取较小 seed。
```

不得选择 development 最好 seed。

### D.6 Direct Seed 指标

### IK 与安全

- raw strict-exact rate；
- post-refinement strict-exact rate；
- fallback rate；
- raw/post position error p50/p90/p95/p99；
- raw/post cone angle p50/p95/p99；
- collision rate、joint-margin violation rate；
- DLS convergence rate；
- active DLS iterations mean/p50/p95；
- 真实 macro controller rollouts、replay samples、critic/actor updates；
- macro-Q calibration error 与 twin-Q disagreement；
- DIRECT/REFINE/FALLBACK 随训练样本数的学习曲线；
- 与最近训练 IK 解、同任务 IKPool 解的 joint-space distance；
- 新 branch/novel seed rate。

### 下游结果

- final progress mean / median / 5% trimmed；
- paired ±50 mm clipped delta；
- `>1 mm` harm / win rate；
- episode length 和 termination modes；
- 20k geometry-bootstrap 95% CI。

IKPool complete-candidate best 只能标为 **IKPool reference oracle**。Direct
generator 有可能生成池外更好 seed，因此相对它的 capture 允许超过 100%，但
不能把该参考值称为全局 IK oracle。

### 初始化成本

分别测量：

- neural forward；
- FK/strict validation；
- DLS refinement；
- collision validation；
- fallback；
- IKPool enumeration；
- FPS；
- SetSel forward；
- 总 seed-stage latency。

报告：

- online batch `B=1` 的 p50/p95/p99 latency；
- batched evaluation `B=512` 的 tasks/s 和每任务均摊时间；
- 实际 IK attempts、active DLS iterations、FK/Jacobian calls；
- peak CUDA memory。

计时前 warm up 100 batches，每个区段前后 `torch.cuda.synchronize()`，同一张
GPU 上不得有并发训练任务。

---

## 附录 E：统一计时口径

每个 controller run 同时报两种墙钟：

### E.1 Cold end-to-end wall

从进程启动到最后 checkpoint 落盘，包括：

- import；
- env/cache load；
- CUDA context；
- `torch.compile`；
- training；
- evaluation；
- checkpoint serialization。

用外层 `/usr/bin/time -v` 记录。

### E.2 Core training wall

只累计 environment collection 和 learner update：

```text
core_train_s = env_step_s + learner_update_s
```

暂停计时器以排除：

- evaluation；
- checkpoint/save；
- logging flush；
- 一次性 pool construction。

每个 GPU 段落前后同步 CUDA。另报：

```text
startup_s
compile_s
pool_load_s
env_step_s
learner_update_s
evaluation_s
checkpoint_s
logging_s
```

不允许只报去掉 compile 的 FlashSAC 时间、却给 PPO 报完整时间。论文主表同时
展示 cold end-to-end 与 core training。

### E.3 吞吐与样本使用

必须区分：

```text
environment transitions
vector steps
completed episodes
learner updates
optimizer samples consumed
replay samples drawn
```

其中：

```text
FlashSAC optimizer_samples = learner_updates * sample_batch_size
PPO optimizer_samples = 每个实际 minibatch 的样本数之和
```

主 FlashSAC 在 steady state 下理论上：

```text
learner_updates / env_transitions = 1/512
replay_samples / env_transitions = 2048/512 = 4
```

`FlashSAC-highUTD` 理论上为：

```text
learner_updates / env_transitions = 1/128
replay_samples / env_transitions = 16
```

日志中的实测比率必须与理论值一致；否则 run 不合格。

---

## 附录 F：必需日志字段

每个 run 保存一个 append-only `metrics.jsonl`。公共字段至少为：

```text
run_id
timestamp
phase
method
algorithm
train_seed
task_rng_seed
exploration_rng_seed
network_rng_seed
git_sha
git_dirty_diff_sha
flashsac_upstream_sha
config_sha256
train_cache_sha256
eval_cache_sha256
checkpoint_sha256
split_fingerprint_sha256
hostname
gpu_name
gpu_uuid
driver_version
cuda_version
torch_version
python_version
compile_enabled
compile_mode
amp_enabled
n_envs
obs_dim
act_dim
```

controller 训练每个日志点：

```text
vector_steps
env_transitions
completed_episodes
unique_task_ids_seen
learner_updates
optimizer_samples
replay_size
replay_samples_drawn
sample_batch_size
update_interval_vector_steps
updates_on_tick
core_train_s
cold_e2e_s
startup_s
compile_s
pool_load_s
env_step_s
learner_update_s
evaluation_s
checkpoint_s
logging_s
transitions_per_core_s
updates_per_core_s
max_cuda_memory_bytes
reward_normalizer_state_hash
```

evaluation 日志：

```text
eval_at_transitions
eval_at_core_s
eval_at_e2e_s
n_eval_tasks
mean_progress_m
median_progress_m
trimmed5_progress_m
mean_episode_len
term_collision_pct
term_cone_pct
term_joint_limit_pct
term_lateral_pct
term_truncated_pct
```

Direct Seed 每个 aggregate 日志：

```text
generator_forwards
generated_candidates
raw_exact_rate
refine_attempt_rate
refine_success_rate
fallback_rate
invalid_executed_count
mean_active_ik_iters
p95_active_ik_iters
mean_position_error_pre_m
p95_position_error_pre_m
mean_position_error_post_m
p95_position_error_post_m
mean_cone_angle_pre_deg
p95_cone_angle_post_deg
ik_attempts_total
fk_calls_total
jacobian_calls_total
generator_forward_s
strict_validation_s
ik_refinement_s
fallback_s
seed_stage_total_s
```

同时保存逐任务 `.npz`，至少包含：

```text
task_id / geometry fingerprint
p0 / line_dir / n_target
selected_or_generated_q
used_raw_q
refinement_attempted
refinement_converged
active_ik_iters
fallback_used
position_error_pre/post
cone_angle_pre/post
collision_free
joint_margin
final_progress_m
episode_len
term_reason
```

只保存 aggregate JSON 而不保存逐任务数组的 run，不能用于 paired CI。

---

## 附录 G：正式实验的最终报告表

### G.1 Controller 主表

| 方法 | 0--5M sample AUC | 到 T75 transitions | 到 T75 core wall | 30M progress |
|---|---:|---:|---:|---:|
| fresh PPO |  |  |  |  |
| FlashSAC official-density |  |  |  |  |
| FlashSAC-highUTD |  |  |  |  |

另表报告 cold end-to-end、core wall、transitions/s、optimizer samples 和显存。

### G.2 Seed 主表

| 方法 | IK attempts | strict success | seed-stage B=1 p50 | final progress |
|---|---:|---:|---:|---:|
| Random-IK1 shared target | 1 |  |  |  |
| Direct-0IK | 0 |  |  |  |
| Direct-IK1 | ≤1 |  |  |  |
| IKPool128 + SetSel | 128 |  |  |  |

另画：

- progress vs IK attempts；
- progress vs measured B=1 latency；
- refinement success vs training tasks；
- paired Direct minus IKPool progress distribution。

---

## 附录 H：主要污染和误判风险清单

1. **用历史 C0 代表 PPO。** 必须 fresh-vs-fresh。
2. **把 vector steps 当 transitions。** 统一乘 `n_envs`。
3. **FlashSAC update density 算错。** 128 env 主臂是每 4 vector steps 一次
   update；每步一次是 4× highUTD。
4. **terminal next_obs 错用 reset obs。** 会污染 off-policy TD target。
5. **eval transition 进入 replay 或 normalizer。** 必须物理隔离。
6. **只给 FlashSAC 去掉 compile 时间。** cold 与 core 两套时间同时报。
7. **并发 GPU 或不同 batch/chunk。** 会同时污染速度和 GPU-SVD 数值。
8. **在 sealed 上挑 best checkpoint/seed。** 最终只用预注册 last checkpoint
   和 median-run publication rule。
9. **重复查看 external 后仍称 confirmation。** 重复使用后它就是普通 dev。
10. **新方法读取旧 sealed。** 只能 post-hoc，必须生成新 sealed。
11. **直接执行 approximate q。** 可能改变任务起点或制造免费进度；主线禁止。
12. **fallback 掩盖 generator 失败。** intent-to-treat 与 no-fallback conditional
    指标都报，主结论使用 intent-to-treat。
13. **把 IKPool reference oracle 称为全局 oracle。** Direct 可能产生池外解。
14. **Direct teacher 使用枚举标签却声称训练阶段无需 IK。** 应称 amortized
    deployment。
15. **在 C0 returns 上训练 generator 后直接宣称对 FlashSAC 通用。** selector/
    generator 的 return supervision 与 controller 绑定；跨 controller 必须单独
    验证，必要时重建 return labels。

---

## 附录 I：推荐执行顺序

```text
A0 Flash adapter correctness
  -> A1 2M single-seed smoke
  -> A2 5M x 6-seed gate
  -> A3 30M formal controller development

B0 Direct 500-task smoke
  -> B1 IK1 causal test
  -> B2 training-size scaling
  -> B3 validation + external
  -> B4 six-seed publication rule

A、B 均冻结
  -> 写两个 freeze manifests
  -> 生成全新 10k sealed set
  -> 一次性运行正交主比较
  -> 可选 Direct + FlashSAC 组合 cell
```

如果某条线没有通过 promotion gate，保留负结果并停止扩大 sealed 评测；不得为了
论文叙事临时降低门槛。

---

## 附录 J：截至 2026-07-28 的完整 development 实验结果

本节记录当前实现和开发集结果，不改变前述正式实验协议。所有数值均为
development 实验，不能替代多训练种子、独占 GPU 和新 sealed set 的正式结果。

### J.1 已实现的统一 Direct Seed 路径

当前主路径为：

```text
9-D task geometry
  -> contextual tanh-Gaussian actor 的 deterministic mean
  -> 一个 7-D q_raw
  -> DIRECT / 至多一次 deterministic IK REFINE / safe FALLBACK
  -> 一次真实 controller rollout
  -> 完整下游 progress 反向训练 macro-Q 与 seed actor
  -> 可选 PPO controller forward phase
  -> controller 更新后清空旧 macro-return replay
```

实现满足：

- 不依赖 diffusion；
- 不枚举、排序或 probe 多个 seed；
- 历史 IKPool 只提供任务几何和显式安全 fallback，候选 joint 在数据入口被丢弃；
- 部署时始终只生成一个 seed，最多一次 IK，恰好一次 controller rollout；
- actor 的 Q、FK 精度和自蒸馏全部作用于部署使用的 deterministic mean；
- 采集噪声只存在于训练期，并可退火到 0；
- controller 更新后清空旧回报，避免用旧 controller 的 return 更新新 seed；
- resume 会显式记录 loss-config 和 optimizer-LR 覆盖，不再静默忽略 CLI 学习率；
- 可选的 precision-only update 只在训练期调用可微 FK/约束，不增加部署成本。

### J.2 fresh PPO vs official-density FlashSAC：2M 单种子 pilot

两侧使用相同 34-D observation、128 environments、训练任务分布和 200 个固定
评测任务。固定任务与 FlashSAC upstream commit 的 SHA-256/SHA 分别为：

```text
evaluation tasks ef9d58ffd39349cb55c830f898ea94e4cd8cbb067406579f0c2a09ae01e8c4ca
FlashSAC commit 87edc9061150ae9e962dd84e6544e27a1554b3ab
```

FlashSAC 核心来自用户指定的官方仓库（MIT）；本地改动限于环境 adapter、
checkpoint/resume、计时和本项目任务接口。
PPO 在 2M nominal milestone 因 4096-transition rollout 粒度实际多执行
2,944 transitions。

| 指标 | fresh PPO | FlashSAC official-density |
|---|---:|---:|
| 实际 transitions | 2,002,944 | 2,000,000 |
| 最终平均进度 | 0.375553 m | 0.416333 m |
| transition AUC | 0.371921 m | 0.389479 m |
| core train | 431.532 s | 447.351 s |
| cold end-to-end | 443.908 s | 468.359 s |

该 2M PPO pilot 使用固定 `3e-4` 学习率。若把 30M 配置的线性退火直接截短到
2M，PPO 学习率会在 pilot 末端提前降为 0，与仍按 30M schedule 运行的 FlashSAC
不公平；因此这里关闭退火只是一项明确标注的短预算近似，不替代 30M 正式配置。

在这一个 training seed 上：

- FlashSAC 最终平均进度高 40.779 mm（+10.86%）；
- 200 个共享任务的 paired bootstrap 95% CI 为
  `[+18.308, +65.265] mm`，但 win rate 只有 52%，paired median 只有
  +0.198 mm，提升分布并不均匀；
- transition AUC 高 17.558 mm（+4.72%）；
- FlashSAC 在 1M transitions 首次达到 0.4 m；PPO 到 2M 仍未达到；
- 一次性 cold run 中 FlashSAC core 慢 3.67%、end-to-end 慢 5.51%。

FlashSAC 的 cold core 包含 108.180 s 首次编译。仅作反事实分解，扣除这笔
一次性成本后 core 为 339.170 s，相对 PPO 少 21.40%，约为 1.272× throughput。
更保守地观察 500k--2M 区间斜率，FlashSAC 为约 5.87k transitions/s，
PPO 为约 4.75k transitions/s，即 steady throughput 约 1.236×。

这些时间来自有其他任务占用的共享 RTX 4090，不是独占 GPU 正式结果。
此外当前只有 seed 0，任务 bootstrap CI 不包含训练随机种子不确定性。

### J.3 Direct Seed 固定开发集结果

共同固定集包含 1,956 个通过同一 strict fallback gate 的任务；92 个 fallback
不满足 gate 的任务预先排除。保留集 geometry fingerprint SHA-256：

```text
4e1cca62481f52bd3d823d5d9ab26d1cb109981303d15989579b92615ba00be8
```

以下每个方法均为一个生成 seed、最多一次 IK、一次 controller rollout。

| 方法 | raw position mean | DIRECT | REFINE | FALLBACK | 最终进度 |
|---|---:|---:|---:|---:|---:|
| random actor | 0.9965 m | 0.00% | 57.98% | 42.02% | 0.420400 m |
| P1 mean-aligned RL | 0.4290 m | 0.00% | 74.44% | 25.56% | 0.454747 m |
| P2 precision-heavy | 0.2871 m | 0.00% | 76.18% | 23.82% | 0.451134 m |
| P4 frozen-feedback RL | 0.4152 m | 0.00% | 66.46% | 33.54% | **0.465996 m** |
| P5 controller 3e-5 / 250k | 0.4152 m | 0.00% | 66.46% | 33.54% | 0.466286 m |

当前 seed actor 主结果取 P4 frozen-feedback：

- 相对 random actor 提升 45.596 mm，paired 95% CI
  `[+34.963, +56.332] mm`；
- 相对 P1 再提升 11.249 mm，paired 95% CI
  `[+2.116, +20.190] mm`；
- 相对固定 fallback 提升 28.788 mm，paired 95% CI
  `[+20.132, +37.541] mm`；
- 候选池 reference oracle 为 0.621990 m；P4 达到相对
  fallback-to-oracle gap 的 15.58%，尚未接近 oracle；
- generator 为 0.0282 ms/task，strict gate 加至多一次 IK 为
  0.9902 ms/task，controller 为 15.1720 ms/task。该 latency 同样受共享 GPU
  负载影响。

P2 把 raw FK 误差明显降到 0.2871 m，却比 P1 少 3.613 mm progress
（95% CI `[-11.585, +4.922] mm`）。这证明“更像精确 IK”不等于“对下游
controller 更好”；seed objective 必须保留下游 return，而不能只优化几何误差。

### J.4 联合训练结论边界

当前结果支持“统一框架可运行并可执行 backward/forward/backward 闭环”，但还不
支持“controller 联合更新本身显著优于 frozen controller”：

- 第一次 1M-transition PPO forward 后，同一个 actor 的 joint-vs-frozen
  差值为 -0.312 mm，95% CI `[-3.707, +3.069] mm`；
- 随后的第二次 backward 将 joint 分支从 0.449237 m 提升到
  0.463992 m，即 +14.755 mm，95% CI `[+5.333, +24.369] mm`；
- 但是严格 matched 对照保持 actor/critic、任务序列和 RNG 相同，只把 backward
  return 的 controller 换回 frozen C0，最终达到 0.465996 m，比 joint-feedback
  高 2.078 mm，95% CI `[-3.224, +7.322] mm`；
- 在最佳 actor 上以真正生效的 `3e-5` 再训练 controller 250k transitions，
  只增加 0.290 mm，95% CI `[-2.069, +2.412] mm`。

因此目前最准确的表述是：**多轮 return-aware seed backward 确实有效；现有 PPO
controller 已接近局部饱和，当前 forward update 尚未提供可重复的额外收益。**
若论文需要“联合训练更好”的主结论，必须先改变 forward learner 或联合目标，并用
matched frozen-feedback 对照和多训练种子重新验证，不能仅依赖训练曲线。

### J.5 P6--P8：elite replay、per-task memory 与 paired fitting

P6 以后保持 P4 的 frozen C0、相同 1,956-task grouped validation 和相同 strict
fallback filter。所有表中 `ΔP4` 都是按相同任务逐项与
`P4 frozen-feedback = 0.465996080 m` 比较；每个部署任务仍是零候选枚举、
一个生成 seed、至多一次 IK 和一次 controller rollout。

#### P6：全局 top-25% elite replay

P6 从 macro replay 的成功 `REFINED` 样本中按真实 controller progress 取
top 25%，额外做 186 次 projection actor update。它把 raw position mean 从
P4 的 0.4152 m 降到 0.3847 m、把 REFINE 比例从 66.46% 提到 68.56%，但最终
progress 为 0.464152 m：

```text
P6 - P4 = -1.844 mm
95% CI = [-9.550, +6.050] mm
```

这再次说明更高的 IK refinement 成功率或更低的 raw FK error 不自动等价于更好的
下游控制初态。跨任务的全局 elite 排序还会让少数易任务反复进入
self-distillation，不能保证每个任务保留自己的最好在线投影。

#### P7：per-task online best projection memory

P7 为 17,588 个训练任务建立按 task id 索引的 CPU memory，只接受
`ROUTE_REFINED` 样本，并只在真实 progress 更高时替换。同一任务内和跨调用都保留
最高回报投影；采样在已有 valid task 之间均匀进行。51,200 次真实单-seed
controller rollout 后：

```text
valid per-task elites = 6,039 / 17,588 = 34.34%
累计 elite improvements = 6,917
per-task projection updates = 186
```

在线 P7 的 validation progress 为 0.463605 m，相对 P4 为
-0.547 mm，95% CI `[-6.209, +5.079] mm`，没有提升。主要问题不是 memory
实现错误，而是 online actor/Q update、稀疏 task coverage 和 projection
imitation 同时作用时仍会互相干扰。

随后冻结同一批 per-task elite，单独做无新增 rollout 的 offline projection fit。
不同 update 数的结果呈明显早停峰值：

| offline projection updates | progress | ΔP4 | REFINE | raw position mean |
|---:|---:|---:|---:|---:|
| 100 | 0.457748 m | -8.248 mm | 67.23% | 0.3546 m |
| 500 | 0.459568 m | -6.428 mm | 70.14% | 0.3531 m |
| **2,000** | **0.469605 m** | **+3.609 mm** | 73.57% | 0.3188 m |
| 3,000 | 0.467544 m | +1.548 mm | 74.08% | 0.3037 m |
| 5,000 | 0.463271 m | -2.725 mm | 73.42% | 0.2800 m |
| 10,000 | 0.461913 m | -4.083 mm | 74.59% | 0.2581 m |

P7-offline-2k 相对 P4 的 +3.609 mm CI 为
`[-5.004, +12.219] mm`，不显著；它相对固定 fallback 为 +32.397 mm，
CI `[+22.734, +41.973] mm`。2k 之后 raw error 继续下降而 progress 反而下降，
是“几何拟合过强、下游 return 退化”的直接负结果。P7-offline-2k 因而是当前
最佳单头 development actor，但不是已确认优于 P4 的模型。

#### P8：覆盖率、paired target 和 positive-only fitting

P8a 先冻结 actor，用固定 cycle 覆盖训练任务并收集投影，再做 2,000 次离线
per-task fit。最终 memory 有 11,724 个 refined target，覆盖
66.66% 的 17,588-task strict train set。无 anchor 和 0.1 anchor 均未提升：

| 方法 | progress | ΔP4 | REFINE | raw position mean |
|---|---:|---:|---:|---:|
| P8a cycle-2k, anchor 0 | 0.461344 m | -4.652 mm | 71.57% | 0.3288 m |
| P8a cycle-2k, anchor 0.1 | 0.461378 m | -4.618 mm | 71.32% | 0.3195 m |

P8b 将 P4 的全任务 deterministic baseline outcome 与 P7 explorer elite 显式
配对，只在 explorer 真实 progress 更高时替换 target。2k paired fit 仍没有超过
P4：

| 方法 | progress | ΔP4 | REFINE | raw position mean |
|---|---:|---:|---:|---:|
| P8b paired-2k, anchor 0 | 0.464482 m | -1.514 mm | 71.47% | 0.3319 m |
| P8b paired-2k, anchor 0.1 | 0.464342 m | -1.654 mm | 73.31% | 0.3225 m |

P8c 只拟合 positive-advantage explorer，并用较强 baseline anchor 约束未选任务。
六个开发点全部低于 P4：

| updates | anchor | progress | ΔP4 |
|---:|---:|---:|---:|
| 100 | 0.5 | 0.459807 m | -6.189 mm |
| 100 | 1.0 | 0.461068 m | -4.928 mm |
| 500 | 0.5 | 0.464249 m | -1.747 mm |
| 500 | 1.0 | 0.460951 m | -5.045 mm |
| 1,000 | 0.5 | 0.462030 m | -3.966 mm |
| 1,000 | 1.0 | 0.463006 m | -2.990 mm |

P6--P8 的共同结论是：在线 RL 确实找到了互补的好投影，但把多模态 joint target
压进一个 deterministic mean 会平均或覆盖模式；单纯提高 target coverage、
延长 imitation、加入普通 L2 anchor 都没有解决这个问题。

### J.6 kNN 只作为可辨识性诊断

为了区分“训练 target 本身没有信号”与“单头网络无法表示局部多模态映射”，做过
一次 training-target kNN 查询：

| kNN target | progress | REFINE | raw position mean | kNN 时间 |
|---|---:|---:|---:|---:|
| baseline-refined | 0.466490 m | 73.21% | 0.21254 m | 0.01761 ms/task |
| paired target | 0.470218 m | 73.72% | 0.21200 m | 0.00128 ms/task |

这些数值只能说明相近 task geometry 的 stored target 带有局部可利用信号。
kNN 依赖训练表检索，数据也已经被查看；它不是 RL 自己生成 seed，不进入主表，
不计作合规的零候选部署方法，也不用于选择最终 checkpoint。

### J.7 P9--P10：naive MoE 和直接 advantage gate 的失败

#### P9：joint-target winner-take-all MoE

P9 将 P4 actor 转为四个 joint heads，用 paired projection target 的归一化
joint-space 距离决定 specialist winner，再让 gate 模仿该 winner。它的
training gate-winner accuracy 从 27.6% 升到 48.9%，但 validation 全程低于
P4：

| update | progress | ΔP4 | REFINE | raw position mean |
|---:|---:|---:|---:|---:|
| 100 | 0.459558 m | -6.438 mm | 66.46% | 0.3772 m |
| 500 | 0.457909 m | -8.087 mm | 67.94% | 0.3611 m |
| 1,000 | 0.456039 m | -9.957 mm | 66.77% | 0.3658 m |
| 2,000 | 0.458357 m | -7.639 mm | 70.71% | 0.3261 m |

2k 时相对 P4 的 CI 为 `[-16.279, +0.726] mm`。P9 证明“能把 q target 分给
不同 expert”不等于“能判断哪个 expert 对真实 controller 更好”：joint-space
nearest winner 是几何标签，不是 downstream advantage 标签。

#### P10：冻结 baseline 的 safe-MoE

P10 冻结 expert 0，只让 specialist 和 gate 学习 `explorer > baseline + 10 mm`
的二分类。四 expert、positive weight 4 的版本在 2,000 updates 后仍把
100% 任务路由给 baseline，positive recall 为 0。两 expert、positive weight 9
虽然开始选择 specialist，但结果不稳定：

| checkpoint | train gate 选 specialist | validation progress | ΔP4 | paired 95% CI |
|---:|---:|---:|---:|---:|
| 500 | 13.75% | 0.466059 m | +0.063 mm | `[-2.419, +2.536]` |
| 1,000 | 30.35% | 0.460642 m | -5.354 mm | `[-10.174, -0.603]` |

1k checkpoint 的 gate positive precision 只有 10.62%。提高 positive class
weight 主要增加 false positive，没有形成安全门控。

#### exact baseline 数值修复

P9/P10 初版 MoE 把原 tanh-Gaussian actor 的 14-output
`[mean, log_std]` 最后一层裁成一个 7-output mean head。权重前七行虽然相同，
但 GEMM 输出宽度改变会产生极小浮点差；在 IK basin 边界上，这种差值足以改变
refinement 分支。因此旧版“expert 0 等于 P4”只是在数学上相同，不是 bit-exact。

P11 修复为：

- expert 0 保留完整 14-output source head，只取前七维 mean；
- mixed-expert batch 中仍在完整 batch 上计算 baseline head，避免切片改变 kernel；
- trunk 和 expert 0 冻结；
- 单测要求 source actor 与 expert 0 的部署 q 逐位相等；
- 训练日志 `frozen_baseline_max_abs_delta = 0.0`。

修复后使用的 exact P4 grouped-validation baseline 是
`0.465996080 m`。P11 的所有 `vs_reference` 数值都来自逐任务执行该 exact
reference，而不是用四舍五入均值相减。

### J.8 P11：outcome-matched frozen specialist gate

P11 先训练一个双分支模型：

```text
task9 -> shared frozen trunk -> task gate
                             -> exact frozen P4 head
                             -> trained specialist head
```

初始 specialist 只学习 online RL 产生的 positive projection target，1,000 次
GPU update 用时 8.176 s。随后冻结整个 specialist，在 17,588 个 strict train
任务上强制执行 expert 1，收集它**实际部署输出**经一次 router/IK 和 C0 后的真实
progress。baseline archive 与 specialist outcome 因而都有 100% task outcome
coverage；不再把“explorer target 的 return”错误当成“拟合后 specialist 输出的
return”。

全 train 结果为：

| 指标 | exact P4 baseline | frozen specialist |
|---|---:|---:|
| mean progress | 0.469692 m | 0.461701 m |
| specialist - baseline |  | -7.990 mm |
| REFINE | 66.66% | 67.49% |
| `specialist > baseline + 10 mm` |  | 3,854 / 17,588 = 21.91% |

specialist 总体更差，但优势任务的平均增益为 +161.211 mm，非优势任务平均为
-55.471 mm。若知道真实 outcome 后逐任务取更优分支，training branch oracle 为
0.505610 m，比 baseline 高 35.918 mm。这给出了双分支 gate 的 retrospective
上限，但这种 oracle 需要真实 return，不能部署。

最终 gate 不再学习 joint-nearest winner，而是在冻结 task representation 上拟合
outcome-matched advantage。部署阈值按 training quota 表示：

- `q075`：训练期选择约 7.5%，原始计划由 training OOF diagnostic 选出；
- `q05`：训练期选择约 5%，是在查看 validation 和 external-dev 后保守化的
  P11 阶段 development candidate。

训练 archive 上，q05 的实际净增益为 +1.783 mm，所选任务平均
+35.676 mm；q075 为 +2.025 mm，所选任务平均 +26.982 mm。这个差距也表明
task-only gate 只能回收 35.918 mm branch-oracle 的一小部分。

需要区分训练与部署成本：training outcome collection 会分别运行冻结分支来建立
监督，但部署时不会比较两个 q、不会查询 return，也不会 controller probe。
P11 checkpoint 明确记录：

```text
one_seed = true
candidate_enumeration = 0
controller_probes = 0
max_ik_refinements = 1
controller_rollouts_per_task = 1
```

### J.9 P11 的 validation、external-dev 与历史 holdout

#### q075：validation 正结果没有在 external-dev 复现

| 数据域 | P4 | q075 | ΔP4 | paired 95% CI |
|---|---:|---:|---:|---:|
| grouped validation，n=1,956 | 0.465996 m | 0.469587 m | **+3.591 mm** | `[+0.857, +6.379]` |
| external-dev，n=1,961 | 0.474197 m | 0.473054 m | **-1.143 mm** | `[-4.473, +2.161]` |

q075 在 validation 上显著高于 P4，并与 P7-offline-2k 基本持平：
`q075 - P7 = -0.018 mm`，CI `[-8.455, +8.381] mm`。但 external-dev
转为负差，因此不能把 validation 正结果写成已泛化的提升。

#### P11 阶段的保守 development 候选 q05

| 数据域 | n | progress | REFINE / FALLBACK | Δ exact P4 | paired 95% CI |
|---|---:|---:|---:|---:|---:|
| grouped validation | 1,956 | 0.468609 m | 66.92% / 33.08% | +2.613 mm | `[+0.373, +4.888]` |
| external-dev | 1,961 | 0.475404 m | 66.34% / 33.66% | +1.206 mm | `[-1.575, +4.103]` |
| 历史 holdout v2 | 9,560 | 0.465755 m | 67.65% / 32.35% | +1.314 mm | `[+0.034, +2.597]` |

三个域中 `DIRECT=0`、`INVALID=0`，所以每个任务都尝试一次 IK；失败后使用预先
通过 strict gate 的共同 fallback。每个任务始终零候选枚举、至多一次 IK、
一次 controller rollout。q05 相对 P4 的 overall harm `>1 mm` 分别为
2.45%、1.94% 和 1.94%，overall win 分别为 3.53%、2.40% 和 2.46%。
大多数任务因 gate 选择 exact baseline 而逐位不变。

q05 在 validation 上仍低于 P7-offline-2k：

```text
q05 - P7-offline-2k = -0.995 mm
95% CI = [-9.511, +7.527] mm
```

因此 q05 的优点是相对 P4 的小幅、低覆盖保守改动，不是明显超过最佳单头。

#### 与 IKPool reference oracle 的距离

在 grouped validation：

| 方法 | progress | 距 pool oracle 0.621990 m |
|---|---:|---:|
| P4 exact baseline | 0.465996 m | 155.994 mm |
| P7-offline-2k | 0.469605 m | 152.385 mm |
| P11 q075 | 0.469587 m | 152.403 mm |
| P11 q05 | 0.468609 m | 153.380 mm |

q05 的 fallback-to-pool capture 为 16.99%，P7-offline-2k 为 17.53%。
当前方法离 candidate-pool reference oracle 仍远，不能使用“接近 oracle”表述。
该 pool oracle 也只是在已有候选中的 complete-candidate best，不是全局 IK
oracle。

external-dev 上 q05 为 0.475404 m，reference pool oracle 为 0.637408 m，
仍差 162.005 mm。历史 holdout v2 没有在本次 q05 artifact 中构建同口径 pool
oracle，因此不补写无法核验的 oracle gap。

### J.10 P11 时间成本

训练与数据采集必须分开报告：

| 阶段 | 时间 |
|---|---:|
| P11 specialist，1,000 GPU updates | 8.176 s |
| 17,588-task specialist generator | 0.040 s |
| 17,588-task router + 至多一次 IK | 3.305 s |
| 17,588-task C0 controller rollout | 67.172 s |
| 上述三段合计 | 70.517 s |

70.517 s 是进入已过滤任务后的分段计时，不包括加载、初始化和在 18,432 个源任务
上动态重算 strict fallback filter 的墙钟。filter 是另一个必要阶段；当前 JSON
artifact 没有保存其独立 timer，因此本报告不编造一个“完整 end-to-end”总秒数。
此外 baseline archive 与 explorer target 的历史收集成本也不能被 8.176 s
specialist SGD 时间掩盖。

在 9,560-task 历史 holdout v2 上，P11 q05 的批量均摊推理时间为：

| 区段 | ms/task |
|---|---:|
| generator + task gate | 0.00299 |
| strict gate + 至多一次 IK | 0.13918 |
| controller rollout | 3.09799 |
| 合计 | 3.24016 |

这里是 batched evaluation 均摊值，不是 online `B=1` latency。所有 P6--P11
实验和 FlashSAC/PPO pilot 都在共享 RTX 4090 上运行，存在其他进程、CUDA warmup、
batch size 和首次编译影响；不同 run 的小幅时间差不能解释为算法本身的稳定
speedup。正式时间结论仍需独占 GPU、固定 warmup 和多次重复。

### J.11 可辨识性与 branch 上限

固定 validation 上，P4 与 P7-offline-2k 的真实逐任务 return 具有很强互补性：

```text
exact max(P4, P7)               = 0.509205 m
只在 P7 > P4 + 10 mm 时切换    = 0.508573 m
10 mm safe-gate 上限相对 P4     = +42.577 mm
```

27.76% 的任务上 P7 优势超过 10 mm，这部分平均优势 +153.4 mm；其余任务平均
优势 -53.9 mm。扩展到八个代表性 P4--P9 branch，retrospective oracle 为
0.542479 m；使用所有已评估 snapshot 的事后 oracle 为 0.561279 m，比 P4 高
95.283 mm，填补 P4 到 pool-reference gap 的 61.1%。这些都需要知道真实
controller outcome，只能作为分支互补性的诊断上限，不能作为可部署结果。

可部署 gate 的困难也被 CPU 诊断量化：

- 原 P7 explorer elite 只覆盖 34.34% 训练任务；
- 用 task9 预测 10 mm explorer advantage 的 HGB 5-fold AUC 约 0.63，
  AP 约 0.154；
- 在 validation 内做乐观 OOF，P4/P7 gate 最多只回收约 9.62 mm，
  且被选任务仍有约 30.8% 出现 `>1 mm` harm；
- 从旧 explorer archive 训练再迁移到 validation，AUC 只有约
  0.565--0.584，多数选择 quota 的净收益为负；
- outcome-matched P11 将 positive 标签增加到 21.91%，改善了标签一致性，
  但 q05 在 train 上仍只回收 +1.783 mm / +35.918 mm branch-oracle。

所以当前主要瓶颈已经从“能否生成不同 IK seed”转为“仅凭 task context 能否可靠
识别哪个 posture 对下游 controller 更好”。P12/P13 进一步执行了
geometry-grouped、outcome-matched 的多分支 return 学习，结果见下；它们提高了
可部署点估计，但仍只回收事后 branch oracle 的很小部分。继续堆 imitation
update、branch 数或 gate 容量，不会自动接近 oracle。

### J.12 P12：固定四分支 outcome bank 与保守线性 gate

P12 保留 bit-exact P4 为 expert 0，并在同一个冻结 trunk 后训练三个不同的
specialist head。specialist bank 的 2,000 次 GPU update 用时 6.035 s。随后在
17,588 个 strict train 任务上分别强制执行每个冻结 specialist，收集其实际
seed 经同一个 router、至多一次 IK 和同一个 C0 controller 后的 outcome。这个
步骤只用于训练 gate；部署不会轮流执行四个 branch。

| branch | train progress | validation progress |
|---|---:|---:|
| expert 0：exact P4 | 0.469692 m | 0.465996 m |
| expert 1 | 0.451572 m | 0.439693 m |
| expert 2 | 0.450804 m | 0.444120 m |
| expert 3 | 0.454467 m | 0.446512 m |
| 逐任务事后 branch oracle | **0.544628 m** | **0.532720 m** |
| oracle - expert 0 | **+74.936 mm** | **+66.724 mm** |

三个 specialist 的总体均值都明显低于 baseline，但 validation 上它们分别在约
24% 的任务取得 `>10 mm` 优势，且 specialist advantage 的两两相关性只有
0.34--0.39。因此，增加 branch 的主要价值是互补性，不是任何一个 branch
单独更强。上表 oracle 必须读取每个 branch 的真实 controller outcome，不能
部署，也不能与 IKPool complete-candidate reference oracle 混称。

#### 从全分支到 `{expert 1, expert 3}` pruning

线性 gate 使用冻结 trunk 的 256-D task feature；每个 fold 内独立拟合
`StandardScaler -> balanced logistic regression`，再把 scaler 精确折叠进
hard gate。OOF 按 exact-float32 task bytes 做 grouped 5-fold，正标签要求
specialist 比 baseline 至少高 10 mm，quota 阈值只由 training OOF 分数确定。

最初允许三个 specialist 的 scaled linear gate，在 validation 上没有通过：

| all-branch gate | training OOF ΔP4 | validation progress | validation ΔP4 |
|---|---:|---:|---:|
| q15 | +4.361 mm | 0.467021 m | +1.025 mm，CI `[-3.929, +6.055]` |
| q20 | +4.686 mm | 0.466683 m | +0.687 mm，CI `[-4.871, +6.349]` |

training OOF 的 branch-wise ablation 显示 expert 2 的边际贡献为负，于是后续
development candidate 禁用 expert 2，只保留 `{expert 1, expert 3}`。pruned
gate 的 training OOF 提升为：

| pruned gate | training OOF ΔP4 | OOF specialist quota |
|---|---:|---:|
| q15 | +4.9925 mm | 15.00% |
| q20 | +5.3364 mm | 20.00% |

必须强调：虽然 pruning 的依据可以在 training OOF 中解释，但全分支
validation 已经先被查看，因此 `{1,3}` 不是预注册的独立确认。checkpoint
将其明确标记为 `post-validation-viewed-development-candidate`；禁用的 expert 2
gate 权重为精确 0、bias 为 `-1e6`，按 fail-closed 方式永不被选择。

#### P12-pruned 的实际结果

| 方法与数据域 | progress | Δ exact P4 | paired 95% CI |
|---|---:|---:|---:|
| q15，grouped validation | **0.469842 m** | +3.846 mm | `[-0.477, +8.177]` |
| q20，grouped validation | 0.470317 m | +4.321 mm | `[-0.688, +9.371]` |
| q15，external-dev | **0.477941 m** | +3.744 mm | `[-1.461, +8.988]` |
| q20，external-dev | 0.476891 m | +2.694 mm | `[-2.938, +8.414]` |
| q15，历史 holdout v2 | **0.469530 m** | +5.088 mm | `[+3.128, +7.079]` |

q20 在 validation 的点估计略高，q15 在 external-dev 更高且 specialist 暴露更
低，因此当前保留 P12-pruned-q15 作为相对稳健的 development candidate。
这个选择已经看过 validation 和 external-dev，只能用于下一次全新 sealed
evaluation，不能把表中跨 0 的 CI 写成已确认优越。

P12-pruned-q15 的生成精度边界为：

| 数据域 | DIRECT | REFINE | FALLBACK | raw position error |
|---|---:|---:|---:|---:|
| grouped validation | 0% | 67.638% | 32.362% | mean 0.410168 m，p50 0.363077 m |
| external-dev | 0% | 66.344% | 33.656% | mean 0.402559 m，p50 0.351796 m |

因此，P12 目前不是 solver-free exact IK generator，也不能因称为“近似 seed”
就暗示误差只有毫米量级。它更接近一个学习得到的 IK 初始猜测与姿态先验；最终严格
合法性仍由一次确定性 IK refinement 和共同安全 fallback 保证。

相对上一阶段 P11 q05：

| 数据域 | P12 q15 - P11 q05 | paired 95% CI |
|---|---:|---:|
| grouped validation | +1.232 mm | `[-3.070, +5.650]` |
| external-dev | +2.537 mm | `[-2.522, +7.547]` |
| 历史 holdout v2 | +3.774 mm | `[+1.696, +5.830]` |

历史 holdout 上的 5% trimmed 差值只有 +0.598 mm；P12 相对 P11 的
`>1 mm` harm 为 5.795%，win 为 8.546%。这说明均值提升由少数任务的较大
收益驱动，不能只报均值而省略风险分布。该 holdout 已经在此前研究中被读取，
仍然只是 post-hoc historical diagnostic。

P12-pruned-q15 在 validation 的 pool capture 为 17.66%，progress 距
0.621990 m 的 IKPool reference oracle 仍有 152.148 mm。四分支事后 oracle
虽然把相对 P4 的可恢复上限增至 66.724 mm，却仍距 pool reference oracle
89.270 mm；更重要的是，可部署 q15 只兑现其中 3.846 mm。结果支持“branch
多样性存在”，不支持“已接近 oracle”。

#### P12 训练与部署时间

三次 17,588-task frozen-specialist outcome collection 的分段计时合计为：

| training-only outcome collection | 时间 |
|---|---:|
| 三个 branch generator | 0.230 s |
| 三个 branch router + 至多一次 IK | 7.292 s |
| 三个 branch C0 controller rollout | 163.627 s |
| 合计 | 171.149 s |

这些是进入已审计 strict-task subset 后的 core 分段时间，不含加载、初始化和首次
构建 fallback filter manifest。它们是训练期构造 outcome matrix 的成本，不是
部署成本。P12-pruned-q15 在 9,560-task 历史 holdout v2 上的 batched 均摊为：

| 部署区段 | ms/task |
|---|---:|
| generator + hard task gate | 0.00360 |
| strict gate + 至多一次 IK | 0.23183 |
| controller rollout | 4.32999 |
| 合计 | 4.56543 |

部署仍只硬选择并执行一个 head，候选枚举、return query 和 controller probe
均为 0；每任务至多一次 IK、恰好一次 controller rollout。上述计时来自共享
RTX 4090 的 batched evaluation，不能与 P11 的另一时刻数据解释成可靠的相对
加速或减速。

### J.13 P13：非线性 advantage gate 没有晋级

P13 保持 P12 的四个 seed branch 完全冻结，只把线性 gate 换成小型 MLP。
训练期 regressor 为
`Linear(256,64) -> SiLU -> Linear(64,3)`，直接回归三个
`specialist - baseline` 的真实 progress advantage；部署 checkpoint 将其嵌入
四输出 hard gate，并把 baseline logit 固定为精确 0。训练仍采用
exact-geometry grouped 5-fold，部署仍是 task-only hard argmax 后只执行一个
seed head。

| MLP gate | training OOF ΔP4 | validation progress | validation ΔP4 |
|---|---:|---:|---:|
| q10 | +5.468747 mm | 0.467319 m | +1.323 mm，CI `[-0.813, +3.494]` |
| q15 | +6.249040 mm | 0.468452 m | +2.456 mm，CI `[-0.304, +5.230]` |

尽管 OOF 高于线性 gate，两个 MLP checkpoint 在 validation 上都低于
P12-pruned-q15 的 0.469842 m，且 CI 均跨 0。因此 P13 被拒绝，不再运行
external-dev 或历史 holdout，避免用更多已查看域继续追逐 quota。

单进程 CPU、batch 1,024 的微基准中，线性 gate 的“task trunk + gate +
最终单 seed 输出”中位数为 1.059 us/task，P13-q10 为 1.106 us/task，仅多
0.047 us/task。该数字排除 IK 和 controller，也不是 `B=1` latency 保证；
它只说明拒绝 P13 的原因是结果没有提升，而不是推理成本不可接受。

### J.14 Evaluation manifest 的复杂度与 provenance 修复

P12 需要对多个冻结 branch 使用完全相同的 strict fallback task subset。
evaluation 现在可复用先前生成的 audited filter manifest，避免每个 branch
重新运行同一批 fallback kinematics/collision 检查。实现过程中还修复了一个
复杂度问题：旧 helper 在 kept/excluded 两个 list comprehension 的每一行都访问
`dataset.task_fingerprints`；该 property 每次都会为全数据集重新构造 fingerprint
tuple，因而总复杂度为 \(O(N^2)\)。现在先缓存一次 tuple 再按 row 索引，复杂度
降为 \(O(N)\)。初始调用“长时间无输出”、修复后为数秒级只是交互观察，没有保存
可靠的 before/after 墙钟 artifact，因此本文不报告精确 speedup。

manifest 复用默认 fail closed，并审计：

- kept/excluded row 必须无重复地精确划分 source rows；
- row 与 task index 必须保持一一对应，整数向量不允许有损 dtype 转换；
- kept/excluded geometry、row list 和 task-index list 分别保存 SHA-256；
- manifest 的 sibling JSON、manifest 本体和当前 candidate 文件都保存并核对
  SHA-256，运行中被修改会直接报错；
- candidate SHA 缺失的旧 artifact 默认拒绝。只有显式启用
  `--allow-legacy-filter-manifest` 才能用于历史诊断，并在输出中标记
  `legacy_unverified=true`；
- 若上游 manifest 曾来自未验证的 legacy lineage，该状态继续向下传播，不能因
  新一轮保存而被“洗白”。

这个修复不改变任何已执行 branch 的 seed、route 或 progress，只减少重复的
训练期预处理，并把 P12/P13 的 candidate identity 和 filter lineage 变成可审计
边界。

### J.15 P12 与旧 IKPool+SetSel 的 strict-safe 同口径结果

前述 P4、P7、P11 和 P12 表格主要比较 Direct Seed 家族内部的改进。为回答
Direct Seed 是否已经能替代旧候选池系统，另在 P12 使用的 strict-safe 共同任务
子集上，与旧 `IKPool + SetSel (S0)` 做逐任务比较。

比较满足：

- validation 1,956 个 task index 和 external-dev 1,961 个 task index 均
  100% 对齐，且各自内部唯一；
- 两侧 controller 都是同一个 `C0 = r2_grouped_best`；
- P12 每任务一个生成 seed、至多一次确定性 IK refinement、一次 C0 rollout；
- 旧 IKPool 每任务做 16 orientations × 8 restarts，即 128 次 DLS IK
  attempts，去重/FPS 后最多保留 32 个生成式候选，再追加一个共同安全 fallback，
  由 SetSel 静态选一个，最终仍只执行一次 C0 rollout；
- 在共同子集上，SetSel 选择该 fallback 的比例为 validation 4.45%、
  external-dev 4.08%，所以旧系统的 33 槽动作集并非只在形式上包含 fallback；
- P12 分析中使用的 pool reference oracle 逐任务复核为 validation
  0.6219897 m、external-dev 0.6374083 m，与对应候选 cache 口径一致。

| 数据域 | P12-pruned-q15 | IKPool+SetSel S0 | P12 - S0 | paired 95% CI |
|---|---:|---:|---:|---:|
| grouped validation，n=1,956 | 0.469842 m | 0.564730 m | -94.888 mm | `[-105.610,-84.463] mm` |
| external-dev，n=1,961 | 0.477941 m | 0.582358 m | -104.418 mm | `[-115.626,-92.799] mm` |

CI 使用 `direct_seed_eval._paired_summary` 的默认 5,000 次逐任务 paired
bootstrap，bootstrap seed 为 20260728。分布稳健指标同样为负：

| 数据域 | 5% trimmed delta | paired median | P12 harm `>1 mm` | P12 win `>1 mm` |
|---|---:|---:|---:|---:|
| grouped validation | -76.038 mm | -14.191 mm | 71.217% | 21.524% |
| external-dev | -80.420 mm | -11.937 mm | 70.066% | 22.947% |

validation 的 1,956 行对应 1,759 个唯一 exact-float32 几何，external-dev
的 1,961 行几何全部唯一。将每个唯一几何先聚合、再以几何为单位等权 bootstrap
5,000 次后，validation 的差值为 -95.743 mm，95% CI
`[-106.663,-84.548] mm`；与逐行分析结论一致。

用于复核的核心 artifact 为：

```text
P12 validation:
runs/direct_seed_rl_p12_exact_specialist_moe4_seed20260728/
validation_outcome_gate_ovr_enabled1_3_q15.npz
SHA-256 37646fb3a05b93b19386dd40b576cb3190474ca5fde8f54588325dbc93f3e845

P12 external:
runs/direct_seed_rl_p12_exact_specialist_moe4_seed20260728/
external_outcome_gate_ovr_enabled1_3_q15.npz
SHA-256 5fae38ca1bcc252bb607d82496cce0befece189fbe290ee83c7404d68c8b3ee1

S0 validation:
runs/ikpool_full_v1/gate_validation.npz
SHA-256 f26b81b72fc202dbeecf3f4c50759e6b76667c3a46aa75a4fe58582b1800ddc5

S0 external:
runs/ikpool_full_v1/gate_external.npz
SHA-256 bebb3af99188b0161a69277954bf75a6a91df7987fa253442a099ea0527db830
```

P12 manifest 记录并现场复核的 C0 SHA-256 为：

```text
agent.pt    8fb3a9b9f08dd3c0173e5ee1936961200d8851e0017b750ad8f4b095aed6aa6f
config.yaml 0a54077641c3342a4c9fc8ce8555ce47dc18f0464ba2feca80c76d9af03e88f8
```

此外，本次现场只读复核使用当前代码和同一 C0 重新执行 S0 所选的单个 seed，
validation 与 external-dev 均与历史 `s0c0` cache 逐元素、逐位一致；这排除了
旧缓存由代码漂移造成约 95--104 mm 差距的解释。本次复核没有另存一份独立 rerun
artifact，因此正式发布前仍应把命令、逐任务输出和 SHA 固化到 provenance 包中。

所以当前结果不是“以一次网络前向无损替代候选池”，而是一个明确的
quality--compute trade-off：Direct Seed 将 128 次 IK attempts、最多 32 个生成
候选、一个安全 fallback 和 SetSel 压缩为一个近似 seed 与至多一次 IK，但平均
progress 仍低约 95--104 mm。由于 validation 和 external-dev 都已经被查看，这个
对照属于 development evidence；未来 sealed 应沿用同一 controller、同一 strict
subset 和逐任务 paired 统计，不得只报告 Direct 家族内部的 P4 增益。

---

## 附录 K：科学边界、复现状态与最终表述

### K.1 当前可以声称什么

1. 已实现一个不依赖 diffusion、部署不访问 IK 候选池的 Direct Seed 框架。
2. 每次部署严格为一个 seed、零候选/return/controller probe、至多一次 IK、
   一次真实 controller rollout；所有已执行 seed 都通过 strict gate 或安全
   fallback。
3. 多轮 return-aware backward training 从 random actor 的 0.420400 m 提升到
   P4 的 0.465996 m，说明下游 controller return 对 seed 学习有实际价值。
4. P7-offline-2k 达到 0.469605 m，但相对 P4 的 CI 跨 0。
5. P12 的固定四分支 outcome bank 把 validation 事后 branch oracle 提高到
   0.532720 m，说明 return-informed specialist 确实形成了互补姿态；该
   oracle 不可部署。
6. P12-pruned-q15 在三个已查看域上相对 exact P4 的点估计均为正，且历史
   holdout 上为 +5.088 mm；这是有价值但尚未 sealed 的 development evidence。
7. P13 的非线性 gate 虽然提高 training OOF，却没有提高 actual validation；
   该负结果进一步定位了 task-only branch selection 的泛化瓶颈。
8. 单种子 2M pilot 中 FlashSAC 的最终进度和 transition AUC 高于 fresh PPO，
   值得进入正式多种子实验。
9. 在同一 C0 与 strict-safe 共同任务上，P12 把 seed-stage 结构从 128 次 IK
   attempts、最多 32 个生成式候选、一个安全 fallback 和 SetSel，降为一个近似
   seed 与至多一次 IK；这是明确的规划结构成本下降。
10. 同一个 head-to-head 也明确显示当前质量代价：P12 在 validation 和
    external-dev 分别比旧 IKPool+SetSel 低 94.888 mm 和 104.418 mm。

### K.2 当前不能声称什么

1. **不能称 P12-pruned-q15 为 sealed superior。** 全分支 validation 在
   expert-2 pruning 前已被查看，q15/q20 又读取了 validation 和 external-dev；
   validation/external 上相对 P4 的 CI 都跨 0。
2. **不能把历史 holdout v2 当新 sealed。** 它在此前研究中已经被读取；其
   P12 相对 P4 的 `+5.088 mm` 和相对 P11 的 `+3.774 mm` 只能作 post-hoc
   historical diagnostic。
3. **不能声称 P12 已确认超过 P11。** validation 和 external-dev 的 paired CI
   都跨 0；历史 holdout 的正 CI 不能替代新 sealed confirmation。
4. **不能声称接近 oracle。** grouped validation 仍距 pool reference
   152.148 mm；事后四分支 oracle 本身也仍距该 reference 89.270 mm。
5. **不能声称 direct exact IK 已解决。** 当前主结果 `DIRECT=0`；收益来自网络
   生成 approximate seed 后的一次 IK refinement 或安全 fallback。
6. **不能声称联合 controller 训练已经优于 frozen。** matched 对照没有建立
   forward controller update 的额外收益。
7. **不能把一个 training seed 的 FlashSAC/PPO 结果推广为总体算法结论。**
   任务 bootstrap CI 没有包含训练随机种子不确定性，GPU 也非独占。
8. **不能用 P13 的 training OOF 写成非线性 gate 更好。** actual validation
   没有超过 P12 线性 gate，P13 已按 promotion rule 拒绝。
9. **不能把 Direct Seed 写成旧 IKPool+SetSel 的 non-inferior 替代。** 同一
   C0、同一 strict-safe subset 上，两个 paired CI 都完全低于 0，trimmed 和
   median delta 也都为负。
10. **不能用低 seed-stage 结构成本掩盖约 95--104 mm 的 progress 差距。**
    “规划更轻”和“当前任务质量更低”必须在同一结论中同时报告。

### K.3 后续正式验证

在写论文主结论前，至少还需要：

- 冻结 P12-pruned-q15 的 branch subset、quota、权重、router 和所有 artifact
  SHA，不再查看 validation/external 调参；
- 使用全新、与所有历史数据 exact-geometry 零重叠的 sealed set；
- 在新 sealed 上同时运行 P12 和旧 IKPool+SetSel，保持同一个 controller、
  strict subset 与逐任务 paired 分析；
- 报告多个 generator/controller training seeds，而不是只 bootstrap 任务；
- 在独占 GPU 上复测训练与 `B=1`/batched 推理时间；
- 对 FlashSAC 与 PPO 完成协议中的六种子 5M gate 和 30M 正式 development；
- 对联合训练保留 matched frozen-feedback 对照，只有跨种子 CI 支持时才写成
  “joint training 更好”。

截至 2026-07-28，最稳妥的总括是：**统一 RL 框架已经把 seed 生成、真实下游
回报和可选 controller forward phase 放在同一训练闭环内；P12 用多个
return-informed specialist 扩大了可恢复 branch 上限，并在 Direct Seed
家族内部、没有增加候选枚举或 controller probe 的条件下取得该家族当前最好的
跨域 development 点估计。
但同一
C0 和 strict-safe 任务上的 head-to-head 显示，它仍比旧 IKPool+SetSel 低
94.888--104.418 mm。P12 因而是一个显著降低 seed-stage 规划结构、但任务质量
尚未追平旧系统的 development Pareto 点，不是无损替代。其 pruning/quota 已使用
开发域，validation/external 相对 P4 的 CI 仍跨 0，可部署 gate 距 oracle 也很远。
P13 说明单纯增加 gate 容量没有解决泛化问题。正式结论仍需在不继续使用已查看
开发域调参的前提下缩小质量差距，再用全新 sealed、多训练种子和同口径旧系统对照
验证。**

# Self-Improvement 实验总报告(2026-07-02 ~ 07-03)

**目标**:单一策略网络在 10k 评估集上超越「π₀ + 滞回切换 + classical」混合系统。
**内部口径**:逐任务 `L/L_oracle` 均值(q0_seed 起点,classical 标签 oracle);胜负线 = hybrid(π₀) = **1.0503**。

## 最终成绩

| 交付物 | 成绩 | 说明 |
|---|---|---|
| **最终单网络 π_final**(`runs/distill_soup2`,r8+r9 权重平均,512 宽 MLP) | **1.0492**(难题 0.977 / 易题 1.070) | 与胜负线差 0.0011 < 配对 SE(~0.004),**统计上与混合系统打平**;比 π₀ (0.990) +6.0pp;难题层 0.687→0.977 |
| **最优系统**(hybrid(r8), τ=0.98 无滞回) | **1.0615** | 超旧纪录 1.0503 +1.1pp |
| **论文口径**(DP 种子 × pct oracle′) | pure **88.6%** / +switch **90.9%** | 原表:RL 85.0 / Hybrid 90.6 → **完整方法行更新为 90.9** |
| 鲁棒性(q0 扰动 σ=0/0.02/0.05) | 1.059/0.678/0.241 | 与 π₀ 同衰减斜率,优势全程保持 |

## 路线图与证据链(按时间)

1. **RL+BC ExIt ×4 轮:失败**(0.990→0.971)。win-filter 蒸馏 + 联合 PPO,BC 与 PPO 同 trunk 互噬。
2. **Guided switch-in-loop PPO:失败**(@51% 0.607)。免费安全网 + 救援期照发 progress reward → **moral hazard**(guide_frac 0.13→0.36)。
3. **+课程退火:失败**(@54% 0.644)。倚赖机械消除但无内化,无保护 episode 学成回避。
4. **+critic floor(V≥V_cls 精确下界):机制有效但不足**(完整 30M → 0.902)。floor 是 guided 系最大单项增益(+0.26),防回避崩塌验证成立。V_cls 基建:159k 精确标注 + 保守分位拟合(高估率 1.5%)。
5. **熵修复**(tanh 压扁分布真熵,替换 Normal 熵代理):**根因之一确认**。旧配方 σ 膨胀顶死 1.65(bang-bang 探索),修复后难题 0.687→0.856(Run A 终检 0.975)。但纯 RL 在 30M 内无法补完:advantage 在危险带信噪比过低,**既发现不了也保不住**(蒸馏后 PPO 微调必侵蚀,v1/v2 对照)。
6. **危险开局+floor(Run B):净负**(0.952)。40% 数据分流拖累易题,难题无额外收益。
7. **蒸馏 ExIt(主线,胜)**:π_{k+1}=distill(hybrid(π_k))。r1 1.030 → r5(热启动)1.045 → r6(软边界,val MSE 0.0013)1.044——classical 老师渐近线 1.045。
8. **MPC 老师**(确定性仿真精确前瞻:K 候选持有 10 步 + classical 接续,精确折扣回报):验证门 **+8.6% over classical,94% 最优动作来自搜索**。r7(深带 MPC 标签)难题 0.970;r8(合并:干净易题标签 + 深带 MPC)**1.048**;r9(K=32 DAgger)1.047;soup2 **1.0492**。
9. 失败的收尾尝试:1024 宽从零拟合(1.031,热启动>容量);π₀ 直接做安全区标签(易题反降)。

## 关键机制结论(可复用)

- 危险带技能「可表示、可监督拟合(R²=0.91)、不可被 PPO 发现或保持」——蒸馏是必经通道。
- 蒸馏保真度三杠杆:热启动(+0.007)≫ 软边界(val MSE 0.019→0.0013)> 容量(负)。
- classical 局部梯度在危险带平均次优 8.6%;MPC 标签有效但带搜索抖动,只应用于深带(qn≥0.975)并与干净标签合并。
- Model soup(同盆地热启动链)白拿 +0.001~0.002。
- 单网络与混合系统的残差差距集中在长任务(论文口径差 2.3pp vs 内部口径打平)。

## 工件索引

- 最终策略:`runs/distill_soup2/`(agent.pt + config.yaml,标准 ckpt_dir)
- 最优系统:`runs/distill_r8_merged` + τ=0.98 无滞回切换
- 数据集缓存:`runs/distill_r6_soft/distill_dataset.npz`(2.16M 干净标签)、`runs/distill_r7_mpc/dataset_round*.npz`(MPC 标签)
- 基建:`self_improve/{distill,mpc_teacher,mpc_distill,vcls,danger_starts,collect,loop}.py`;ppo.py 新增 squashed_entropy / guided switch / critic floor / actor warmup(均默认关闭,向后兼容)
- 对照与失败留档:`runs/self_improve/`(RL+BC)、`runs/p0_guided_*`(guided 系)、`runs/p0_entfix_*`(纯 RL 终检)、`runs/pid_finetune_v1_eroded|v2`(侵蚀对照)

## 后续方向(按预期收益)

1. **闭环 MPC 蒸馏加深**:扩大 MPC 标注区(qn≥0.955)、更长 hold、CEM 迭代精化——单网络突破 1.05 最直接的路。→ 已执行,见下方续战章节
2. **切换器学习化**:V_cls vs V_π 价值比较切换替换手工阈值(基建已备)。→ 已测试并否决,见下方
3. **完整系统重评**:diffusion 种子 × 新控制器全 cell 重跑,论文表整体刷新(90.9% 只是首行)。
4. DART(噪声注入蒸馏)提升真机部署冗余。

---

# 续战(2026-07-03 下午):持续超越 SOTA

在上午收官成绩(单网 1.0492 / 系统 1.0615)基础上继续推进,全部四条上午建议中的前两条已执行完毕。

## 最终成绩(全部 10k, 逐任务缓存)

| 交付物 | 成绩 | 对照与显著性 |
|---|---|---|
| **最优系统** = hybrid(`distill_r12m_b0.965_soup2`, τ=0.985/0.96) | **1.0669** | vs 旧纪录 1.0615:**+0.54pp, 3.9σ 显著** |
| **最优单网** = `distill_soup3_s2_b975`(= avg(soup2, r12m_b0.975_soup2)) | **1.0510** | vs soup2 1.0492:+0.18pp(1.4σ,方向为正未过门槛);vs 旧混合线 1.0503:首次名义反超(+0.0013±0.0023,统计打平) |
| **论文口径**(DP 种子 × pct oracle′) | pure **89.1%** / soup3+switch **91.1%** / **r12m965+switch 91.4%** | 原表 Hybrid 行 90.6 → 上午 90.9 → **91.4** |

## 路线与证据链

1. **τ 精扫**(~30 对 × 3 底座,缓存 `runs/*/tau_sweep/`):宽滞回(enter 0.985 / exit 0.96-0.965)一致优于无滞回,且切换次数 0.84→0.31/ep;τ 面在 1.065 附近平台化——系统在阈值维度已榨干。
2. **r11a 零成本边界下探**:r7 的 dataset 当时已把 qn≥0.955 全部 MPC 标注(r8 只用了 ≥0.975 部分),于是重做合并即可下探边界,无需新标注。0.965 是甜点(hyb 1.0658),0.955 回落。**发现 pure 与切换底座的最优边界分裂**:0.975 利 pure,0.965 利 hybrid。
3. **价值切换器:否决**(上午遗留的 salvage idea,现有确定性答案)。163k 状态成对精确标注(G_cls, G_π):带内 qn≥0.95 有 35.3% 状态 classical 真值更优——头寸真实存在;但 V 网合成 RMSE ≈7.8 对真值差 std 10.5,SNR 不足,最优变体 1.0456 ≪ τ 切换 1.0653(无 qn 门控时 1.0268,sw/ep 7.5,纯噪声驱动)。要兑现头寸需要 RMSE<3 的价值网。工件保留:`distill_r11_belt0.965/vswitch/`(paired_dataset 163k + vcls_sym/vpi_sym + 4 变体缓存)。
4. **r12 深带重标注**:K 16→32、hold 10→16、soup2 visitation、DAgger 2 轮,166k 深带 MPC 标签(`distill_r12_mpc32/dataset_round*.npz`)。合并(r6 干净 + r12 深带):b0.965+soup2 热启动 → **hyb 1.0669(新系统纪录)**;b0.975+soup2 → pure 1.0486。
5. **soup3**:avg(soup2, r12m_b0.975_soup2)(同盆地,r12m 从 soup2 热启动)→ pure **1.0510**,soup 再次白拿 +0.0018。加权 soup(0.3/0.7)与三成员均无额外增益。

## 新增机制结论

- **深带合并边界存在 pure/hybrid 分裂**:更深的 MPC 标签(0.965)让学生的救援行为更早介入,伤 pure(易题路径税)但作为切换底座更强——最终交付物因此是两个不同的网。
- **价值切换的可行性边界量化了**:头寸 35%、幅度 std 10.5,回归噪声 7.8——不是"想法不对",是"网不够准";若未来 V 网 RMSE 降到 ~3 可复活(paired_dataset 已备)。
- **标签质量(K、hold)主要改善切换底座而非 pure**——pure 的残差瓶颈不在深带标签噪声,与"长任务残差"结论一致。
- 宽滞回是免费午餐:成绩微升 + 切换次数降 2/3(真机部署友好)。

## 工件索引(新增)

- 最优系统底座:`runs/distill_r12m_b0.965_soup2`(采用 τ=0.985/0.96)
- 最优单网:`runs/distill_soup3_s2_b975`
- r12 标签库:`runs/distill_r12_mpc32/dataset_round{0,1}.npz`(K=32/hold16,1.08M 行含 166k 深带)
- τ 面缓存:`runs/{distill_soup2,distill_r8_merged,distill_r11_belt0.965}/tau_sweep/`
- 价值切换全套:`runs/distill_r11_belt0.965/vswitch/`
- 论文口径缓存:`system_eval/runs/eval_10k_systematic/sweeps/main_{soup3_pure,soup3+switch0.985-0.965,r12m965+switch0.985-0.96}.npz`
- 其余对照:`distill_r11_belt{0.965,0.955}`,`distill_r12m_b*`,`distill_soup3_*`,`distill_soup_r*`(加权 soup 平手留档)

## 剩余方向

1. 单网 >1.05 显著化:pure 瓶颈已不在深带——需要长任务专项(论文口径 Easy 桶 83% 是最大失血点)或 CEM 精化 MPC 老师。
2. 完整系统重评(cell 全刷)与论文表整体更新(91.4% 只是 Hybrid 行)。
3. DART 真机冗余。

## 补充(同日晚):双模选择器否决 + 灾难尾归因

**A. 任务级双模选择器:否决。** 逐任务 max(soup3_pure, r12m_hyb) = 1.0814(+1.45pp 头寸真实存在),但 t=0 初始 obs 预测"哪个模式赢"的信号太弱:训练分布 val 上只捕获 ~12% 头寸,10k 上 1.0665 vs always-hyb 1.0669(平手)。原因:切换是否有利取决于轨迹后段进不进危险带,起点不可见。与价值切换器同归——头寸存在但不可从可观测量兑现。工件 `runs/mode_selector/`。

**B. Easy 桶灾难尾归因:主要是种子问题,且死因是 cone 而非 jl。** 405 个 pct<50% 的 Easy 任务(占论文口径总失血 62% 的主体):
- 死因分布(DP 种子):**cone 64.7% / jl 21.7% / collision 12.3%**,93% 在 50 步内早死(中位 31 步),init_qn≈0.86(不在关节带内)——**整个 campaign 优化的 jl 救援已非瓶颈,尾部死于丢锥**。
- 换 classical 最优种子(max_label_q):46.4% 恢复到 ≥90% oracle′;剩余 stuck 156 个换 oracle_hyb 逐任务最优种子后又有 66.7% 恢复 ≥90%。**合计 ~85-90% 的灾难尾是 DP 种子选择问题,控制器从对的种子出发能解**。真正控制器解不了的只剩 ~40 任务(尾部 10%,Easy 桶 1.5%)。
- 粗算头寸:尾部若种子修复(29.7%→80%),论文口径 All +2.0pp(91.4→93.4),Medium 桶同类修复或再 +1pp。

**结论:控制器战役收敛,论文口径的下一座矿在 seed_selection(DP 推理端 K 候选生成 + 学习排序,或扩散种子模型本身),不在 RL_controller。** 附带一条新技能方向留档:cone-keeping(近锥危险态的 MPC 标注)——但其独立收益上限只有 Easy 桶 1.5% 任务,优先级低于种子侧。

## 补充 2(同日夜):seed-ranking 落地,论文口径 91.41 → 93.24%

按 `Yuan/seed_selection/DESIGN_seed_ranking_0703.md` 四阶段执行完毕(设计文档头部有结果摘要):
- **Phase 0 头寸**:真实 DP 采样器 K=8 候选,best-of-2/4/8 = 94.2/98.4/100.6% vs first-valid 91.41(精确复现主表锚点);候选间可交换(mean≈first)——现部署的 lazy first-IK-valid 等价随机抽,选择动作本身价值 +9.2pp。
- **Phase 2/3 排序器**:pointwise L̂ 回归(obs0 31 维,小 MLP)held-out capture 26.3% > min-qn 启发式 16.2% ≫ V 网基线 ≈0。10k 端到端:**All 91.41→93.24%(+1.83pp±0.21,显著),Easy 84.0→87.7%,灾难尾 385 任务 mean pct 30→65.5%(58% 脱离 <50%)**——灾难尾归因的预言直接闭环。
- 部署成本:K 次 DDIM 批量采样 + K 次 Newton + K 次 MLP 前向,零额外 rollout。
- 剩余头寸 ~7.4pp(capture 20%),下一步杠杆:锥余量特征、修复 pairwise 损失(本次实现有 bug)、K=16、不确定度感知混采。

**全 campaign 最终论文口径:Full method(DP ranked seed × hybrid r12m@0.985/0.96)= 93.2%(起点 90.6)。**

### v2 排序器(继续迭代,零新增 rollout)

候选集扩到 9(8 DP + pilot 种子,后者部署时同样免费)、特征 obs31+log-manipulability+标准化、修复 pairwise 损失(任务内全配对 hinge)、5 网集成:
- held-out capture 26.3% → **40.8%**(pair 35.8 > list 30.2 > point 26.7——v1 的 pairwise 是实现 bug 所致);
- **10k:95.77%**(+4.36pp±0.23 vs first-valid;v1 93.24),capture 44%;Easy 89.9 / Medium 97.1 / **Difficult 99.4**;灾难尾 mean 30→73.5%,仅 128/385 仍 <50%;pilot 候选被选中 14.9%。
- 工件:`rank_train/ranker_v2.pt`(ens5-pair + 标准化参数)、`rank_phase0/phase3_ranked_v2.npz`。

### v3 终局(训练数据 40k + K=16 + ens10)

- 训练数据 20k→40k 任务:val capture 40.8%→**56.8%**(集成在 ens5-7 饱和);
- eval 侧 K=16(+8 槽候选,排序器逐候选打分零重训):单独贡献 +0.5pp,上限 best-of-17 = 102.46%;
- **10k 终局:97.77%**(+6.36pp±0.23 vs first-valid 91.41;capture 57.5%,val/test 一致无过拟合);
- 分桶:Easy **93.2%** / Medium 98.5% / Difficult **101.2%**(难桶已穿透旧 oracle′);灾难尾 mean 30→79.0%,仅 97/385 仍 <50%(75% 脱离)。
- 工件:`rank_train/ranker_v3.pt`(ens10-pair)、`rank_phase0/phase3_ranked_v3.npz`、`candidates_ext8.npz` + `L_slot8-15`。

**全 campaign 论文口径终线:Full method = 97.8%(一天内 90.6 → 97.8,+7.2pp)。**
里程碑链:90.6(π₀ hybrid)→ 91.4(r12m 控制器)→ 93.2(v1 排序)→ 95.8(v2:pilot 候选+pairwise+集成)→ 96.3(K=16)→ **97.8**(40k 数据)。

剩余头寸 ~4.7pp(至 best-of-17)。已验证的 scaling 规律:训练数据翻倍 ≈ capture +13pp(≈ 论文口径 +1.5pp,再翻倍预计 +0.5-0.7pp);K 翻倍 ≈ 上限 +1pp、实得 +0.5pp;集成 >7 网无增益。下一批杠杆(未做):多 w 混采提升候选多样性、候选间去重特征、listwise+pairwise 复合损失。

### v4(60k 数据 + pair+list 复合损失 + 25 路混 w 候选)

- 损失对决(40k 单网):pair+list 51.0% > pair(m=0.05) 48.6% > pair(m=0.10) 47.0% > pair(m=0.02) 44.6% > pair+point 39.1%;
- 60k + pair+list:val capture 60.4%(单网)→ **65.6%**(ens6 饱和);
- 候选多样性:w=1.0 低引导 8 槽(IK ok 87.3%,单独用略弱),混入后上限 102.46→**103.22%**;
- **10k 终数:25 路 98.38%**(+6.96pp±0.24 vs first-valid;17 路 98.09,数据+损失贡献 +0.32,混 w 贡献 +0.29);
- 分桶:Easy **94.1** / Medium 98.6 / Difficult **102.2**;灾难尾 mean 30→82.6%,仅 83/385 仍 <50%(78% 脱离)。
- 工件:`rank_train/ranker_v4.pt`(ens10-pair+list)、`rank_phase0/phase3_ranked_v4.npz`、`candidates_extw1.npz` + `L_slot16-23`。

**全 campaign 论文口径终线:Full method = 98.4%(一天 90.6 → 98.4,+7.8pp)。**
完整里程碑:90.6 → 91.4(控制器)→ 93.2(v1)→ 95.8(v2)→ 97.8(v3)→ **98.4(v4)**。

### 处女集终验(0704,封版实验)

10k 集在 v1→v4 迭代中被多次查看,存在基准适应风险。终验:从 pilot 抽取从未进入 10k 的 2048 任务(排除 src_idx),oracle′ 按论文原协议重算(SMM top-K′ × π₀ hybrid 0.98/0.94,只滚有效槽):
- π₀+first-valid **89.95**(10k 90.6)/ 新控制器+first **90.55**(91.41)/ **ranked-25 98.29**(98.38)/ 上限 103.04(103.22)——四行全部复现,**无基准适应,98.4 封版**。ranked−first = +7.74pp±0.57。
- 鲁棒性(执行种子加噪):σ=0.02 时 ranked 62.3% vs first 57.5%(+4.9pp),σ=0.05 时 +2.9pp——排序种子衰减略大(36.0 vs 33.1pp)但**优势在噪声下保持**;大幅衰减本身是任务性质(约束边缘作业),与 π₀ 时代衰减同量级。
- 工件:`seed_selection/runs/fresh_holdout/`(fresh_set_2k + oracle_prime + 全行 L 缓存),脚本 `seed_selection/ranking/fresh_holdout.py`。

剩余未做(按优先级):完整 cell 矩阵重评(3 控制器 × 3 种子源全表)、真机验证(replay 基建已备,DART 留档)、论文 TeX。

## 补充 3(0706):r13 混合续跑老师——用户驱动的概念修正,单网再进一步

**动机(用户提出)**:MPC 老师给候选打分的续跑是"classical 一路到终局",偏离 hybrid 原意("出壳即还权给学生");学生已训练充分,评估应模拟部署真相。
**Gate**(2048 深带状态 × K=32 同候选集,双口径):argmax 分歧 **63.7%**,旧标签留下 **+3.0%** 部署价值(P90 8.3%)。工件 `distill_r12m_b0.965_soup2/hybcont_gate.npz`。
**r13 全量**(混合续跑老师,冻结 π_D 续跑,τ=0.985/0.96,2 轮 DAgger,166k 深带行,与 r12m 同配方对照):

| 边界 | pure | hyb | frac(h>p) |
|---|---|---|---|
| r13m_b0.965 | **1.0515**(r12m 1.0470,+0.45pp) | 1.0675(平手) | 9.5% |
| r13m_b0.975 | **1.0517**(r12m 1.0486,+0.31pp) | 1.0636(平手) | 7.8% |
| soup5_r13_s3(+soup3 三方平均) | **1.0525**(单网名义新高,vs soup3 +0.94σ) | 1.0638 | 7.3% |

**四猜想判决**:①主猜想成立但增益全落 pure——机理:hybrid 部署时运行时切换本来就实现"还权",旧标签缺陷被其补偿;单网独立跑时标签口径与处境自洽,直接受益。②pure/hyb 边界分裂消失(1.0515≈1.0517)——确证为旧标签风格伪影。③切换再内化:frac 6.9~9.5%(起点 22.8%)。④冻结续跑学生的自举偏差未兑现。
**论文口径**:单网 89.1→**89.4%**(soup5;Easy 80.8/Med 91.4/Diff 94.7);系统 91.4% 不变;头条 98.4% 不变(部署控制器未换,ranked 缓存无需重滚)。
**论文更新**:IV-D 增补 scoring-continuation 精化段(gate 数字 + 建议"单网为交付物时用部署续跑"),结果节同步;Algorithm 2 保持 classical 续跑口径(与主表一致),精化以消融文字呈现。
**工件**:`distill_r13_hybcont/dataset_round{0,1}.npz`(hybcont 标签库)、`distill_r13m_b{0.965,0.975}`、`distill_soup5_*`、脚本 `scripts_0703/`+scratchpad `r13_hybcont.py`/`mpc_hybcont_gate.py`。

边际收益已明显递减(本轮 +0.6pp / ~6h GPU)。要再上台阶的候选思路:短程 rollout 探针特征(每候选仿真 5-10 步再打分,部署成本小幅上升但特征质量跃升)、更大训练集(120k)、K=32 混 w。

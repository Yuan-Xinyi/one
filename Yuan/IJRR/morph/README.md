# 跨构型零空间策略（分支 morph-general）

**状态：已封存（2026-10-01），不再延伸。** 结论成立、结果完整，但这个方向不进论文主线。
分支不合并，与主线 `rl-single-traj-clean` 和任务条件化探索分支 `task-conditioned-line` 互相独立。
本文档是这条线的完整留存：问题、做法、结果、结论、没做的事、复现方法、产物位置。

## 1. 问题

论文的零空间策略（DirFrac 接口 + PPO）是一臂一模型。这里问的是：既然运动学几何显式已知，
能不能把机械臂的构型本身编码进观测，训练一个跨构型的通用零空间策略，并零样本用于没见过的臂？

借用的是 Ha、Liu、Song 的 Transformer Transformer（arXiv 2607.25798）的思想：按部件切 token、
用可学习嵌入表示 token 之间的指向关系、补齐加掩码、构型随机化训一个模型。只借了思想，
没有跑它的代码，也没有和它对比（第 7 节）。

## 2. 做法

分支相对主线 94c186a 共改 11 个文件：环境 14 行，PPO 循环未动，新代码都在本目录。

| 文件 | 作用 |
|---|---|
| `chain_specs.py` | 串联链规格（FR3、xArm7、Cobotta）、随机变体、数组式碰撞球、注册表、静态关节特征、FR3 规格比对 |
| `token_env.py` | 单臂环境包装成 token 观测（`TokenEnv`）；多臂拼成一个批环境（`MultiArmEnv`） |
| `token_agent.py` | transformer 演员评论家（`TokenAgent`） |
| `train_morph.py` | 训练入口：按配置展开臂组、建池、调用主线 PPO |
| `config_morph_all.yaml` | 全族模型配置：FR3 + xArm7 各加 3 个变体 |
| `config_morph_xarm_only.yaml` | 留一臂配置：只用 xArm7 加 6 个变体，从未见过 FR3 形状 |
| `morph_eval.py` | 评测：三条真臂 10k 任务对各自旗舰与经典律；已见/未见变体对经典律 |
| `morph_stage.sh` | 两次训练加两次评测的队列脚本 |
| `../env/env.py` | `NSRLBatchedEnv(kin=, collision=)` 与 `CUSTOM_ARMS` 注册表 |
| `../env/line_distribution.py` | 可行性过滤的临时环境强制用基元盒接口 |

### 2.1 机械臂规格

每条臂是一个字典：每个关节一项（父系中的旋转、平移、自身转轴），加上下限、速度限、法兰位置、笔长、任务速度 v。
字典直接喂给已有的 `BatchedChainKinematics`，FK、雅可比、连杆位姿都是现成代码。

- 三条真臂。xArm7 和 Cobotta 取 `kinematics/batched_chain_kin.py` 的 SPECS，补笔长 0.10 m 与各自的 v（0.2、0.05）。
  FR3 原来是单独手写的 `BatchedFR3Kinematics`，这里按 URDF 翻成同样的链式规格；
  `verify_fr3_spec` 对 256 个随机位形比对笔尖、姿态、雅可比与各连杆位姿，最大差 1.8e-7。
- 碰撞球。按连杆读 json（FR3 读 one 里的球文件，另两条读 `kinematics/spheres/`）；
  `ArraySphereCollision` 用数组做自碰撞，接口与原来的类相同，同连杆、相邻、隔一个连杆的球对忽略。
- 随机变体 `perturb_spec`。每个关节的平移乘一个独立因子（0.75 到 1.25），每个关节的速度限乘一个独立因子（0.7 到 1.3）。
  关节 i 的平移写在连杆 i 坐标系里，所以连杆 i 的球心乘同一个因子，底座球不动。变体名形如 `fr3_v1_2`，由种子决定。
- 静态特征 `static_joint_features`：每关节 16 维，描述"这个关节是什么"。

### 2.2 环境侧改动

1. `NSRLBatchedEnv` 多了 `kin=`、`collision=` 参数和 `CUSTOM_ARMS` 注册表，按 `cfg.robot` 名查到工厂函数就用它建臂。
   需要注册表是因为任务池的可行性过滤会从配置里的机器人名重建一个临时环境。
2. 可行性过滤用经典律，经典律只出基元盒动作，训练配置是 dir_frac 2，所以临时环境强制 `dir_frac_action=0`。

### 2.3 token 观测

`TokenEnv` 拿单臂环境原本的 rm 布局观测（q、q²、任务 13 维、a_prev、四个余量、三个投影尺度）重新拼成三段，共 252 维：

| 段 | 内容 | 维数 |
|---|---|---|
| 全局 token | 任务 13 维（d、z、n、cos、z×n）、4 个余量、3 个投影尺度、关节数比例 | 21 |
| 关节 token × 7 | 动态 16：归一化角、其平方、上一步执行量、限位余量、世界系转轴、关节原点相对笔尖、位置雅可比列、姿态雅可比列；静态 16：局部位置、局部旋转前两列、转轴、限位除以 π、速度限除以 3.14、序号比例 | 32 × 7 |
| 掩码 | 真实关节 1，补零关节 0 | 7 |

动态部分里的世界系转轴、相对笔尖位置、两列雅可比每步从 FK 重新算。这是"这个关节现在怎样影响笔尖"的显式几何，
网络不必从关节角自己推 FK。六关节臂第 7 个 token 补零、掩码为 0。
`step` 里终止观测用重置前的状态构造：先 `auto_reset=False` 跑底层一步，把终止观测转成 token，再手动重置。

`MultiArmEnv` 把若干 `TokenEnv` 顺序步进，观测、奖励、终止拼接，7 维动作按各臂关节数切片，统计按环境数或完成回合数加权。
对 PPO 来说它就是一个 8192 环境的普通环境。

### 2.4 策略网络

`TokenAgent` 继承主线 `Agent`，只重写前向：

- 关节 token 过两层 MLP 嵌入到 192 维，加可学习的关节序号嵌入和类型嵌入；全局 token 另一组 MLP 加类型嵌入。
- 1 + 7 个 token 进 3 层 4 头 TransformerEncoder（pre-norm、无 dropout），补零关节通过 `src_key_padding_mask` 从注意力剔除。
- 动作头：每个关节 token 的输出过共享 MLP 出一个标量，7 个标量拼成关节空间方向 u，再 tanh。
  环境按 dir_frac 2 处理：位置雅可比的精确零空间投影器投影 u，投影后的方向是执行方向，模长给出速度限使用比例（rho_from_norm）。
  接口与旗舰完全相同，只把出数的网络从 MLP 换成了 transformer。
- 价值头：读全局 token 的输出。
- 补零关节的均值强制为 0，并从对数概率和熵里乘掩码剔除。
- 参数量 1.62M。

### 2.5 训练入口与配置

配置的 `arms` 是若干组：一条真臂、若干变体、每条的环境数。`build_specs` 展开成臂列表，`register_specs` 注册到环境，再逐条建 `TokenEnv`。
每条臂有自己的任务池：`LineDistribution.load_or_build` 显式传入该臂的 kin 和 collision，缓存按臂名存，
真臂 10 万条、变体 2 万条，都过 0.1 m 可行性过滤。每条臂另建 200 任务的留出环境，每 2000 万步评一次。

`MultiArmEnv` 与 `TokenAgent` 交给主线 `stage2_traj/ppo.py` 的 `train`（它通过 `agent=` 接收外部网络、通过 `act_dim_policy` 取动作维数）。
PPO 配方与旗舰一致（lr 6e-4、n_steps 4、γ 0.99、归一化回报、压缩熵），只把 epoch 从 15 降到 8、总步数 160M。
环境设置与旗舰一致（dt 0.05、a_max 0.5、锥 30°、500 步），每条臂保留自己的 v、笔长与速度限。

### 2.6 评测

第一部分走论文协议：三条真臂各 10k 直线任务，dt 减半、步数加倍、每个决策保持两个子步，
与各臂旗舰和经典律同任务同协议跑，参考值取逐点上界、见证、所有已记录方法与本次三者的逐任务最大值，报 Ratio 均值与 p10。
第二部分对变体：训练种子的变体算已见，新种子算未见，各建 2000 任务池（过滤后约 1800 到 1900 条），和经典律比增益与胜负率。
变体没有自己的旗舰，只能和经典律比。

## 3. 两次训练

| 运行 | 构型 | 环境数 | 步数 | 用时 |
|---|---|---|---|---|
| 全族模型 `morph_all` | FR3、xArm7 各 2048 环境，各加 3 个变体（种子 1、2）各 682 环境 | 8188 | 160M | 3.7 h |
| 留一臂模型 `morph_xarm_only` | xArm7 3072 环境加 6 个变体（种子 2）各 853 环境，从未见过 FR3 形状 | 8190 | 160M | 3.3 h |

每次更新 2.33 s。留出行程各臂同步上升到约 0.5 到 0.6 m。

## 4. 结果

真臂，10k 直线任务，Ratio 均值 / p10：

| 测试臂 | 经典律 | 自家旗舰（单臂 240M） | 全族模型 | 留一臂模型 |
|---|---|---|---|---|
| FR3 | 59.9 / 17.8 | 93.2 / 77.2 | 89.8 / 68.0 | 85.3 / 57.5（零样本） |
| xArm7 | 56.0 / 17.8 | 91.6 / 68.4 | 88.0 / 62.1 | 88.9 / 63.4 |
| Cobotta，6 关节，两个模型都没训过 | 58.1 / 26.8 | 90.5 / 75.8 | 80.7 / 39.9（零样本） | 80.5 / 41.7（零样本） |

变体，对经典律的行程增益（倍）与胜 / 负率（%，差 1 cm 以上计）：

| 变体 | 全族模型 | 留一臂模型 |
|---|---|---|
| fr3_v1_0（全族已见） | 1.78，78.9 / 2.8 | 1.65，73.1 / 8.0 |
| fr3_v1_1（全族已见） | 1.77，80.1 / 2.6 | 1.64，73.9 / 7.1 |
| fr3_v7_0 … v7_3（未见） | 1.86、1.85、1.80、1.79，胜 81.5 到 83.8，负 2.0 到 2.7 | 1.74、1.73、1.68、1.66，胜 76.2 到 79.3，负 5.2 到 6.5 |
| xarm7_v2_0（已见） | 1.97，84.3 / 1.3 | 2.00，85.5 / 1.3 |
| xarm7_v2_1（已见） | 1.98，84.7 / 1.3 | 2.02，86.1 / 1.4 |
| xarm7_v8_0 … v8_3（未见） | 2.01、1.90、1.92、1.97，胜 82.5 到 83.9，负 1.8 到 2.4 | 2.03、1.91、1.94、2.00，胜 84.3 到 84.9，负 1.2 到 1.7 |

留一臂模型从未训过任何 FR3 形状，所以对它而言 fr3_v1 也是未见。

## 5. 结论

1. 跨构型泛化成立。没见过 FR3 这族链的策略在 FR3 上零样本达到逐点上界的 85%，是经典律的 1.4 倍；关节数和轴排布都不同的 Cobotta 零样本 80%。
   未见变体和已见变体的增益一样，没有分布内外之差。
2. 把目标臂族放进训练收回大部分差距（FR3 85.3 到 89.8）。单臂 240M 旗舰仍领先 3 到 4 点，p10 领先约 10 点。
   全族模型分到每条真臂的数据只有约 40M 步，而单臂旗舰 120M 步时是 89.1 / 60.2，数据效率上跨构型模型更高。
3. 关节 token 加雅可比列的编码让网络学的是"关节怎样影响工具"的规律而不是某条臂的记忆。这是对跨构型成立原因的推测，没有消融支持（第 7 节）。

## 6. 工程限制

每条臂是一个独立的 `NSRLBatchedEnv` 实例，一步的开销按实例算（约 45 ms，与批量无关），构型数直接等于步时间倍数。
第一版 18 条臂一次更新 11.7 s，缩到 8 条后 2.33 s。要扩到几十条构型，得把不同构型批进同一个实例（运动学参数按环境逐条存），这条线没有做。

## 7. 没做的事：与 Transformer Transformer 的对比

没有对比。它的设计空间是 ViperX、轮式双臂、四足，控制器是对 Mink 微分逆运动学或 RL 专家做行为克隆（扩散输出动作序列），
物理是 MJX，数据是几百 GB 的专家回放；我们的任务是极限运动，没有专家可克隆，直接跑它的仓库不构成公平对比。
与它的主要区别：它的 token 分连杆、固定关节、动态关节、电机、状态、动作六类，关节 token 只含物理参数，不含运动学导出量，网络要自己推运动学；
我们只有关节 token 与全局 token，雅可比显式写进 token；它的任务空间由网络直接出关节命令跟踪末端轨迹，我们的任务空间由零空间投影解析解决。

如果以后重开，两条可行的对比都在我们自己的环境和 PPO 下做：

1. 表示法消融：把关节 token 里的 FK 导出量去掉，只留链规格和关节角。这是"RoboTokens 式"表示在我们任务上的样子，
   直接回答显式雅可比是不是跨构型成立的原因。改动只在 `TokenEnv._tokens`，一次 160M 训练约 3 小时。
2. 方法基线：按它的方式拆成连杆、关节、电机三类 token，指向关系做可学习嵌入，主干换 DiT 块，训练目标仍用我们的 PPO。约一天。

## 8. 复现

在工作树 `/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05` 下、conda 环境 `one`
（`/home/lqin/miniconda3/envs/one/bin/python`）、`MKL_THREADING_LAYER=GNU`：

```bash
python Yuan/IJRR/morph/chain_specs.py        # FR3 规格比对 + 变体自检
bash Yuan/IJRR/morph/morph_stage.sh           # 两次训练加两次评测，约 8 小时
# 或分步：
python -m Yuan.IJRR.morph.train_morph --config Yuan/IJRR/morph/config_morph_all.yaml --out-dir Yuan/IJRR/runs/morph_all
python Yuan/IJRR/morph/morph_eval.py all Yuan/IJRR/runs/morph_all/agent.pt Yuan/IJRR/morph/config_morph_all.yaml
```

`morph_eval.py` 和 `morph_stage.sh` 里写死了工作树路径与主检出的 `runs/paper_fill` 路径，换环境要改。

## 9. 产物位置（都不入库）

- 检查点与训练日志（工作树 `Yuan/IJRR/runs/`）：`morph_all/{agent.pt,arms.json,train.log}`、`morph_xarm_only/{agent.pt,arms.json,train.log}`、
  `morph_all.log`、`morph_xarm_only.log`、`morph_all_eval.log`、`morph_xarm_only_eval.log`、`morph_stage.log`。
- 评测数据（主检出 `/home/lqin/one/Yuan/IJRR/runs/paper_fill/fam_unify/`）：`morph_{all,xarm_only}_rows.json`，
  `morph_{all,xarm_only}_{fr3,xarm7,cobotta}_10k.npz`（逐任务 tok / flag / cls / ref），`morph_smoketest_*`（2M 步冒烟检查点的同款评测）。
- 任务池缓存：`LineDistribution` 的默认缓存目录，按臂名（真臂 10 万条，变体 2 万条）。

# Cameras / RS435 — D435 eye-to-hand 标定

## 点云配准标定(xArm7 + XHand,交互式)

`register_camera_extrinsics.py`:把 D435 实测点云与仿真 xArm7+XHand 表面点云
在 base 系下交互配准,直接得到 `T_base_cam`。不需要标定板。

**反向模式(默认):相机点云固定不动,键盘移动的是渲染里的机械臂**。相机点云用
一个冻结外参 `T_view` 投到 base 系显示一次、之后不动;机械臂在显示里的刚体位姿
记为 `M`(初始为单位阵)。把机械臂调到与相机点云重合后,外参反算得到:

    显示重合:  M @ sim_pts  ≈  T_view @ cam_pts
    反解外参:  T_base_cam = inv(M) @ T_view

```bash
conda activate one

# 在线:连 xArm 读关节角 + D435 采点云
python register_camera_extrinsics.py --ip 192.168.1.205

# 离线:载入上次保存的采集数据继续调(不需要硬件)
python register_camera_extrinsics.py --load camera_extrinsics_capture.npz

# 从已有外参出发作为冻结视角 T_view
python register_camera_extrinsics.py --ip 192.168.1.205 --init camera_extrinsics.yaml
```

键位(移动的是机械臂):`W/S A/D Q/E` 沿 base XYZ 平移;`U/O I/K J/L` 绕 XYZ 旋转;
`-`/`=` 步长减半/加倍;`G` 把机械臂平移到与相机点云质心重合;`T` ICP 精配准;
`C` 重新采集;`B` 回初始位姿(`M=` 单位阵);**ENTER 保存** `camera_extrinsics.yaml`
(+ `*_capture.npz`);ESC 退出。

流程建议:启动后先 `G` 粗对齐 → 键盘把机械臂调到与相机点云大致重合 → `T` 跑 ICP →
ENTER 保存。再次运行会自动把上次保存的外参当作冻结视角 `T_view` 继续。

- 仿真点云由 `sim_xarm7_xhand_cloud.py` 生成(子进程自动用 **wrs** 环境的
  python 跑,依赖 `/home/lqin/wrs_xinyi` 的 `XArm7XHR` 模型)。
- XHand 手指在仿真里是张开(零位)姿态,采集前把真手也张开。
- 相机点云在 **color 光学系**(与下面 ChArUco 流程同一约定),`T_base_cam`
  两种方法可互相对照/互为初值(`--init calibration_result.yaml`)。

---

## ChArUco 手眼标定(历史流程)

Intel RealSense **D435**(固定俯视)+ **xArm7** 的 eye-to-hand 外参标定与验证。
标定板:ChArUco(`DICT_4X4_50`, 6×8, square 0.018 m, marker 0.012 m),固定在法兰上。

## 坐标系约定

`T_A_B` 表示 "B 在 A 中的位姿",`p_A = T_A_B @ p_B`。

| 帧 | 含义 |
|---|---|
| `base`   | xArm7 基座 |
| `flange` | 法兰(xArm `get_position` 返回 base 下的位姿) |
| `board`  | ChArUco 板原点(OpenCV 约定,角点处,Z 朝板外) |
| `cam`    | D435 **color** 光学系(X 右 / Y 下 / Z 朝前射出镜头) |

标定方程(相机固定、板在法兰):

    T_base_flange_i @ T_flange_board = T_base_cam @ T_cam_board_i

求两个常量外参:`T_base_cam`(最终想要的)与 `T_flange_board`。

## 文件

| 文件 | 作用 |
|---|---|
| `realsense_xarm_handeye_calib.py` | 采集(空格存、u撤销、q退出)+ least_squares 标定,输出 yaml |
| `verify_pointcloud_in_base.py`    | 用 `T_base_cam` 把 D435 点云投到 base,open3d 可视化对齐 |
| `calibration_result.yaml`         | 标定结果(外参矩阵/四元数/RPY、内参、误差) |
| `handeye_data.npz`                | 原始采集数据(A=T_base_flange, B=T_cam_board, K, dist) |

## 用法

```bash
conda activate one

# 1) 采集 + 标定(15~25 组含明显旋转的姿态)
python realsense_xarm_handeye_calib.py --ip 192.168.1.205

# 离线重标(不连硬件)
python realsense_xarm_handeye_calib.py --load handeye_data.npz

# 2) 验证:点云投到 base,看与法兰/板是否对齐
python verify_pointcloud_in_base.py            # 连机器人画法兰参考
python verify_pointcloud_in_base.py --no-arm   # 只看点云
python verify_pointcloud_in_base.py --ply cloud.ply
```

## 依赖

```bash
pip install "opencv-contrib-python>=4.7" pyrealsense2 numpy scipy pyyaml open3d
pip install xArm-Python-SDK
```

## 备注

- 内参用 **color** stream 的(不是 depth)。
- xArm 欧拉角约定:`R = Rz(yaw)·Ry(pitch)·Rx(roll)` = scipy extrinsic `'xyz'`;mm→m、deg→rad。
- 验证脚本的点云在 **color 光学系**生成(depth 原生点云 + depth→color 外参),
  不用 align(depth→color) 以避免重采样空洞。
- 当前结果:18 组,平移均值 ~2.5 mm、旋转均值 ~0.43°。

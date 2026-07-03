# Cameras / RS435 — D435 eye-to-hand 标定

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

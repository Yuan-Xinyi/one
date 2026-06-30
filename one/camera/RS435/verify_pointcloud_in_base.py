#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 T_base_cam 把 D435 点云投到 base 系,验证 eye-to-hand 标定
============================================================

原理
----
1. depth 对齐到 color(rs.align),再用 *color 内参* 反投影 -> 点云在
   color 光学系(和标定的 T_base_cam 同一个系,绝不用 depth 系)。
2. p_base = T_base_cam @ p_cam,把点云搬到 base。
3. 叠加参考物:
   - base 原点 + 坐标轴
   - 当前法兰位置(xArm get_position) —— 红点
   - 预测的板原点 = T_base_flange @ T_flange_board —— 绿点
   若点云里"板/法兰"那一块正好落在红/绿点附近,说明外参对齐良好。

用法
----
    python verify_pointcloud_in_base.py                 # 连机器人画法兰参考
    python verify_pointcloud_in_base.py --no-arm        # 不连机器人,只看点云
    python verify_pointcloud_in_base.py --ply out.ply   # 另存 base 系点云

依赖: pyrealsense2 numpy scipy pyyaml matplotlib  (+ xArm SDK 若连机器人)
"""

import argparse
import numpy as np
import yaml
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R

W, H, FPS = 1280, 720, 30          # color
DW, DH = 848, 480                  # depth: D435 原生最佳分辨率(带宽更省)


def load_calib(path):
    y = yaml.safe_load(open(path))
    X = np.array(y["T_base_cam"]["matrix"])
    Z = np.array(y["T_flange_board"]["matrix"])
    return X, Z


def get_T_base_flange(ip):
    from xarm.wrapper import XArmAPI
    arm = XArmAPI(ip, is_radian=False)
    code, pose = arm.get_position(is_radian=False)
    arm.disconnect()
    if code != 0:
        raise RuntimeError(f"xArm get_position code={code}")
    x, y, z, roll, pitch, yaw = pose
    T = np.eye(4)
    T[:3, :3] = R.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()
    T[:3, 3] = np.array([x, y, z]) / 1000.0
    return T


def grab_cloud_in_cam(n_accum=8):
    """
    返回 (pts Nx3 in color optical frame [m], colors Nx3 uint8)。

    用 rs.pointcloud() 在 depth 原生帧上算稠密点云(点在 depth 光学系),
    经 RealSense 滤波链去噪/填洞,再用 depth->color 外参搬到 color 光学系
    (= 标定 T_base_cam 所在的系)。不做 align(depth->color),避免重采样空洞。
    """
    TMO = 10000                                   # 取流超时(ms)

    def _start():
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, DW, DH, rs.format.z16, FPS)
        cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
        pipe.start(cfg)
        for _ in range(20):                       # 等自动曝光稳定
            pipe.wait_for_frames(TMO)
        return pipe

    # 首次取流失败 -> 硬件复位再试一次(设备常因上次崩溃卡死)
    try:
        pipe = _start()
    except RuntimeError as e:
        print(f"[rs] 取流失败({e}),硬件复位后重试...")
        ctx = rs.context()
        if len(ctx.query_devices()):
            ctx.query_devices()[0].hardware_reset()
        import time
        time.sleep(6)
        pipe = _start()

    # 后处理滤波链(标准顺序):转视差 -> 空间 -> 时间 -> 转回深度 -> 填洞
    to_disp = rs.disparity_transform(True)
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    to_depth = rs.disparity_transform(False)
    hole = rs.hole_filling_filter(1)

    try:
        depth_f = color_f = None
        for _ in range(n_accum):                  # 多帧让 temporal 滤波累积
            frames = pipe.wait_for_frames(TMO)
            df = frames.get_depth_frame()
            df = to_disp.process(df)
            df = spatial.process(df)
            df = temporal.process(df)
            df = to_depth.process(df)
            df = hole.process(df)
            depth_f = df.as_depth_frame()
            color_f = frames.get_color_frame()

        pc = rs.pointcloud()
        pc.map_to(color_f)
        points = pc.calculate(depth_f)
        vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        tex = np.asanyarray(points.get_texture_coordinates()).view(
            np.float32).reshape(-1, 2)
        color = np.asanyarray(color_f.get_data())            # HxWx3 BGR

        # depth -> color 外参(rs rotation 为列主序 9 元素)
        ext = depth_f.profile.get_extrinsics_to(color_f.profile)
        Rdc = np.array(ext.rotation, dtype=np.float64).reshape(3, 3).T
        tdc = np.array(ext.translation, dtype=np.float64)
    finally:
        pipe.stop()

    pts_cam = (Rdc @ vtx.T).T + tdc                          # -> color 光学系

    # 用纹理坐标取颜色
    ch, cw = color.shape[:2]
    u = np.clip((tex[:, 0] * cw).astype(int), 0, cw - 1)
    v = np.clip((tex[:, 1] * ch).astype(int), 0, ch - 1)
    cols = color[v, u][:, ::-1]                              # BGR -> RGB

    z = vtx[:, 2]
    valid = (z > 0.2) & (z < 1.5) & \
            (tex[:, 0] >= 0) & (tex[:, 0] <= 1) & \
            (tex[:, 1] >= 0) & (tex[:, 1] <= 1)
    return pts_cam[valid], cols[valid]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default="calibration_result.yaml")
    ap.add_argument("--ip", default="192.168.1.205")
    ap.add_argument("--no-arm", action="store_true")
    ap.add_argument("--ply", default=None)
    args = ap.parse_args()

    X, Z = load_calib(args.yaml)     # X=T_base_cam, Z=T_flange_board
    pts_cam, cols = grab_cloud_in_cam()

    # 变换到 base: p_base = T_base_cam @ p_cam
    pts_base = (X[:3, :3] @ pts_cam.T).T + X[:3, 3]

    # 可选另存 base 系点云
    if args.ply:
        with open(args.ply, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(pts_base)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for p, c in zip(pts_base, cols):
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} "
                        f"{c[0]} {c[1]} {c[2]}\n")
        print(f"[ply] saved base-frame cloud -> {args.ply}")

    # 参考物
    T_bf = None
    board_o = None
    if not args.no_arm:
        try:
            T_bf = get_T_base_flange(args.ip)
            board_o = (T_bf @ Z)[:3, 3]    # 预测板原点 in base
            print(f"[arm] flange pos in base = {T_bf[:3,3].round(3)} m")
            print(f"[arm] predicted board origin in base = {board_o.round(3)} m")
        except Exception as e:
            print(f"[arm] 读法兰失败({e}),只画点云")

    # ---- open3d 交互可视化 ----
    import open3d as o3d
    geoms = []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_base)
    pcd.colors = o3d.utility.Vector3dVector(cols / 255.0)
    geoms.append(pcd)

    # base 坐标轴(大)
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.15, origin=[0, 0, 0]))

    if T_bf is not None:
        # 法兰坐标轴(随姿态)
        flange_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
        flange_axes.transform(T_bf)
        geoms.append(flange_axes)
        # 法兰位置:红球
        s = o3d.geometry.TriangleMesh.create_sphere(radius=0.012)
        s.translate(T_bf[:3, 3]); s.paint_uniform_color([1, 0, 0])
        geoms.append(s)
        # 预测板原点:绿球
        g = o3d.geometry.TriangleMesh.create_sphere(radius=0.012)
        g.translate(board_o); g.paint_uniform_color([0, 1, 0])
        geoms.append(g)
        print("[viz] 红球=法兰位置, 绿球=预测板原点; 看点云中实物是否落在球附近")

    o3d.visualization.draw_geometries(
        geoms, window_name="D435 cloud in BASE frame",
        width=1280, height=800)


if __name__ == "__main__":
    main()

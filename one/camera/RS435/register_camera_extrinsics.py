"""D435 外参标定:把真实点云与仿真 xArm7+XHand 点云做交互式配准。

反向模式(默认):**相机点云固定不动**,键盘移动的是渲染里的机械臂(仿真点云)。
相机点云用一个冻结的外参 T_view 投到 base 系显示一次、之后不再动;机械臂在显示
里的刚体位姿记为 M(初始为单位阵,即真值位置)。当机械臂被调到与相机点云重合时,
外参由 M 反算得到:

    显示重合:  M @ sim_pts  ≈  T_view @ cam_pts
    反解外参:  T_base_cam = inv(M) @ T_view   (使 T_base_cam @ cam_pts ≈ sim_pts)

这样调整时动的是机械臂,相机点云静止、便于看清对齐。

用法(conda activate one):

    # 在线:连 xArm 读关节角 + D435 采点云
    python register_camera_extrinsics.py --ip 192.168.1.205

    # 手动给关节角(度),不连机械臂
    python register_camera_extrinsics.py --jnts-deg 20,-90,120,30,0,40,0

    # 离线:载入上次保存的采集数据调外参,不需要任何硬件
    python register_camera_extrinsics.py --load camera_extrinsics_capture.npz

    # 从已有外参出发继续调(作为冻结视角 T_view)
    python register_camera_extrinsics.py --ip ... --init camera_extrinsics.yaml

键位(在 open3d 窗口内按,移动的是机械臂):

    W/S   base X +/-        U/O   绕 X 转 +/-
    A/D   base Y +/-        I/K   绕 Y 转 +/-
    Q/E   base Z +/-        J/L   绕 Z 转 +/-
    - / = 步长 减半 / 加倍
    G     质心粗对齐(把机械臂平移到与相机点云质心重合)
    T     ICP 精配准(先粗对齐到大致重合再按)
    C     重新采集(重读关节角 + 重拍点云)
    B     回到初始位姿(机械臂 M=单位阵)
    ENTER 保存外参 yaml + 采集 npz
    ESC   退出

注意:XHand 手指在仿真里是张开(零位)姿态,采集前把真手也张开。
"""
import argparse
import datetime
import os
import subprocess
import sys
import tempfile

import numpy as np
import open3d as o3d
import yaml
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
WRS_PYTHON = "/home/lqin/miniconda3/envs/wrs/bin/python"
SIM_SCRIPT = os.path.join(HERE, "sim_xarm7_xhand_cloud.py")

# GLFW key codes
K_ENTER, K_ESC, K_MINUS, K_EQUAL = 257, 256, 45, 61


# ---------------------------------------------------------------- capture ---

def capture_realsense(z_min, z_max, voxel=0.004, warmup=15):
    """拍一帧,返回 (pts(N,3) color光学系, colors(N,3) 0-1, intrinsics dict)。

    与旧脚本同一约定:depth 原生点云 + depth→color 外参变换到 color 光学系,
    不做 align(depth→color) 以避免重采样空洞。
    """
    import pyrealsense2 as rs

    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)
    profile = pipe.start(cfg)
    try:
        for _ in range(warmup):  # 等自动曝光稳定
            frames = pipe.wait_for_frames()
        depth, color = frames.get_depth_frame(), frames.get_color_frame()

        pc = rs.pointcloud()
        pc.map_to(color)
        points = pc.calculate(depth)
        verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        tex = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)

        d_prof = depth.get_profile().as_video_stream_profile()
        c_prof = color.get_profile().as_video_stream_profile()
        ext = d_prof.get_extrinsics_to(c_prof)  # rotation 按列主序存储
        R = np.asarray(ext.rotation).reshape(3, 3).T
        t = np.asarray(ext.translation)

        intr = c_prof.get_intrinsics()
        intrinsics = dict(width=intr.width, height=intr.height,
                          fx=float(intr.fx), fy=float(intr.fy),
                          cx=float(intr.ppx), cy=float(intr.ppy),
                          model=str(intr.model), coeffs=[float(v) for v in intr.coeffs])

        img = np.asanyarray(color.get_data())
    finally:
        pipe.stop()

    valid = verts[:, 2] > 1e-6
    verts, tex = verts[valid], tex[valid]
    pts = verts @ R.T + t  # -> color 光学系
    keep = (pts[:, 2] > z_min) & (pts[:, 2] < z_max)
    pts, tex = pts[keep], tex[keep]

    h, w = img.shape[:2]
    u = np.clip((tex[:, 0] * w).astype(int), 0, w - 1)
    v = np.clip((tex[:, 1] * h).astype(int), 0, h - 1)
    colors = img[v, u].astype(np.float32) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    pcd = pcd.voxel_down_sample(voxel)
    return (np.asarray(pcd.points, dtype=np.float32),
            np.asarray(pcd.colors, dtype=np.float32), intrinsics)


def read_arm_jnts_rad(ip):
    from xarm.wrapper import XArmAPI

    arm = XArmAPI(ip)
    try:
        code, angles = arm.get_servo_angle()  # 度
        if code != 0:
            raise RuntimeError(f"get_servo_angle failed, code={code}")
    finally:
        arm.disconnect()
    return np.radians(np.asarray(angles[:7], dtype=float))


def sim_cloud(jnts_rad, n=20000):
    """子进程调 wrs 环境生成仿真机械臂表面点云(base 系)。"""
    fd, path = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    try:
        subprocess.run(
            [WRS_PYTHON, SIM_SCRIPT, "--jnts-rad", ",".join(f"{j:.8f}" for j in jnts_rad),
             "--n", str(n), "--out", path],
            check=True, stdout=subprocess.DEVNULL)
        return np.load(path)["points"]
    finally:
        os.unlink(path)


# ---------------------------------------------------------------- helpers ---

def load_init_T(path):
    with open(path) as f:
        d = yaml.safe_load(f)
    return np.asarray(d["T_base_cam"], dtype=float)


def rough_overhead_guess():
    """无任何先验时的粗略俯视相机外参:base 前方 0.6 m、高 0.8 m,朝下看。"""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", [180, 0, 90], degrees=True).as_matrix()
    T[:3, 3] = [0.6, 0.0, 0.8]
    return T


def rot_about(axis, ang, center):
    """绕过 center、方向为 axis 的轴转 ang 的 4x4 齐次矩阵。"""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(np.asarray(axis, float) * ang).as_matrix()
    T[:3, 3] = center - T[:3, :3] @ center
    return T


def apply_T(T, pts):
    return pts @ T[:3, :3].T + T[:3, 3]


def fmt_T(T):
    rpy = Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
    t = T[:3, 3]
    return (f"t=[{t[0]:+.4f} {t[1]:+.4f} {t[2]:+.4f}] m  "
            f"rpy=[{rpy[0]:+7.2f} {rpy[1]:+7.2f} {rpy[2]:+7.2f}] deg")


# ------------------------------------------------------------------- app ---

class RegisterApp:
    """反向配准:相机点云用 T_view 冻结显示,键盘移动机械臂(位姿 M),
    外参 T_base_cam = inv(M) @ T_view。"""

    def __init__(self, args):
        self.args = args
        self.trans_step = 0.02    # m
        self.rot_step = np.radians(2.0)
        self.intrinsics = None

        if args.load:
            d = np.load(args.load)
            self.cam_pts = d["cam_pts"]
            self.cam_colors = d["cam_colors"]
            self.jnts_rad = d["jnts_rad"]
            self.sim_pts = d["sim_pts"]
            if "intrinsics_yaml" in d:
                self.intrinsics = yaml.safe_load(str(d["intrinsics_yaml"]))
            print(f"[load] {args.load}: cam {len(self.cam_pts)} pts, "
                  f"sim {len(self.sim_pts)} pts")
        else:
            self.acquire()

        # 冻结显示用的外参 T_view:优先 --init > 上次结果 > load 里带的 > 粗略俯视
        if args.init and os.path.exists(args.init):
            self.T_view = load_init_T(args.init)
            print(f"[init] 冻结视角外参来自 {args.init}")
        elif os.path.exists(args.out):
            self.T_view = load_init_T(args.out)
            print(f"[init] 冻结视角外参来自上次结果 {args.out}")
        elif args.load and "T_base_cam" in np.load(args.load):
            self.T_view = np.asarray(np.load(args.load)["T_base_cam"], dtype=float)
            print(f"[init] 冻结视角外参来自 {args.load} 里的 T_base_cam")
        else:
            self.T_view = rough_overhead_guess()
            print("[init] 无先验外参,使用粗略俯视猜测,先按 G 粗对齐再手动转")

        # M:机械臂在显示 base 系里的位姿,初始为单位阵(= 真值位置)
        self.M = np.eye(4)
        self.M0 = self.M.copy()

        # 相机点云用 T_view 投到 base 后固定不动
        self.cam_view = apply_T(self.T_view, self.cam_pts)
        self.pcd_cam = o3d.geometry.PointCloud()
        self.pcd_cam.points = o3d.utility.Vector3dVector(self.cam_view)
        self.pcd_cam.colors = o3d.utility.Vector3dVector(self.cam_colors)
        # 机械臂(仿真)点云,随 M 移动
        self.pcd_sim = o3d.geometry.PointCloud()
        self.pcd_sim.points = o3d.utility.Vector3dVector(self.robot_in_view())
        self.pcd_sim.paint_uniform_color([0.1, 0.85, 0.1])

    # ---- data ----
    def acquire(self):
        if self.args.jnts_deg:
            self.jnts_rad = np.radians([float(x) for x in self.args.jnts_deg.split(",")])
        else:
            print(f"[arm] 连接 {self.args.ip} 读关节角 ...")
            self.jnts_rad = read_arm_jnts_rad(self.args.ip)
        print(f"[arm] jnts(deg) = {np.degrees(self.jnts_rad).round(2)}")
        print("[sim] 生成仿真机械臂点云(wrs 子进程)...")
        self.sim_pts = sim_cloud(self.jnts_rad, n=self.args.n_sim)
        print(f"[sim] {len(self.sim_pts)} pts")
        print("[cam] 采集 D435 点云 ...")
        self.cam_pts, self.cam_colors, self.intrinsics = capture_realsense(
            self.args.z_min, self.args.z_max, voxel=self.args.voxel)
        print(f"[cam] {len(self.cam_pts)} pts (z ∈ [{self.args.z_min}, {self.args.z_max}] m)")

    # ---- geometry ----
    def robot_in_view(self):
        """机械臂点云在显示 base 系(= 施加位姿 M)。"""
        return apply_T(self.M, self.sim_pts)

    def T_base_cam(self):
        """当前外参:使 T @ cam_pts ≈ sim_pts。"""
        return np.linalg.inv(self.M) @ self.T_view

    def centroid_align(self):
        """保持朝向不变,平移机械臂使其质心与相机点云质心重合。"""
        d = self.cam_view.mean(axis=0) - self.robot_in_view().mean(axis=0)
        self.M = self.M.copy()
        self.M[:3, 3] += d

    def icp_refine(self):
        """把机械臂对齐到固定的相机点云。先裁掉离机械臂太远的相机点(背景),
        再做多分辨率 ICP,增量作用在机械臂位姿 M 上。"""
        robot_pcd = o3d.geometry.PointCloud()
        robot_pcd.points = o3d.utility.Vector3dVector(self.robot_in_view())
        cam_pcd = o3d.geometry.PointCloud()
        cam_pcd.points = o3d.utility.Vector3dVector(self.cam_view)
        dist = np.asarray(cam_pcd.compute_point_cloud_distance(robot_pcd))
        mask = dist < self.args.icp_crop
        if mask.sum() < 100:
            print(f"[icp] 只有 {mask.sum()} 个相机点落在机械臂 "
                  f"{self.args.icp_crop*1000:.0f}mm 内,先手动粗对齐再按 T")
            return False
        # source = 裁剪后的相机点(固定 target 参照物是它们该贴到的机械臂表面)
        cam_src = o3d.geometry.PointCloud()
        cam_src.points = o3d.utility.Vector3dVector(self.cam_view[mask])
        dT = np.eye(4)
        for max_corr in (self.args.icp_crop, self.args.icp_crop / 2, self.args.icp_crop / 4):
            reg = o3d.pipelines.registration.registration_icp(
                cam_src, robot_pcd, max_corr, dT,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
            dT = reg.transformation
        # dT 把相机点拉向机械臂;要动的是机械臂,故对 M 施加其逆
        self.M = np.linalg.inv(np.asarray(dT)) @ self.M
        print(f"[icp] 用 {mask.sum()} 点, fitness={reg.fitness:.3f}, "
              f"inlier_rmse={reg.inlier_rmse*1000:.2f} mm")
        return True

    # ---- io ----
    def save(self):
        T = self.T_base_cam()
        quat = Rotation.from_matrix(T[:3, :3]).as_quat()  # xyzw
        rpy = Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
        out = dict(
            frame_convention="T_base_cam: p_base = T @ p_cam; cam = D435 color 光学系(X右/Y下/Z前)",
            T_base_cam=T.tolist(),
            T_cam_base=np.linalg.inv(T).tolist(),
            quat_xyzw=quat.tolist(),
            rpy_deg_extrinsic_xyz=rpy.tolist(),
            jnts_rad=self.jnts_rad.tolist(),
            color_intrinsics=self.intrinsics,
            method="reverse (move-robot) keyboard + ICP registration (register_camera_extrinsics.py)",
            date=datetime.datetime.now().isoformat(timespec="seconds"),
        )
        with open(self.args.out, "w") as f:
            yaml.safe_dump(out, f, sort_keys=False, allow_unicode=True)
        cap = os.path.splitext(self.args.out)[0] + "_capture.npz"
        np.savez_compressed(cap, cam_pts=self.cam_pts, cam_colors=self.cam_colors,
                            jnts_rad=self.jnts_rad, sim_pts=self.sim_pts,
                            T_base_cam=T,
                            intrinsics_yaml=yaml.safe_dump(self.intrinsics))
        print(f"\n[save] 外参 -> {self.args.out}")
        print(f"[save] 采集数据 -> {cap}(之后可用 --load {os.path.basename(cap)} 离线继续调)")
        print(f"[save] {fmt_T(T)}")

    # ---- ui ----
    def run(self):
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window("register D435  <->  sim xArm7+XHand  (移动机械臂, ESC 退出)",
                          1280, 800)
        vis.add_geometry(self.pcd_cam)   # 相机点云(固定)
        vis.add_geometry(self.pcd_sim)   # 机械臂点云(移动)
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15))
        self.refresh(vis, reset_view=True)

        def on_key(fn):
            def cb(vis_, action, mods):
                if action == 0:  # release
                    return False
                fn()
                self.refresh(vis_)
                return False
            return cb

        def move(axis, sign):
            def f():
                self.M = self.M.copy()
                self.M[:3, 3][axis] += sign * self.trans_step
            return f

        def rot(axis, sign):
            def f():
                ax = np.eye(3)[axis]
                center = self.robot_in_view().mean(axis=0)
                self.M = rot_about(ax, sign * self.rot_step, center) @ self.M
            return f

        def scale_step(k):
            def f():
                self.trans_step = float(np.clip(self.trans_step * k, 1e-4, 0.2))
                self.rot_step = float(np.clip(self.rot_step * k, np.radians(0.02),
                                              np.radians(20)))
            return f

        def recapture():
            if self.args.load:
                print("[c] --load 离线模式,无法重新采集")
                return
            self.acquire()
            self.cam_view = apply_T(self.T_view, self.cam_pts)
            self.pcd_cam.points = o3d.utility.Vector3dVector(self.cam_view)
            self.pcd_cam.colors = o3d.utility.Vector3dVector(self.cam_colors)
            self.M = self.M0.copy()

        def reset():
            self.M = self.M0.copy()

        keymap = {
            ord("W"): move(0, +1), ord("S"): move(0, -1),
            ord("A"): move(1, +1), ord("D"): move(1, -1),
            ord("Q"): move(2, +1), ord("E"): move(2, -1),
            ord("U"): rot(0, +1), ord("O"): rot(0, -1),
            ord("I"): rot(1, +1), ord("K"): rot(1, -1),
            ord("J"): rot(2, +1), ord("L"): rot(2, -1),
            K_MINUS: scale_step(0.5), K_EQUAL: scale_step(2.0),
            ord("G"): self.centroid_align, ord("T"): self.icp_refine,
            ord("C"): recapture, ord("B"): reset,
            K_ENTER: self.save,
        }
        for k, fn in keymap.items():
            vis.register_key_action_callback(k, on_key(fn))

        print(__doc__.split("键位")[1].split("注意")[0])
        print(f"[T] {fmt_T(self.T_base_cam())}")
        vis.run()
        vis.destroy_window()

    def refresh(self, vis, reset_view=False):
        self.pcd_sim.points = o3d.utility.Vector3dVector(self.robot_in_view())
        vis.update_geometry(self.pcd_sim)
        vis.update_geometry(self.pcd_cam)
        if reset_view:
            vis.reset_view_point(True)
        print(f"\r[T] {fmt_T(self.T_base_cam())}   step: {self.trans_step*1000:.1f} mm / "
              f"{np.degrees(self.rot_step):.2f} deg      ", end="", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", default="192.168.1.205", help="xArm IP(读关节角)")
    ap.add_argument("--jnts-deg", help="手动给 7 个关节角(度),跳过连机械臂")
    ap.add_argument("--load", help="离线:载入之前保存的 *_capture.npz")
    ap.add_argument("--init", help="冻结视角外参 yaml(含 T_base_cam)")
    ap.add_argument("--out", default=os.path.join(HERE, "camera_extrinsics.yaml"),
                    help="输出外参 yaml 路径")
    ap.add_argument("--n-sim", type=int, default=20000, help="仿真点云点数")
    ap.add_argument("--z-min", type=float, default=0.15, help="相机点云最近距离 m")
    ap.add_argument("--z-max", type=float, default=1.5, help="相机点云最远距离 m")
    ap.add_argument("--voxel", type=float, default=0.004, help="相机点云体素降采样 m")
    ap.add_argument("--icp-crop", type=float, default=0.05,
                    help="ICP 前保留距机械臂多少 m 内的相机点")
    args = ap.parse_args()

    app = RegisterApp(args)
    app.run()


if __name__ == "__main__":
    main()

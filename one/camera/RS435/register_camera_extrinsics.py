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

键位(在 open3d 窗口内按,移动的是机械臂;平移/旋转默认按**屏幕(相机视角)**参考系):

    W/S   屏幕 上/下           U/O   绕屏幕水平轴 俯仰 +/-
    A/D   屏幕 左/右           I/K   绕屏幕竖直轴 偏航 +/-
    Q/E   靠近/远离(纵深)     J/L   绕视线轴 滚转 +/-
    F     切换 屏幕系 / base 系   - / = 步长 减半 / 加倍
    G     质心粗对齐(把机械臂平移到与相机点云质心重合)
    T     ICP 精配准(先粗对齐到大致重合再按)
    C     重新采集(重读关节角 + 重拍点云)
    B     回到初始位姿(机械臂 M=单位阵)
    ENTER 保存外参 yaml + 采集 npz
    ESC   退出

XHand 手指在仿真里是全 0(张开)位姿。在线采集时脚本会自动连真手并用
one/control/end_effector/xhand 驱动把 12 个手指命令到全 0,使真手与仿真一致;
不想连手可加 --no-hand-connect。
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


N_XHAND_DOF = 12
XHAND_CTRL_DIR = os.path.normpath(
    os.path.join(HERE, "..", "..", "control", "end_effector", "xhand"))


def connect_hand(port, baud):
    """连接真实 XHand(dex-hand 自带 one/control/end_effector/xhand 的 RS-485 驱动)。
    返回 XHandX 句柄;连接失败返回 None、标定流程照常继续。"""
    if not os.path.isfile(os.path.join(XHAND_CTRL_DIR, "xhand_x.py")):
        print(f"[hand] 找不到驱动 {XHAND_CTRL_DIR}/xhand_x.py,跳过连接手")
        return None
    if XHAND_CTRL_DIR not in sys.path:
        sys.path.insert(0, XHAND_CTRL_DIR)  # 驱动内部相对导入 data_type 需要它在路径上
    try:
        import xhand_x
    except Exception as e:
        print(f"[hand] 载入 xhand 驱动失败({e}),跳过连接手")
        return None
    hand = xhand_x.XHandX(port=port, baudrate=baud)
    if getattr(hand, "ser", None) is None:
        print(f"[hand] 打开串口 {port} 失败,跳过(标定不受影响)")
        return None
    print(f"[hand] 已连接 {port}")
    return hand


def command_zero_read_measured(hand):
    """把 12 个手指命令到全 0(张开),并回读它们的**实测**关节角。

    真手的关节零点与 wrs 仿真模型的零点不一定是同一物理姿态,所以只发 0 未必和
    仿真渲染重合。这里回读实测角度,交由仿真按同样的数值渲染手指(sim 跟随真手),
    使渲染的手与真手尽量一致。返回 (12,) ndarray 或 None(回读失败)。
    """
    states = hand.goto_given_conf([0.0] * N_XHAND_DOF)  # 发 0 并读回状态
    if not states or len(states) < N_XHAND_DOF:
        print("[hand] 已发全 0,但没读回手指状态(仿真手指用全 0 渲染)")
        return None
    meas = np.array([float(s.position) for s in states[:N_XHAND_DOF]])
    print(f"[hand] 手指实测角(rad)= {np.round(meas, 3).tolist()}")
    return meas


def sim_cloud(jnts_rad, hand_rad=None, n=20000):
    """子进程调 wrs 环境生成仿真机械臂表面点云(base 系)。
    hand_rad(12,)给定时,仿真手指按该角度渲染(用于让 sim 跟随真手)。"""
    fd, path = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    cmd = [WRS_PYTHON, SIM_SCRIPT,
           "--jnts-rad", ",".join(f"{j:.8f}" for j in jnts_rad),
           "--n", str(n), "--out", path]
    if hand_rad is not None:
        cmd += ["--hand-rad", ",".join(f"{h:.8f}" for h in hand_rad)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
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
        self.hand = None          # 真实 XHand 句柄(在线采集时连接)
        self.hand_meas = None     # 真手 12 指实测关节角(用于让 sim 跟随真手)
        self.screen_frame = True  # 平移/旋转按屏幕(相机视角)参考系;F 键切到 base

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
    def sim_hand_pose(self):
        """仿真手指该用的 12 关节角:手动 --hand-rad > 真手实测(--sim-hand measured)
        > None(仿真用模型零位)。"""
        if self.args.hand_rad:
            v = np.array([float(x) for x in self.args.hand_rad.split(",")])
            if v.shape != (N_XHAND_DOF,):
                raise ValueError(f"--hand-rad 需要 {N_XHAND_DOF} 个值,收到 {v.shape}")
            return v
        if self.args.sim_hand == "measured" and self.hand_meas is not None:
            return self.hand_meas
        return None

    def acquire(self):
        if self.args.jnts_deg:
            self.jnts_rad = np.radians([float(x) for x in self.args.jnts_deg.split(",")])
        else:
            print(f"[arm] 连接 {self.args.ip} 读关节角 ...")
            self.jnts_rad = read_arm_jnts_rad(self.args.ip)
        print(f"[arm] jnts(deg) = {np.degrees(self.jnts_rad).round(2)}")
        # 连接真手、命令到全 0 并回读实测角(放在拍点云之前),让仿真手指跟随真手
        if not self.args.no_hand_connect:
            if self.hand is None:
                self.hand = connect_hand(self.args.hand_port, self.args.hand_baud)
            if self.hand is not None:
                self.hand_meas = command_zero_read_measured(self.hand)
        hand_rad = self.sim_hand_pose()
        print(f"[sim] 生成仿真机械臂点云(wrs 子进程, 手指{'零位' if hand_rad is None else '跟随真手/手动'})...")
        self.sim_pts = sim_cloud(self.jnts_rad, hand_rad=hand_rad, n=self.args.n_sim)
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
    def verify_report(self):
        """用当前外参(M=单位阵 => T_base_cam=T_view)投影这帧,报与仿真机器人的
        重合残差。返回 True 表示重合良好。适合换一个手臂姿态跑,做多姿态一致性检查。"""
        T = self.T_base_cam()
        cp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.cam_view))
        sp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.robot_in_view()))
        dd = np.asarray(cp.compute_point_cloud_distance(sp))
        print("\n==== 验证:用给定外参投影这帧,机器人固定在真值 ====")
        print(f"  相机原点(base)= {np.round(T[:3, 3], 3)}  "
              f"jnts(deg)= {np.degrees(self.jnts_rad).round(1)}")
        ok = False
        for thr in (0.01, 0.03):
            mm = dd < thr
            if mm.sum():
                med = 1000 * float(np.median(dd[mm]))
                print(f"  ≤{thr*100:.0f}cm: {int(mm.sum()):6d} 点 ({100*mm.mean():4.1f}%)  "
                      f"rmse={1000*np.sqrt((dd[mm]**2).mean()):5.2f}mm  中位={med:5.2f}mm")
                if thr == 0.03:
                    ok = med < 10.0  # 近区中位 <1cm 视为一致
            else:
                print(f"  ≤{thr*100:.0f}cm: 0 点 —— 外参不适用于这帧(相机移动过?姿态错了?)")
        print("  判定:", "一致 ✓(外参在这帧下重合良好)" if ok else
              "偏差偏大 ✗(查相机是否移动 / 手臂姿态 / 手指同步)")
        return ok

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

        def cam_basis():
            """当前视角(屏幕)的三个方向在 base 系里的单位向量:右、上、进屏幕。"""
            p = vis.get_view_control().convert_to_pinhole_camera_parameters()
            R = np.asarray(p.extrinsic)[:3, :3]  # world->camera,行 = 相机轴在 world
            right, up, into = R[0], -R[1], R[2]  # 相机 X 右 / Y 下(取负=上) / Z 进屏幕
            return right, up, into

        def trans_dirs():
            # 屏幕系:[竖直(上), 水平(右), 纵深(进屏幕)];base 系:[X, Y, Z]
            if self.screen_frame:
                right, up, into = cam_basis()
                return [up, right, into]
            return [np.eye(3)[0], np.eye(3)[1], np.eye(3)[2]]

        def rot_axes():
            # 屏幕系:绕[水平轴(俯仰), 竖直轴(偏航), 视线轴(滚转)];base 系:绕[X,Y,Z]
            if self.screen_frame:
                right, up, into = cam_basis()
                return [right, up, into]
            return [np.eye(3)[0], np.eye(3)[1], np.eye(3)[2]]

        def move(chan, sign):
            def f():
                d = trans_dirs()[chan]
                self.M = self.M.copy()
                self.M[:3, 3] += sign * self.trans_step * d
            return f

        def rot(chan, sign):
            def f():
                ax = rot_axes()[chan]
                # 绕相机点云质心转(= 正在对齐的可见区域),旋转时该区域基本不动
                center = self.cam_view.mean(axis=0)
                self.M = rot_about(ax, sign * self.rot_step, center) @ self.M
            return f

        def toggle_frame():
            self.screen_frame = not self.screen_frame
            print(f"\n[frame] 参考系切换为: "
                  f"{'屏幕(相机视角)' if self.screen_frame else 'base'}")

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
            # 平移:屏幕系下 W/S=上/下, D/A=右/左, E/Q=远离/靠近(base 系时为 X/Y/Z)
            ord("W"): move(0, +1), ord("S"): move(0, -1),
            ord("D"): move(1, +1), ord("A"): move(1, -1),
            ord("E"): move(2, +1), ord("Q"): move(2, -1),
            # 旋转:屏幕系下绕 水平轴(U/O)/竖直轴(I/K)/视线轴(J/L)
            ord("U"): rot(0, +1), ord("O"): rot(0, -1),
            ord("I"): rot(1, +1), ord("K"): rot(1, -1),
            ord("J"): rot(2, +1), ord("L"): rot(2, -1),
            K_MINUS: scale_step(0.5), K_EQUAL: scale_step(2.0),
            ord("G"): self.centroid_align, ord("T"): self.icp_refine,
            ord("C"): recapture, ord("B"): reset, ord("F"): toggle_frame,
            K_ENTER: self.save,
        }
        for k, fn in keymap.items():
            vis.register_key_action_callback(k, on_key(fn))

        print(__doc__.split("键位")[1].split("XHand")[0])
        print(f"[T] {fmt_T(self.T_base_cam())}")
        vis.run()
        vis.destroy_window()

    def refresh(self, vis, reset_view=False):
        self.pcd_sim.points = o3d.utility.Vector3dVector(self.robot_in_view())
        vis.update_geometry(self.pcd_sim)
        vis.update_geometry(self.pcd_cam)
        if reset_view:
            vis.reset_view_point(True)
        frame = "屏幕" if self.screen_frame else "base"
        print(f"\r[T] {fmt_T(self.T_base_cam())}   帧:{frame}  step: "
              f"{self.trans_step*1000:.1f} mm / {np.degrees(self.rot_step):.2f} deg   ",
              end="", flush=True)


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
    ap.add_argument("--hand-port", default="/dev/ttyUSB0",
                    help="XHand RS-485 串口(在线采集时连手用)")
    ap.add_argument("--hand-baud", type=int, default=3000000, help="XHand 波特率")
    ap.add_argument("--no-hand-connect", action="store_true",
                    help="不连真手/不命令手指到全 0(手没接或已手动张开时用)")
    ap.add_argument("--sim-hand", choices=["measured", "zero"], default="measured",
                    help="仿真手指姿态:measured=跟随真手实测角(默认),zero=模型零位")
    ap.add_argument("--hand-rad", default=None,
                    help="手动指定仿真 12 指关节角(rad,逗号分隔),覆盖 --sim-hand")
    ap.add_argument("--verify", action="store_true",
                    help="验证模式:用给定外参(--init/上次结果)投影这帧、机器人固定在真值,"
                         "直接报重合残差,不进入编辑。换个姿态跑可做多姿态一致性检查")
    ap.add_argument("--no-window", action="store_true",
                    help="验证模式下只打印残差,不弹可视化窗口")
    args = ap.parse_args()

    app = RegisterApp(args)
    if args.verify:
        ok = app.verify_report()
        if not args.no_window:
            print("(窗口里机器人固定在真值位姿;肉眼确认是否重合,关闭即可。别按 ENTER 以免覆盖)")
            app.run()
        raise SystemExit(0 if ok else 1)
    app.run()


if __name__ == "__main__":
    main()

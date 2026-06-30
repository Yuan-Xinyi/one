#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intel RealSense D435  +  xArm7   eye-to-hand 外参标定
=====================================================

场景
----
- 相机 D435 固定在桌面/支架上(不随机器人动)。
- ChArUco 标定板固定在 xArm7 法兰盘上,随机器人运动。
- 用 color 图像检测 ChArUco,solvePnP 求板在相机系下的位姿。

坐标系(T_A_B 表示 "B 在 A 中的位姿",p_A = T_A_B @ p_B)
-----------------------------------------------------------
- base   : xArm7 基座
- flange : 法兰(xArm get_position 返回 base 下的法兰位姿)
- board  : ChArUco 板原点(OpenCV 约定:原点在 (0,0) 角点,Z 朝板外)
- cam    : D435 color 光学系(OpenCV 约定:X 右 / Y 下 / Z 朝前射出镜头)

标定方程(eye-to-hand):
    T_base_flange_i @ T_flange_board = T_base_cam @ T_cam_board_i
            A_i     @       Z        =     X      @      B_i

未知:  X = T_base_cam     (相机 -> base,最终想要的外参)
        Z = T_flange_board (板   -> flange)

依赖
----
    pip install pyrealsense2 opencv-contrib-python>=4.7 numpy scipy pyyaml
    pip install xArm-Python-SDK     # xArm7 SDK

用法
----
    python realsense_xarm_handeye_calib.py --ip 192.168.1.xxx

    实时窗口里:
        空格 = 保存当前一组 (T_base_flange, T_cam_board)
        u    = 撤销上一组
        q    = 退出并开始标定
    采集 15~25 组,姿态间要有 *明显的旋转*(不要只平移)。
"""

import argparse
import os
import time

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

import pyrealsense2 as rs


# --------------------------------------------------------------------------- #
# ChArUco 板参数(按题目给定)
# --------------------------------------------------------------------------- #
ARUCO_DICT     = cv2.aruco.DICT_4X4_50
SQUARES_X      = 6        # 横向方格数
SQUARES_Y      = 8        # 纵向方格数
SQUARE_LENGTH  = 0.018    # 棋盘格边长 (m)
MARKER_LENGTH  = 0.012    # aruco 边长 (m)

MIN_CORNERS    = 6        # solvePnP 至少需要的 charuco 角点数
TARGET_W, TARGET_H = 1280, 720
FPS            = 30


# --------------------------------------------------------------------------- #
# 小工具:位姿 <-> 4x4
# --------------------------------------------------------------------------- #
def rt_to_T(rvec, tvec):
    """rvec(罗德里格斯,3) + tvec(3) -> 4x4"""
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))[0]
    T[:3, 3]  = np.asarray(tvec, float).reshape(3)
    return T


def T_to_rt(T):
    rvec = cv2.Rodrigues(T[:3, :3])[0].reshape(3)
    tvec = T[:3, 3].reshape(3)
    return rvec, tvec


def inv_T(T):
    Ri = T[:3, :3].T
    ti = -Ri @ T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = Ri
    out[:3, 3]  = ti
    return out


# --------------------------------------------------------------------------- #
# ChArUco 检测器(兼容 OpenCV 新/旧 API)
# --------------------------------------------------------------------------- #
class CharucoFinder:
    def __init__(self):
        self.dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self._new_api = hasattr(cv2.aruco, "CharucoDetector")
        if self._new_api:
            # OpenCV >= 4.7
            self.board = cv2.aruco.CharucoBoard(
                (SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH,
                self.dictionary)
            self.detector = cv2.aruco.CharucoDetector(self.board)
        else:
            # OpenCV 4.2 ~ 4.6
            self.board = cv2.aruco.CharucoBoard_create(
                SQUARES_X, SQUARES_Y, SQUARE_LENGTH, MARKER_LENGTH,
                self.dictionary)
            self.aruco_params = cv2.aruco.DetectorParameters_create()

    def detect(self, gray):
        """返回 (charuco_corners, charuco_ids) 或 (None, None)"""
        if self._new_api:
            ch_corners, ch_ids, _, _ = self.detector.detectBoard(gray)
        else:
            m_corners, m_ids, _ = cv2.aruco.detectMarkers(
                gray, self.dictionary, parameters=self.aruco_params)
            if m_ids is None or len(m_ids) == 0:
                return None, None
            _, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
                m_corners, m_ids, gray, self.board)
        if ch_ids is None or len(ch_ids) < MIN_CORNERS:
            return None, None
        return ch_corners, ch_ids

    def solve_pnp(self, ch_corners, ch_ids, K, dist):
        """
        用 charuco 角点 + solvePnP 求 T_cam_board。
        返回 (T_cam_board, rvec, tvec) 或 None
        """
        if self._new_api:
            obj_pts, img_pts = self.board.matchImagePoints(ch_corners, ch_ids)
        else:
            # 旧 API:从板取 3D 角点,按 id 匹配
            all_obj = self.board.chessboardCorners  # (N,3)
            obj_pts = np.array([all_obj[i[0]] for i in ch_ids], dtype=np.float32)
            img_pts = ch_corners.reshape(-1, 2).astype(np.float32)
            obj_pts = obj_pts.reshape(-1, 1, 3)
            img_pts = img_pts.reshape(-1, 1, 2)

        if obj_pts is None or len(obj_pts) < MIN_CORNERS:
            return None
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        return rt_to_T(rvec, tvec), rvec, tvec


# --------------------------------------------------------------------------- #
# RealSense:只取 color,并读 color intrinsics(不要 depth)
# --------------------------------------------------------------------------- #
class RealSenseColor:
    def __init__(self):
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, TARGET_W, TARGET_H,
                          rs.format.bgr8, FPS)
        self.profile = self.pipeline.start(cfg)

        # ---- 关键:读 color stream 的 intrinsics(绝不是 depth) ----
        color_stream = self.profile.get_stream(rs.stream.color)
        intr = color_stream.as_video_stream_profile().get_intrinsics()
        self.K = np.array([[intr.fx, 0,        intr.ppx],
                           [0,        intr.fy,  intr.ppy],
                           [0,        0,        1]], dtype=np.float64)
        self.dist = np.array(intr.coeffs, dtype=np.float64)  # [k1,k2,p1,p2,k3]
        self.intr = intr
        # 丢几帧让自动曝光稳定
        for _ in range(15):
            self.pipeline.wait_for_frames()

    def get_color(self):
        frames = self.pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            return None
        return np.asanyarray(color.get_data())

    def stop(self):
        self.pipeline.stop()


# --------------------------------------------------------------------------- #
# xArm7:读法兰位姿 T_base_flange(mm->m, deg->rad)
# --------------------------------------------------------------------------- #
class XArm7:
    def __init__(self, ip):
        from xarm.wrapper import XArmAPI
        self.arm = XArmAPI(ip, is_radian=False)  # 这里用 deg/mm,自己转换
        self.arm.motion_enable(enable=True)
        self.arm.clean_warn()
        self.arm.clean_error()
        # 标定时机器人手动示教即可,不需要使能运动;若需要:
        # self.arm.set_mode(0); self.arm.set_state(0)

    def get_T_base_flange(self):
        """
        xArm get_position() -> code, [x,y,z(mm), roll,pitch,yaw(deg)]
        姿态约定: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)  (绕固定基轴 x->y->z,
                  即 scipy 的 extrinsic 'xyz')。
        返回 T_base_flange (4x4),平移单位 m。
        """
        code, pose = self.arm.get_position(is_radian=False)
        if code != 0:
            raise RuntimeError(f"xArm get_position failed, code={code}")
        x, y, z, roll, pitch, yaw = pose
        t = np.array([x, y, z]) / 1000.0                      # mm -> m
        rot = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)  # deg -> rad
        T = np.eye(4)
        T[:3, :3] = rot.as_matrix()
        T[:3, 3]  = t
        return T

    def disconnect(self):
        self.arm.disconnect()


# --------------------------------------------------------------------------- #
# 采集回路
# --------------------------------------------------------------------------- #
def collect(ip, out_npz):
    rsc = RealSenseColor()
    finder = CharucoFinder()
    print("[intr] color K =\n", rsc.K)
    print("[intr] color dist =", rsc.dist)

    arm = XArm7(ip)

    A_list, B_list = [], []   # A=T_base_flange, B=T_cam_board
    win = "charuco (space=save  u=undo  q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        while True:
            img = rsc.get_color()
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ch_corners, ch_ids = finder.detect(gray)

            disp = img.copy()
            pnp = None
            if ch_ids is not None:
                cv2.aruco.drawDetectedCornersCharuco(disp, ch_corners, ch_ids)
                pnp = finder.solve_pnp(ch_corners, ch_ids, rsc.K, rsc.dist)
                if pnp is not None:
                    T_cb, rvec, tvec = pnp
                    cv2.drawFrameAxes(disp, rsc.K, rsc.dist, rvec, tvec,
                                      0.03)  # 画 3cm 坐标轴
                    n = 0 if ch_ids is None else len(ch_ids)
                    cv2.putText(disp, f"corners={n}  PnP OK", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(disp, "board not found", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.putText(disp, f"saved={len(A_list)}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('u'):
                if A_list:
                    A_list.pop(); B_list.pop()
                    print(f"[undo] now {len(A_list)} samples")
            elif key == ord(' '):
                if pnp is None:
                    print("[skip] 当前帧没有有效 PnP,未保存")
                    continue
                T_bf = arm.get_T_base_flange()
                T_cb = pnp[0]
                A_list.append(T_bf)
                B_list.append(T_cb)
                print(f"[save] sample {len(A_list)}  "
                      f"flange t={T_bf[:3,3].round(3)}  "
                      f"board_in_cam t={T_cb[:3,3].round(3)}")
    finally:
        cv2.destroyAllWindows()
        rsc.stop()
        arm.disconnect()

    A = np.array(A_list)
    B = np.array(B_list)
    np.savez(out_npz, A=A, B=B, K=rsc.K, dist=rsc.dist)
    print(f"[data] saved {len(A)} samples -> {out_npz}")
    return A, B, rsc.K, rsc.dist


# --------------------------------------------------------------------------- #
# 初值:用 cv2.calibrateHandEye 求 eye-to-hand 初值(失败则用单位阵)
# --------------------------------------------------------------------------- #
def init_guess(A_list, B_list):
    """
    返回 (X0=T_base_cam, Z0=T_flange_board) 初值。
    eye-to-hand 技巧:把 base2flange(= inv(A)) 当作 'gripper2base' 喂给
    cv2.calibrateHandEye,target2cam = B,返回的 cam2gripper 即 T_base_cam。
    """
    try:
        R_b2f, t_b2f, R_t2c, t_t2c = [], [], [], []
        for A, B in zip(A_list, B_list):
            Ainv = inv_T(A)              # T_flange_base
            R_b2f.append(Ainv[:3, :3]); t_b2f.append(Ainv[:3, 3])
            R_t2c.append(B[:3, :3]);    t_t2c.append(B[:3, 3])
        R_x, t_x = cv2.calibrateHandEye(
            R_b2f, t_b2f, R_t2c, t_t2c,
            method=cv2.CALIB_HAND_EYE_TSAI)
        X0 = np.eye(4); X0[:3, :3] = R_x; X0[:3, 3] = t_x.reshape(3)
    except Exception as e:
        print(f"[init] calibrateHandEye 失败({e}),用单位阵初值")
        X0 = np.eye(4)
    # Z0 = inv(A0) @ X0 @ B0
    Z0 = inv_T(A_list[0]) @ X0 @ B_list[0]
    return X0, Z0


# --------------------------------------------------------------------------- #
# least_squares 精炼
# --------------------------------------------------------------------------- #
def pack(X, Z):
    return np.concatenate([T_to_rt(X)[0], T_to_rt(X)[1],
                           T_to_rt(Z)[0], T_to_rt(Z)[1]])


def unpack(p):
    X = rt_to_T(p[0:3], p[3:6])
    Z = rt_to_T(p[6:9], p[9:12])
    return X, Z


def make_residual(A_list, B_list, rot_weight=1.0):
    """
    每组残差 = relative pose log:  M = inv(A@Z) @ (X@B)
    残差 = [rotvec(M_R)*rot_weight, M_t]   (理想全 0)
    rot_weight 把 rad 与 m 量纲拉到可比(1 rad ~= 1 m 量级时设 1.0)。
    """
    def residual(p):
        X, Z = unpack(p)
        res = []
        for A, B in zip(A_list, B_list):
            M = inv_T(A @ Z) @ (X @ B)
            rotvec = R.from_matrix(M[:3, :3]).as_rotvec()
            res.append(rotvec * rot_weight)
            res.append(M[:3, 3])
        return np.concatenate(res)
    return residual


def calibrate(A_list, B_list):
    X0, Z0 = init_guess(A_list, B_list)
    res_fn = make_residual(A_list, B_list, rot_weight=1.0)
    sol = least_squares(res_fn, pack(X0, Z0), method='lm', max_nfev=2000)
    X, Z = unpack(sol.x)
    print(f"[opt] success={sol.success}  cost={sol.cost:.3e}  "
          f"nfev={sol.nfev}")
    return X, Z


# --------------------------------------------------------------------------- #
# 验证:每组的平移/旋转误差
# --------------------------------------------------------------------------- #
def evaluate(X, Z, A_list, B_list):
    trans_err, rot_err = [], []
    for A, B in zip(A_list, B_list):
        left  = A @ Z          # T_base_board (经机器人)
        right = X @ B          # T_base_board (经相机)
        dt = np.linalg.norm(left[:3, 3] - right[:3, 3])
        dR = left[:3, :3].T @ right[:3, :3]
        ang = np.degrees(np.linalg.norm(R.from_matrix(dR).as_rotvec()))
        trans_err.append(dt)
        rot_err.append(ang)
    trans_err = np.array(trans_err)
    rot_err   = np.array(rot_err)
    print("\n========== 每组误差 ==========")
    for i, (t, a) in enumerate(zip(trans_err, rot_err)):
        print(f"  #{i:02d}  trans={t*1000:7.2f} mm   rot={a:6.3f} deg")
    print("------------------------------")
    print(f"  mean trans err = {trans_err.mean()*1000:.2f} mm "
          f"(max {trans_err.max()*1000:.2f})")
    print(f"  mean rot   err = {rot_err.mean():.3f} deg "
          f"(max {rot_err.max():.3f})")
    print("==============================\n")
    return trans_err, rot_err


# --------------------------------------------------------------------------- #
# 保存 yaml
# --------------------------------------------------------------------------- #
def save_yaml(path, X, Z, K, dist, trans_err, rot_err, n):
    Xr = R.from_matrix(X[:3, :3])
    Zr = R.from_matrix(Z[:3, :3])
    data = {
        "description": "eye-to-hand calibration: D435 (fixed) + xArm7, "
                       "ChArUco on flange",
        "convention": "T_A_B maps points B->A (pose of B in A)",
        "n_samples": int(n),
        "T_base_cam": {
            "matrix": X.tolist(),
            "translation_m": X[:3, 3].tolist(),
            "quaternion_xyzw": Xr.as_quat().tolist(),
            "rpy_deg_xyz": Xr.as_euler('xyz', degrees=True).tolist(),
        },
        "T_flange_board": {
            "matrix": Z.tolist(),
            "translation_m": Z[:3, 3].tolist(),
            "quaternion_xyzw": Zr.as_quat().tolist(),
        },
        "color_intrinsics": {
            "K": K.tolist(),
            "dist": dist.tolist(),
        },
        "errors": {
            "mean_trans_mm": float(trans_err.mean() * 1000),
            "max_trans_mm":  float(trans_err.max() * 1000),
            "mean_rot_deg":  float(rot_err.mean()),
            "max_rot_deg":   float(rot_err.max()),
        },
    }
    with open(path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"[yaml] saved -> {path}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=False, default="192.168.1.205",
                    help="xArm7 IP(默认 192.168.1.205);若用 --load 离线标定可不填")
    ap.add_argument("--out", default="handeye_data.npz")
    ap.add_argument("--yaml", default="calibration_result.yaml")
    ap.add_argument("--load", default=None,
                    help="跳过采集,直接读已保存的 npz 做标定")
    args = ap.parse_args()

    if args.load:
        d = np.load(args.load)
        A, B, K, dist = d["A"], d["B"], d["K"], d["dist"]
    else:
        if not args.ip:
            raise SystemExit("采集模式需要 --ip <xArm7 IP>")
        A, B, K, dist = collect(args.ip, args.out)

    A_list = list(A)
    B_list = list(B)
    if len(A_list) < 8:
        raise SystemExit(f"样本太少({len(A_list)}),建议 15~25 组含明显旋转")

    X, Z = calibrate(A_list, B_list)
    trans_err, rot_err = evaluate(X, Z, A_list, B_list)

    print("T_base_cam (相机 -> base) =\n", np.round(X, 5))
    print("T_flange_board (板 -> flange) =\n", np.round(Z, 5))

    save_yaml(args.yaml, X, Z, K, dist, trans_err, rot_err, len(A_list))


if __name__ == "__main__":
    main()

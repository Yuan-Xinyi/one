#!/usr/bin/env python3
"""Play back a saved joint-trajectory (.npy, shape (T, 7)) through the pen-FR3 robot in Panda3D.

The trajectory is the output of the DiT inference script:
  ``fr3_dit/experiments/outputs/infer_task<idx>_q_pred.npy``

Also draws the TCP path (derived via in-scene FK) and the shared desk.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw

from Yuan.fr3_dit.core.pen_fr3_robot import PenFrankaResearch3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--q-npy", type=Path, required=True, help="Saved q-trajectory (.npy of shape (T, 7))")
    p.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "pen_fr3_composite_tasks_50k.hdf5",
                   help="Composite HDF5 (used only to read desk config).")
    p.add_argument("--stride", type=int, default=4, help="Stride for robot animation frames.")
    return p.parse_args()


def rotation_matrix_from_normal(normal: np.ndarray) -> np.ndarray:
    z = normal / max(float(np.linalg.norm(normal)), 1e-12)
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(float(z[0])) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
    x = np.cross(helper, z); x /= max(float(np.linalg.norm(x)), 1e-12)
    y = np.cross(z, x); y /= max(float(np.linalg.norm(y)), 1e-12)
    return np.column_stack((x, y, z)).astype(np.float32)


def main() -> None:
    args = parse_args()
    q_path = np.load(args.q_npy).astype(np.float32)
    print(f"[q] shape={q_path.shape} from {args.q_npy}")

    world = ovw.World(cam_pos=[1.8, -1.6, 1.1], cam_lookat_pos=[0.2, 0.0, 0.35])
    ossop.frame().attach_to(world.scene)

    # Desk
    if args.data.exists():
        with h5py.File(args.data, "r") as f:
            ma = f["meta"].attrs
            if str(ma.get("source_sampling_mode", "")) == "fixed_desk":
                desk_center = np.asarray(ma["source_desk_center"], dtype=np.float32)
                desk_normal = np.asarray(ma["source_desk_normal"], dtype=np.float32)
                desk_rot = rotation_matrix_from_normal(desk_normal)
                DESK_HALF = 0.85
                ossop.box(
                    half_extents=(DESK_HALF, DESK_HALF, 0.0015),
                    pos=desk_center, rotmat=desk_rot,
                    rgb=np.array([0.82, 0.78, 0.68], dtype=np.float32),
                    alpha=0.55,
                ).attach_to(world.scene)

    # Quick FK for TCP path so we can draw the stroke.
    robot = PenFrankaResearch3(name="pen", enable_cc=False)
    tcp_pts = []
    for q in q_path:
        robot.goto_given_conf(q.astype(np.float32))
        tcp_pts.append(np.asarray(robot.manipulator.gl_tcp_pos, dtype=np.float32))
    tcp_pts = np.stack(tcp_pts, axis=0)

    for i in range(tcp_pts.shape[0] - 1):
        ossop.cylinder(
            spos=tcp_pts[i], epos=tcp_pts[i + 1],
            radius=0.0035,
            rgb=np.array([0.90, 0.30, 0.20], dtype=np.float32),
            alpha=0.9,
        ).attach_to(world.scene)
    ossop.sphere(pos=tcp_pts[0], radius=0.012,
                 rgb=np.array([0.15, 0.45, 1.0], dtype=np.float32),
                 alpha=1.0).attach_to(world.scene)
    ossop.sphere(pos=tcp_pts[-1], radius=0.013,
                 rgb=np.array([1.00, 0.20, 0.20], dtype=np.float32),
                 alpha=1.0).attach_to(world.scene)

    stride = max(1, int(args.stride))
    pose_indices = list(range(0, q_path.shape[0], stride))
    if pose_indices[-1] != q_path.shape[0] - 1:
        pose_indices.append(q_path.shape[0] - 1)

    from Yuan.fr3_dit.core import viz_utils
    viz_utils.visualize_anime_path(world, robot, q_path[pose_indices])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw

from Yuan.fr3_dit.core.pen_fr3_robot import PEN_LENGTH, PenFrankaResearch3

DEFAULT_H5 = Path(__file__).resolve().parents[1] / "data" / "pen_fr3_plane_trajectories.hdf5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize one generated FR3 plane-constrained trajectory.")
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--traj-idx", type=int, default=None, help="Trajectory index. If omitted, sample one at random.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stride", type=int, default=10, help="Render every k-th robot pose along the trajectory.")
    parser.add_argument("--plane-size", type=float, default=0.70)
    return parser.parse_args()


def rotation_matrix_from_normal(normal: np.ndarray) -> np.ndarray:
    z_axis = normal / max(float(np.linalg.norm(normal)), 1e-12)
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(float(z_axis[0])) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
    x_axis = np.cross(helper, z_axis)
    x_axis = x_axis / max(float(np.linalg.norm(x_axis)), 1e-12)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / max(float(np.linalg.norm(y_axis)), 1e-12)
    return np.column_stack((x_axis, y_axis, z_axis)).astype(np.float32)


def load_trajectory(path: Path, traj_idx: int | None, seed: int | None) -> tuple[int, dict]:
    rng = np.random.default_rng(seed)
    with h5py.File(path, "r") as f:
        keys = sorted(f.keys())
        if not keys:
            raise RuntimeError(f"No trajectories found in {path}")
        if traj_idx is None:
            traj_idx = int(rng.integers(0, len(keys)))
        if traj_idx < 0 or traj_idx >= len(keys):
            raise IndexError(f"traj-idx={traj_idx} out of range [0, {len(keys) - 1}]")
        g = f[keys[traj_idx]]
        traj = {
            "start_q": g["start_q"][()],
            "plane_point": g["plane_point"][()],
            "plane_normal": g["plane_normal"][()],
            "plane_side": float(g.attrs["plane_side"]),
            "direction": g["direction"][()],
            "termination_code": int(g.attrs["termination_code"]),
            "termination_reason": str(g.attrs["termination_reason"]),
            "num_points": int(g.attrs["num_points"]),
            "total_projected_length": float(g.attrs["total_projected_length"]),
            "q": g["q"][()],
            "tcp_pos": g["tcp_pos"][()],
            "progress_length": g["progress_length"][()],
        }
    return traj_idx, traj


def main() -> None:
    args = parse_args()
    traj_idx, traj = load_trajectory(args.h5, args.traj_idx, args.seed)

    plane_normal = np.asarray(traj["plane_normal"], dtype=np.float32)
    plane_point = np.asarray(traj["plane_point"], dtype=np.float32)
    direction = np.asarray(traj["direction"], dtype=np.float32)
    tcp_pos = np.asarray(traj["tcp_pos"], dtype=np.float32)
    q_path = np.asarray(traj["q"], dtype=np.float32)

    print(
        f"[pen traj {traj_idx}] num_points={traj['num_points']} "
        f"total_projected_length={traj['total_projected_length']:.4f}m "
        f"termination={traj['termination_reason']} ({traj['termination_code']})"
    )
    print(
        f"[termination] reason={traj['termination_reason']} "
        f"code={traj['termination_code']} "
        f"length={traj['total_projected_length']:.4f}m "
        f"num_points={traj['num_points']}"
    )
    print(f"start_q      = {np.array2string(traj['start_q'], precision=4, suppress_small=True)}")
    print(f"plane_point  = {np.array2string(plane_point, precision=4, suppress_small=True)}")
    print(f"plane_normal = {np.array2string(plane_normal, precision=4, suppress_small=True)}")
    print(f"direction    = {np.array2string(direction, precision=4, suppress_small=True)}")

    world = ovw.World(cam_pos=[1.8, -1.6, 1.1], cam_lookat_pos=[0.2, 0.0, 0.35])
    ossop.frame().attach_to(world.scene)

    plane_rotmat = rotation_matrix_from_normal(plane_normal)
    plane_center = plane_point + 0.5 * float(args.plane_size) * direction
    half = 0.5 * float(args.plane_size)
    ossop.box(
        half_extents=(half, half, 0.0005),
        pos=plane_center,
        rotmat=plane_rotmat,
        rgb=np.array([0.80, 0.85, 0.90], dtype=np.float32),
        alpha=1.0,
    ).attach_to(world.scene)

    ossop.sphere(pos=plane_point, radius=0.01,
                 rgb=np.array([0.0, 0.7, 1.0], dtype=np.float32),
                 alpha=1.0).attach_to(world.scene)
    ossop.arrow(spos=plane_point, epos=plane_point + direction * 0.20,
                rgb=np.array([1.0, 0.15, 0.15], dtype=np.float32),
                alpha=0.95).attach_to(world.scene)
    ossop.arrow(spos=plane_point, epos=plane_point + plane_normal * 0.20,
                rgb=np.array([0.10, 0.45, 1.0], dtype=np.float32),
                alpha=0.95).attach_to(world.scene)
    ossop.frame(pos=plane_point, rotmat=plane_rotmat,
                length_scale=0.45).attach_to(world.scene)

    for i in range(tcp_pos.shape[0] - 1):
        alpha = 0.25 + 0.65 * (i / max(1, tcp_pos.shape[0] - 2))
        ossop.cylinder(
            spos=tcp_pos[i],
            epos=tcp_pos[i + 1],
            radius=0.0035,
            rgb=np.array([0.10, 0.85, 0.20], dtype=np.float32),
            alpha=float(alpha),
        ).attach_to(world.scene)
    ossop.sphere(pos=tcp_pos[0], radius=0.008,
                 rgb=np.array([0.15, 0.45, 1.0], dtype=np.float32),
                 alpha=0.95).attach_to(world.scene)
    ossop.sphere(pos=tcp_pos[-1], radius=0.009,
                 rgb=np.array([0.10, 0.85, 0.20], dtype=np.float32),
                 alpha=0.95).attach_to(world.scene)

    stride = max(1, int(args.stride))
    pose_indices = list(range(0, q_path.shape[0], stride))
    if pose_indices[-1] != q_path.shape[0] - 1:
        pose_indices.append(q_path.shape[0] - 1)

    # static keyframes (snapshots along the trajectory)
    for idx in pose_indices:
        alpha = 1.0 if idx == 0 or idx == pose_indices[-1] else 0.05
        snapshot = PenFrankaResearch3(name="pen", enable_cc=False)
        snapshot.goto_given_conf(q_path[idx].astype(np.float32))
        snapshot.gen_meshmodel(
            alpha=float(alpha),
            toggle_tcp_frame=(idx == pose_indices[-1]),
        ).attach_to(world)

    # animated trajectory replay (drives one extra robot)
    anim_robot = PenFrankaResearch3(name="pen", enable_cc=False)
    from Yuan.fr3_dit.core import viz_utils
    viz_utils.visualize_anime_path(world, anim_robot, q_path[pose_indices])


if __name__ == "__main__":
    main()

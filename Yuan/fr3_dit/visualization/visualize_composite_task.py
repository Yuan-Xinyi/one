#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw

from Yuan.fr3_dit.core.pen_fr3_robot import PenFrankaResearch3


DEFAULT_H5 = Path(__file__).resolve().parents[1] / "data" / "pen_fr3_composite_tasks.hdf5"

TOKEN_KIND_START = 0
TOKEN_KIND_SEGMENT = 1
TOKEN_KIND_CORNER = 2

SEGMENT_COLORS = np.array(
    [
        [0.10, 0.85, 0.20],
        [0.10, 0.45, 1.00],
        [0.95, 0.55, 0.15],
        [0.85, 0.20, 0.70],
        [0.65, 0.80, 0.20],
        [0.20, 0.80, 0.80],
    ],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a composite (stitched) task.")
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--task-idx", type=int, default=None,
                        help="Composite task index. If omitted, pick one at random.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Animation speed multiplier (2.0 = 2x faster).")
    parser.add_argument("--only-composites", action="store_true",
                        help="Only pick tasks with seg_count >= 2.")
    parser.add_argument("--min-segs", type=int, default=1,
                        help="Only pick tasks with seg_count >= min_segs.")
    parser.add_argument("--no-desk", action="store_true",
                        help="Skip drawing the shared desk plane.")
    parser.add_argument("--no-save-qtraj", action="store_true",
                        help="Skip auto-saving the 7-joint trajectory plot.")
    parser.add_argument("--qtraj-out", type=Path, default=None,
                        help="Custom path for the saved q-trajectory plot (default: experiments/outputs/qtraj_task<idx>.svg).")
    return parser.parse_args()


def load_task(path: Path, task_idx: int | None, seed: int | None, only_composites: bool, min_segs: int = 1) -> tuple[int, dict]:
    rng = np.random.default_rng(seed)
    with h5py.File(path, "r") as f:
        ts = f["tasks"]
        seg_count = ts["seg_count"][()]
        n = int(seg_count.shape[0])
        if task_idx is None:
            floor = max(min_segs, 2 if only_composites else 1)
            pool = np.where(seg_count >= floor)[0]
            if pool.size == 0:
                raise RuntimeError(f"No tasks with seg_count >= {floor}; re-run stitch with lower eps or check data.")
            task_idx = int(rng.choice(pool))
        if task_idx < 0 or task_idx >= n:
            raise IndexError(f"task_idx={task_idx} out of range [0, {n - 1}]")

        tok_off = ts["token_offset"][()]
        q_off = ts["qtraj_offset"][()]
        ss_off = ts["subseg_offset"][()]
        t_lo, t_hi = int(tok_off[task_idx]), int(tok_off[task_idx + 1])
        q_lo, q_hi = int(q_off[task_idx]), int(q_off[task_idx + 1])
        s_lo, s_hi = int(ss_off[task_idx]), int(ss_off[task_idx + 1])

        subseg_meta = np.asarray(ts["subseg_meta_flat"][s_lo:s_hi], dtype=np.int32)  # (K, 3): traj_id, start, end

        task = {
            "tokens": np.asarray(ts["token_flat"][t_lo:t_hi], dtype=np.float32),
            "token_kind": np.asarray(ts["token_kind"][t_lo:t_hi], dtype=np.uint8),
            "qtraj": np.asarray(ts["qtraj_flat"][q_lo:q_hi], dtype=np.float32),
            "tcp": np.asarray(ts["tcp_flat"][q_lo:q_hi], dtype=np.float32),
            "start_q": np.asarray(ts["start_q"][task_idx], dtype=np.float32),
            "local_frame": np.asarray(ts["local_frame"][task_idx], dtype=np.float32),
            "local_origin": np.asarray(ts["local_origin"][task_idx], dtype=np.float32),
            "plane_normal": np.asarray(ts["plane_normal"][task_idx], dtype=np.float32),
            "seg_count": int(ts["seg_count"][task_idx]),
            "total_length": float(ts["total_length"][task_idx]),
            "subseg_meta": subseg_meta,
        }

        raw = f["raw_trajs"]
        raw_off = raw["offset"][()]
        raw_plane_normals = raw["plane_normal"][()]
        raw_plane_points = raw["plane_point"][()]
        raw_segments = []
        for traj_id, start, end in subseg_meta:
            r_lo, r_hi = int(raw_off[traj_id]), int(raw_off[traj_id + 1])
            full_tcp = np.asarray(raw["tcp_flat"][r_lo:r_hi], dtype=np.float32)
            full_q = np.asarray(raw["q_flat"][r_lo:r_hi], dtype=np.float32)
            sub_tcp = full_tcp[int(start):int(end)]
            sub_q = full_q[int(start):int(end)]
            direction = np.asarray(raw["direction"][traj_id], dtype=np.float32)
            length = float(np.linalg.norm(sub_tcp[-1] - sub_tcp[0])) if sub_tcp.shape[0] > 1 else 0.0
            raw_segments.append(
                {
                    "seg_id": int(traj_id),
                    "start": int(start),
                    "end": int(end),
                    "plane_normal": np.asarray(raw_plane_normals[int(traj_id)], dtype=np.float32),
                    "plane_point": np.asarray(raw_plane_points[int(traj_id)], dtype=np.float32),
                    "tcp": sub_tcp,
                    "q": sub_q,
                    "direction": direction,
                    "length": length,
                }
            )
        task["raw_segments"] = raw_segments

        meta_attrs = f["meta"].attrs
        if str(meta_attrs.get("source_sampling_mode", "")) == "fixed_desk":
            task["desk"] = {
                "center": np.asarray(meta_attrs["source_desk_center"], dtype=np.float32),
                "normal": np.asarray(meta_attrs["source_desk_normal"], dtype=np.float32),
                "x_half": float(meta_attrs["source_desk_x_half"]),
                "y_half": float(meta_attrs["source_desk_y_half"]),
            }
        else:
            task["desk"] = None
    return task_idx, task


def rotation_matrix_from_normal(normal: np.ndarray) -> np.ndarray:
    z_axis = normal / max(float(np.linalg.norm(normal)), 1e-12)
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(float(z_axis[0])) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
    x_axis = np.cross(helper, z_axis); x_axis /= max(float(np.linalg.norm(x_axis)), 1e-12)
    y_axis = np.cross(z_axis, x_axis); y_axis /= max(float(np.linalg.norm(y_axis)), 1e-12)
    return np.column_stack((x_axis, y_axis, z_axis)).astype(np.float32)


def main() -> None:
    args = parse_args()
    task_idx, task = load_task(args.h5, args.task_idx, args.seed, args.only_composites, args.min_segs)

    print(
        f"[composite {task_idx}] seg_count={task['seg_count']} "
        f"total_length={task['total_length']:.4f}m "
        f"subseg_meta(traj_id,start,end)={task['subseg_meta'].tolist()}"
    )
    kinds = task["token_kind"]
    corners = np.where(kinds == TOKEN_KIND_CORNER)[0]
    for ci in corners:
        tok = task["tokens"][ci]
        sin_t, cos_t = float(tok[7]), float(tok[8])
        delta_theta_deg = float(np.degrees(np.arctan2(sin_t, cos_t)))
        cum_len_norm = float(tok[18])
        print(f"[corner token {ci}] Δθ={delta_theta_deg:+.2f}deg cum_len_norm={cum_len_norm:.3f}")

    tcp_pos = task["tcp"]
    q_path = task["qtraj"]

    world = ovw.World(cam_pos=[1.8, -1.6, 1.1], cam_lookat_pos=[0.2, 0.0, 0.35])
    ossop.frame().attach_to(world.scene)

    # Shared desk plane (one per composite, since all sub-segments live on the same desk).
    # Rendered larger than the sampling region so the full FR3 reachable workspace is covered.
    if not args.no_desk and task.get("desk") is not None:
        desk = task["desk"]
        desk_rot = rotation_matrix_from_normal(desk["normal"])
        DESK_RENDER_HALF = 0.85  # FR3 reach ≈ 0.855 m → a ~1.7 m × 1.7 m slab covers the workspace
        ossop.box(
            half_extents=(DESK_RENDER_HALF, DESK_RENDER_HALF, 0.0015),
            pos=desk["center"],
            rotmat=desk_rot,
            rgb=np.array([0.82, 0.78, 0.68], dtype=np.float32),
            alpha=0.55,
        ).attach_to(world.scene)
        print(
            f"[desk] center={desk['center'].tolist()} normal={desk['normal'].tolist()} "
            f"rendered={2*DESK_RENDER_HALF:.2f}×{2*DESK_RENDER_HALF:.2f}m"
        )

    # TCP path per sub-segment, color-coded. Consecutive segments share a corner vertex
    # (midpoint of the two TCP samples bracketing the intersection) so the composite path
    # renders as a single continuous line with sharp corners instead of two disjoint strokes.
    segments = task["raw_segments"]
    segment_tcps = [rs["tcp"].copy() for rs in segments]
    for k in range(len(segments) - 1):
        corner = 0.5 * (segment_tcps[k][-1] + segment_tcps[k + 1][0])
        segment_tcps[k][-1] = corner
        segment_tcps[k + 1][0] = corner

    for seg_i, (rs, tcp_seg) in enumerate(zip(segments, segment_tcps)):
        color = SEGMENT_COLORS[seg_i % SEGMENT_COLORS.shape[0]]
        for j in range(tcp_seg.shape[0] - 1):
            ossop.cylinder(
                spos=tcp_seg[j], epos=tcp_seg[j + 1],
                radius=0.0035, rgb=color, alpha=0.95,
            ).attach_to(world.scene)
        if seg_i == 0:
            ossop.sphere(pos=tcp_seg[0], radius=0.008, rgb=color, alpha=0.95).attach_to(world.scene)
        if seg_i == len(segments) - 1:
            ossop.sphere(pos=tcp_seg[-1], radius=0.008, rgb=color, alpha=0.95).attach_to(world.scene)
        else:
            ossop.sphere(pos=tcp_seg[-1], radius=0.010,
                         rgb=np.array([1.0, 1.0, 0.2], dtype=np.float32),
                         alpha=1.0).attach_to(world.scene)
        print(
            f"[seg {seg_i}] traj_id={rs['seg_id']} range=[{rs['start']},{rs['end']}) "
            f"length={rs['length']:.4f}m color={color.tolist()}"
        )

    # Overall start / end markers.
    ossop.sphere(pos=tcp_pos[0], radius=0.012,
                 rgb=np.array([0.15, 0.45, 1.0], dtype=np.float32),
                 alpha=1.0).attach_to(world.scene)
    ossop.sphere(pos=tcp_pos[-1], radius=0.013,
                 rgb=np.array([1.00, 0.20, 0.20], dtype=np.float32),
                 alpha=1.0).attach_to(world.scene)

    # Auto-save the joint-space curve plot alongside the live viz.
    if not args.no_save_qtraj:
        from Yuan.fr3_dit.visualization.plot_composite_qtraj import save_qtraj_plot
        save_qtraj_plot(args.h5, task_idx, args.qtraj_out)

    # Animate the robot along the stitched joint-space trajectory.
    robot = PenFrankaResearch3(name="pen", enable_cc=True)
    stride = max(1, int(args.stride))
    pose_indices = list(range(0, q_path.shape[0], stride))
    if pose_indices[-1] != q_path.shape[0] - 1:
        pose_indices.append(q_path.shape[0] - 1)
    from Yuan.fr3_dit.core import viz_utils
    frame_delay = max(0.02, 0.2 / max(args.speed, 1e-3))
    viz_utils.visualize_anime_path(world, robot, q_path[pose_indices], frame_delay=frame_delay)


if __name__ == "__main__":
    main()

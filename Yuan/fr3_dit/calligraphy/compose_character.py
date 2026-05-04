"""Compose a Chinese character via retrieval + per-frame IK on the canonical polyline.

Pipeline:
  1. Place character → canonical polylines (perfect geometry: closed boxes etc.).
  2. For each stroke:
       a) Densify polyline at 1 mm spacing → ordered TCP waypoints.
       b) Build pen-into-desk TCP rotation (z-axis = -desk_normal, x-axis = first
          segment direction).
       c) Retrieval: find a training task whose intrinsic shape (length / corners)
          matches the target stroke. Load its first stored q as the IK *seed*.
       d) IK each waypoint sequentially with the previous q as the running seed
          (joint-space continuity preserved).
  3. Concatenate strokes with Cartesian-IK pen-lift between them.
  4. Animate in panda3d.

The canonical geometry guarantees the character closes; the retrieval seeds the
IK with a known-feasible joint configuration so we land on a "natural" branch.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw

from Yuan.fr3_dit.calligraphy.character_def import list_characters, place_character
from Yuan.fr3_dit.calligraphy.retrieve_strokes import StrokeRetrieval
from Yuan.fr3_dit.core.pen_fr3_robot import PenFrankaResearch3
from Yuan.fr3_dit.core.viz_utils import visualize_anime_path


STROKE_COLORS = np.array([
    [0.10, 0.85, 0.20], [0.10, 0.45, 1.00], [0.95, 0.55, 0.15],
    [0.85, 0.20, 0.70], [0.65, 0.80, 0.20],
], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--char", type=str, default="中",
                   help=f"Character to compose. Known: {list_characters()}")
    p.add_argument("--size", type=float, default=0.30,
                   help="Character bounding-box half-extent in meters (default 30 cm so that all "
                        "single-segment strokes are ≥ 30 cm — within the 1-seg training distribution).")
    p.add_argument("--x", type=float, default=0.5)
    p.add_argument("--y", type=float, default=0.0)
    p.add_argument("--z", type=float, default=-0.05)
    p.add_argument("--desk-normal", type=float, nargs=3, default=[0.0, 0.0, 1.0])
    p.add_argument("--theta-deg", type=float, default=0.0)
    p.add_argument("--lift-height", type=float, default=0.04)
    p.add_argument("--interp-frames", type=int, default=40)
    p.add_argument("--step-mm", type=float, default=1.0,
                   help="Densification spacing for IK along each stroke (mm). Default 1.0.")
    p.add_argument("--data", type=Path, default=None,
                   help="HDF5 path for retrieval index. Default: project default.")
    p.add_argument("--no-retrieval-seed", action="store_true", default=False,
                   help="Skip retrieval and use the FR3 home configuration as IK seed.")
    p.add_argument("--frame-delay", type=float, default=0.01)
    p.add_argument("--playback-stride", type=int, default=5)
    p.add_argument("--no-animate", action="store_true")
    return p.parse_args()


def densify_polyline(polyline: np.ndarray, step_m: float) -> np.ndarray:
    """Resample a polyline to evenly-spaced points at ``step_m`` along the arc length.

    Returns shape (N, 3). Always includes original start and end vertices.
    """
    pts = np.asarray(polyline, dtype=np.float64)
    out = [pts[0].copy()]
    for i in range(pts.shape[0] - 1):
        a, b = pts[i], pts[i + 1]
        seg_len = float(np.linalg.norm(b - a))
        if seg_len < 1e-9:
            continue
        n_steps = max(1, int(np.ceil(seg_len / step_m)))
        for k in range(1, n_steps + 1):
            alpha = k / n_steps
            out.append((1.0 - alpha) * a + alpha * b)
    return np.stack(out, axis=0).astype(np.float64)


def build_pen_rotmat(desk_normal: np.ndarray, first_seg_dir_world: np.ndarray) -> np.ndarray:
    """Pen-into-desk TCP rotation. z = -desk_normal, x = first_seg_dir (projected),
    y = z × x."""
    z = -desk_normal / max(float(np.linalg.norm(desk_normal)), 1e-12)
    d = np.asarray(first_seg_dir_world, dtype=np.float64)
    x = d - z * float(np.dot(z, d))
    nx = float(np.linalg.norm(x))
    if nx < 1e-9:
        helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = helper - z * float(np.dot(z, helper))
        x /= max(float(np.linalg.norm(x)), 1e-12)
    else:
        x /= nx
    y = np.cross(z, x); y /= max(float(np.linalg.norm(y)), 1e-12)
    return np.column_stack((x, y, z)).astype(np.float64)


def ik_polyline(
    pen_robot,
    tcp_waypoints: np.ndarray,            # (N, 3) world XYZ
    tcp_rotmat: np.ndarray,               # (3, 3) — same orientation throughout the stroke
    seed_q: np.ndarray,                   # (7,) — initial IK seed
) -> tuple[np.ndarray, int]:
    """IK each waypoint sequentially. Returns (q_traj (N, 7), n_failures)."""
    out = np.zeros((tcp_waypoints.shape[0], 7), dtype=np.float32)
    n_fail = 0
    seed = np.asarray(seed_q, dtype=np.float64).reshape(7)
    for i, p in enumerate(tcp_waypoints):
        try:
            sol = pen_robot.ik(tgt_pos=p.astype(np.float64),
                                tgt_rotmat=tcp_rotmat, seed_jnt_values=seed)
        except Exception:
            sol = None
        if sol is None:
            n_fail += 1
            sol = seed   # fall back to previous q (frame frozen)
        out[i] = np.asarray(sol, dtype=np.float32)
        seed = np.asarray(sol, dtype=np.float64)
    return out, n_fail


def cartesian_lift_q_path(
    pen_robot, q_at_stroke_end: np.ndarray, q_at_next_start: np.ndarray,
    desk_normal: np.ndarray, lift_height: float, n_frames: int,
) -> np.ndarray:
    """Cartesian inverted-U pen-lift between strokes (same idea as draw_character)."""
    q_a = np.asarray(q_at_stroke_end, dtype=np.float64).reshape(7)
    q_b = np.asarray(q_at_next_start, dtype=np.float64).reshape(7)
    n_normal = np.asarray(desk_normal, dtype=np.float64).reshape(3)
    n_normal /= max(float(np.linalg.norm(n_normal)), 1e-12)

    pen_robot.goto_given_conf(q_a)
    tcp_a = np.asarray(pen_robot.manipulator.gl_tcp_pos, dtype=np.float64)
    rot_a = np.asarray(pen_robot.manipulator.gl_tcp_rotmat, dtype=np.float64)
    pen_robot.goto_given_conf(q_b)
    tcp_b = np.asarray(pen_robot.manipulator.gl_tcp_pos, dtype=np.float64)
    rot_b = np.asarray(pen_robot.manipulator.gl_tcp_rotmat, dtype=np.float64)

    tcp_a_lift = tcp_a + n_normal * float(lift_height)
    tcp_b_lift = tcp_b + n_normal * float(lift_height)

    n_each = max(4, n_frames // 3)

    def waypts(s, e, n):
        alphas = np.linspace(0, 1, n + 1)[1:]
        return [(1.0 - a) * s + a * e for a in alphas]

    out = []
    seed = q_a
    for p in waypts(tcp_a, tcp_a_lift, n_each) + waypts(tcp_a_lift, tcp_b_lift, n_each):
        sol = pen_robot.ik(tgt_pos=p, tgt_rotmat=rot_a, seed_jnt_values=seed)
        if sol is None:
            sol = seed + 0.05 * (q_b - seed)
        sol = np.asarray(sol, dtype=np.float64)
        out.append(sol); seed = sol
    n_down = len(waypts(tcp_b_lift, tcp_b, n_each))
    for i, p in enumerate(waypts(tcp_b_lift, tcp_b, n_each)):
        a = (i + 1) / n_down
        rot_blend = (1.0 - a) * rot_a + a * rot_b
        u, _, vt = np.linalg.svd(rot_blend); rot_blend = u @ vt
        if np.linalg.det(rot_blend) < 0:
            u[:, -1] *= -1; rot_blend = u @ vt
        sol = pen_robot.ik(tgt_pos=p, tgt_rotmat=rot_blend, seed_jnt_values=seed)
        if sol is None:
            sol = seed + (1.0 / max(n_down - i, 1)) * (q_b - seed)
        sol = np.asarray(sol, dtype=np.float64)
        out.append(sol); seed = sol
    out.append(q_b)
    return np.stack(out, axis=0).astype(np.float32)


def rotation_matrix_from_normal(normal: np.ndarray) -> np.ndarray:
    z = normal / max(float(np.linalg.norm(normal)), 1e-12)
    helper = np.array([1.0, 0.0, 0.0]) if abs(float(z[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(helper, z); x /= max(float(np.linalg.norm(x)), 1e-12)
    y = np.cross(z, x); y /= max(float(np.linalg.norm(y)), 1e-12)
    return np.column_stack((x, y, z)).astype(np.float32)


def main() -> None:
    args = parse_args()
    desk_center = np.array([args.x, args.y, args.z], dtype=np.float32)
    desk_normal = np.asarray(args.desk_normal, dtype=np.float32)
    desk_normal /= max(float(np.linalg.norm(desk_normal)), 1e-12)
    theta_rad = float(np.deg2rad(args.theta_deg))

    print(f"[char] '{args.char}' @ size={args.size*100:.1f}cm  "
          f"centre=({args.x:.2f},{args.y:.2f},{args.z:.2f})  theta={args.theta_deg:.1f}°")
    strokes_world = place_character(args.char, desk_center, desk_normal,
                                     size_m=args.size, theta_rad=theta_rad)
    print(f"[char] {len(strokes_world)} strokes")

    pen_robot = PenFrankaResearch3(name="pen", enable_cc=False)
    home_q = np.zeros(7, dtype=np.float32)   # safe fallback seed

    retriever = None
    if not args.no_retrieval_seed:
        retriever = StrokeRetrieval(args.data) if args.data is not None else StrokeRetrieval()

    stroke_q_trajs: list[np.ndarray] = []
    print("\n[compose] per-stroke IK along canonical polyline...")
    for i, poly in enumerate(strokes_world):
        # 1. Densify polyline.
        wpts = densify_polyline(poly, step_m=args.step_mm / 1000.0)
        # 2. TCP rotation: pen pointing into desk along first seg dir.
        first_seg_dir = poly[1] - poly[0]
        first_seg_dir /= max(float(np.linalg.norm(first_seg_dir)), 1e-12)
        rot = build_pen_rotmat(desk_normal, first_seg_dir)
        # 3. Get IK seed: retrieval q0 (best shape) or home.
        if retriever is not None:
            ms = retriever.query(poly, k=1, position_weight=0.0,
                                  len_weight=10.0, corner_weight=3.0,
                                  max_scale_dev=0.30, max_angle_dev_rad=0.52)
            if ms:
                with h5py.File(retriever.h5_path, "r") as f:
                    q_off = retriever.qtraj_offset
                    qtraj_flat = f["tasks/qtraj_flat"]
                    q_lo = int(q_off[ms[0].task_idx])
                    seed_q = np.asarray(qtraj_flat[q_lo], dtype=np.float32)
                src_label = f"retrieval task #{ms[0].task_idx}"
            else:
                seed_q = home_q.copy()
                src_label = "home (no retrieval match)"
        else:
            seed_q = home_q.copy()
            src_label = "home (--no-retrieval-seed)"

        # 4. IK each waypoint.
        q_traj, n_fail = ik_polyline(pen_robot, wpts, rot, seed_q)
        stroke_q_trajs.append(q_traj)
        print(f"  stroke {i+1}/{len(strokes_world)}: {wpts.shape[0]} waypoints, "
              f"IK_fail={n_fail}, seed={src_label}")

    if args.no_animate:
        return

    # 5. Concatenate with pen-lifts.
    full_q = [stroke_q_trajs[0]]
    for k in range(1, len(stroke_q_trajs)):
        q_end = stroke_q_trajs[k - 1][-1]
        q_start = stroke_q_trajs[k][0]
        if not np.allclose(q_end, q_start, atol=1e-3):
            transition = cartesian_lift_q_path(
                pen_robot, q_end, q_start, desk_normal,
                args.lift_height, args.interp_frames,
            )
            full_q.append(transition)
        full_q.append(stroke_q_trajs[k])
    full_q = np.concatenate(full_q, axis=0).astype(np.float32)
    print(f"[compose] concatenated trajectory: {full_q.shape[0]} frames "
          f"({len(stroke_q_trajs)} strokes + {max(0, len(stroke_q_trajs)-1)} pen-lifts)")

    # 6. Visualize.
    world = ovw.World(cam_pos=[1.8, -1.6, 1.1],
                      cam_lookat_pos=[args.x, args.y, args.z + 0.4])
    ossop.frame().attach_to(world.scene)
    rot = rotation_matrix_from_normal(desk_normal)
    ossop.box(half_extents=(0.85, 0.85, 0.0015), pos=desk_center, rotmat=rot,
              rgb=np.array([0.82, 0.78, 0.68], dtype=np.float32),
              alpha=0.4).attach_to(world.scene)
    for i, poly in enumerate(strokes_world):
        c = STROKE_COLORS[i % STROKE_COLORS.shape[0]]
        for j in range(poly.shape[0] - 1):
            ossop.cylinder(spos=poly[j], epos=poly[j + 1],
                           radius=0.0035, rgb=c, alpha=0.85).attach_to(world.scene)

    # Black TCP trace for verification.
    rollout_tcp = []
    for q in full_q:
        pen_robot.goto_given_conf(q.astype(np.float32))
        ee_pos, _ = pen_robot.fk(q.astype(np.float32))
        rollout_tcp.append(np.asarray(ee_pos, dtype=np.float32))
    rollout_tcp = np.stack(rollout_tcp, axis=0)
    for j in range(rollout_tcp.shape[0] - 1):
        ossop.cylinder(spos=rollout_tcp[j], epos=rollout_tcp[j + 1],
                       radius=0.0018,
                       rgb=np.array([0.05, 0.05, 0.05], dtype=np.float32),
                       alpha=0.6).attach_to(world.scene)

    stride = max(1, int(args.playback_stride))
    anim_path = full_q[::stride]
    if anim_path.shape[0] == 0 or not np.array_equal(anim_path[-1], full_q[-1]):
        anim_path = np.concatenate([anim_path, full_q[-1:]], axis=0)
    print(f"\n[viz] {anim_path.shape[0]}/{full_q.shape[0]} frames @ {args.frame_delay}s "
          f"→ ~{anim_path.shape[0]*args.frame_delay:.1f}s playback.")
    visualize_anime_path(world, pen_robot, list(anim_path), frame_delay=float(args.frame_delay))


if __name__ == "__main__":
    main()

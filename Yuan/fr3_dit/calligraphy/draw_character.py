"""End-to-end Chinese-character drawing demo.

For a named character at a chosen (size, position, orientation):
1. Place character → list of stroke polylines in world frame.
2. For each stroke: tokenize → query oracle → pick best candidate → grab its
   full q-trajectory.
3. Between strokes: lift pen along desk_normal, q-space-interpolate to next
   stroke's start q (after IK refine), lower pen.
4. Animate the entire concatenated joint trajectory in panda3d, with the GT
   stroke polylines drawn for spatial context.

Usage:
    python -m fr3_dit.calligraphy.draw_character \
        --char 中 --size 0.08 --x 0.5 --y 0.0 --theta-deg 0
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch

import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw

from Yuan.fr3_dit.calligraphy.character_def import list_characters, place_character
from Yuan.fr3_dit.calligraphy.feasibility_check import FeasibilityOracle, StrokeResult
from Yuan.fr3_dit.calligraphy.polyline_to_tokens import tokenize_stroke
from Yuan.fr3_dit.core.pen_fr3_robot import PenFrankaResearch3
from Yuan.fr3_dit.core.viz_utils import visualize_anime_path


# Per-stroke palette (cycled for n>5 strokes).
STROKE_COLORS = np.array([
    [0.10, 0.85, 0.20], [0.10, 0.45, 1.00], [0.95, 0.55, 0.15],
    [0.85, 0.20, 0.70], [0.65, 0.80, 0.20],
], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--char", type=str, default="中",
                   help=f"Character to draw. Known: {list_characters()}")
    p.add_argument("--size", type=float, default=0.08,
                   help="Character bounding-box half-extent in meters (default 8cm).")
    p.add_argument("--x", type=float, default=0.5,
                   help="Desk-frame X (m) of character centre.")
    p.add_argument("--y", type=float, default=0.0,
                   help="Desk-frame Y (m) of character centre.")
    p.add_argument("--z", type=float, default=-0.05,
                   help="Desk-frame Z (m) — should equal source_desk_center[2] (default -0.05).")
    p.add_argument("--desk-normal", type=float, nargs=3, default=[0.0, 0.0, 1.0])
    p.add_argument("--theta-deg", type=float, default=0.0,
                   help="Rotate the character about desk_normal by this many degrees.")
    p.add_argument("--lift-height", type=float, default=0.04,
                   help="Pen-lift height (m) between strokes (default 4cm above desk).")
    p.add_argument("--interp-frames", type=int, default=40,
                   help="Number of q-interp frames per pen-up / pen-down + horizontal move.")
    p.add_argument("--ckpt", type=Path, default=None,
                   help="DiT ckpt path (defaults to dit_q0_v5_ckpts/final.pt).")
    p.add_argument("--n-candidates", type=int, default=8,
                   help="Number of q0 candidates DiT samples per stroke.")
    p.add_argument("--top-k-rollout", type=int, default=None,
                   help="Rollout only the top-K candidates by DiT self-score (option B). "
                        "Default = n_candidates (no filtering). Setting K=2 gives a ~4x rollout "
                        "speedup on 8 candidates with little hit-rate loss.")
    p.add_argument("--cfg-w", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for DiT sampling (controls candidate diversity).")
    p.add_argument("--frame-delay", type=float, default=0.01)
    p.add_argument("--playback-stride", type=int, default=5)
    p.add_argument("--no-animate", action="store_true",
                   help="Skip panda3d animation; print feasibility only.")
    return p.parse_args()


def linear_q_interp(q_start: np.ndarray, q_end: np.ndarray, n_steps: int) -> np.ndarray:
    """Joint-space linear interpolation. Returns (n_steps, 7), excluding q_start."""
    q_start = np.asarray(q_start, dtype=np.float32).reshape(7)
    q_end = np.asarray(q_end, dtype=np.float32).reshape(7)
    if n_steps <= 0:
        return np.zeros((0, 7), dtype=np.float32)
    alphas = np.linspace(0.0, 1.0, n_steps + 1, dtype=np.float32)[1:]
    return (1.0 - alphas[:, None]) * q_start + alphas[:, None] * q_end


def cartesian_lift_q_path(
    pen_robot,
    q_at_stroke_end: np.ndarray,
    q_at_next_start: np.ndarray,
    desk_normal: np.ndarray,
    lift_height: float,
    n_frames: int,
) -> np.ndarray:
    """Pen-lift trajectory: TCP follows an inverted-U in Cartesian space, IK each frame.

    Phase 1: lift up   — TCP moves from current → current + lift_height·desk_normal
    Phase 2: traverse  — TCP moves from current_lifted → next_lifted (above desk plane)
    Phase 3: lower     — TCP moves from next_lifted → next_start

    Each frame solves IK with seed = previous q so the joint trajectory stays smooth.
    The TCP rotation is held at q_at_stroke_end's rotation through the lift; on the
    final descent it's interpolated to q_at_next_start's rotation in q-space (handled
    by IK seed continuity + a final blend frame).

    Returns (M, 7) joint trajectory, EXCLUDING ``q_at_stroke_end`` (caller already has
    that frame), but ENDING with ``q_at_next_start`` so the next stroke continues
    cleanly.
    """
    q_a = np.asarray(q_at_stroke_end, dtype=np.float32).reshape(7).astype(np.float64)
    q_b = np.asarray(q_at_next_start, dtype=np.float32).reshape(7).astype(np.float64)
    n_normal = np.asarray(desk_normal, dtype=np.float64).reshape(3)
    n_normal = n_normal / max(float(np.linalg.norm(n_normal)), 1e-12)

    # FK at both endpoints (use the CPU pen robot already constructed elsewhere).
    pen_robot.goto_given_conf(q_a)
    tcp_a = np.asarray(pen_robot.manipulator.gl_tcp_pos, dtype=np.float64)
    rot_a = np.asarray(pen_robot.manipulator.gl_tcp_rotmat, dtype=np.float64)
    pen_robot.goto_given_conf(q_b)
    tcp_b = np.asarray(pen_robot.manipulator.gl_tcp_pos, dtype=np.float64)
    rot_b = np.asarray(pen_robot.manipulator.gl_tcp_rotmat, dtype=np.float64)

    tcp_a_lifted = tcp_a + n_normal * float(lift_height)
    tcp_b_lifted = tcp_b + n_normal * float(lift_height)

    n_each = max(4, n_frames // 3)

    def waypoints(p_start, p_end, n):
        alphas = np.linspace(0.0, 1.0, n + 1)[1:]
        return [(1.0 - a) * p_start + a * p_end for a in alphas]

    wpts_up      = waypoints(tcp_a, tcp_a_lifted, n_each)
    wpts_across  = waypoints(tcp_a_lifted, tcp_b_lifted, n_each)
    wpts_down    = waypoints(tcp_b_lifted, tcp_b, n_each)

    out: list[np.ndarray] = []
    seed = q_a
    # Phase 1 + 2: hold rotation = rot_a.
    for p in wpts_up + wpts_across:
        sol = pen_robot.ik(tgt_pos=p, tgt_rotmat=rot_a, seed_jnt_values=seed)
        if sol is None:
            # Fall back: linear-q from previous to a tiny step toward q_b (best-effort).
            sol = seed + 0.05 * (q_b - seed)
        sol = np.asarray(sol, dtype=np.float64)
        out.append(sol)
        seed = sol
    # Phase 3: blend rotation from rot_a → rot_b along the descent.
    n_down = len(wpts_down)
    for i, p in enumerate(wpts_down):
        a = (i + 1) / n_down
        # Slerp via shortest-path quaternion blend (cheap fallback: linear blend + reorthonormalize).
        rot_blend = (1.0 - a) * rot_a + a * rot_b
        # Re-orthonormalize via SVD.
        u, _, vt = np.linalg.svd(rot_blend)
        rot_blend = u @ vt
        if np.linalg.det(rot_blend) < 0:
            u[:, -1] *= -1
            rot_blend = u @ vt
        sol = pen_robot.ik(tgt_pos=p, tgt_rotmat=rot_blend, seed_jnt_values=seed)
        if sol is None:
            sol = seed + (1.0 / max(n_down - i, 1)) * (q_b - seed)
        sol = np.asarray(sol, dtype=np.float64)
        out.append(sol)
        seed = sol

    # Append the exact q_b as the final frame so the next stroke's segments line up.
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
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    desk_center = np.array([args.x, args.y, args.z], dtype=np.float32)
    desk_normal = np.asarray(args.desk_normal, dtype=np.float32)
    desk_normal /= max(float(np.linalg.norm(desk_normal)), 1e-12)
    theta_rad = float(np.deg2rad(args.theta_deg))

    print(f"[char] '{args.char}' @ size={args.size*100:.1f}cm  "
          f"centre=({args.x:.2f},{args.y:.2f},{args.z:.2f})  theta={args.theta_deg:.1f}°")

    # 1) Place character into world.
    strokes_world = place_character(args.char, desk_center, desk_normal,
                                     size_m=float(args.size), theta_rad=theta_rad)
    print(f"[char] {len(strokes_world)} strokes")

    # 2) Build oracle (loads DiT + tracker).
    oracle_kwargs = {} if args.ckpt is None else {"ckpt": args.ckpt}
    oracle = FeasibilityOracle(
        n_candidates=args.n_candidates,
        top_k_rollout=args.top_k_rollout,
        cfg_w=args.cfg_w,
        **oracle_kwargs,
    )

    # 3) Per-stroke evaluation; collect best q-trajectories.
    stroke_results: List[StrokeResult] = []
    stroke_q_trajs: List[np.ndarray | None] = []        # None = infeasible stroke, skip in animation
    print("\n[draw] querying DiT prior + IK refine + tracker per stroke...")
    for i, poly in enumerate(strokes_world):
        ts = tokenize_stroke(poly, desk_normal)
        result = oracle.evaluate_stroke(ts, desk_center, desk_normal)
        stroke_results.append(result)
        marker = "✓" if result.feasible else "✗"
        best = result.best()
        print(f"  stroke {i+1}/{len(strokes_world)} {marker}  "
              f"feasible={result.feasible}  n_success={result.n_success}/{result.n_candidates}  "
              f"best_completion={result.best_completion_pct*100:.1f}%  "
              f"best_dit_score={best.dit_score:+.3f}  "
              f"top_fail={best.top_failure_label}")
        # Only retain the q-trajectory if the stroke is feasible. For infeasible strokes,
        # the "best" candidate is whatever got farthest — its TCP often starts at a wrong
        # place (IK refine failed) and its trajectory is short / drifts visibly. Render it
        # as a stroke-skip in the animation instead of polluting the path.
        stroke_q_trajs.append(best.full_q_trajectory if result.feasible else None)

    n_feasible = sum(1 for r in stroke_results if r.feasible)
    print(f"\n[draw] character feasibility: {n_feasible}/{len(stroke_results)} strokes feasible")
    if n_feasible < len(stroke_results):
        print("       infeasible strokes are SKIPPED in the animation (their best candidate's")
        print("       q-trajectory typically starts from a wrong TCP and drifts visibly).")
        print("       Try a smaller --size or different --theta-deg.")
    if args.no_animate:
        return
    if n_feasible == 0:
        print("[draw] nothing to animate — all strokes infeasible.")
        return

    # 4) Concatenate ONLY the feasible strokes' q-trajectories with Cartesian-IK pen-lift
    # transitions between them. Infeasible strokes (entries == None) are skipped: pen-lift
    # bridges the previous feasible stroke's end → next feasible stroke's start.
    # Reuse oracle.ik_robot so we don't trigger a second SELIK CVT generation (~60s).
    pen_robot_lift = oracle.ik_robot
    feasible_trajs = [t for t in stroke_q_trajs if t is not None]
    full_q = [feasible_trajs[0]]
    for k in range(1, len(feasible_trajs)):
        q_end = feasible_trajs[k - 1][-1]
        q_start = feasible_trajs[k][0]
        if not np.allclose(q_end, q_start, atol=1e-3):
            transition = cartesian_lift_q_path(
                pen_robot_lift, q_end, q_start,
                desk_normal, args.lift_height, args.interp_frames,
            )
            full_q.append(transition)
        full_q.append(feasible_trajs[k])
    full_q = np.concatenate(full_q, axis=0).astype(np.float32)
    print(f"[draw] concatenated trajectory: {full_q.shape[0]} frames "
          f"({len(feasible_trajs)} stroke segments + {max(0, len(feasible_trajs)-1)} pen-lifts)")

    # 5) Visualize.
    world = ovw.World(cam_pos=[1.8, -1.6, 1.1],
                      cam_lookat_pos=[args.x, args.y, args.z + 0.4])
    ossop.frame().attach_to(world.scene)
    rot = rotation_matrix_from_normal(desk_normal)
    ossop.box(
        half_extents=(0.85, 0.85, 0.0015), pos=desk_center, rotmat=rot,
        rgb=np.array([0.82, 0.78, 0.68], dtype=np.float32), alpha=0.4,
    ).attach_to(world.scene)

    # GT polylines (target stroke shapes) in world frame, colored per stroke.
    for i, poly in enumerate(strokes_world):
        c = STROKE_COLORS[i % STROKE_COLORS.shape[0]]
        for j in range(poly.shape[0] - 1):
            ossop.cylinder(spos=poly[j], epos=poly[j + 1],
                           radius=0.0035, rgb=c, alpha=0.85).attach_to(world.scene)

    # Rollout TCP trace (black) by FK-ing every frame.
    fr3_cpu = PenFrankaResearch3(name="pen", enable_cc=False)
    rollout_tcp = []
    for q in full_q:
        fr3_cpu.goto_given_conf(q.astype(np.float32))
        ee_pos, _ = fr3_cpu.fk(q.astype(np.float32))
        rollout_tcp.append(np.asarray(ee_pos, dtype=np.float32))
    rollout_tcp = np.stack(rollout_tcp, axis=0)
    for j in range(rollout_tcp.shape[0] - 1):
        ossop.cylinder(spos=rollout_tcp[j], epos=rollout_tcp[j + 1],
                       radius=0.0018,
                       rgb=np.array([0.05, 0.05, 0.05], dtype=np.float32),
                       alpha=0.6).attach_to(world.scene)

    anim_robot = PenFrankaResearch3(name="pen", enable_cc=False)
    stride = max(1, int(args.playback_stride))
    anim_path = full_q[::stride]
    if anim_path.shape[0] == 0 or not np.array_equal(anim_path[-1], full_q[-1]):
        anim_path = np.concatenate([anim_path, full_q[-1:]], axis=0)
    print(f"\n[viz] animation: {anim_path.shape[0]}/{full_q.shape[0]} frames "
          f"(stride={stride}) @ {args.frame_delay}s/frame  →  "
          f"~{anim_path.shape[0]*args.frame_delay:.1f}s playback.")
    visualize_anime_path(world, anim_robot, list(anim_path), frame_delay=float(args.frame_delay))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Animate the FR3 starting from a predicted q0 and rolling out via PlaneConstrainedTracker.

Picks the best-of-N candidate by default (or any --rank-k), runs the same per-segment
rollout that eval_tracker does, concatenates the resulting joint trajectory, and animates
the robot in panda3d. Prints per-segment success/failure with the termination label.

Use this to *see* whether a predicted q0 actually completes the planned stroke.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch

import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw
from one.robots.manipulators.franka.fr3_pen import make_pen_collision_helpers

from Yuan.fr3_dit.core.pen_fr3_robot import PenFrankaResearch3, PenFrankaResearch3GPU
from Yuan.fr3_dit.core.viz_utils import visualize_anime_path
from Yuan.fr3_dit.data_generation.generate_fr3_plane_dataset import (
    PlaneConstrainedTracker,
    TrackerConfig,
    termination_label,
)
from Yuan.fr3_dit.training.eval_tracker import decode_segments_world
from Yuan.fr3_dit.training.ik_refine import refine_q0_seed


DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "outputs"
DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "pen_fr3_composite_tasks_50k_minseg10_anchored.hdf5"


SEG_COLORS = np.array([
    [0.10, 0.85, 0.20], [0.10, 0.45, 1.00], [0.95, 0.55, 0.15],
    [0.85, 0.20, 0.70], [0.65, 0.80, 0.20],
], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task-idx", type=int, required=True)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out-prefix", type=str, default="infer_q0_v5",
                   help="Filename prefix used at inference time.")
    p.add_argument("--rank-k", type=int, default=0,
                   help="Which candidate to roll out, 0 = best by RMSE (default), -1 = use GT q0 instead.")
    p.add_argument("--refine-ik", action="store_true", default=False,
                   help="Run wrs IK from the chosen q0 seed targeting the exact path-start TCP "
                        "(orientation = seed's own TCP rotation), then roll out from the refined q.")
    p.add_argument("--length-ref", type=float, default=0.30,
                   help="Length normalization used when generating the tokens.")
    p.add_argument("--theta-max-deg", type=float, default=30.0)
    p.add_argument("--angle-null-gain", type=float, default=1.0,
                   help="Strength of boundary brake (active near/past the theta_max cone). "
                        "Default 1.0 (training/data-gen used 0.4).")
    p.add_argument("--angle-attract-gain", type=float, default=2.0,
                   help="Always-on interior attractor: pulls TCP_z toward -desk_normal at every step "
                        "proportional to angle deviation (radians). Default 2.0 (5.0 was too aggressive "
                        "after IK refine — pushed wrist into self-collision in batch eval).")
    p.add_argument("--max-steps-buffer", type=int, default=30)
    p.add_argument("--frame-delay", type=float, default=0.01,
                   help="Seconds between animation frames (smaller = faster playback). Default 0.01.")
    p.add_argument("--playback-stride", type=int, default=5,
                   help="Subsample the rollout trajectory: draw every Nth frame. Default 5 = 5x faster "
                        "than full-rate playback. Use 1 to see every tracker step.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def rotation_matrix_from_normal(normal: np.ndarray) -> np.ndarray:
    z = normal / max(float(np.linalg.norm(normal)), 1e-12)
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(float(z[0])) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
    x = np.cross(helper, z); x /= max(float(np.linalg.norm(x)), 1e-12)
    y = np.cross(z, x); y /= max(float(np.linalg.norm(y)), 1e-12)
    return np.column_stack((x, y, z)).astype(np.float32)


def load_task(h5_path: Path, idx: int):
    with h5py.File(h5_path, "r") as f:
        ts = f["tasks"]
        tok_off = ts["token_offset"][()]
        ss_off = ts["subseg_offset"][()]
        t_lo, t_hi = int(tok_off[idx]), int(tok_off[idx + 1])
        s_lo, s_hi = int(ss_off[idx]), int(ss_off[idx + 1])
        tokens = np.asarray(ts["token_flat"][t_lo:t_hi], dtype=np.float32)
        kinds = np.asarray(ts["token_kind"][t_lo:t_hi], dtype=np.uint8)
        local_frame = np.asarray(ts["local_frame"][idx], dtype=np.float32)
        local_origin = np.asarray(ts["local_origin"][idx], dtype=np.float32)
        seg_count = int(ts["seg_count"][idx])
        total_length = float(ts["total_length"][idx])
        start_q = np.asarray(ts["start_q"][idx], dtype=np.float32)
        subseg_meta = np.asarray(ts["subseg_meta_flat"][s_lo:s_hi], dtype=np.int32)
        raw = f["raw_trajs"]
        raw_off = raw["offset"][()]
        raw_tcp = raw["tcp_flat"]
        gt_tcp_chunks = []
        for traj_id, st, en in subseg_meta:
            r_lo = int(raw_off[int(traj_id)])
            gt_tcp_chunks.append(np.asarray(raw_tcp[r_lo + int(st): r_lo + int(en)], dtype=np.float32))
        ma = f["meta"].attrs
        desk_center = np.asarray(ma["source_desk_center"], dtype=np.float32)
        desk_normal = np.asarray(ma["source_desk_normal"], dtype=np.float32)
    return {
        "tokens": tokens, "kinds": kinds, "local_frame": local_frame,
        "local_origin": local_origin,
        "seg_count": seg_count, "total_length": total_length,
        "start_q": start_q, "gt_tcp_chunks": gt_tcp_chunks,
        "desk_center": desk_center, "desk_normal": desk_normal,
    }


def rollout_from_q0(
    tracker: PlaneConstrainedTracker,
    config: TrackerConfig,
    q0: np.ndarray,
    segments,
    desk_center: np.ndarray,
    desk_normal: np.ndarray,
    device: torch.device,
    max_steps_buffer: int = 30,
):
    """Single-candidate per-segment rollout — returns concatenated q-trajectory + per-seg result."""
    pen_axis = -desk_normal / max(float(np.linalg.norm(desk_normal)), 1e-12)
    step_dist = config.task_speed * config.dt
    q_now = torch.from_numpy(q0.astype(np.float32)).unsqueeze(0).to(device)  # (1, 7)
    plane_point = torch.from_numpy(desk_center.astype(np.float32)).unsqueeze(0).to(device)
    plane_normal = torch.from_numpy(pen_axis.astype(np.float32)).unsqueeze(0).to(device)
    plane_side = -torch.ones(1, dtype=torch.float32, device=device)

    full_traj = [q0.astype(np.float32).reshape(1, 7).copy()]
    seg_results = []
    failed = False
    for seg_idx, (dir_world, length_m) in enumerate(segments):
        if failed:
            seg_results.append({"completed": False, "term_label": "skipped (prior fail)",
                                 "steps": 0, "length_m": length_m})
            continue
        # max_steps gives the tracker headroom to reach the target (use ceil + buffer);
        # trim_steps is the exact integer-step length we want the segment to be (use round
        # so float imprecision in length_m doesn't add 1 step / 1mm of cumulative drift).
        max_target_steps = max(1, int(math.ceil(length_m / step_dist)))
        trim_steps = max(1, int(round(length_m / step_dist)))
        orig_max = config.max_steps
        config.max_steps = max_target_steps + max_steps_buffer
        dir_b = torch.from_numpy(dir_world.astype(np.float32)).unsqueeze(0).to(device)
        try:
            trajs = tracker.collect_batch_trajectories(q_now, plane_point, dir_b, plane_normal, plane_side)
        finally:
            config.max_steps = orig_max
        traj = trajs[0]
        steps = int(traj["num_points"]) - 1
        term_code = int(traj["termination_code"])
        q_seg = np.asarray(traj["q"], dtype=np.float32)            # (steps+1, 7)

        # When max_steps fires the tracker has overshot by buffer steps. Trim to exactly
        # ``trim_steps`` so traveled = target with no cumulative drift across segments.
        if term_code == 3 and steps > trim_steps:
            q_seg = q_seg[: trim_steps + 1]
            steps = trim_steps

        traveled = steps * step_dist
        full_traj.append(q_seg[1:])                                  # skip first (== q_now)
        completed = (term_code == 3) and (traveled >= length_m * 0.95)
        seg_results.append({
            "completed": completed,
            "term_code": term_code,
            "term_label": termination_label(term_code),
            "steps": steps, "length_m": length_m, "traveled_m": traveled,
        })
        if completed:
            q_now = torch.from_numpy(q_seg[-1]).unsqueeze(0).to(device)
        else:
            failed = True

    return np.concatenate(full_traj, axis=0), seg_results


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    idx = int(args.task_idx)

    task = load_task(args.data, idx)
    segments = decode_segments_world(task["tokens"], task["kinds"], task["local_frame"], args.length_ref)
    print(f"[task {idx}] seg_count={task['seg_count']} total_len={task['total_length']*100:.1f}cm "
          f"n_segments_decoded={len(segments)}")
    print(f"[debug] args.length_ref={args.length_ref}  decoded segments lengths (cm):",
          [round(L*100, 2) for _, L in segments])

    # --- Pick q0 ---
    if args.rank_k == -1:
        q0 = task["start_q"].copy()
        candidate_label = "GT q0 (rank=-1)"
        candidate_rmse = 0.0
    else:
        npy = args.out_dir / f"{args.out_prefix}_task{idx:06d}_q0_pred.npy"
        meta = args.out_dir / f"{args.out_prefix}_task{idx:06d}_meta.json"
        if not npy.exists():
            raise FileNotFoundError(f"{npy} not found — run inference with --out-prefix {args.out_prefix} first.")
        q0_preds = np.load(npy).astype(np.float32)
        rmses = np.asarray(json.loads(meta.read_text())["per_sample_rmse_rad"], dtype=np.float32)
        order = np.argsort(rmses)
        k = int(args.rank_k)
        if k < 0 or k >= len(q0_preds):
            raise IndexError(f"rank-k={k} out of range [0, {len(q0_preds)-1}]")
        chosen = int(order[k])
        q0 = q0_preds[chosen]
        candidate_label = f"v5 cand rank={k} (sample {chosen})"
        candidate_rmse = float(rmses[chosen])

    print(f"[q0 source] {candidate_label}  RMSE_to_GT={candidate_rmse:.3f}rad")
    print(f"[q0]        {q0.round(3).tolist()}")

    # --- Optional IK refine (preserve seed's TCP rotation, snap TCP to local_origin) ---
    if args.refine_ik:
        ik_robot = PenFrankaResearch3(name="pen", enable_cc=False)
        q_ref, ok, info = refine_q0_seed(
            ik_robot, q0, task["local_origin"],
            target_rotmat=None, desk_normal=task["desk_normal"],
            theta_max_deg=float(args.theta_max_deg),
        )
        print(f"[ik]    seed TCP err={info['tcp_err_seed_m']*100:.2f}cm  "
              f"refined TCP err={info['tcp_err_refined_m']*100:.3f}cm  "
              f"in_cone={info.get('seed_in_cone')}  ok={ok}")
        if ok:
            q0 = q_ref
            print(f"[q0]   (refined) {q0.round(3).tolist()}")
        else:
            print("[ik]   refine failed → falling back to seed q0")

    # --- Set up tracker ---
    fr3_gpu = PenFrankaResearch3GPU(device)
    helpers = make_pen_collision_helpers(fr3_gpu.robot)
    config = TrackerConfig(
        theta_max_deg=float(args.theta_max_deg),
        angle_null_gain=float(args.angle_null_gain),
        angle_attract_gain=float(args.angle_attract_gain),
    )
    print(f"[tracker] angle_null_gain={config.angle_null_gain}  "
          f"angle_attract_gain={config.angle_attract_gain}")
    tracker = PlaneConstrainedTracker(
        robot=fr3_gpu.robot,
        self_collision_fn=helpers.self_collision_fn,
        sphere_positions_fn=helpers.sphere_positions_fn,
        sphere_radii=helpers.sphere_radii,
        sphere_link_indices=helpers.sphere_link_indices,
        config=config,
    )

    # --- Rollout ---
    full_q, seg_results = rollout_from_q0(
        tracker, config, q0, segments, task["desk_center"], task["desk_normal"],
        device=device, max_steps_buffer=args.max_steps_buffer,
    )
    print(f"\n[rollout] full trajectory shape: {full_q.shape}  ({full_q.shape[0]} frames)")
    n_done = sum(1 for r in seg_results if r["completed"])
    target_total = sum(L for _, L in segments)
    traveled = sum(r.get("traveled_m", 0.0) for r in seg_results)
    print(f"[rollout] segments completed: {n_done}/{len(segments)}   "
          f"distance: {traveled*100:.1f}/{target_total*100:.1f}cm "
          f"({100*traveled/max(target_total,1e-9):.1f}% completion)")
    for i, r in enumerate(seg_results):
        c = SEG_COLORS[i % SEG_COLORS.shape[0]]
        sym = "✓" if r["completed"] else "✗"
        print(f"  seg {i+1}/{len(segments)} {sym}  "
              f"target={r['length_m']*100:5.1f}cm  traveled={r.get('traveled_m',0.0)*100:5.1f}cm  "
              f"term={r['term_label']}")

    # --- Visualize ---
    world = ovw.World(cam_pos=[1.8, -1.6, 1.1], cam_lookat_pos=[0.2, 0.0, 0.35])
    ossop.frame().attach_to(world.scene)

    rot = rotation_matrix_from_normal(task["desk_normal"])
    ossop.box(
        half_extents=(0.85, 0.85, 0.0015),
        pos=task["desk_center"], rotmat=rot,
        rgb=np.array([0.82, 0.78, 0.68], dtype=np.float32),
        alpha=0.4,
    ).attach_to(world.scene)

    # GT TCP path — segmented colors
    for k, tcp_seg in enumerate(task["gt_tcp_chunks"]):
        c = SEG_COLORS[k % SEG_COLORS.shape[0]]
        for j in range(tcp_seg.shape[0] - 1):
            ossop.cylinder(spos=tcp_seg[j], epos=tcp_seg[j + 1],
                           radius=0.0035, rgb=c, alpha=0.85).attach_to(world.scene)

    # Rollout TCP trace — black thin sticks
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
                       alpha=0.7).attach_to(world.scene)
    # Mark failure point if any
    first_fail = next((i for i, r in enumerate(seg_results) if not r["completed"]), None)
    if first_fail is not None:
        # Red sphere where rollout died
        ossop.sphere(pos=rollout_tcp[-1], radius=0.012,
                     rgb=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                     alpha=1.0).attach_to(world.scene)

    # Animation: robot follows the full q-trajectory
    anim_robot = PenFrankaResearch3(name="pen", enable_cc=False)
    stride = max(1, int(args.playback_stride))
    anim_path = full_q[::stride]
    if anim_path.shape[0] == 0 or not np.array_equal(anim_path[-1], full_q[-1]):
        anim_path = np.concatenate([anim_path, full_q[-1:]], axis=0)
    print(f"\n[viz] starting panda3d window. Animation: {anim_path.shape[0]}/{full_q.shape[0]} frames "
          f"(stride={stride}) @ {args.frame_delay}s/frame  →  ~{anim_path.shape[0]*args.frame_delay:.1f}s playback.")
    print("[viz] black trace = rollout TCP path; colored sticks = GT segments; "
          "red sphere = where rollout failed (if any).")
    visualize_anime_path(world, anim_robot, list(anim_path), frame_delay=float(args.frame_delay))


if __name__ == "__main__":
    main()

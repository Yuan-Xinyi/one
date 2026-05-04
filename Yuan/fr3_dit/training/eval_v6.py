"""v6 evaluation: validate the IK-interpolated trajectory directly (no tracker
rollout). For each candidate's full q-trajectory, check:
  - Joint limits (no joint over its limit at any frame)
  - Self-collision (sphere collision check on FR3 + pen)
  - TCP cone (TCP_z within ±theta_max from -desk_normal at every frame)
  - TCP plane clearance (no TCP below desk plane outside the segment-painting region)
  - Smoothness (max |Δq| per step < threshold)

A candidate is **fully feasible** if it passes ALL frames; per-task best-of-N
mirrors the tracker eval format so we can drop in the same comparison machinery.

Usage:
    python -m fr3_dit.training.eval_v6 --task-indices 234088 207043 ... \
        --prefix infer_q0_v6 --report-out /tmp/eval_v6.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from one.robots.manipulators.franka.fr3_pen import make_pen_collision_helpers
from Yuan.fr3_dit.core.pen_fr3_robot import PenFrankaResearch3GPU
from Yuan.fr3_dit.training.task_cond_dit_q0 import FR3_JOINT_LIMITS


DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "pen_fr3_composite_tasks_50k_minseg10_anchored.hdf5"
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "outputs"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--task-indices", type=int, nargs="+", required=True)
    p.add_argument("--prefix", type=str, default="infer_q0_v6")
    p.add_argument("--n-candidates", type=int, default=8)
    p.add_argument("--theta-max-deg", type=float, default=30.0)
    p.add_argument("--max-step-rad", type=float, default=0.05,
                   help="Max allowed |Δq_i| per consecutive frame (rad). Smoothness gate.")
    p.add_argument("--joint-margin", type=float, default=0.02,
                   help="Required margin (rad) above joint limits.")
    p.add_argument("--report-out", type=Path, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def evaluate_trajectory(
    q_traj: np.ndarray,                  # (T, 7)
    fr3_gpu,                              # PenFrankaResearch3GPU
    self_collision_fn,                    # callable q (B, 7) -> cost
    desk_center: np.ndarray,
    desk_normal: np.ndarray,
    target_total_m: float,
    theta_cos: float,
    max_step_rad: float,
    joint_margin: float,
    device: torch.device,
) -> dict:
    """Per-frame validation. Returns dict with feasibility flags + first-failure info."""
    T = q_traj.shape[0]
    q_t = torch.from_numpy(q_traj).to(device, dtype=torch.float32)

    # Joint-limit check (with margin).
    lo = torch.tensor(FR3_JOINT_LIMITS[:, 0] + joint_margin, device=device, dtype=torch.float32)
    hi = torch.tensor(FR3_JOINT_LIMITS[:, 1] - joint_margin, device=device, dtype=torch.float32)
    in_limits = ((q_t >= lo) & (q_t <= hi)).all(dim=-1)                # (T,)

    # Smoothness check.
    if T > 1:
        d = (q_t[1:] - q_t[:-1]).abs().max(dim=-1).values              # (T-1,)
        smooth_ok = d <= max_step_rad
        smooth_ok = torch.cat([torch.tensor([True], device=device), smooth_ok])
    else:
        smooth_ok = torch.ones(T, dtype=torch.bool, device=device)

    # FK in batches for cone check + plane clearance.
    tcp_pos, tcp_rot = fr3_gpu.robot.fk_batch(q_t)                      # (T, 3), (T, 3, 3)
    tcp_z = tcp_rot[:, :, 2]                                             # (T, 3)
    pen_axis = torch.tensor(-desk_normal / max(float(np.linalg.norm(desk_normal)), 1e-12),
                              device=device, dtype=torch.float32)
    cos_theta = (tcp_z * pen_axis).sum(dim=-1)                          # (T,)
    cone_ok = cos_theta >= theta_cos                                    # (T,)

    # Plane clearance: TCP must lie at-or-above desk plane (i.e., (tcp - center) . desk_normal ≥ 0).
    desk_center_t = torch.from_numpy(desk_center).to(device, dtype=torch.float32)
    desk_normal_t = torch.from_numpy(desk_normal).to(device, dtype=torch.float32)
    plane_signed = ((tcp_pos - desk_center_t) * desk_normal_t).sum(dim=-1)  # (T,)
    plane_ok = plane_signed >= -0.005                                       # 5mm tolerance

    # Self-collision check (batched JAX).
    sc_cost = self_collision_fn(q_t).cpu().numpy()                          # (T,)
    no_sc = sc_cost <= 0

    all_ok_per_frame = (
        in_limits.cpu().numpy() & smooth_ok.cpu().numpy()
        & cone_ok.cpu().numpy() & plane_ok.cpu().numpy() & no_sc
    )
    first_fail = int(np.argmax(~all_ok_per_frame)) if (~all_ok_per_frame).any() else T
    fully_ok = bool(all_ok_per_frame.all())

    # Reasons for first failure (priority order matches the tracker's termination codes).
    reason = "ok"
    if not fully_ok:
        i = first_fail
        if not in_limits[i].item():        reason = "joint_margin"
        elif not smooth_ok[i].item():      reason = "non_smooth"
        elif not cone_ok[i].item():        reason = "angle_violation"
        elif not plane_ok[i].item():       reason = "plane_clear"
        elif not no_sc[i]:                  reason = "self_collision"
        else:                                reason = "unknown"

    distance_traveled = float(np.linalg.norm(np.diff(tcp_pos.cpu().numpy(), axis=0), axis=1).sum())
    completion_pct = float(min(distance_traveled / max(target_total_m, 1e-9), 1.0))

    return {
        "fully_ok": fully_ok,
        "first_fail_frame": int(first_fail),
        "first_fail_reason": reason,
        "n_frames": int(T),
        "distance_traveled_m": distance_traveled,
        "completion_pct": completion_pct,
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"[setup] FR3 + collision checker (theta_max={args.theta_max_deg}°)")
    fr3 = PenFrankaResearch3GPU(device)
    helpers = make_pen_collision_helpers(fr3.robot)
    self_collision_fn = helpers.self_collision_fn

    print(f"[setup] reading desk meta from {args.data}")
    with h5py.File(args.data, "r") as f:
        ma = f["meta"].attrs
        desk_center = np.asarray(ma["source_desk_center"], dtype=np.float32)
        desk_normal = np.asarray(ma["source_desk_normal"], dtype=np.float32)
        desk_normal /= max(float(np.linalg.norm(desk_normal)), 1e-12)
        ts = f["tasks"]
        per_task_meta = {}
        for idx in args.task_indices:
            per_task_meta[idx] = {
                "seg_count": int(ts["seg_count"][idx]),
                "total_length": float(ts["total_length"][idx]),
            }

    theta_cos = float(np.cos(np.deg2rad(args.theta_max_deg)))
    report = {}
    print("\n" + "=" * 100)
    header = f"{'task':>7} {'segs':>4} {'len_cm':>7} {'cand':>4} {'best_done':>9} {'all_ok_cand':>11} {'top_fail':>14}"
    print(header); print("-" * len(header))

    for idx in args.task_indices:
        meta_path = args.out_dir / f"{args.prefix}_task{idx:06d}_meta.json"
        if not meta_path.exists():
            print(f"  task {idx}: missing {meta_path}")
            continue
        meta = json.loads(meta_path.read_text())
        n_cand = int(meta["n_candidates"])
        per_cand = []
        for ci in range(n_cand):
            qpath = args.out_dir / f"{args.prefix}_task{idx:06d}_cand{ci}_qtraj.npy"
            if not qpath.exists():
                continue
            q_traj = np.load(qpath).astype(np.float32)
            r = evaluate_trajectory(
                q_traj, fr3, self_collision_fn,
                desk_center=desk_center, desk_normal=desk_normal,
                target_total_m=per_task_meta[idx]["total_length"],
                theta_cos=theta_cos, max_step_rad=args.max_step_rad,
                joint_margin=args.joint_margin, device=device,
            )
            per_cand.append(r)
        if not per_cand:
            print(f"  task {idx}: no candidates loaded")
            continue
        best_completion = max(c["completion_pct"] for c in per_cand)
        n_all_ok = sum(1 for c in per_cand if c["fully_ok"])
        top_fail = max(
            (c["first_fail_reason"] for c in per_cand if not c["fully_ok"]),
            key=lambda r: sum(1 for c in per_cand if c["first_fail_reason"] == r),
            default="-",
        )
        print(f"{idx:>7} {per_task_meta[idx]['seg_count']:>4} "
              f"{per_task_meta[idx]['total_length']*100:>7.1f} {n_cand:>4} "
              f"{best_completion*100:>8.1f}% {n_all_ok:>4}/{n_cand:<3}    {top_fail:>14}")
        report[idx] = {
            "n_candidates": n_cand,
            "best_completion_pct": best_completion,
            "any_full_success": bool(n_all_ok > 0),
            "n_full_success": int(n_all_ok),
            "per_candidate": per_cand,
            "seg_count_planned": per_task_meta[idx]["seg_count"],
            "total_length_m": per_task_meta[idx]["total_length"],
            "top_failure_label": top_fail,
        }

    if not report:
        print("[done] no tasks evaluated"); return

    bcps = np.array([r["best_completion_pct"] for r in report.values()])
    any_succ = sum(1 for r in report.values() if r["any_full_success"])
    full_succ_rates = np.array([r["n_full_success"] / r["n_candidates"] for r in report.values()])
    print("-" * len(header))
    print(f"\n[summary] tasks={len(report)}  "
          f"best_completion: median={np.median(bcps)*100:.1f}%  mean={bcps.mean()*100:.1f}%\n"
          f"          per-cand full-success rate: median={np.median(full_succ_rates)*100:.1f}%  "
          f"mean={full_succ_rates.mean()*100:.1f}%\n"
          f"          tasks with any successful candidate: {any_succ}/{len(report)}")

    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_out, "w") as f:
            json.dump({"per_task": report,
                        "summary": {"best_completion_median": float(np.median(bcps)),
                                     "full_success_median_rate": float(np.median(full_succ_rates)),
                                     "any_success_count": any_succ,
                                     "total_tasks": len(report)}},
                       f, indent=2)
        print(f"[saved] {args.report_out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Roll out the plane-constrained tracker from each predicted q₀ and report
**task-completion** statistics — the metric we ultimately care about, not q-RMSE.

For every task:
    1. Decode the token sequence to a per-segment (direction_world, length_m) list
    2. For every q₀ candidate saved by ``infer_dit_q0`` (or freshly sampled),
       loop over segments and run the existing ``PlaneConstrainedTracker`` with
       a per-segment ``max_steps = ceil(length / (task_speed·dt)) + buffer``
    3. A segment is "completed" iff the rollout went the full target arc without
       triggering joint-margin / collision / plane / angle / low-mu / pos-error
       termination

Aggregates:
    - per-task best-of-N completion %  (longest arc divided by planned total)
    - per-task # of full-task successes (all K segments completed)
    - termination-reason distribution at first failure
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch

from one.robots.manipulators.franka.fr3_pen import make_pen_collision_helpers
from Yuan.fr3_dit.core.pen_fr3_robot import PenFrankaResearch3, PenFrankaResearch3GPU
from Yuan.fr3_dit.data_generation.generate_fr3_plane_dataset import (
    PlaneConstrainedTracker,
    TrackerConfig,
    termination_label,
)
from Yuan.fr3_dit.training.ik_refine import refine_batch


DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "pen_fr3_composite_tasks_50k_minseg10_anchored.hdf5"
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "outputs"
TOKEN_KIND_SEGMENT = 1
DIR_LOCAL_OFFSET = 3
LEN_NORM_OFFSET = 6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--task-indices", type=int, nargs="+", required=True,
                   help="Composite task indices to evaluate.")
    p.add_argument("--prefix", type=str, default="infer_q0_task",
                   help="Filename prefix for the saved q0 predictions (npy + meta).")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-steps-buffer", type=int, default=30,
                   help="Extra rollout steps beyond planned segment length, to absorb tracker lag.")
    p.add_argument("--theta-max-deg", type=float, default=30.0)
    p.add_argument("--angle-null-gain", type=float, default=1.0,
                   help="Boundary brake gain (active near/past theta_max cone). data-gen used 0.4.")
    p.add_argument("--angle-attract-gain", type=float, default=2.0,
                   help="Always-on interior attractor gain — pulls TCP_z toward -desk_normal "
                        "proportional to angle deviation (radians), suppressing angle drift "
                        "accumulation. data-gen default 0.0; eval default 2.0 (5.0 was too "
                        "aggressive after IK refine — pushed wrist into self-collision).")
    p.add_argument("--length-ref", type=float, default=0.30,
                   help="Length normalization used when generating the tokens.")
    p.add_argument("--report-out", type=Path, default=None,
                   help="Optional JSON file to write the full per-task report.")
    p.add_argument("--refine-ik", action="store_true", default=False,
                   help="Run wrs IK from each predicted q0 (seed) targeting the exact path-start TCP "
                        "+ pen-into-desk orientation, then roll out from the refined q. This separates "
                        '"good IK seed" (DiT job) from "TCP precision" (IK job).')
    return p.parse_args()


def decode_segments_world(
    tokens: np.ndarray,                # (T_tok, 32)
    token_kind: np.ndarray,            # (T_tok,)
    local_frame: np.ndarray,           # (3, 3) columns = (x̂, ŷ, ẑ)
    length_ref: float,
) -> list[tuple[np.ndarray, float]]:
    """Return list of (direction_world (3,), length_m) for every SEGMENT token."""
    out = []
    for i, k in enumerate(token_kind):
        if int(k) != TOKEN_KIND_SEGMENT:
            continue
        dir_local = tokens[i, DIR_LOCAL_OFFSET:DIR_LOCAL_OFFSET + 3].astype(np.float32)
        length = float(tokens[i, LEN_NORM_OFFSET]) * float(length_ref)
        dir_world = local_frame @ dir_local
        n = float(np.linalg.norm(dir_world))
        if n < 1e-9 or length < 1e-3:
            continue
        out.append((dir_world / n, length))
    return out


def evaluate_task(
    tracker: PlaneConstrainedTracker,
    config: TrackerConfig,
    q0_candidates: np.ndarray,                 # (N, 7) raw rad
    segments: list[tuple[np.ndarray, float]],
    desk_center: np.ndarray,                   # (3,) world
    desk_normal: np.ndarray,                   # (3,) world (outward)
    device: torch.device,
    max_steps_buffer: int = 30,
) -> dict:
    """Roll out each q₀ candidate through every segment; collect per-candidate stats."""
    N = q0_candidates.shape[0]
    K = len(segments)
    pen_axis = -desk_normal / max(float(np.linalg.norm(desk_normal)), 1e-12)
    target_total = sum(L for _, L in segments)

    q = torch.from_numpy(q0_candidates.astype(np.float32)).to(device)
    plane_point = torch.from_numpy(desk_center.astype(np.float32)).expand(N, 3).contiguous().to(device)
    plane_normal = torch.from_numpy(pen_axis.astype(np.float32)).expand(N, 3).contiguous().to(device)
    plane_side = -torch.ones(N, dtype=torch.float32, device=device)

    seg_completed = np.zeros(N, dtype=np.int32)
    distance_traveled = np.zeros(N, dtype=np.float32)
    termination_at = np.full(N, -1, dtype=np.int32)
    failure_seg = np.full(N, -1, dtype=np.int32)
    active = np.ones(N, dtype=bool)

    step_dist = config.task_speed * config.dt   # arc per tracker step

    for seg_idx, (dir_world, length_m) in enumerate(segments):
        if not active.any():
            break
        target_steps = max(1, int(math.ceil(length_m / step_dist)))
        # Temporarily swap max_steps for this segment
        orig_max = config.max_steps
        config.max_steps = target_steps + max_steps_buffer

        idx_active = np.where(active)[0]
        q_active = q[idx_active]
        pp_active = plane_point[idx_active]
        pn_active = plane_normal[idx_active]
        ps_active = plane_side[idx_active]
        dir_b = torch.from_numpy(dir_world.astype(np.float32)).expand(len(idx_active), 3).contiguous().to(device)

        try:
            trajs = tracker.collect_batch_trajectories(q_active, pp_active, dir_b, pn_active, ps_active)
        finally:
            config.max_steps = orig_max

        for j, traj in enumerate(trajs):
            i = int(idx_active[j])
            steps = int(traj["num_points"]) - 1     # number of completed integration steps
            d = steps * step_dist
            distance_traveled[i] += d
            term_code = int(traj["termination_code"])
            # "Completed segment" condition: rolled the full planned arc with no early termination.
            # In our config the only termination triggered after target_steps when max_steps =
            # target_steps + buffer is termination_code 3 (max_steps) — i.e. we exited cleanly.
            completed_target = (term_code == 3) and (d >= length_m * 0.95)
            if completed_target:
                seg_completed[i] += 1
                # Update q to end-of-rollout for next segment
                q[i] = torch.from_numpy(traj["q"][-1].astype(np.float32)).to(device)
            else:
                # Early termination → fail this segment
                termination_at[i] = term_code
                failure_seg[i] = seg_idx
                active[i] = False

    completion_pct = (distance_traveled / max(target_total, 1e-9)).clip(0.0, 1.0)
    full_success = (seg_completed == K)
    return {
        "n_candidates": N,
        "n_segments": K,
        "target_total_m": target_total,
        "seg_completed": seg_completed.tolist(),
        "distance_traveled_m": distance_traveled.tolist(),
        "completion_pct": completion_pct.tolist(),
        "termination_at_first_fail": termination_at.tolist(),
        "failure_seg": failure_seg.tolist(),
        "full_success": full_success.tolist(),
        "best_completion_pct": float(completion_pct.max()),
        "best_seg_completed": int(seg_completed.max()),
        "any_full_success": bool(full_success.any()),
        "n_full_success": int(full_success.sum()),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print(f"[setup] tracker (theta_max={args.theta_max_deg}°, length_ref={args.length_ref})")
    fr3 = PenFrankaResearch3GPU(device)
    helpers = make_pen_collision_helpers(fr3.robot)
    config = TrackerConfig(
        theta_max_deg=float(args.theta_max_deg),
        angle_null_gain=float(args.angle_null_gain),
        angle_attract_gain=float(args.angle_attract_gain),
    )
    print(f"[tracker] angle_null_gain={config.angle_null_gain}  "
          f"angle_attract_gain={config.angle_attract_gain}")
    tracker = PlaneConstrainedTracker(
        robot=fr3.robot,
        self_collision_fn=helpers.self_collision_fn,
        sphere_positions_fn=helpers.sphere_positions_fn,
        sphere_radii=helpers.sphere_radii,
        sphere_link_indices=helpers.sphere_link_indices,
        config=config,
    )

    print(f"[setup] reading desk meta from {args.data}")
    with h5py.File(args.data, "r") as f:
        ma = f["meta"].attrs
        desk_center = np.asarray(ma["source_desk_center"], dtype=np.float32)
        desk_normal = np.asarray(ma["source_desk_normal"], dtype=np.float32)
        desk_normal /= max(float(np.linalg.norm(desk_normal)), 1e-12)
        ts = f["tasks"]
        per_task_data = {}
        for idx in args.task_indices:
            tok_off = ts["token_offset"][()]
            t_lo, t_hi = int(tok_off[idx]), int(tok_off[idx + 1])
            tokens = np.asarray(ts["token_flat"][t_lo:t_hi], dtype=np.float32)
            kinds = np.asarray(ts["token_kind"][t_lo:t_hi], dtype=np.uint8)
            local_frame = np.asarray(ts["local_frame"][idx], dtype=np.float32)
            local_origin = np.asarray(ts["local_origin"][idx], dtype=np.float32)
            seg_count = int(ts["seg_count"][idx])
            total_length = float(ts["total_length"][idx])
            segments = decode_segments_world(tokens, kinds, local_frame, args.length_ref)
            per_task_data[idx] = {
                "segments": segments, "seg_count": seg_count, "total_length": total_length,
                "local_origin": local_origin,
            }

    # Use canonical name "pen" so this instance shares the SELIK CVT cache with the rest
    # of the project (avoid the ~60s rebuild whenever a new identifier_str hits disk).
    pen_robot_cpu = PenFrankaResearch3(name="pen", enable_cc=False) if args.refine_ik else None
    if args.refine_ik:
        print("[setup] --refine-ik on: per-candidate wrs IK from predicted q0 seed → exact path-start TCP")

    report = {}
    print("\n" + "=" * 100)
    header = f"{'task':>7} {'segs':>4} {'len_cm':>7} {'cand':>4} {'best_done':>9} {'best_segs':>9} {'full_succ':>9} {'top_fail':>14}"
    print(header); print("-" * len(header))

    for idx in args.task_indices:
        npy_path = args.out_dir / f"{args.prefix}{idx:06d}_q0_pred.npy"
        if not npy_path.exists():
            print(f"  task {idx}: missing {npy_path}")
            continue
        q0 = np.load(npy_path).astype(np.float32)
        td = per_task_data[idx]
        if len(td["segments"]) == 0:
            print(f"  task {idx}: no segments decoded")
            continue

        refine_summary = None
        q0_for_rollout = q0
        if args.refine_ik:
            q0_refined, ok_mask, info_list = refine_batch(
                pen_robot_cpu, q0, td["local_origin"], desk_normal,
                theta_max_deg=float(args.theta_max_deg),
            )
            tcp_seed = np.array([i["tcp_err_seed_m"] for i in info_list])
            tcp_ref = np.array([i["tcp_err_refined_m"] for i in info_list])
            in_cone = sum(1 for i in info_list if i.get("seed_in_cone"))
            refine_summary = {
                "n_ik_ok": int(ok_mask.sum()),
                "n_candidates": int(q0.shape[0]),
                "n_seed_in_cone": in_cone,
                "tcp_err_seed_cm_mean": float(tcp_seed.mean() * 100),
                "tcp_err_seed_cm_max":  float(tcp_seed.max()  * 100),
                "tcp_err_refined_cm_mean": float(tcp_ref.mean() * 100),
                "tcp_err_refined_cm_max":  float(tcp_ref.max()  * 100),
            }
            q0_for_rollout = q0_refined

        result = evaluate_task(
            tracker, config, q0_for_rollout, td["segments"], desk_center, desk_normal,
            device=device, max_steps_buffer=args.max_steps_buffer,
        )
        # Most common failure reason across candidates
        terms = [t for t in result["termination_at_first_fail"] if t >= 0]
        top_fail = max(set(terms), key=terms.count) if terms else None
        top_fail_label = termination_label(top_fail) if top_fail is not None else "-"
        ik_tag = ""
        if refine_summary is not None:
            ik_tag = (f" | IK ok={refine_summary['n_ik_ok']}/{refine_summary['n_candidates']} "
                      f"tcp_err {refine_summary['tcp_err_seed_cm_mean']:.2f}→{refine_summary['tcp_err_refined_cm_mean']:.3f}cm")
        print(
            f"{idx:>7} {td['seg_count']:>4} {td['total_length']*100:>7.1f} "
            f"{q0.shape[0]:>4} {result['best_completion_pct']*100:>8.1f}% "
            f"{result['best_seg_completed']:>4}/{td['seg_count']:<3} "
            f"{result['n_full_success']:>4}/{q0.shape[0]:<3}  {top_fail_label:>14}"
            f"{ik_tag}"
        )
        report[idx] = {
            **result,
            "seg_count_planned": td["seg_count"],
            "total_length_m": td["total_length"],
            "top_failure_label": top_fail_label,
            "refine_ik": refine_summary,
        }

    if not report:
        print("[done] no tasks evaluated")
        return

    best_pcts = np.array([r["best_completion_pct"] for r in report.values()])
    full_succ_rates = np.array([r["n_full_success"] / r["n_candidates"] for r in report.values()])
    any_succ = sum(1 for r in report.values() if r["any_full_success"])
    print("-" * len(header))
    print(
        f"\n[summary] tasks={len(report)}  "
        f"best_completion_pct: median={np.median(best_pcts)*100:.1f}%  mean={best_pcts.mean()*100:.1f}%  "
        f"min={best_pcts.min()*100:.1f}%\n"
        f"          per-candidate full-task success rate: median={np.median(full_succ_rates)*100:.1f}%  "
        f"mean={full_succ_rates.mean()*100:.1f}%\n"
        f"          tasks with any successful candidate: {any_succ}/{len(report)}"
    )

    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_out, "w") as f:
            json.dump({"per_task": report,
                        "summary": {"best_completion_median": float(np.median(best_pcts)),
                                    "full_success_median_rate": float(np.median(full_succ_rates)),
                                    "any_success_count": any_succ, "total_tasks": len(report)}},
                       f, indent=2)
        print(f"[saved] {args.report_out}")


if __name__ == "__main__":
    main()

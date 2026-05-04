"""1-D bisection on the length of a SINGLE straight line drawable on the desk.

This is the simplest baseline of "find max writable thing": one segment, one
direction, one position. No multi-stroke, no pen-lift. Identifies how long a
straight stroke DiT + tracker can complete given a fixed start point and
in-plane direction.

Usage:
    python -m fr3_dit.calligraphy.find_max_line                         # default placement
    python -m fr3_dit.calligraphy.find_max_line --direction-deg 90      # vertical line
    python -m fr3_dit.calligraphy.find_max_line --x 0.6 --y 0.1
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from Yuan.fr3_dit.calligraphy.feasibility_check import FeasibilityOracle
from Yuan.fr3_dit.calligraphy.polyline_to_tokens import tokenize_stroke


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--x", type=float, default=0.5,
                   help="Desk-frame X (m) of stroke START point.")
    p.add_argument("--y", type=float, default=0.0,
                   help="Desk-frame Y (m) of stroke START point.")
    p.add_argument("--z", type=float, default=-0.05,
                   help="Desk-frame Z (m) — should equal source_desk_center[2].")
    p.add_argument("--desk-normal", type=float, nargs=3, default=[0.0, 0.0, 1.0])
    p.add_argument("--direction-deg", type=float, default=0.0,
                   help="In-plane direction (degrees CCW from desk +x). 0=along x; 90=along y.")
    p.add_argument("--len-min", type=float, default=0.05,
                   help="Lower length bound in meters (default 5 cm — below DiT's training distribution).")
    p.add_argument("--len-max", type=float, default=0.60,
                   help="Upper length bound in meters (default 60 cm).")
    p.add_argument("--precision", type=float, default=0.005,
                   help="Stop bisection when interval < this (m). Default 5 mm.")
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--n-candidates", type=int, default=8)
    p.add_argument("--top-k-rollout", type=int, default=2,
                   help="Use option-B DiT score to select top-K → roll out only those.")
    p.add_argument("--cfg-w", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    desk_normal = np.asarray(args.desk_normal, dtype=np.float32)
    desk_normal /= max(float(np.linalg.norm(desk_normal)), 1e-12)

    # Build in-plane direction by rotating +x_world about desk_normal by direction_deg.
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(desk_normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
    u = helper - desk_normal * float(np.dot(helper, desk_normal))
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    v = np.cross(desk_normal, u)
    a = float(np.deg2rad(args.direction_deg))
    direction = np.cos(a) * u + np.sin(a) * v
    direction = direction.astype(np.float32)

    start = np.array([args.x, args.y, args.z], dtype=np.float32)
    desk_center = start  # use start as the tracker's desk_center reference

    print(f"[find_max_line] start=({args.x:.2f},{args.y:.2f},{args.z:.2f}) "
          f"dir_deg={args.direction_deg:.1f}° dir_world={direction.round(3).tolist()}")
    print(f"[find_max_line] bisect range=[{args.len_min*100:.1f}, {args.len_max*100:.1f}]cm "
          f"precision={args.precision*1000:.0f}mm")

    oracle_kwargs = {} if args.ckpt is None else {"ckpt": args.ckpt}
    oracle = FeasibilityOracle(
        n_candidates=args.n_candidates, top_k_rollout=args.top_k_rollout,
        cfg_w=args.cfg_w, verbose=False, **oracle_kwargs,
    )
    print(f"[oracle] DiT loaded; n_cand={args.n_candidates} top-K={oracle.top_k_rollout} "
          f"tracker(g_null={oracle.tracker_config.angle_null_gain}, "
          f"g_attract={oracle.tracker_config.angle_attract_gain})\n")

    def feasible(length_m: float):
        end = start + direction * float(length_m)
        polyline = np.stack([start, end], axis=0).astype(np.float32)
        ts = tokenize_stroke(polyline, desk_normal)
        r = oracle.evaluate_stroke(ts, desk_center, desk_normal)
        return r

    def fmt(L_m, r):
        mark = "✓" if r.feasible else "✗"
        return (f"  L={L_m*100:5.1f}cm  {mark}  "
                f"n_succ={r.n_success}/{r.n_candidates}  "
                f"best_completion={r.best_completion_pct*100:.1f}%  "
                f"top_fail={r.best().top_failure_label}")

    # 1) Bound checks.
    t0 = time.time()
    print(f"[bisect 0] testing lower bound...")
    r_lo = feasible(args.len_min)
    print(fmt(args.len_min, r_lo))
    if not r_lo.feasible:
        print(f"\n[result] line is INFEASIBLE even at lower bound {args.len_min*100:.1f}cm.\n"
              f"         Either --len-min too short for DiT training distribution\n"
              f"         (training segments were 10-50cm), or this position/direction is OOD.")
        return

    print(f"[bisect 1] testing upper bound...")
    r_hi = feasible(args.len_max)
    print(fmt(args.len_max, r_hi))
    if r_hi.feasible:
        elapsed = time.time() - t0
        print(f"\n[result] max line length ≥ {args.len_max*100:.1f}cm "
              f"(upper bound exhausted; raise --len-max).  elapsed={elapsed:.1f}s")
        return

    # 2) Bisect.
    lo, hi = float(args.len_min), float(args.len_max)
    last_ok_L = lo
    iter_idx = 1
    while hi - lo > args.precision:
        iter_idx += 1
        mid = 0.5 * (lo + hi)
        r = feasible(mid)
        print(f"[bisect {iter_idx}]" + fmt(mid, r))
        if r.feasible:
            lo = mid; last_ok_L = mid
        else:
            hi = mid

    elapsed = time.time() - t0
    print(f"\n[result] max writable straight line = {last_ok_L*100:.1f}cm "
          f"@ start=({args.x:.2f},{args.y:.2f},{args.z:.2f}) dir={args.direction_deg:.1f}°  "
          f"({iter_idx} oracle queries, {elapsed:.1f}s)")


if __name__ == "__main__":
    main()

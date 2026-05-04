"""1-D bisection search for the maximum writable size of a character at a fixed
desk placement.

Assumes feasibility is monotonic in size (smaller character = easier; we always
have an interior lower bound where it works). Bisects on ``size_m`` until precision
is reached. Each query asks the oracle whether **all** strokes of the character
are feasible (≥1 candidate completes) at that size.

Usage:
    python -m fr3_dit.calligraphy.find_max_size --char 中
    python -m fr3_dit.calligraphy.find_max_size --char 中 --top-k-rollout 2
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from Yuan.fr3_dit.calligraphy.character_def import list_characters, place_character
from Yuan.fr3_dit.calligraphy.feasibility_check import FeasibilityOracle
from Yuan.fr3_dit.calligraphy.polyline_to_tokens import tokenize_stroke


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--char", type=str, default="中",
                   help=f"Character to size-search. Known: {list_characters()}")
    p.add_argument("--x", type=float, default=0.5,
                   help="Desk-frame X (m) of character centre.")
    p.add_argument("--y", type=float, default=0.0)
    p.add_argument("--z", type=float, default=-0.05)
    p.add_argument("--desk-normal", type=float, nargs=3, default=[0.0, 0.0, 1.0])
    p.add_argument("--theta-deg", type=float, default=0.0)
    p.add_argument("--size-min", type=float, default=0.04,
                   help="Lower bound for bisection in meters (default 4 cm).")
    p.add_argument("--size-max", type=float, default=0.25,
                   help="Upper bound for bisection in meters (default 25 cm).")
    p.add_argument("--precision", type=float, default=0.005,
                   help="Stop bisection when interval < this (m). Default 5 mm.")
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--n-candidates", type=int, default=8)
    p.add_argument("--top-k-rollout", type=int, default=2,
                   help="Use option-B DiT score to select top-K → roll out only those. "
                        "Default 2 (4× speedup vs full 8). Set =--n-candidates to disable.")
    p.add_argument("--cfg-w", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    desk_center = np.array([args.x, args.y, args.z], dtype=np.float32)
    desk_normal = np.asarray(args.desk_normal, dtype=np.float32)
    desk_normal /= max(float(np.linalg.norm(desk_normal)), 1e-12)
    theta_rad = float(np.deg2rad(args.theta_deg))

    print(f"[find_max] '{args.char}'  centre=({args.x:.2f},{args.y:.2f},{args.z:.2f})  "
          f"theta={args.theta_deg:.1f}°  bisect range=[{args.size_min*100:.1f}, "
          f"{args.size_max*100:.1f}]cm  precision={args.precision*1000:.0f}mm")

    oracle_kwargs = {} if args.ckpt is None else {"ckpt": args.ckpt}
    oracle = FeasibilityOracle(
        n_candidates=args.n_candidates, top_k_rollout=args.top_k_rollout,
        cfg_w=args.cfg_w, verbose=False, **oracle_kwargs,
    )
    print(f"[oracle] DiT loaded; n_cand={args.n_candidates} top-K rollout={oracle.top_k_rollout} "
          f"tracker(g_null={oracle.tracker_config.angle_null_gain}, "
          f"g_attract={oracle.tracker_config.angle_attract_gain})")

    def all_feasible(size_m: float) -> tuple[bool, list]:
        """Return (all_strokes_feasible, per_stroke_results)."""
        strokes_world = place_character(args.char, desk_center, desk_normal,
                                          size_m=size_m, theta_rad=theta_rad)
        per = []
        for k, poly in enumerate(strokes_world):
            ts = tokenize_stroke(poly, desk_normal)
            r = oracle.evaluate_stroke(ts, desk_center, desk_normal)
            per.append((k, r.feasible, r.n_success, r.best_completion_pct))
            if not r.feasible:
                # Short-circuit: one infeasible stroke is enough.
                while k + 1 < len(strokes_world):
                    k += 1
                    per.append((k, None, 0, 0.0))   # skipped
                return False, per
        return True, per

    def fmt_per(per: list, size_m: float) -> str:
        parts = []
        for k, ok, ns, bcp in per:
            if ok is None:
                parts.append(f"s{k+1}=skip")
            else:
                mark = "✓" if ok else "✗"
                parts.append(f"s{k+1}={mark}({ns}/8,{bcp*100:.0f}%)")
        return f"  size={size_m*100:5.1f}cm  " + "  ".join(parts)

    # 1) Sanity-check both bounds.
    t0 = time.time()
    print(f"\n[bisect 0/?] testing lower bound...")
    ok_lo, per_lo = all_feasible(args.size_min)
    print(fmt_per(per_lo, args.size_min) + f"  → {'feasible' if ok_lo else 'INFEASIBLE'}")
    if not ok_lo:
        print(f"\n[result] character is INFEASIBLE even at lower bound {args.size_min*100:.1f}cm. "
              f"Try a larger size_min or different placement.")
        return

    print(f"[bisect 1/?] testing upper bound...")
    ok_hi, per_hi = all_feasible(args.size_max)
    print(fmt_per(per_hi, args.size_max) + f"  → {'feasible' if ok_hi else 'infeasible'}")
    if ok_hi:
        elapsed = time.time() - t0
        print(f"\n[result] max writable size ≥ {args.size_max*100:.1f}cm "
              f"(upper bound exhausted; raise --size-max).  elapsed={elapsed:.1f}s")
        return

    # 2) Bisect.
    lo, hi = float(args.size_min), float(args.size_max)
    iter_idx = 1
    last_ok_size = lo
    last_ok_per = per_lo
    while hi - lo > args.precision:
        iter_idx += 1
        mid = 0.5 * (lo + hi)
        ok, per = all_feasible(mid)
        print(f"[bisect {iter_idx}] " + fmt_per(per, mid)
              + f"  → {'feasible' if ok else 'infeasible'}")
        if ok:
            lo = mid; last_ok_size = mid; last_ok_per = per
        else:
            hi = mid

    elapsed = time.time() - t0
    print(f"\n[result] '{args.char}' max writable size = {last_ok_size*100:.1f}cm "
          f"@ centre=({args.x:.2f},{args.y:.2f},{args.z:.2f}) theta={args.theta_deg:.1f}°  "
          f"({iter_idx} oracle queries, {elapsed:.1f}s)")
    print(f"[result] suggested visualization command:")
    print(f"  python -m fr3_dit.calligraphy.draw_character "
          f"--char {args.char} --size {last_ok_size:.4f} "
          f"--x {args.x} --y {args.y} --z {args.z} --theta-deg {args.theta_deg}")


if __name__ == "__main__":
    main()

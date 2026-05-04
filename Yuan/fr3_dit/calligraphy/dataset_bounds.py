"""Read the composite-task training dataset and report the distribution of
segment lengths, segment counts, and TCP-target XY positions.

The DiT can only generate good q0 candidates for strokes whose ``(length, position)``
fall inside the training distribution. So instead of running an oracle search
(N seconds each) for "is length L feasible?", we just look up where L sits in the
training histogram. This collapses the feasibility question to a 1-microsecond
percentile lookup.

Usage:
    # Just print the bounds
    python -m fr3_dit.calligraphy.dataset_bounds

    # Query: is a 25cm single line writable?
    python -m fr3_dit.calligraphy.dataset_bounds --query-line 0.25

    # Query: is 中 @ size_m=0.15 writable? (= longest seg = 2 * 0.15 = 0.30m)
    python -m fr3_dit.calligraphy.dataset_bounds --query-char 中 --size 0.15
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from Yuan.fr3_dit.calligraphy.character_def import (
    get_canonical, list_characters, place_character, stroke_segments,
)


DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "pen_fr3_composite_tasks_50k_minseg10_anchored.hdf5"


def load_segment_lengths(h5_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (per_segment_lengths_m, per_task_seg_counts, per_task_xy_origin)."""
    with h5py.File(h5_path, "r") as f:
        ts = f["tasks"]
        token_flat = np.asarray(ts["token_flat"][()], dtype=np.float32)
        token_kind = np.asarray(ts["token_kind"][()], dtype=np.uint8)
        seg_count = np.asarray(ts["seg_count"][()], dtype=np.int32)
        local_origin = np.asarray(ts["local_origin"][()], dtype=np.float32)
        length_ref = float(f["meta"].attrs["length_ref"])
    # len_norm at offset 6; segment tokens are kind==1
    is_seg = (token_kind == 1)
    seg_lens = token_flat[is_seg, 6] * length_ref
    return seg_lens.astype(np.float32), seg_count, local_origin


def percentile_position(x: float, arr: np.ndarray) -> float:
    """Return what fraction of arr is ≤ x. (0 = below all, 1 = above all)."""
    return float(np.mean(arr <= x))


def print_distribution(name: str, arr: np.ndarray, unit_factor: float, unit: str) -> None:
    a = arr * unit_factor
    print(f"  {name:<20} min={a.min():.2f}{unit}  "
          f"p1={np.percentile(a, 1):.2f}  p5={np.percentile(a, 5):.2f}  "
          f"p50={np.percentile(a, 50):.2f}  "
          f"p95={np.percentile(a, 95):.2f}  p99={np.percentile(a, 99):.2f}  "
          f"max={a.max():.2f}{unit}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--query-line", type=float, default=None,
                   help="Query: is a single straight line of this length (m) in training distribution?")
    p.add_argument("--query-char", type=str, default=None,
                   help=f"Query: is this character at --size in distribution? Known: {list_characters()}")
    p.add_argument("--size", type=float, default=0.10,
                   help="size_m for --query-char (default 10 cm).")
    p.add_argument("--desk-x", type=float, default=0.5)
    p.add_argument("--desk-y", type=float, default=0.0)
    p.add_argument("--desk-z", type=float, default=-0.05)
    p.add_argument("--desk-normal", type=float, nargs=3, default=[0.0, 0.0, 1.0])
    p.add_argument("--theta-deg", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[data] reading {args.data}")
    seg_lens, seg_count, local_origin = load_segment_lengths(args.data)
    n_seg = seg_lens.shape[0]; n_tasks = seg_count.shape[0]
    print(f"[data] {n_tasks} tasks, {n_seg} segments total\n")

    # ---- Length distribution ----
    print("Segment-length distribution (cm) — what DiT was trained on:")
    print_distribution("seg_length_cm", seg_lens, 100, "cm")

    # ---- Seg count distribution ----
    counts, edges = np.histogram(seg_count, bins=np.arange(seg_count.min(), seg_count.max() + 2))
    print(f"\nSeg-count histogram:")
    for c, e in zip(counts, edges[:-1]):
        print(f"  {int(e)}-seg: {c:7d}  ({100*c/n_tasks:5.1f}%)")

    # ---- Position distribution ----
    print("\nTask-start XY (path-start TCP) distribution in world frame:")
    print_distribution("X (m)", local_origin[:, 0], 1, "m")
    print_distribution("Y (m)", local_origin[:, 1], 1, "m")
    radii = np.linalg.norm(local_origin[:, :2] - np.array([args.desk_x, args.desk_y]), axis=1)
    print_distribution("R from desk_center (m)", radii, 1, "m")

    # ---- Bounds summary ----
    p1 = float(np.percentile(seg_lens, 1))
    p99 = float(np.percentile(seg_lens, 99))
    p5 = float(np.percentile(seg_lens, 5))
    p95 = float(np.percentile(seg_lens, 95))
    print(f"\n[bounds] DiT comfort zone for SINGLE segment length:")
    print(f"  conservative (p5–p95):  [{p5*100:.1f}cm, {p95*100:.1f}cm]")
    print(f"  aggressive   (p1–p99):  [{p1*100:.1f}cm, {p99*100:.1f}cm]")
    print(f"  hard bounds  (min–max): [{seg_lens.min()*100:.1f}cm, {seg_lens.max()*100:.1f}cm]")

    # ---- Queries ----
    if args.query_line is not None:
        L = float(args.query_line)
        pct = percentile_position(L, seg_lens)
        in_p5_p95 = (p5 <= L <= p95)
        in_min_max = (seg_lens.min() <= L <= seg_lens.max())
        verdict = "✅ in conservative zone" if in_p5_p95 else (
            "🟡 in hard bounds (rare in training)" if in_min_max else "❌ out of training distribution")
        print(f"\n[query line]  length={L*100:.1f}cm  →  percentile={pct*100:.1f}%  →  {verdict}")

    if args.query_char is not None:
        char = args.query_char
        size = float(args.size)
        desk_center = np.array([args.desk_x, args.desk_y, args.desk_z], dtype=np.float32)
        desk_normal = np.asarray(args.desk_normal, dtype=np.float32)
        desk_normal /= max(float(np.linalg.norm(desk_normal)), 1e-12)
        theta_rad = float(np.deg2rad(args.theta_deg))
        strokes = place_character(char, desk_center, desk_normal, size_m=size, theta_rad=theta_rad)
        print(f"\n[query char] '{char}' @ size={size*100:.1f}cm  centre=({args.desk_x:.2f},{args.desk_y:.2f}) "
              f"theta={args.theta_deg:.1f}°")
        any_oop = False
        for k, poly in enumerate(strokes):
            segs = stroke_segments(poly)
            verdict = "✅"
            for j, (_, L) in enumerate(segs):
                pct = percentile_position(L, seg_lens)
                in_zone = (p5 <= L <= p95)
                in_hard = (seg_lens.min() <= L <= seg_lens.max())
                if not in_zone:
                    verdict = "🟡" if in_hard else "❌"
                    if not in_hard: any_oop = True
                print(f"  stroke {k+1} seg {j+1}: L={L*100:5.1f}cm  pct={pct*100:5.1f}%  {verdict}")
        # Origin-position check
        origin_R = float(np.linalg.norm(strokes[0][0, :2] - np.array([args.desk_x, args.desk_y])))
        R_p99 = float(np.percentile(radii, 99))
        if origin_R > R_p99:
            any_oop = True
            print(f"  ⚠ first-stroke start R={origin_R:.3f}m > training p99 R={R_p99:.3f}m")
        print(f"\n[verdict] '{char}' @ {size*100:.1f}cm: "
              f"{'❌ infeasible (OOD segments)' if any_oop else '✅ all segments in DiT distribution'}")


if __name__ == "__main__":
    main()

"""Query the reachability map for the geometric bound on each evaluation task.

A task is a straight line: start at ``p0`` and travel along ``d``. The tip can
never leave the reachable workspace, so the distance from ``p0`` to the point
where the line exits the reachable set is an upper bound on the achievable path
length, and it depends on nothing but the kinematics. Comparing the achieved
length against it separates "the workspace ran out" from "a safety constraint
fired first".

Two variants are reported per task:
    L_reach   line stays inside voxels with D > 0            (reachable set)
    L_dex     line stays inside voxels with D >= --dex-level (well-conditioned)

Usage:
    python -m Yuan.reachability.chord_bound \
        --npz Yuan/reachability/runs/reach_5cm_50dir.npz \
        --tasks Yuan/system_eval/runs/eval_10k_systematic/eval_set_10k.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_lookup(voxel_xyz: np.ndarray, D: np.ndarray, res: float):
    """Dense 3-D array of D indexed by voxel coordinate, plus its origin."""
    origin = voxel_xyz.min(axis=0)
    ijk = np.rint((voxel_xyz - origin) / res).astype(np.int64)
    shape = ijk.max(axis=0) + 1
    grid = np.zeros(shape, dtype=np.float32)
    grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = D
    return grid, origin


def march(grid: np.ndarray, origin: np.ndarray, res: float,
          p0: np.ndarray, d: np.ndarray, level: float,
          step: float, max_len: float) -> np.ndarray:
    """Distance along ``d`` until the voxel value first drops below ``level``."""
    n = int(np.ceil(max_len / step)) + 1
    t = np.arange(n, dtype=np.float32) * step
    pts = p0[:, None, :] + d[:, None, :] * t[None, :, None]      # (N, n, 3)
    ijk = np.rint((pts - origin) / res).astype(np.int64)
    inside = np.ones(ijk.shape[:2], dtype=bool)
    for ax in range(3):
        inside &= (ijk[..., ax] >= 0) & (ijk[..., ax] < grid.shape[ax])
    ijk = np.clip(ijk, 0, np.array(grid.shape) - 1)
    vals = grid[ijk[..., 0], ijk[..., 1], ijk[..., 2]]
    ok = inside & (vals >= level)
    # First failing sample; the bound is the last sample still inside.
    bad = ~ok
    first_bad = np.where(bad.any(axis=1), bad.argmax(axis=1), n)
    return (first_bad - 1).clip(0) * step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default='Yuan/reachability/runs/reach_5cm_50dir.npz')
    ap.add_argument('--tasks',
                    default='Yuan/system_eval/runs/eval_10k_systematic/eval_set_10k.npz')
    ap.add_argument('--step', type=float, default=0.005)
    ap.add_argument('--max-len', type=float, default=2.5)
    ap.add_argument('--dex-level', type=float, default=0.5)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    m = np.load(a.npz, allow_pickle=False)
    res = float(json.loads(str(m['meta']))['res'])
    grid, origin = build_lookup(m['voxel_xyz'], m['D'], res)

    t = np.load(a.tasks, allow_pickle=False)
    p0 = t['cs_p0'].astype(np.float32)
    d = t['cs_line_dir'].astype(np.float32)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)

    L_reach = march(grid, origin, res, p0, d, 1e-6, a.step, a.max_len)
    L_dex = march(grid, origin, res, p0, d, a.dex_level, a.step, a.max_len)

    L_seed = t['L_seed'].astype(np.float32)
    L_oracle = t['max_label_L'].astype(np.float32)
    ok = np.isfinite(L_oracle) & (L_reach > 0)

    def line(name, num, den):
        # Tasks starting outside the queried level have a zero-length chord and
        # simply have no bound of that kind; they are dropped, not clipped.
        sel = ok & (den > 0)
        r = num[sel] / den[sel]
        print(f'  {name:<22s} mean {r.mean():.4f}   median {np.median(r):.4f}   '
              f'p90 {np.quantile(r, 0.90):.4f}   (n={sel.sum()})')

    print(f'tasks: {ok.sum()} / {len(p0)}')
    print(f'  L_reach   mean {L_reach[ok].mean():.4f} m  median {np.median(L_reach[ok]):.4f}')
    print(f'  L_dex     mean {L_dex[ok].mean():.4f} m  median {np.median(L_dex[ok]):.4f}'
          f'   (D >= {a.dex_level})')
    print('ratios against the kinematic bound:')
    line('L_seed / L_reach', L_seed, L_reach)
    line('L_oracle / L_reach', L_oracle, L_reach)
    line('L_oracle / L_dex', L_oracle, L_dex)

    out = Path(a.out or Path(a.npz).with_name('chord_bound_10k.npz'))
    np.savez_compressed(out, L_reach=L_reach, L_dex=L_dex,
                        dex_level=np.float32(a.dex_level), step=np.float32(a.step))
    print(f'[chord] wrote {out}')


if __name__ == '__main__':
    main()

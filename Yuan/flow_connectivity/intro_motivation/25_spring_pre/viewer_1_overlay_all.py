"""Viewer 1 — All SMM branches overlaid (transparent nullspace family).

Static scene: every branch's SMM arc is sampled at N points and a
transparent arm is placed at each sample. Each branch gets its own
color. Result: a "fan" of overlaid postures, all of which satisfy
FK(q) = (p_tgt, R_tgt) — visually demonstrates the 1-D nullspace and
that there are multiple disjoint families of it.

Optional q_start ghost (slightly stronger alpha) marks the starting
posture of the rollout on each branch.

Usage:
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/viewer_1_overlay_all.py
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/viewer_1_overlay_all.py --n-ghosts 12
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/viewer_1_overlay_all.py --alpha 0.18
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[3]
for _p in (str(_REPO), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _shared import DEFAULT_SEED, build_or_load  # noqa: E402
from _viewer_common import (  # noqa: E402
    ARC_GHOST_ALPHA, START_GHOST_ALPHA,
    add_task_path, make_ghost_arm, make_world, sample_arc_indices,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--n-ghosts', type=int, default=8,
                        help='# of SMM-arc ghosts per branch')
    parser.add_argument('--alpha', type=float, default=ARC_GHOST_ALPHA,
                        help='alpha for the arc ghosts (default 0.12)')
    parser.add_argument('--no-start-ghost', action='store_true',
                        help='do not draw the extra q_start ghost')
    args = parser.parse_args()

    d = build_or_load(seed=args.seed, force=args.force)
    n_branches = int(d['meta']['n_branches'])
    task_path = d['task_path']
    plane_normal = d['plane_normal']
    branch_trajs = [d[f'branch_traj_{b}'].astype(np.float32)
                    for b in range(n_branches)]
    q0_best = d['q0_best']

    print(f'\nseed={args.seed}, {n_branches} SMM branches')
    print(f'  arc ghosts/branch = {args.n_ghosts}  alpha = {args.alpha:.2f}')

    base = make_world(task_path)
    add_task_path(base, task_path, plane_normal)
    cmap = plt.get_cmap('tab10')

    for bid in range(n_branches):
        rgb = tuple(float(c) for c in cmap(bid % 10)[:3])
        traj = branch_trajs[bid]
        for k in sample_arc_indices(traj.shape[0], args.n_ghosts):
            make_ghost_arm(base, traj[int(k)], rgb, args.alpha)
        if not args.no_start_ghost:
            make_ghost_arm(base, q0_best[bid], rgb, START_GHOST_ALPHA)
        print(f'    br{bid}: {args.n_ghosts} arc ghosts + 1 q_start ghost')

    base.run()


if __name__ == '__main__':
    main()

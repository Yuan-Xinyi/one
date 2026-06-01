"""Viewer 2 — Single SMM branch, isolated (no overlay across branches).

Like viewer_1 but takes a `--branch BID` parameter and only renders
ghosts for that branch. Other branches are hidden entirely. Use this
when the overlay of all 4 branches becomes too cluttered to read.

Branch colors follow the same tab10 palette as the figures, so br0
shown in isolation here is still the blue br0 in the matplotlib panels.

Usage:
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/viewer_2_single_branch.py --branch 0
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/viewer_2_single_branch.py --branch 2 --n-ghosts 15
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/viewer_2_single_branch.py --branch 3 --alpha 0.25
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
    parser.add_argument('--branch', type=int, default=2,
                        help='which branch to render (0..n_branches-1)')
    parser.add_argument('--n-ghosts', type=int, default=12)
    parser.add_argument('--alpha', type=float, default=ARC_GHOST_ALPHA)
    parser.add_argument('--no-start-ghost', action='store_true')
    args = parser.parse_args()

    d = build_or_load(seed=args.seed, force=args.force)
    n_branches = int(d['meta']['n_branches'])
    if not (0 <= args.branch < n_branches):
        raise SystemExit(f'--branch must be in [0, {n_branches - 1}], '
                          f'got {args.branch}')

    task_path = d['task_path']
    plane_normal = d['plane_normal']
    traj = d[f'branch_traj_{args.branch}'].astype(np.float32)
    q0_best = d['q0_best'][args.branch]
    L_self = float(d['L_self_best'][args.branch])
    closed = bool(d['branch_closed'][args.branch])
    arc = float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))

    # Match viewer_1's default-arm look: keep the FR3 renderer's own
    # color (rgb=None) instead of coloring by branch index.
    rgb = None

    print(f'\nseed={args.seed}, isolated br{args.branch}: '
          f'T={traj.shape[0]}, arc={arc:.2f} rad, '
          f'{"closed" if closed else "open"}, '
          f'best L_self={L_self:.3f}')
    print(f'  arc ghosts = {args.n_ghosts}  alpha = {args.alpha:.2f}')

    base = make_world(task_path)
    add_task_path(base, task_path, plane_normal, draw_plane=False)

    for k in sample_arc_indices(traj.shape[0], args.n_ghosts):
        make_ghost_arm(base, traj[int(k)], rgb, args.alpha)
    if not args.no_start_ghost:
        make_ghost_arm(base, q0_best, rgb, START_GHOST_ALPHA)

    base.run()


if __name__ == '__main__':
    main()

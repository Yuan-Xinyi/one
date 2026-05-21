"""Viewer 5 — SMM with joint limits removed, closed-period view.

Companion to `fig_no_jl_smm_closure.py`. The cached no-JL walk is
bi-directional and over-long; this viewer carves out exactly one
closed period of the SMM by finding the smallest step T* > min_step
after the anchor where  q(T*) − q0 ≈ 2π · k  (k ∈ Z^7), and places
ghost arms along the slice `no_jl_traj[anchor : anchor + T*]`.

Ghosts on the FR3-feasible slice render at `START_GHOST_ALPHA` (more
visible); ghosts on the rest of the closed period render at
`ARC_GHOST_ALPHA` (faint). Because every q on the slice satisfies
FK(q) = (p_tgt, R_tgt), the end-effector stays pinned at the task
start pose — the elbow / forearm traces a closed loop in R^3.

By default a single branch is rendered (br0). Pass `--branch BID` to
pick a different one, or `--all` to overlay every branch in one
scene; ghosts keep the default Franka renderer color and overlay
transparently because every branch lives on the same SMM
(per-branch tab10 coloring is opt-in via `--per-branch-color`).
Pass `--raw` to fall back to ghosts along the full bi-directional
unwrapped walk instead of one closed period.

Usage:
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/viewer_5_no_jl_smm.py
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/viewer_5_no_jl_smm.py --branch 1
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/viewer_5_no_jl_smm.py --branch 2 --n-ghosts 24
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/viewer_5_no_jl_smm.py --all
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/viewer_5_no_jl_smm.py --raw
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


def _anchor_in_no_jl(no_jl_traj: np.ndarray, branch_start: np.ndarray) -> int:
    return int(np.argmin(
        np.linalg.norm(no_jl_traj - branch_start[None, :], axis=1)))


def _branch_slice_in_no_jl(no_jl_traj: np.ndarray, branch_traj: np.ndarray
                            ) -> tuple[int, int]:
    d0 = np.linalg.norm(no_jl_traj - branch_traj[0][None, :], axis=1)
    d1 = np.linalg.norm(no_jl_traj - branch_traj[-1][None, :], axis=1)
    i_a = int(np.argmin(d0))
    i_b = int(np.argmin(d1))
    if i_a > i_b:
        i_a, i_b = i_b, i_a
    return i_a, i_b


def _find_period(traj: np.ndarray, q0: np.ndarray, min_step: int = 30
                 ) -> tuple[int, float, np.ndarray]:
    diffs = traj - q0[None, :]
    k_int = np.round(diffs / (2.0 * np.pi))
    err = np.linalg.norm(diffs - 2.0 * np.pi * k_int, axis=1)
    err[:min_step] = np.inf
    t_star = int(np.argmin(err))
    return t_star, float(err[t_star]), k_int[t_star].astype(int)


def _one_period_slice(no_jl_traj: np.ndarray, branch_traj: np.ndarray,
                       min_step: int = 30
                       ) -> tuple[np.ndarray, int, int, np.ndarray, float]:
    i_anchor = _anchor_in_no_jl(no_jl_traj, branch_traj[0])
    fwd = no_jl_traj[i_anchor:]
    q0 = fwd[0]
    t_star, err, k_int = _find_period(fwd, q0, min_step=min_step)
    slc = fwd[:t_star + 1]
    d_a = np.linalg.norm(slc - branch_traj[0][None, :], axis=1)
    d_b = np.linalg.norm(slc - branch_traj[-1][None, :], axis=1)
    i_a = int(np.argmin(d_a))
    i_b = int(np.argmin(d_b))
    if i_a > i_b:
        i_a, i_b = i_b, i_a
    return slc, i_a, i_b, k_int, err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--branch', type=int, default=2,
                        help='which branch to render (default 2)')
    parser.add_argument('--all', action='store_true',
                        help='overlay every branch in one scene '
                             '(default: single branch chosen by --branch)')
    parser.add_argument('--n-ghosts', type=int, default=18,
                        help='# of ghosts per branch along the closed period')
    parser.add_argument('--alpha', type=float, default=ARC_GHOST_ALPHA,
                        help='alpha for ghosts outside the FR3-feasible slice')
    parser.add_argument('--feasible-alpha', type=float,
                        default=START_GHOST_ALPHA,
                        help='alpha for ghosts inside the FR3-feasible slice')
    parser.add_argument('--raw', action='store_true',
                        help='show the full bi-directional unwrapped walk '
                             'instead of one closed period')
    parser.add_argument('--per-branch-color', action='store_true',
                        help='color each branch separately (tab10). '
                             'Default: ghosts keep the Franka renderer '
                             "default color and overlay transparently "
                             'since every branch lives on the same SMM.')
    args = parser.parse_args()

    d = build_or_load(seed=args.seed, force=args.force)
    n_branches = int(d['meta']['n_branches'])
    if not args.all and not (0 <= args.branch < n_branches):
        raise SystemExit(f'--branch must be in [0, {n_branches - 1}], '
                          f'got {args.branch}')

    task_path = d['task_path']
    plane_normal = d['plane_normal']

    branch_ids = list(range(n_branches)) if args.all else [args.branch]
    cmap = plt.get_cmap('tab10')

    base = make_world(task_path)
    add_task_path(base, task_path, plane_normal)

    no_jl_limit_mult = float(d['meta'].get('no_jl_limit_mult', 5.0))
    mode = 'raw (unwrapped)' if args.raw else 'closed-period'
    print(f'\nseed={args.seed}, no-JL limit = ±{no_jl_limit_mult:.1f}π, '
          f'mode={mode}')

    for bid in branch_ids:
        rgb = (tuple(float(c) for c in cmap(bid % 10)[:3])
               if args.per_branch_color else None)
        traj_fr3 = d[f'branch_traj_{bid}'].astype(np.float32)
        nj_full = d[f'no_jl_traj_{bid}'].astype(np.float32)

        if args.raw:
            slc = nj_full
            i_a, i_b = _branch_slice_in_no_jl(slc, traj_fr3)
            arc_total = float(np.sum(np.linalg.norm(np.diff(slc, axis=0), axis=1)))
            print(f'  br{bid}: T={slc.shape[0]} arc={arc_total:.2f} rad  '
                  f'feasible slice steps [{i_a}, {i_b}]')
        else:
            slc, i_a, i_b, k_int, err = _one_period_slice(nj_full, traj_fr3)
            arc_total = float(np.sum(np.linalg.norm(np.diff(slc, axis=0), axis=1)))
            k_str = ' '.join(f'{ki:+d}' for ki in k_int)
            print(f'  br{bid}: T*={slc.shape[0] - 1} steps, '
                  f'period={arc_total:.2f} rad, k=[{k_str}], '
                  f'wrap-err={err:.4f} rad')

        ghost_idx = sample_arc_indices(slc.shape[0], args.n_ghosts)
        for k in ghost_idx:
            k = int(k)
            in_feasible = (i_a <= k <= i_b)
            alpha = args.feasible_alpha if in_feasible else args.alpha
            make_ghost_arm(base, slc[k], rgb, alpha)

    base.run()


if __name__ == '__main__':
    main()

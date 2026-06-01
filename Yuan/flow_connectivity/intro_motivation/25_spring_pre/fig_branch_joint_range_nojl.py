"""Per-branch no-JL joint-angle trajectories (one figure per SMM branch).

The no-JL companion to `fig_branch_joint_range.py`, matching what
`viewer_5_no_jl_smm.py` renders in 3-D: for each SMM branch, the joint
limits are removed and the walk is carried over exactly ONE closed
period. This script overlays all seven joint curves q1..q7 of that
closed period against arc length, one figure per branch.

Because the limits are removed, the curves run past the FR3 joint
limits (that is the whole point of the no-JL view), so the y-axis is
auto-fit to the data — a single shared range across all branch figures.
No joint-limit lines, feasible-slice highlight, or endpoint markers are
drawn (uniform curves only).

Styling follows Yuan/paper_figures/fig04_joint_trajectories.py.

Usage:
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/fig_branch_joint_range_nojl.py
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/fig_branch_joint_range_nojl.py --seed 17
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

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _shared import DEFAULT_SEED, FIG_DIR, build_or_load  # noqa: E402
from fig_no_jl_per_joint import _one_period_slice  # noqa: E402
from fig_branch_joint_range import JOINT_COLORS, Y_PAD, arc_length  # noqa: E402


def plot_branch(slc, bid, y_lo, y_hi, linewidth, figsize, dpi, out_path):
    x = arc_length(slc)

    fig, ax = plt.subplots(figsize=tuple(figsize))

    for j in range(7):
        ax.plot(x, slc[:, j], '-', color=JOINT_COLORS[j], linewidth=linewidth,
                alpha=0.9, label=fr'$q_{{{j + 1}}}$')

    ax.set_xlabel('arc length along SMM period (rad)', fontsize=14)
    ax.set_ylabel('q  (rad)', fontsize=14)
    ax.set_xlim(-0.02 * float(x[-1]), 1.04 * float(x[-1]))
    ax.set_ylim(y_lo, y_hi)
    ax.tick_params(axis='both', labelsize=12)

    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    ax.legend(ncol=7, fontsize=10, loc='upper center',
              bbox_to_anchor=(0.5, 1.12), frameon=False,
              columnspacing=1.0, handlelength=1.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'saved: {out_path}  (period T={slc.shape[0] - 1}, '
          f'arc={x[-1]:.2f} rad)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--linewidth', type=float, default=2.2)
    parser.add_argument('--figsize', nargs=2, type=float, default=(5, 4))
    parser.add_argument('--dpi', type=int, default=200)
    parser.add_argument('--out-dir', type=str, default=None)
    args = parser.parse_args()

    d = build_or_load(seed=args.seed, force=args.force)
    n_branches = int(d['meta']['n_branches'])

    # Carve one closed period out of each branch's no-JL walk.
    slices = []
    for bid in range(n_branches):
        traj_fr3 = d[f'branch_traj_{bid}'].astype(np.float32)
        nj_full = d[f'no_jl_traj_{bid}'].astype(np.float32)
        slc, _, _, _, _ = _one_period_slice(nj_full, traj_fr3)
        slices.append(slc.astype(np.float32))

    # Shared y-axis auto-fit to the data (curves exceed the FR3 limits).
    all_q = np.concatenate(slices, axis=0)
    y_lo = float(all_q.min()) - Y_PAD
    y_hi = float(all_q.max()) + Y_PAD
    print(f'seed={args.seed}, {n_branches} branches; '
          f'shared y in [{y_lo:.3f}, {y_hi:.3f}]')

    out_dir = Path(args.out_dir) if args.out_dir else FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    for bid, slc in enumerate(slices):
        out_path = out_dir / f'fig_branch{bid}_joints_nojl_seed{args.seed}.png'
        plot_branch(slc, bid, y_lo, y_hi,
                    args.linewidth, args.figsize, args.dpi, out_path)


if __name__ == '__main__':
    main()

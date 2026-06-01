"""Per-branch joint-angle trajectories (one figure per SMM branch).

For the representative task (seed 17 by default), each SMM branch is an
arc through joint space. This script renders ONE figure per branch, with
all seven joint curves q1..q7 overlaid against arc length along the SMM.

Styling follows Yuan/paper_figures/fig04_joint_trajectories.py:
  - clean axes (top/right spines hidden), dpi 200
  - the y-axis spans the GLOBAL joint-limit envelope
    [min over joints of lmt_lo, max over joints of lmt_up], so the same
    y range is shared across all three branch figures.

Joint-limit contact: an SMM arc terminates when some joint reaches its
limit, so each arc endpoint sits within JOINT_MARGIN of a limit on one
joint. Wherever a joint curve touches its limit at an endpoint, a small
hollow circle (same color as that joint) is drawn on the endpoint.

Usage:
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/fig_branch_joint_range.py
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/fig_branch_joint_range.py --seed 17
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
from Yuan.flow_connectivity.intro_motivation.v18_smm_core import (  # noqa: E402
    JOINT_MARGIN,
)


# Per-joint palette (q1..q7), supplied by the paper's color scheme.
JOINT_COLORS = [
    '#F27970',  # q1
    '#BB9727',  # q2
    '#54B345',  # q3
    '#32B897',  # q4
    '#05B9E2',  # q5
    '#8983BF',  # q6
    '#C76DA2',  # q7
]

# Extra y-axis headroom beyond the global joint-limit envelope (rad).
Y_PAD = 0.4


def arc_length(traj: np.ndarray) -> np.ndarray:
    """Cumulative 7-D arc length (rad) along an SMM trajectory."""
    diffs = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(diffs)])


def plot_branch(traj, bid, closed, lo, hi, y_lo, y_hi,
                linewidth, figsize, dpi, out_path):
    x = arc_length(traj)
    tol = float(JOINT_MARGIN) + 1e-3

    fig, ax = plt.subplots(figsize=tuple(figsize))

    for j in range(7):
        color = JOINT_COLORS[j]
        ax.plot(x, traj[:, j], '-', color=color, linewidth=linewidth,
                alpha=0.9, label=fr'$q_{{{j + 1}}}$')
        # Mark either arc endpoint where this joint is at its limit.
        for idx in (0, -1):
            q = float(traj[idx, j])
            if abs(q - lo[j]) < tol or abs(q - hi[j]) < tol:
                ax.plot(x[idx], q, marker='o', markersize=11,
                        markerfacecolor='white', markeredgecolor=color,
                        markeredgewidth=2.0, zorder=6)

    ax.set_xlabel('arc length along SMM (rad)', fontsize=14)
    ax.set_ylabel('q  (rad)', fontsize=14)
    ax.set_xlim(-0.02 * float(x[-1]), 1.04 * float(x[-1]))
    ax.set_ylim(y_lo - Y_PAD, y_hi + Y_PAD)
    ax.tick_params(axis='both', labelsize=12)

    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    ax.legend(ncol=7, fontsize=10, loc='upper center',
              bbox_to_anchor=(0.5, 1.12), frameon=False,
              columnspacing=1.0, handlelength=1.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'saved: {out_path}  ({"closed" if closed else "open"}, '
          f'T={traj.shape[0]}, arc={x[-1]:.2f} rad)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--linewidth', type=float, default=2.2)
    parser.add_argument('--figsize', nargs=2, type=float, default=(6, 4))
    parser.add_argument('--dpi', type=int, default=200)
    parser.add_argument('--out-dir', type=str, default=None)
    args = parser.parse_args()

    d = build_or_load(seed=args.seed, force=args.force)
    n_branches = int(d['meta']['n_branches'])
    lo = np.asarray(d['lmt_lo'], dtype=np.float32)
    hi = np.asarray(d['lmt_up'], dtype=np.float32)

    # Global joint-limit envelope -> shared y-axis bounds across branches.
    y_lo = float(lo.min())
    y_hi = float(hi.max())
    print(f'seed={args.seed}, {n_branches} branches; '
          f'shared y in [{y_lo:.3f}, {y_hi:.3f}]')

    out_dir = Path(args.out_dir) if args.out_dir else FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    for bid in range(n_branches):
        traj = d[f'branch_traj_{bid}'].astype(np.float32)
        closed = bool(d['branch_closed'][bid])
        out_path = out_dir / f'fig_branch{bid}_joints_seed{args.seed}.png'
        plot_branch(traj, bid, closed, lo, hi, y_lo, y_hi,
                    args.linewidth, args.figsize, args.dpi, out_path)


if __name__ == '__main__':
    main()

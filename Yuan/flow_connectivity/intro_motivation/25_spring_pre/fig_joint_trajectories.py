"""Per-joint SMM-arc trajectories (7-up grid + branch summary).

For the same representative task as the other Part-1 figures (seed 118),
plot q_j as a function of arc length along the SMM, one subplot per
joint j=0..6, with FR3 joint limits drawn as red dashed lines. This is
the most direct visual evidence that the {n_branches} SMM arcs trace
DIFFERENT curves in joint space — each subplot shows the per-branch
curves taking different shapes, never coinciding.

Layout:
  2x4 grid of subplots: 7 joint panels + 1 info panel (branch summary).

Mimics `save_smm_joint_curves` from v18_smm_task.py but reads from the
shared cache so no recomputation is needed.

Usage:
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/fig_joint_trajectories.py
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/fig_joint_trajectories.py --seed 118
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    d = build_or_load(seed=args.seed, force=args.force)
    n_branches = int(d['meta']['n_branches'])
    branches = [{
        'traj': d[f'branch_traj_{b}'].astype(np.float32),
        'closed': bool(d['branch_closed'][b]),
    } for b in range(n_branches)]
    lo = d['lmt_lo']; hi = d['lmt_up']

    cmap = plt.get_cmap('tab10')
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
    axes = axes.flatten()

    for j in range(7):
        ax = axes[j]
        ax.axhspan(lo[j] - 1, lo[j], color='red', alpha=0.10)
        ax.axhspan(hi[j], hi[j] + 1, color='red', alpha=0.10)
        ax.axhline(lo[j], color='red', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.axhline(hi[j], color='red', linestyle='--', linewidth=0.8, alpha=0.7)
        for bid, b in enumerate(branches):
            traj = b['traj']
            diffs = np.linalg.norm(np.diff(traj, axis=0), axis=1)
            x = np.concatenate([[0.0], np.cumsum(diffs)])
            ax.plot(x, traj[:, j], '-', color=cmap(bid % 10), alpha=0.85,
                    linewidth=1.6,
                    label=f'br{bid} ({"closed" if b["closed"] else "open"})')
            ax.scatter([x[0], x[-1]], [traj[0, j], traj[-1, j]],
                       s=40, c=[cmap(bid % 10)],
                       edgecolors='black', linewidths=0.5, zorder=5)
        ax.set_title(f'j{j}  limits [{lo[j]:.2f}, {hi[j]:.2f}]', fontsize=10)
        ax.set_xlabel('arc length along SMM (rad)', fontsize=8)
        ax.set_ylabel('q [rad]', fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
        all_y = np.concatenate([b['traj'][:, j] for b in branches])
        ymin = min(lo[j], float(all_y.min())) - 0.2
        ymax = max(hi[j], float(all_y.max())) + 0.2
        ax.set_ylim(ymin, ymax)
        if j == 0:
            ax.legend(fontsize=8, loc='best')

    # Info panel
    ax_info = axes[7]
    ax_info.axis('off')
    info = [f'seed = {args.seed}', f'{n_branches} SMM branches', '']
    for bid, b in enumerate(branches):
        arc = float(np.sum(np.linalg.norm(np.diff(b['traj'], axis=0), axis=1)))
        info.append(f'  br{bid}: T={b["traj"].shape[0]}, arc={arc:.2f} rad, '
                    f'{"closed" if b["closed"] else "open"}')
    info += ['',
             'dashed red = FR3 joint limits',
             'star markers = SMM arc endpoints',
             '',
             'reading the figure:',
             '  each subplot = one joint;',
             '  curves of different colors do not coincide ⇒',
             '  the branches are physically distinct postures.']
    ax_info.text(0.0, 1.0, '\n'.join(info), fontsize=9,
                 family='monospace', verticalalignment='top')

    fig.suptitle(f'Per-joint SMM-arc trajectories  '
                 f'(seed={args.seed}, {n_branches} branches)',
                 fontsize=12, y=1.005)
    fig.tight_layout()

    out_path = (Path(args.out) if args.out else
                FIG_DIR / f'fig_joints_seed{args.seed}.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved: {out_path}')


if __name__ == '__main__':
    main()

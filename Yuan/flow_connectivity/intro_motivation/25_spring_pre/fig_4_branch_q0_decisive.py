"""Fig 4 — Initial joint angle is decisive: same task, different q0 branch
=> dramatically different length-to-failure.

Same representative task as the other Part-1 figures (seed 118). For each
SMM branch at the task start pose, we pick the BEST q0 on that branch
(highest L_self over 50 samples along the branch arc), record the 6-DOF
strict rollout from that q0, and plot the resulting end-effector
trajectory.

Three panels:

  (A) 3D EE trajectory: task path in gray, per-branch EE trajectories
      colored by branch, ending where the rollout died. Visually shows
      how far each branch's best q0 was able to follow the line before
      the controller hit a joint-limit or pose-tracking failure.

  (B) Bar chart of L_self / L_max per branch (best q0). Best branch
      annotated; ratio of best-to-worst printed in the title.

  (C) σ_min(J(t)) and joint-limit margin along the alive portion of
      each rollout, sharing x = step. Surfaces the geometric reason the
      worst branches die early (low σ_min and/or low margin).

Usage:
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/fig_4_branch_q0_decisive.py
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/fig_4_branch_q0_decisive.py --seed 118 --force
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, E402  (registers 3d)
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
    task_path = d['task_path']
    L_max = float(d['L_max'])
    ee_xyz = d['ee_xyz_best']   # (T+1, B, 3)
    fail_step = d['fail_step']
    L_self = d['L_self_best']
    fail_reasons = d['meta']['fail_reasons']
    sig_t = d['sigma_min_t']    # list of (T_alive,)
    mgn_t = d['margin_min_t']
    p_tgt = d['p_tgt']

    cmap = plt.get_cmap('tab10')
    fig = plt.figure(figsize=(18, 6))
    ax3d = fig.add_subplot(1, 3, 1, projection='3d')
    ax_bar = fig.add_subplot(1, 3, 2)
    ax_diag = fig.add_subplot(1, 3, 3)

    # --- Panel A: 3D EE trajectories ---
    # Zoom the view to the rollout region: rollouts die at ~0.1-0.3 m so
    # showing the full extended task line (1.5 m) would compress them.
    ee_extent = np.concatenate(
        [ee_xyz[:max(1, int(fail_step[b]) + 1), b, :] for b in range(n_branches)],
        axis=0)
    lo3 = ee_extent.min(axis=0); hi3 = ee_extent.max(axis=0)
    ctr = 0.5 * (lo3 + hi3)
    span = float((hi3 - lo3).max()) * 0.65 + 0.03
    box_lo = ctr - span; box_hi = ctr + span
    # Clip the task line to the visible cube so the rollouts dominate.
    line_in = np.all((task_path >= box_lo) & (task_path <= box_hi), axis=1)
    if line_in.any():
        idxs = np.where(line_in)[0]
        s, e = int(idxs.min()), int(idxs.max()) + 1
        ax3d.plot(task_path[s:e, 0], task_path[s:e, 1], task_path[s:e, 2],
                  '-', color='0.3', linewidth=1.2, alpha=0.6, label='task line')
    ax3d.scatter([task_path[0, 0]], [task_path[0, 1]], [task_path[0, 2]],
                 c='black', marker='o', s=40, label='start')
    for bid in range(n_branches):
        T_alive = max(1, min(int(fail_step[bid]) + 1, ee_xyz.shape[0]))
        xyz = ee_xyz[:T_alive, bid, :]
        col = cmap(bid % 10)
        ax3d.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], '-',
                  color=col, linewidth=2.2, alpha=0.9,
                  label=f'br{bid}: L={L_self[bid]:.3f}')
        ax3d.scatter([xyz[-1, 0]], [xyz[-1, 1]], [xyz[-1, 2]],
                     c=[col], s=80, edgecolors='black', linewidths=0.6,
                     marker='X', zorder=6)
    ax3d.set_xlabel('x [m]', fontsize=9)
    ax3d.set_ylabel('y [m]', fontsize=9)
    ax3d.set_zlabel('z [m]', fontsize=9)
    ax3d.set_title('A) EE trajectories from each branch best-q0\n'
                   '(X = rollout death, view zoomed to rollout region)',
                   fontsize=11)
    ax3d.legend(loc='upper left', fontsize=8)
    ax3d.set_xlim(box_lo[0], box_hi[0])
    ax3d.set_ylim(box_lo[1], box_hi[1])
    ax3d.set_zlim(box_lo[2], box_hi[2])
    ax3d.view_init(elev=22, azim=-58)

    # --- Panel B: L per branch ---
    ax = ax_bar
    x = np.arange(n_branches)
    colors = [cmap(b % 10) for b in range(n_branches)]
    bars = ax.bar(x, L_self, color=colors, edgecolor='black', alpha=0.85)
    best_b = int(np.argmax(L_self))
    bars[best_b].set_linewidth(2.5)
    for xi, v, r in zip(x, L_self, fail_reasons):
        ax.text(xi, v + 0.005, f'{v:.3f}', ha='center', fontsize=10,
                fontweight='bold')
        ax.text(xi, -0.012, r.replace('joint_limit', 'JL'),
                ha='center', fontsize=7, color='0.35', rotation=15)
    ax.set_xticks(x); ax.set_xticklabels([f'br{b}' for b in range(n_branches)])
    ax.set_ylabel('L_self / L_max  (best q0 in branch)')
    ymax = float(L_self.max()) * 1.25
    ax.set_ylim(-0.03, ymax)
    L_min = float(L_self.min())
    ratio = float(L_self.max()) / max(L_min, 1e-6)
    ax.set_title(f'B) best-q0 length per branch\n'
                 f'best/worst ratio = {ratio:.2f}x  '
                 f'(L_max = {L_max:.2f} m)', fontsize=11)
    ax.grid(alpha=0.3, axis='y')

    # --- Panel C: σ_min(J(t)) and JL margin(t) along each rollout ---
    ax = ax_diag
    twin = ax.twinx()
    for bid in range(n_branches):
        col = cmap(bid % 10)
        sig = sig_t[bid]
        mgn = mgn_t[bid]
        t = np.arange(len(sig))
        ax.plot(t, sig, '-', color=col, linewidth=1.6, alpha=0.85,
                label=f'br{bid}')
        twin.plot(t, mgn, '--', color=col, linewidth=1.2, alpha=0.6)
    ax.set_xlabel('rollout step (alive prefix)')
    ax.set_ylabel('sigma_min( J(t) )  [solid]')
    twin.set_ylabel('min joint-limit margin [rad]  [dashed]', color='0.35')
    ax.set_title('C) why branches die early\n'
                 'low sigma_min or low JL margin\n'
                 '  solid = sigma_min(J),  dashed = JL margin',
                 fontsize=11)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f'Initial joint angle q0 is decisive  —  seed {args.seed}, '
        f'{n_branches} SMM branches  (same task, same controller)',
        fontsize=12, y=1.02,
    )
    fig.tight_layout()

    out_path = Path(args.out) if args.out else FIG_DIR / f'fig4_seed{args.seed}.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved: {out_path}')


if __name__ == '__main__':
    main()

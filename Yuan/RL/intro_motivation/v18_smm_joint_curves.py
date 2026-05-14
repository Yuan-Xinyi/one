"""Plot each joint's angle along the SMM arc, overlaid per branch.

Reads data/smm_branches.jsonl, computes cumulative arc length per
branch, plots 7 subplots (one per joint). Each subplot overlays the
3 branches in tab10 colors, with dashed horizontal lines at the FR3
joint limits — so you can see directly which JL boundary cuts each
branch.

Usage:
    python -m Yuan.RL.intro_motivation.v18_smm_joint_curves
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl', type=str,
                        default='Yuan/RL/intro_motivation/data/smm_branches.jsonl')
    parser.add_argument('--out-png', type=str,
                        default='Yuan/RL/intro_motivation/data/smm_joint_curves.png')
    parser.add_argument('--x-mode', choices=['arc', 'index_norm'],
                        default='arc',
                        help='arc=cumulative arc length (rad); '
                             'index_norm=normalized sample index 0..1')
    args = parser.parse_args()

    branches = []
    meta = None
    with open(args.jsonl) as f:
        for line in f:
            d = json.loads(line)
            if d.get('type') == 'meta':
                meta = d
            else:
                branches.append(d)
    print(f'loaded {len(branches)} branches')

    # Get joint limits from the live kinematics object (matches what was used
    # for ODE termination, not hardcoded).
    device = torch.device('cpu')
    kin = BatchedFR3Kinematics(device=device)
    lo = kin.lmt_lo.detach().cpu().numpy()
    hi = kin.lmt_up.detach().cpu().numpy()

    cmap = plt.get_cmap('tab10')
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    axes = axes.flatten()

    for j in range(7):
        ax = axes[j]
        # JL bands
        ax.axhspan(lo[j] - 1, lo[j], color='red', alpha=0.10)
        ax.axhspan(hi[j], hi[j] + 1, color='red', alpha=0.10)
        ax.axhline(lo[j], color='red', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.axhline(hi[j], color='red', linestyle='--', linewidth=0.8, alpha=0.7)

        for bid, b in enumerate(branches):
            traj = np.array(b['traj_subsampled'])
            if args.x_mode == 'arc':
                diffs = np.linalg.norm(np.diff(traj, axis=0), axis=1)
                x = np.concatenate([[0.0], np.cumsum(diffs)])
                xlabel = 'arc length along SMM (rad)'
            else:
                x = np.linspace(0, 1, traj.shape[0])
                xlabel = 'normalized arc position'
            ax.plot(x, traj[:, j], '-', color=cmap(bid % 10), alpha=0.85,
                    linewidth=1.6,
                    label=f'br{bid} ({"closed" if b["closed"] else "open"})')
            # Mark endpoints with stars
            ax.scatter([x[0], x[-1]], [traj[0, j], traj[-1, j]],
                       s=40, c=[cmap(bid % 10)],
                       edgecolors='black', linewidths=0.5, zorder=5)

        ax.set_title(f'j{j} (limits [{lo[j]:.2f}, {hi[j]:.2f}] rad)',
                     fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel('rad', fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
        # Y range: a bit of padding around the union of JL range and observed
        ymin = min(lo[j], min(np.array(b['traj_subsampled'])[:, j].min()
                              for b in branches)) - 0.2
        ymax = max(hi[j], max(np.array(b['traj_subsampled'])[:, j].max()
                              for b in branches)) + 0.2
        ax.set_ylim(ymin, ymax)
        if j == 0:
            ax.legend(fontsize=8, loc='best')

    # 8th panel: legend / summary
    ax_sum = axes[7]
    ax_sum.axis('off')
    p_tgt = meta['p_tgt']
    R_tgt = np.array(meta['R_tgt'])
    z_tgt = R_tgt[:, 2]
    info = [
        f"target pose:",
        f"  p = [{p_tgt[0]:+.2f}, {p_tgt[1]:+.2f}, {p_tgt[2]:+.2f}]",
        f"  z = [{z_tgt[0]:+.2f}, {z_tgt[1]:+.2f}, {z_tgt[2]:+.2f}]",
        f"",
        f"{len(branches)} branches:",
    ]
    for bid, b in enumerate(branches):
        info.append(f"  br{bid} ({'closed' if b['closed'] else 'open'}): "
                    f"arc={b['arc_length_rad']:.2f} rad, "
                    f"T={b['n_steps']}")
    info += [
        f"",
        f"dashed red = FR3 joint limits",
        f"star = arc endpoint (touches JL or singularity)",
    ]
    ax_sum.text(0.0, 1.0, '\n'.join(info), fontsize=9,
                family='monospace', verticalalignment='top')

    fig.suptitle('Joint angles along each SMM branch', fontsize=12, y=1.005)
    fig.tight_layout()
    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved: {out_png}')


if __name__ == '__main__':
    main()

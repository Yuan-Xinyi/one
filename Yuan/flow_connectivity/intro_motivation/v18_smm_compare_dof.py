"""Compare 6-DOF strict vs 5-DOF Žlajpah rollout on one task seed.

Shared:
  - Same task (sample_line_task at --seed)
  - Same SMM enumeration at 6-DOF locked pose
  - Same q0 samples per branch

Different:
  - 6-DOF: track full SO(3) tightly, 1D null space, JL avoidance only
  - 5-DOF: track pos + z-axis (with deadzone), 2D null space, JL + H_dir

Output: 2-row PNG comparing per-branch L distributions side-by-side,
plus per-branch bar chart.

Usage:
    python -m Yuan.flow_connectivity.intro_motivation.v18_smm_compare_dof --seed 118
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from Yuan.flow_connectivity.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.flow_connectivity.batched_rollout import _branch_seed_bank
from Yuan.flow_connectivity.intro_motivation.v18_smm_core import (
    DEDUP_RAD, DEFAULT_H, JOINT_MARGIN,
    as_tensor, enumerate_branches, get_task_target_pose, path_length,
    project_and_filter,
)
from Yuan.flow_connectivity.intro_motivation.v18_smm_rollout_5dof import rollout_lengths_5dof
from Yuan.flow_connectivity.intro_motivation.v18_smm_rollout_6dof import rollout_lengths_6dof
from Yuan.flow_connectivity.intro_motivation.v18_smm_task import sample_branch_q0s
from Yuan.flow_connectivity.v18_data_prep import _dense_ik_at


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=118)
    parser.add_argument('--n-per-branch', type=int, default=100)
    parser.add_argument('--n-ik-seeds', type=int, default=256)
    parser.add_argument('--h', type=float, default=DEFAULT_H)
    parser.add_argument('--free-task', action='store_true')
    parser.add_argument('--out-dir', type=str,
                        default='Yuan/flow_connectivity/intro_motivation/data')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kin = BatchedFR3Kinematics(device=device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    p_tgt, R_tgt, task = get_task_target_pose(args.seed, kin, rng,
                                                free=args.free_task)
    task_path = task['fine_path_pts']
    L_max = path_length(task_path)
    track_pts = as_tensor(task_path, device)
    plane_normal_np = task['plane_normal']
    plane_normal_t = as_tensor(plane_normal_np, device)
    print(f'seed={args.seed}, L_max={L_max:.3f}m, p_tgt={p_tgt}')

    # SMM branches (6-DOF locked pose, shared by both rollout modes)
    p_t = torch.as_tensor(p_tgt, device=device, dtype=torch.float32)
    R_t = torch.as_tensor(R_tgt, device=device, dtype=torch.float32)
    extra = _branch_seed_bank(kin).detach().cpu().numpy()
    Q_seed_t, _ = _dense_ik_at(kin, p_t, R_t, args.n_ik_seeds, rng, extra_seeds=extra)
    Q = project_and_filter(kin, Q_seed_t.detach().cpu().numpy(), p_tgt, R_tgt,
                            kin.lmt_lo.detach().cpu().numpy(),
                            kin.lmt_up.detach().cpu().numpy(),
                            joint_margin=JOINT_MARGIN, dedup_rad=DEDUP_RAD)
    branches, _ = enumerate_branches(kin, Q, p_tgt, R_tgt, args.h)
    print(f'  {len(branches)} branches at task start')
    for bid, b in enumerate(branches):
        arc = float(np.sum(np.linalg.norm(np.diff(b['traj'], axis=0), axis=1)))
        print(f'    br{bid}: T={b["traj"].shape[0]}, arc={arc:.2f} rad, '
              f'{"closed" if b["closed"] else "open"}')

    all_q, all_bid, all_arc = sample_branch_q0s(branches, args.n_per_branch)
    q_batch = torch.as_tensor(all_q, device=device, dtype=torch.float32)
    print(f'\n  sampled {q_batch.shape[0]} q0 (per-branch ≈ {q_batch.shape[0] // len(branches)})')

    print('  running 6-DOF strict rollout ...')
    L_abs_6 = rollout_lengths_6dof(kin, q_batch, track_pts, plane_normal_t)
    L_rel_6 = L_abs_6 / L_max
    print('  running 5-DOF Žlajpah rollout ...')
    L_abs_5 = rollout_lengths_5dof(kin, q_batch, track_pts, plane_normal_t)
    L_rel_5 = L_abs_5 / L_max

    # Per-branch stats
    print(f'\n  {"branch":<7}{"n":<5}{"6-DOF mean±std":<20}{"5-DOF mean±std":<20}{"Δ(5-6)":<10}')
    stats_6 = []
    stats_5 = []
    for bid in range(len(branches)):
        L6 = L_rel_6[all_bid == bid]
        L5 = L_rel_5[all_bid == bid]
        m6, s6 = float(L6.mean()), float(L6.std())
        m5, s5 = float(L5.mean()), float(L5.std())
        stats_6.append((m6, s6, L6))
        stats_5.append((m5, s5, L5))
        delta = m5 - m6
        print(f'  br{bid:<5}{len(L6):<5}{m6:.3f}±{s6:.3f}      {m5:.3f}±{s5:.3f}      '
              f'{delta:+.3f}')

    # Plot: top row violins, bottom row bar chart
    cmap = plt.get_cmap('tab10')
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    # (top-left) 6-DOF violins
    ax = axes[0][0]
    data_6 = [s[2] for s in stats_6]
    pos = list(range(len(branches)))
    parts = ax.violinplot(data_6, positions=pos, showmeans=True, widths=0.6)
    for k, pc in enumerate(parts['bodies']):
        pc.set_facecolor(cmap(k % 10)); pc.set_alpha(0.55); pc.set_edgecolor('black')
    parts['cmeans'].set_color('black'); parts['cmeans'].set_linewidth(2)
    ax.set_xticks(pos); ax.set_xticklabels([f'br{b}' for b in pos])
    ax.set_ylabel('L_self / L_max')
    ax.set_title('6-DOF strict (1D null, JL only)')
    ax.grid(alpha=0.3, axis='y')

    # (top-right) 5-DOF violins
    ax = axes[0][1]
    data_5 = [s[2] for s in stats_5]
    parts = ax.violinplot(data_5, positions=pos, showmeans=True, widths=0.6)
    for k, pc in enumerate(parts['bodies']):
        pc.set_facecolor(cmap(k % 10)); pc.set_alpha(0.55); pc.set_edgecolor('black')
    parts['cmeans'].set_color('black'); parts['cmeans'].set_linewidth(2)
    ax.set_xticks(pos); ax.set_xticklabels([f'br{b}' for b in pos])
    ax.set_ylabel('L_self / L_max')
    ax.set_title('5-DOF Žlajpah (2D null, JL + H_dir, 5° deadzone, 15° hard)')
    ax.grid(alpha=0.3, axis='y')

    # Common y-range
    ymax = max(max(L_rel_6.max(), L_rel_5.max()) * 1.15, 0.1)
    for ax in axes[0]:
        ax.set_ylim(-0.02, ymax)

    # (bottom-left) bar chart: mean ± std per branch, 6-DOF vs 5-DOF
    ax = axes[1][0]
    x = np.arange(len(branches))
    w = 0.35
    m6 = [s[0] for s in stats_6]; s6 = [s[1] for s in stats_6]
    m5 = [s[0] for s in stats_5]; s5 = [s[1] for s in stats_5]
    ax.bar(x - w/2, m6, w, yerr=s6, label='6-DOF strict',
            color='C0', alpha=0.8, capsize=4)
    ax.bar(x + w/2, m5, w, yerr=s5, label='5-DOF Žlajpah',
            color='C3', alpha=0.8, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels([f'br{b}' for b in pos])
    ax.set_ylabel('mean L_self / L_max')
    ax.set_title('per-branch mean L (with std bars)')
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    # (bottom-right) gain (Δ = 5-DOF - 6-DOF) per branch
    ax = axes[1][1]
    delta = [m5_i - m6_i for m5_i, m6_i in zip(m5, m6)]
    colors = ['C2' if d > 0 else 'C3' for d in delta]
    ax.bar(x, delta, 0.6, color=colors, alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f'br{b}' for b in pos])
    ax.set_ylabel('Δ L_self (5-DOF − 6-DOF)')
    ax.set_title('extra reach from spin DOF + H_dir')
    ax.grid(alpha=0.3, axis='y')

    fig.suptitle(f'6-DOF vs 5-DOF Žlajpah on seed {args.seed}  '
                 f'(same SMM, same q0, {args.n_per_branch} per branch)',
                 fontsize=12, y=1.01)
    fig.tight_layout()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f'task_seed{args.seed}_compare_dof.png'
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'\nsaved: {out_png}')


if __name__ == '__main__':
    main()

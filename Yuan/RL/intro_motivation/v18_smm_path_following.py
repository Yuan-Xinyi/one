"""Are SMM branches fundamentally different in path-following capability?

For a v18 line task:
  1. Derive task start pose (p_tgt, R_tgt) from the seed.
  2. Enumerate SMM branches at this pose (using v18_smm_enumerate's API).
  3. From each branch, sample N q0 uniformly along the SMM arc.
  4. Run pos_priority rollout (JL avoidance ON) for every q0.
  5. Compare L_self / L_max distributions:
       - intra-branch spread (within one branch's q0 sweep)
       - inter-branch spread (across the 3 branches)
  6. Pairwise Mann-Whitney U test to assert that branches differ
     significantly.

Output: violin (per-branch distribution) + scatter (L vs arc position),
plus printed per-branch (mean, 95% CI) and pairwise p-values.

Usage:
    python -m Yuan.RL.intro_motivation.v18_smm_path_following --seed 118
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
from scipy.stats import mannwhitneyu

from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.RL.batched_rollout import _branch_seed_bank
from Yuan.RL.intro_motivation.v18_motivation_core import (
    LINE_L_RANGE,
    ROLLOUT_THETA_MAX,
    TARGET_PATH_M,
    as_tensor,
    extend_task_path,
    path_length,
    rollout_lengths,
    sample_line_task,
)
from Yuan.RL.intro_motivation.v18_smm_enumerate import (
    DEFAULT_H,
    DEDUP_RAD,
    JOINT_MARGIN,
    enumerate_branches,
    project_and_filter,
)
from Yuan.RL.intro_motivation.v18_smm_rollout_5dof_strict import (
    EPS_ORI_5DOF_STRICT, EPS_POS_5DOF_STRICT,
    rollout_lengths_5dof_strict,
)
from Yuan.RL.intro_motivation.v18_smm_rollout_6dof import (
    EPS_ORI_6DOF, EPS_POS_6DOF,
    rollout_lengths_6dof,
)
from Yuan.RL.v18_data_prep import _build_R_from_normal_direction, _dense_ik_at


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=118)
    parser.add_argument('--n-per-branch', type=int, default=40)
    parser.add_argument('--n-ik-seeds', type=int, default=256)
    parser.add_argument('--h', type=float, default=DEFAULT_H)
    parser.add_argument('--task-dof', type=str, choices=['5', '5strict', '6'],
                        default='5',
                        help="'5': v18 pos_priority (pos+z-axis, 30° dead zone); "
                             "'5strict': pos+z-axis tight 3°, spin free (2D null); "
                             "'6': full pose tight (1D null, SMM tangent only)")
    parser.add_argument('--out-png', type=str,
                        default='Yuan/RL/intro_motivation/data/smm_path_following.png')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kin = BatchedFR3Kinematics(device=device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # --- task path setup (matches branch_comparison / hardness_scan) ---
    task = sample_line_task(rng, kin, l_range=LINE_L_RANGE)
    task = extend_task_path(task, TARGET_PATH_M)
    task_path = task['fine_path_pts']
    L_max = path_length(task_path)
    track_pts = as_tensor(task_path, device)
    plane_normal_np = task['plane_normal']
    plane_normal_t = as_tensor(plane_normal_np, device)

    seg_dir = task_path[1] - task_path[0]
    seg_dir = seg_dir / max(np.linalg.norm(seg_dir), 1e-12)
    R_tgt = _build_R_from_normal_direction(plane_normal_np, seg_dir).astype(np.float32)
    p_tgt = task_path[0].astype(np.float32)
    print(f'seed={args.seed}, |task_path|={L_max:.3f}m')
    print(f'  p_tgt = {p_tgt}')
    print(f'  z_tgt = {R_tgt[:, 2]}')

    # --- enumerate SMM branches at task start ---
    p_t = torch.as_tensor(p_tgt, device=device, dtype=torch.float32)
    R_t = torch.as_tensor(R_tgt, device=device, dtype=torch.float32)
    extra_bank = _branch_seed_bank(kin).detach().cpu().numpy()
    Q_seed_t, _ = _dense_ik_at(kin, p_t, R_t, args.n_ik_seeds, rng,
                                extra_seeds=extra_bank)
    Q_seed = Q_seed_t.detach().cpu().numpy()
    lo = kin.lmt_lo.detach().cpu().numpy()
    hi = kin.lmt_up.detach().cpu().numpy()
    Q = project_and_filter(kin, Q_seed, p_tgt, R_tgt, lo, hi,
                            joint_margin=JOINT_MARGIN, dedup_rad=DEDUP_RAD)
    print(f'  IK candidates after filtering: {Q.shape[0]}')
    branches, assigned = enumerate_branches(kin, Q, p_tgt, R_tgt, args.h)
    print(f'  SMM branches: {len(branches)}')
    for bid, b in enumerate(branches):
        arc = float(np.sum(np.linalg.norm(np.diff(b['traj'], axis=0), axis=1)))
        print(f'    br{bid}: T={b["traj"].shape[0]}, arc={arc:.2f} rad, '
              f'closed={b["closed"]}, members={int((assigned == bid).sum())}')

    # --- sample q0 from each branch + rollout ---
    all_q, all_bid, all_arc = [], [], []
    for bid, b in enumerate(branches):
        traj = b['traj'].astype(np.float32)
        n_avail = traj.shape[0]
        n_sample = min(args.n_per_branch, n_avail)
        idxs = np.linspace(0, n_avail - 1, n_sample).astype(int)
        for i in idxs:
            all_q.append(traj[i])
            all_bid.append(bid)
            all_arc.append(float(i) / max(n_avail - 1, 1))
    all_q = np.array(all_q, dtype=np.float32)
    all_bid = np.array(all_bid)
    all_arc = np.array(all_arc)
    print(f'\n  rollout: {all_q.shape[0]} q0 samples '
          f'({args.n_per_branch} per branch nominal)')

    q_batch = torch.as_tensor(all_q, device=device, dtype=torch.float32)
    if args.task_dof == '6':
        print(f'  rollout: 6-DOF strict '
              f'(eps_pos={EPS_POS_6DOF*1000:.0f}mm, '
              f'eps_ori={np.rad2deg(EPS_ORI_6DOF):.1f}° full SO(3))')
        L_abs = rollout_lengths_6dof(kin, q_batch, track_pts, plane_normal_t,
                                     enforce_init_pose=True)
    elif args.task_dof == '5strict':
        print(f'  rollout: 5-DOF strict (spin free) '
              f'(eps_pos={EPS_POS_5DOF_STRICT*1000:.0f}mm, '
              f'eps_ori={np.rad2deg(EPS_ORI_5DOF_STRICT):.1f}° z-axis)')
        L_abs = rollout_lengths_5dof_strict(
            kin, q_batch, track_pts, plane_normal_t, enforce_init_pose=True)
    else:
        print(f'  rollout: 5-DOF pos_priority '
              f'(theta_max={np.rad2deg(ROLLOUT_THETA_MAX):.0f}°)')
        L_abs = rollout_lengths(kin, q_batch, track_pts, plane_normal_t,
                                theta_max_rad=ROLLOUT_THETA_MAX,
                                enforce_init_pose=True, pos_priority=True)
    L_rel = L_abs / L_max

    # --- per-branch stats ---
    print('\nPer-branch L_self / L_max:')
    print(f'  {"branch":<8}{"n":<5}{"mean":<8}{"std":<8}{"min":<8}{"max":<8}{"95% CI":<22}')
    branch_data = []
    for bid in range(len(branches)):
        mask = all_bid == bid
        L = L_rel[mask]
        if len(L) == 0:
            continue
        m, s = float(L.mean()), float(L.std())
        ci = 1.96 * s / max(np.sqrt(len(L)), 1.0)
        branch_data.append((bid, L))
        print(f'  br{bid:<6}{len(L):<5}{m:<8.3f}{s:<8.3f}'
              f'{float(L.min()):<8.3f}{float(L.max()):<8.3f}'
              f'[{m - ci:.3f}, {m + ci:.3f}]')

    print('\nPairwise Mann-Whitney U (alternative=two-sided):')
    for i in range(len(branch_data)):
        for j in range(i + 1, len(branch_data)):
            bid_a, La = branch_data[i]
            bid_b, Lb = branch_data[j]
            stat, p = mannwhitneyu(La, Lb, alternative='two-sided')
            ratio_means = float(La.mean()) / max(float(Lb.mean()), 1e-6)
            print(f'  br{bid_a} vs br{bid_b}: U={stat:.0f}, p={p:.3g}, '
                  f'mean_ratio={ratio_means:.2f}, '
                  f'|Δmean|={abs(La.mean() - Lb.mean()):.3f}')

    # --- plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap('tab10')

    data_list, pos, lbls = [], [], []
    for bid, L in branch_data:
        data_list.append(L)
        pos.append(bid)
        lbls.append(f'br{bid}')
    parts = ax1.violinplot(data_list, positions=pos, showmeans=True,
                           showmedians=False, widths=0.6)
    for k, pc in enumerate(parts['bodies']):
        pc.set_facecolor(cmap(pos[k] % 10))
        pc.set_alpha(0.55)
        pc.set_edgecolor('black')
    parts['cmeans'].set_color('black')
    parts['cmeans'].set_linewidth(2)
    # overlay individual samples as jittered scatter
    rng_jitter = np.random.default_rng(0)
    for bid, L in branch_data:
        jitter = rng_jitter.uniform(-0.10, 0.10, size=len(L))
        ax1.scatter(np.full(len(L), bid) + jitter, L,
                    c=[cmap(bid % 10)], s=18, alpha=0.8,
                    edgecolors='black', linewidths=0.3, zorder=3)
    ax1.set_xticks(pos)
    ax1.set_xticklabels(lbls)
    ax1.set_ylabel('L_self / L_max')
    ax1.set_title(f'Path-following per SMM branch  (task-dof={args.task_dof})\n'
                  f'seed={args.seed}, {args.n_per_branch} q0 each, JL avoidance ON')
    ax1.set_ylim(-0.02, max(float(L_rel.max()) * 1.15, 0.15))
    ax1.grid(alpha=0.3, axis='y')

    for bid, L in branch_data:
        mask = all_bid == bid
        ax2.scatter(all_arc[mask], L,
                    c=[cmap(bid % 10)], s=30, alpha=0.8,
                    edgecolors='black', linewidths=0.4,
                    label=f'br{bid} (n={len(L)})')
    ax2.set_xlabel('normalized arc position within branch (0=start, 1=end)')
    ax2.set_ylabel('L_self / L_max')
    ax2.set_title('Intra-branch variation along SMM')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(-0.02, max(float(L_rel.max()) * 1.15, 0.15))

    fig.suptitle('Same TCP start, 3 SMM branches → different path-following capability',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'\nsaved: {out_png}')


if __name__ == '__main__':
    main()

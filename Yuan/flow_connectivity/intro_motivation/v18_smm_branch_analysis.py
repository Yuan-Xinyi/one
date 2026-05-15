"""Cross-seed verification: does branch-level path-following ranking align
with σ_min(J(t)) and joint-limit margin geometry?

For each seed:
  1. Enumerate SMM branches at task start.
  2. Pick each branch's BEST q0 (highest L under 6-DOF strict rollout).
  3. Record rollout, compute σ_min(J(t)) and min JL margin(t) along each
     surviving trajectory.
  4. Sort branches by L_self_norm and print σ_min and JL stats.
  5. Flag whether the best branch beats the worst on either metric.

Usage:
    python -m Yuan.flow_connectivity.intro_motivation.v18_smm_branch_analysis
    python -m Yuan.flow_connectivity.intro_motivation.v18_smm_branch_analysis --seeds 42,100,999
    python -m Yuan.flow_connectivity.intro_motivation.v18_smm_branch_analysis --free-task
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from Yuan.flow_connectivity.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.flow_connectivity.batched_rollout import _branch_seed_bank
from Yuan.flow_connectivity.intro_motivation.v18_smm_core import (
    DEDUP_RAD, DEFAULT_H, JOINT_MARGIN,
    as_tensor, enumerate_branches, get_task_target_pose, path_length,
    project_and_filter,
)
from Yuan.flow_connectivity.intro_motivation.v18_smm_rollout_6dof import record_rollout_6dof
from Yuan.flow_connectivity.intro_motivation.v18_smm_task import pick_representative_q0
from Yuan.flow_connectivity.v18_data_prep import _dense_ik_at


def traj_metrics(kin, q_traj_b, fail_step):
    """Along the alive portion of one branch's recorded trajectory, compute
    σ_min(J) and min(q - lmt_lo, lmt_up - q) per step. Returns (σ_min[t], margin[t])."""
    device = kin.device
    lo = kin.lmt_lo.detach().cpu().numpy()
    hi = kin.lmt_up.detach().cpu().numpy()
    T_alive = max(1, min(fail_step + 1, q_traj_b.shape[0]))
    q_alive = q_traj_b[:T_alive]
    q_t = torch.as_tensor(q_alive, device=device, dtype=torch.float32)
    _, _, J_t, _ = kin.tcp_fk_jac(q_t)
    J_np = J_t.detach().cpu().numpy()
    sigmas = np.array([float(np.linalg.svd(J_np[t], compute_uv=False).min())
                        for t in range(T_alive)])
    margins = np.array([float(np.min(np.minimum(q_alive[t] - lo, hi - q_alive[t])))
                         for t in range(T_alive)])
    return sigmas, margins


def analyze_seed(seed: int, kin, free_task: bool, n_ik_seeds: int):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    try:
        p_tgt, R_tgt, task = get_task_target_pose(seed, kin, rng, free=free_task)
    except RuntimeError:
        return None
    task_path = task['fine_path_pts']
    L_max = path_length(task_path)
    track_pts = as_tensor(task_path, kin.device)
    plane_normal_np = task['plane_normal']
    plane_normal_t = as_tensor(plane_normal_np, kin.device)

    p_t = torch.as_tensor(p_tgt, device=kin.device, dtype=torch.float32)
    R_t = torch.as_tensor(R_tgt, device=kin.device, dtype=torch.float32)
    extra = _branch_seed_bank(kin).detach().cpu().numpy()
    Q_seed_t, _ = _dense_ik_at(kin, p_t, R_t, n_ik_seeds, rng, extra_seeds=extra)
    if Q_seed_t.shape[0] == 0:
        return None
    Q = project_and_filter(kin, Q_seed_t.detach().cpu().numpy(), p_tgt, R_tgt,
                            kin.lmt_lo.detach().cpu().numpy(),
                            kin.lmt_up.detach().cpu().numpy(),
                            joint_margin=JOINT_MARGIN, dedup_rad=DEDUP_RAD,
                            verbose=False)
    if Q.shape[0] == 0:
        return None
    branches, _ = enumerate_branches(kin, Q, p_tgt, R_tgt, DEFAULT_H)
    if len(branches) < 2:
        return None

    rep = pick_representative_q0(branches, kin, track_pts, plane_normal_t,
                                  L_max, mode='best')
    q_init = torch.as_tensor(np.array([r['q0'] for r in rep]),
                              device=kin.device, dtype=torch.float32)
    q_traj, fail_infos = record_rollout_6dof(kin, q_init, track_pts, plane_normal_np)
    q_traj_np = q_traj.detach().cpu().numpy()

    rows = []
    for bid in range(len(branches)):
        traj = q_traj_np[:, bid, :]
        fs = fail_infos[bid]['fail_step']
        sigmas, margins = traj_metrics(kin, traj, fs)
        rows.append({
            'bid': bid,
            'L': rep[bid]['L_self_norm'],
            'die_at': fs,
            'reason': fail_infos[bid]['reason'],
            'sigma_min': float(sigmas.min()),
            'sigma_avg': float(sigmas.mean()),
            'margin_min': float(margins.min()),
            'margin_avg': float(margins.mean()),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=str,
                        default='118,42,100,200,500,999,12345,7')
    parser.add_argument('--free-task', action='store_true')
    parser.add_argument('--n-ik-seeds', type=int, default=128)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kin = BatchedFR3Kinematics(device=device)

    n_seeds = 0
    n_sigma_consistent = 0
    n_margin_consistent = 0
    n_either = 0

    for seed in seeds:
        rows = analyze_seed(seed, kin, args.free_task, args.n_ik_seeds)
        if rows is None:
            print(f'\nseed={seed}: skipped (no branches / IK failure)')
            continue
        rows.sort(key=lambda r: -r['L'])
        print(f'\n=== seed={seed}  ({len(rows)} branches, sorted by L) ===')
        hdr = (f'  {"rank":<5}{"bid":<5}{"L":<7}{"die@":<6}{"reason":<22}'
                f'{"σ_min":<9}{"σ_avg":<9}{"JL_min":<9}{"JL_avg":<9}')
        print(hdr)
        for k, r in enumerate(rows):
            print(f'  {k+1:<5}br{r["bid"]:<3}{r["L"]:<7.3f}'
                   f'{r["die_at"]:<6d}{r["reason"]:<22}'
                   f'{r["sigma_min"]:<9.4f}{r["sigma_avg"]:<9.4f}'
                   f'{r["margin_min"]:<9.4f}{r["margin_avg"]:<9.4f}')

        # Spearman-style: is best (rank 1) also best on σ_avg / margin_avg?
        # And is worst (last rank) also worst?
        best, worst = rows[0], rows[-1]
        sigma_ok = best['sigma_avg'] > worst['sigma_avg']
        margin_ok = best['margin_avg'] > worst['margin_avg']
        verdict = '✓✓' if (sigma_ok and margin_ok) else ('✓ ' if (sigma_ok or margin_ok) else '✗ ')
        print(f'  {verdict} best vs worst: '
              f'σ_avg {best["sigma_avg"]:.3f} vs {worst["sigma_avg"]:.3f} '
              f'({"✓" if sigma_ok else "✗"}), '
              f'JL_avg {best["margin_avg"]:.3f} vs {worst["margin_avg"]:.3f} '
              f'({"✓" if margin_ok else "✗"})')
        n_seeds += 1
        n_sigma_consistent += int(sigma_ok)
        n_margin_consistent += int(margin_ok)
        n_either += int(sigma_ok or margin_ok)

    if n_seeds > 0:
        print(f'\n=== aggregate across {n_seeds} seeds ===')
        print(f'  best beats worst on σ_avg:    {n_sigma_consistent}/{n_seeds} '
              f'({100*n_sigma_consistent/n_seeds:.0f}%)')
        print(f'  best beats worst on JL_avg:   {n_margin_consistent}/{n_seeds} '
              f'({100*n_margin_consistent/n_seeds:.0f}%)')
        print(f'  best beats worst on either:   {n_either}/{n_seeds} '
              f'({100*n_either/n_seeds:.0f}%)')


if __name__ == '__main__':
    main()

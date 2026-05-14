"""Analyze v18_hardness_scan output to characterize hard tasks.

Questions answered:
  * Which joint dominates joint_limit failures across hard tasks?
  * Does hardness correlate with task geometry (pN, path length, z_tgt tilt)?
  * Within a hard task, do good vs bad q0 differ by which joint dies?
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.RL.intro_motivation.v18_motivation_core import (
    LINE_L_RANGE,
    ROLLOUT_THETA_MAX,
    TARGET_PATH_M,
    as_tensor,
    enumerate_start_iks,
    extend_task_path,
    path_length,
    record_rollout,
    rollout_lengths,
    sample_line_task,
)
from Yuan.RL.intro_motivation.v18_branch_comparison import farthest_point_pick


def reload_failed_joint_per_anchor(seed: int, kin, n_picks=16):
    """Re-run record_rollout to get fail_joint per q0 (not stored in jsonl)."""
    device = kin.device
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    task = sample_line_task(rng, kin, l_range=LINE_L_RANGE)
    task = extend_task_path(task, TARGET_PATH_M)
    task_path = task['fine_path_pts']
    L_max = path_length(task_path)
    track_pts = as_tensor(task_path, device)
    plane_normal_np = task['plane_normal']
    plane_normal_t = as_tensor(plane_normal_np, device)

    q_set = enumerate_start_iks(kin, rng, task, track_pts)
    lo_np = kin.lmt_lo.detach().cpu().numpy()
    hi_np = kin.lmt_up.detach().cpu().numpy()
    q_set_np = q_set.detach().cpu().numpy()
    inbounds = ((q_set_np - lo_np > 0.15)
                & (hi_np - q_set_np > 0.15)).all(axis=1)
    if int(inbounds.sum()) < n_picks:
        inbounds = np.ones(q_set.shape[0], dtype=bool)
    q_good = q_set[inbounds]
    q_good_np = q_good.detach().cpu().numpy()
    L_start = rollout_lengths(kin, q_good, track_pts, plane_normal_t,
                              theta_max_rad=ROLLOUT_THETA_MAX,
                              enforce_init_pose=True, pos_priority=True)
    seed_idx = int(np.argmax(L_start))
    picks = farthest_point_pick(q_good_np, min(n_picks, q_good.shape[0]), seed_idx)
    q_picks = q_good[picks]

    q_traj_all, fail_infos = record_rollout(kin, q_picks, track_pts, plane_normal_np)
    L_self = L_start[picks].copy() / L_max

    rows = []
    for b in range(q_picks.shape[0]):
        q_b = q_traj_all[:, b, :]
        changed = np.any(np.abs(np.diff(q_b, axis=0)) > 1e-9, axis=1)
        last_alive = int(np.where(changed)[0].max()) + 1 if changed.any() else 0
        q_fail = q_b[last_alive]
        margin = np.minimum(q_fail - lo_np, hi_np - q_fail)
        fail_joint = int(margin.argmin())
        rows.append({
            'L_self': float(L_self[b]),
            'fail_reason': fail_infos[b]['reason'],
            'fail_joint': fail_joint,
            'final_margin': float(margin[fail_joint]),
            'T_alive': last_alive,
            'pN_x': float(task_path[-1][0]),
            'pN_norm': float(np.linalg.norm(task_path[-1])),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scan-jsonl', type=str,
                        default='Yuan/RL/intro_motivation/data/hardness_scan.jsonl')
    parser.add_argument('--max-hard-tasks', type=int, default=10,
                        help='re-run this many top-spread hard tasks for joint analysis')
    args = parser.parse_args()

    scan = []
    with open(args.scan_jsonl) as f:
        for line in f:
            scan.append(json.loads(line))
    print(f'loaded {len(scan)} scanned tasks')

    n_total = len(scan)
    n_hard = sum(1 for r in scan if r['is_hard'])
    print(f'\noverall hard fraction: {n_hard}/{n_total} = {n_hard / n_total:.2%}')

    # Aggregate failure reasons
    hard = [r for r in scan if r['is_hard']]
    easy = [r for r in scan if not r['is_hard']]
    print('\n=== Failure-reason distribution (mean per task, 16 anchors each) ===')
    for label, group in [('HARD', hard), ('EASY', easy)]:
        if not group:
            continue
        joint_lim_fracs = []
        pos_err_fracs = []
        for r in group:
            counts = r['reason_counts']
            total = sum(counts.values())
            joint_lim_fracs.append(counts.get('joint_limit', 0) / total)
            pos_err_fracs.append(counts.get('pos_err_exceeded', 0) / total)
        print(f'  {label:<5} ({len(group)} tasks): '
              f'joint_limit frac mean={np.mean(joint_lim_fracs):.2f} '
              f'std={np.std(joint_lim_fracs):.2f}; '
              f'pos_err frac mean={np.mean(pos_err_fracs):.2f}')

    print('\n=== Hard tasks by rel_spread (top, with task geometry) ===')
    top_hard = sorted(hard, key=lambda r: -r['L_self_rel_spread'])[:args.max_hard_tasks]
    print(f"{'seed':<6}{'rel_sp':<9}{'L_min':<8}{'L_max':<8}{'pN_x':<7}{'pN_y':<7}{'pN_z':<7}{'pN_norm':<9}{'JL_frac':<9}")
    for r in top_hard:
        pN = r['pN']
        jl_frac = r['reason_counts'].get('joint_limit', 0) / 16
        pN_norm = float(np.linalg.norm(pN))
        print(f"{r['seed']:<6}{r['L_self_rel_spread']:<9.3f}{r['L_self_min']:<8.3f}"
              f"{r['L_self_max']:<8.3f}{pN[0]:<7.2f}{pN[1]:<7.2f}{pN[2]:<7.2f}{pN_norm:<9.2f}{jl_frac:<9.2f}")

    print('\n=== Per-joint death rate in top hard tasks ===')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kin = BatchedFR3Kinematics(device=device)
    joint_death_counts = Counter()
    good_joint_counts = Counter()  # joint at fail for top-half L_self anchors
    bad_joint_counts = Counter()   # joint at fail for bottom-half L_self anchors
    for r in top_hard:
        seed = r['seed']
        rows = reload_failed_joint_per_anchor(seed, kin)
        rows.sort(key=lambda x: -x['L_self'])
        median_L = np.median([row['L_self'] for row in rows])
        for row in rows:
            if row['fail_reason'] != 'joint_limit':
                continue
            joint_death_counts[row['fail_joint']] += 1
            if row['L_self'] >= median_L:
                good_joint_counts[row['fail_joint']] += 1
            else:
                bad_joint_counts[row['fail_joint']] += 1
        print(f"  seed{seed}: rows analyzed")

    print('\nJoint-limit deaths by joint index (joint_limit fails only):')
    for j in range(7):
        good = good_joint_counts.get(j, 0)
        bad = bad_joint_counts.get(j, 0)
        print(f'  joint {j}: total={joint_death_counts.get(j, 0):3d}, '
              f'in good-half anchors={good:3d}, in bad-half anchors={bad:3d}')

    print('\n=== Correlation: rel_spread vs |pN| (workspace edge proximity) ===')
    rel_spreads = np.array([r['L_self_rel_spread'] for r in scan])
    pN_norms = np.array([float(np.linalg.norm(r['pN'])) for r in scan])
    pN_xs = np.array([r['pN'][0] for r in scan])
    pN_zs = np.array([r['pN'][2] for r in scan])
    if len(scan) > 2:
        print(f'  corr(rel_spread, |pN|)    = {np.corrcoef(rel_spreads, pN_norms)[0, 1]:.3f}')
        print(f'  corr(rel_spread, pN_x)    = {np.corrcoef(rel_spreads, pN_xs)[0, 1]:.3f}')
        print(f'  corr(rel_spread, |pN_x|)  = {np.corrcoef(rel_spreads, np.abs(pN_xs))[0, 1]:.3f}')
        print(f'  corr(rel_spread, pN_z)    = {np.corrcoef(rel_spreads, pN_zs)[0, 1]:.3f}')


if __name__ == '__main__':
    main()

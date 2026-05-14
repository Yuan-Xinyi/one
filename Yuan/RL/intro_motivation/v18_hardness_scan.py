"""Hardness scan: how often does q0 still matter under JL avoidance?

For each random seed:
  * Sample a line task (per v18 pipeline conventions).
  * Enumerate IK candidates at the start TCP, farthest-point pick 16 anchors.
  * Run pos-priority rollout (JL avoidance ON) for all 16.
  * Record per-anchor: L_self, fail_reason, jl_margin_init,
    sigma_pos_min, fail_joint.
  * Per task: report spread of L_self and dominant failure mode.

Hardness signal: tasks where L_self spread / mean > THRESHOLD are 'hard'.

Usage:
    python -m Yuan.RL.intro_motivation.v18_hardness_scan --n-seeds 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.RL.intro_motivation.v18_branch_comparison import farthest_point_pick
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

JOINT_MARGIN = 0.15
N_PICKS = 16
HARD_SPREAD_THRESHOLD = 0.15  # L_self spread / mean


def joint_limit_margin(q: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[float, int]:
    margins = np.minimum(q - lo, hi - q)
    return float(margins.min()), int(margins.argmin())


def scan_one_seed(kin: BatchedFR3Kinematics, seed: int) -> dict:
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
    inbounds = ((q_set_np - lo_np > JOINT_MARGIN)
                & (hi_np - q_set_np > JOINT_MARGIN)).all(axis=1)
    if int(inbounds.sum()) < N_PICKS:
        inbounds = np.ones(q_set.shape[0], dtype=bool)
    q_good = q_set[inbounds]
    q_good_np = q_good.detach().cpu().numpy()
    L_start = rollout_lengths(kin, q_good, track_pts, plane_normal_t,
                              theta_max_rad=ROLLOUT_THETA_MAX,
                              enforce_init_pose=True, pos_priority=True)
    seed_idx = int(np.argmax(L_start))
    picks = farthest_point_pick(q_good_np, min(N_PICKS, q_good.shape[0]), seed_idx)
    q_picks = q_good[picks]

    q_traj_all, fail_infos = record_rollout(kin, q_picks, track_pts, plane_normal_np)
    L_self = L_start[picks] / L_max

    anchors = []
    fail_reasons = []
    init_jl = []
    for b in range(q_picks.shape[0]):
        q_b = q_traj_all[:, b, :]
        changed = np.any(np.abs(np.diff(q_b, axis=0)) > 1e-9, axis=1)
        last_alive = int(np.where(changed)[0].max()) + 1 if changed.any() else 0
        q_init = q_b[0]
        init_margin, init_j = joint_limit_margin(q_init, lo_np, hi_np)
        anchors.append({
            'L_self': float(L_self[b]),
            'fail_reason': fail_infos[b]['reason'],
            'init_jl_margin': init_margin,
            'init_jl_joint': init_j,
            'T_alive': last_alive,
        })
        fail_reasons.append(fail_infos[b]['reason'])
        init_jl.append(init_margin)

    L_self_arr = np.array([a['L_self'] for a in anchors])
    spread = float(L_self_arr.max() - L_self_arr.min())
    rel_spread = float(spread / max(L_self_arr.mean(), 1e-6))

    reason_counts = {r: fail_reasons.count(r) for r in set(fail_reasons)}

    p0 = task_path[0]
    pN = task_path[-1]
    return {
        'seed': seed,
        'p0': p0.tolist(),
        'pN': pN.tolist(),
        'path_length': float(np.linalg.norm(pN - p0)),
        'L_max': float(L_max),
        'L_self_min': float(L_self_arr.min()),
        'L_self_max': float(L_self_arr.max()),
        'L_self_mean': float(L_self_arr.mean()),
        'L_self_spread': spread,
        'L_self_rel_spread': rel_spread,
        'is_hard': bool(rel_spread > HARD_SPREAD_THRESHOLD),
        'reason_counts': reason_counts,
        'min_init_jl_margin': float(min(init_jl)),
        'mean_init_jl_margin': float(np.mean(init_jl)),
        'anchors': anchors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-seeds', type=int, default=30)
    parser.add_argument('--seed-start', type=int, default=100)
    parser.add_argument('--out', type=str,
                        default='Yuan/RL/intro_motivation/data/hardness_scan.jsonl')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kin = BatchedFR3Kinematics(device=device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_hard = 0
    print(f"{'seed':<6}{'rel_spread':<12}{'L_min':<8}{'L_max':<8}{'L_mean':<8}"
          f"{'init_jl_min':<13}{'pN_x':<9}{'hard?':<7}{'top_reasons':<40}")
    with open(out_path, 'w') as f:
        for k in range(args.n_seeds):
            seed = args.seed_start + k
            try:
                result = scan_one_seed(kin, seed)
            except Exception as e:
                print(f"seed{seed}: FAILED {e!r}")
                continue
            f.write(json.dumps(result) + '\n')
            f.flush()
            n_hard += int(result['is_hard'])
            reasons = sorted(result['reason_counts'].items(), key=lambda r: -r[1])
            reason_str = ','.join(f'{r}={c}' for r, c in reasons)
            print(f"{seed:<6}{result['L_self_rel_spread']:<12.3f}"
                  f"{result['L_self_min']:<8.3f}{result['L_self_max']:<8.3f}"
                  f"{result['L_self_mean']:<8.3f}"
                  f"{result['min_init_jl_margin']:<13.3f}"
                  f"{result['pN'][0]:<9.2f}"
                  f"{'YES' if result['is_hard'] else 'no':<7}{reason_str:<40}")

    print(f"\nhard fraction (rel_spread > {HARD_SPREAD_THRESHOLD}): "
          f"{n_hard}/{args.n_seeds} = {n_hard / args.n_seeds:.2%}")
    print(f"saved per-seed details: {out_path}")


if __name__ == '__main__':
    main()

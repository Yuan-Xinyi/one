"""Diagnose why q0 still differentiates in some tasks even with JL avoidance.

For each q anchor in a saved branch_comparison_*.jsonl, replay the rollout
with record_traj=True, then post-hoc compute along the trajectory:
  * fail_reason (orient / pos / joint_limit / completed_path)
  * sigma_min(J_pos)  — primary-task dexterity
  * sigma_min(J_full) — overall manipulator dexterity
  * distance to nearest joint limit at fail step

Hypothesis under test: when q0 differentiates despite JL avoidance, the
low-L_self anchors fail with low sigma_min (singularity) rather than
hitting joint limits.

Usage:
    python -m Yuan.RL.intro_motivation.v18_diagnose --seed 2026
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.RL.intro_motivation.v18_motivation_core import (
    OUT_DIR,
    ROLLOUT_THETA_MAX,
    TARGET_PATH_M,
    as_tensor,
    extend_task_path,
    path_length,
    record_rollout,
    sample_line_task,
    LINE_L_RANGE,
)


def trajectory_diagnostics(kin: BatchedFR3Kinematics,
                           q_traj: np.ndarray,
                           fail_info: dict) -> dict:
    device = kin.device
    q_t = torch.as_tensor(q_traj, device=device, dtype=torch.float32)
    _, _, J_t, _ = kin.tcp_fk_jac(q_t)
    J_np = J_t.detach().cpu().numpy()
    sigma_pos = np.array([np.linalg.svd(J_np[t, :3, :], compute_uv=False)[-1]
                          for t in range(J_np.shape[0])])
    sigma_full = np.array([np.linalg.svd(J_np[t, :, :], compute_uv=False)[-1]
                           for t in range(J_np.shape[0])])
    lo = kin.lmt_lo.detach().cpu().numpy()
    hi = kin.lmt_up.detach().cpu().numpy()
    dist_lo = q_traj - lo
    dist_hi = hi - q_traj
    # per-step per-joint margin to nearest limit (positive = inside)
    margin_per_joint = np.minimum(dist_lo, dist_hi)
    margin_t = margin_per_joint.min(axis=-1)
    # which joint is closest to a limit at the final (fail) step
    fail_joint_idx = int(margin_per_joint[-1].argmin())
    # initial margin: how close to limits at q0
    init_margin_per_joint = margin_per_joint[0]
    init_margin_min = float(init_margin_per_joint.min())
    init_margin_min_joint = int(init_margin_per_joint.argmin())
    return {
        'sigma_pos_min_overall': float(sigma_pos.min()),
        'sigma_full_min_overall': float(sigma_full.min()),
        'jl_margin_min_overall': float(margin_t.min()),
        'sigma_pos_at_fail': float(sigma_pos[-1]),
        'sigma_full_at_fail': float(sigma_full[-1]),
        'jl_margin_at_fail': float(margin_t[-1]),
        'jl_margin_init': init_margin_min,
        'jl_margin_init_joint': init_margin_min_joint,
        'fail_joint_idx': fail_joint_idx,
        'fail_reason': fail_info['reason'],
        'pos_err_m': float(fail_info['pos_err_m']),
        'orient_err_deg': float(fail_info['orient_err_deg']),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--data-dir', type=str, default='data')
    args = parser.parse_args()
    seed = int(args.seed)

    seed_dir = Path('Yuan/RL/intro_motivation') / args.data_dir / f'seed{seed}'
    if not seed_dir.exists():
        raise FileNotFoundError(f'no data at {seed_dir}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kin = BatchedFR3Kinematics(device=device)

    # Resample the same task and load saved q_anchors.
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    task = sample_line_task(rng, kin, l_range=LINE_L_RANGE)
    task = extend_task_path(task, TARGET_PATH_M)
    task_path = task['fine_path_pts']
    L_max = path_length(task_path)
    track_pts = as_tensor(task_path, device)
    plane_normal_np = task['plane_normal']

    q_anchors = []
    labels = []
    sigs = []
    L_self_saved = []
    for jpath in sorted(seed_dir.glob('branch_comparison_*.jsonl')):
        with open(jpath) as fh:
            meta = json.loads(fh.readline())
        q_anchors.append(meta['q_anchor'])
        labels.append(meta['anchor_label'])
        sigs.append(tuple(meta['branch_signature']))
        L_self_saved.append(meta['L_self_normalized'])

    q_init = torch.as_tensor(np.array(q_anchors), device=device, dtype=torch.float32)
    q_traj_all, fail_infos = record_rollout(kin, q_init, track_pts, plane_normal_np)
    # q_traj_all: (T, B, 7)
    T = q_traj_all.shape[0]
    B = q_traj_all.shape[1]
    print(f'seed{seed}: replayed {B} anchors, trajectory length T={T}')

    rows = []
    for b in range(B):
        # Crop trajectory to alive steps. record_rollout freezes q at last alive
        # so trailing rows are duplicates. Find last index where q changed.
        q_b = q_traj_all[:, b, :]
        changed = np.any(np.abs(np.diff(q_b, axis=0)) > 1e-9, axis=1)
        if changed.any():
            last_alive = int(np.where(changed)[0].max()) + 1
        else:
            last_alive = 0
        q_b_alive = q_b[:last_alive + 1]
        diag = trajectory_diagnostics(kin, q_b_alive, fail_infos[b])
        rows.append((labels[b], sigs[b], L_self_saved[b], last_alive, diag))

    rows.sort(key=lambda r: -r[2])
    print(f"\n{'label':<6}{'branch':<14}{'L_self':<9}{'T_alive':<9}"
          f"{'reason':<22}{'sig_p_min':<11}"
          f"{'jl0':<10}{'j_init':<8}{'jl_min':<10}{'j_fail':<8}")
    for label, sig, L, T_alive, diag in rows:
        print(f"q_{label:<4}{str(sig):<14}{L:<9.3f}{T_alive:<9d}"
              f"{diag['fail_reason']:<22}{diag['sigma_pos_min_overall']:<11.4f}"
              f"{diag['jl_margin_init']:<10.4f}{diag['jl_margin_init_joint']:<8d}"
              f"{diag['jl_margin_min_overall']:<10.4f}{diag['fail_joint_idx']:<8d}")


if __name__ == '__main__':
    main()

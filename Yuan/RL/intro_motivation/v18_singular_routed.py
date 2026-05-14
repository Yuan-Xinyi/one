"""Scenario 1: singularity-routed tasks. Sample line tasks whose start q
sits near a wrist singularity (q5 ~ 0). FK to get start TCP, generate
line task from there, run hardness scan.

Hypothesis: q0 differentiation persists; failure mode shifts to pos_err
(sigma_min driven) rather than joint_limit.

Usage:
    python -m Yuan.RL.intro_motivation.v18_singular_routed --n-tasks 10
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
    extend_task_path,
    path_length,
    record_rollout,
    rollout_lengths,
)
from Yuan.RL.v18_data_prep import _build_R_from_normal_direction, _dense_ik_at


def sample_near_singular_q(rng: np.random.Generator,
                           kin: BatchedFR3Kinematics,
                           wrist_thresh: float = 0.10) -> torch.Tensor:
    """Sample a q with |q5| < wrist_thresh (near wrist alignment singularity).
    Other joints sampled uniformly within limits. Returns a single q (7,)."""
    lo = kin.lmt_lo.detach().cpu().numpy()
    hi = kin.lmt_up.detach().cpu().numpy()
    for _ in range(200):
        q = rng.uniform(lo + 0.15, hi - 0.15)
        q[5] = rng.uniform(-wrist_thresh, wrist_thresh)
        q_t = torch.as_tensor(q, device=kin.device, dtype=torch.float32)
        _, _, J, _ = kin.tcp_fk_jac(q_t.unsqueeze(0))
        sigma = float(np.linalg.svd(J[0].detach().cpu().numpy(),
                                    compute_uv=False)[-1])
        # near singular but not totally degenerate
        if sigma < 0.05 and sigma > 1e-4:
            return q_t
    return q_t  # fallback


def sample_singular_routed_task(rng: np.random.Generator,
                                kin: BatchedFR3Kinematics,
                                L_range=(0.30, 0.40)):
    """Build a line task whose first point is the FK of a near-singular q.
    Tangent direction sampled uniformly in horizontal plane.
    Returns task dict matching sample_line_task output format."""
    q_seed = sample_near_singular_q(rng, kin)
    p_tcp, R_tcp, _, _ = kin.tcp_fk_jac(q_seed.unsqueeze(0))
    p0 = p_tcp[0].detach().cpu().numpy()
    z0 = R_tcp[0, :, 2].detach().cpu().numpy()
    L = float(rng.uniform(*L_range))
    # tangent perpendicular to z0 (so path stays in the orientation plane)
    for _ in range(30):
        v = rng.normal(size=3)
        v -= v.dot(z0) * z0
        if np.linalg.norm(v) > 1e-3:
            v /= np.linalg.norm(v)
            break
    pN = p0 + L * v
    n_pts = max(120, int(round(120 * L / 1.0)))
    path = np.linspace(p0, pN, n_pts).astype(np.float32)
    return {
        'fine_path_pts': path,
        'plane_normal': z0.astype(np.float32),
        'n_per_step': np.tile(z0[None, :], (path.shape[0] - 1, 1)).astype(np.float32),
    }


def scan_one(kin: BatchedFR3Kinematics, seed: int) -> dict:
    device = kin.device
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    task = sample_singular_routed_task(rng, kin, L_range=LINE_L_RANGE)
    task = extend_task_path(task, TARGET_PATH_M)
    task_path = task['fine_path_pts']
    L_max = path_length(task_path)
    track_pts = as_tensor(task_path, device)
    plane_normal_np = task['plane_normal']
    plane_normal_t = as_tensor(plane_normal_np, device)

    rot0_np = _build_R_from_normal_direction(plane_normal_np, (task_path[1] - task_path[0])
                                              / max(np.linalg.norm(task_path[1] - task_path[0]), 1e-12))
    q_set, _ = _dense_ik_at(kin, track_pts[0], as_tensor(rot0_np, device), 512, rng)
    if q_set.shape[0] == 0:
        return {'seed': seed, 'error': 'no IK at singular start'}
    lo_np = kin.lmt_lo.detach().cpu().numpy()
    hi_np = kin.lmt_up.detach().cpu().numpy()
    q_set_np = q_set.detach().cpu().numpy()
    inbounds = ((q_set_np - lo_np > 0.15)
                & (hi_np - q_set_np > 0.15)).all(axis=1)
    if int(inbounds.sum()) < 8:
        inbounds = np.ones(q_set.shape[0], dtype=bool)
    q_good = q_set[inbounds]
    if q_good.shape[0] < 4:
        return {'seed': seed, 'error': f'only {q_good.shape[0]} inbound IK'}
    q_good_np = q_good.detach().cpu().numpy()
    n_picks = min(16, q_good.shape[0])
    L_start = rollout_lengths(kin, q_good, track_pts, plane_normal_t,
                              theta_max_rad=ROLLOUT_THETA_MAX,
                              enforce_init_pose=True, pos_priority=True)
    seed_idx = int(np.argmax(L_start))
    picks = farthest_point_pick(q_good_np, n_picks, seed_idx)
    q_picks = q_good[picks]

    q_traj_all, fail_infos = record_rollout(kin, q_picks, track_pts, plane_normal_np)
    L_self = L_start[picks] / L_max
    reasons = [fi['reason'] for fi in fail_infos]
    reason_counts = {r: reasons.count(r) for r in set(reasons)}

    # sigma_pos_min per trajectory
    sigma_pos_mins = []
    for b in range(q_picks.shape[0]):
        q_b = q_traj_all[:, b, :]
        changed = np.any(np.abs(np.diff(q_b, axis=0)) > 1e-9, axis=1)
        last_alive = int(np.where(changed)[0].max()) + 1 if changed.any() else 0
        q_t = torch.as_tensor(q_b[:last_alive + 1], device=device, dtype=torch.float32)
        _, _, J_t, _ = kin.tcp_fk_jac(q_t)
        J_np = J_t.detach().cpu().numpy()
        sigmas = np.array([np.linalg.svd(J_np[t, :3, :], compute_uv=False)[-1]
                           for t in range(J_np.shape[0])])
        sigma_pos_mins.append(float(sigmas.min()))

    return {
        'seed': seed,
        'L_self_min': float(L_self.min()),
        'L_self_max': float(L_self.max()),
        'L_self_mean': float(L_self.mean()),
        'L_self_rel_spread': float((L_self.max() - L_self.min()) / max(L_self.mean(), 1e-6)),
        'reason_counts': reason_counts,
        'sigma_pos_min_overall': float(min(sigma_pos_mins)),
        'sigma_pos_min_mean': float(np.mean(sigma_pos_mins)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-tasks', type=int, default=10)
    parser.add_argument('--seed-start', type=int, default=500)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kin = BatchedFR3Kinematics(device=device)
    print(f"{'seed':<6}{'rel_spread':<12}{'L_min':<8}{'L_max':<8}"
          f"{'sig_min_min':<13}{'sig_min_mean':<14}{'reasons':<40}")
    rs = []
    for k in range(args.n_tasks):
        seed = args.seed_start + k
        result = scan_one(kin, seed)
        if 'error' in result:
            print(f"{seed:<6}ERROR: {result['error']}")
            continue
        reasons = sorted(result['reason_counts'].items(), key=lambda r: -r[1])
        rstr = ','.join(f'{r}={c}' for r, c in reasons)
        print(f"{seed:<6}{result['L_self_rel_spread']:<12.3f}"
              f"{result['L_self_min']:<8.3f}{result['L_self_max']:<8.3f}"
              f"{result['sigma_pos_min_overall']:<13.4f}{result['sigma_pos_min_mean']:<14.4f}{rstr:<40}")
        rs.append(result['L_self_rel_spread'])
    if rs:
        rs = np.array(rs)
        print(f"\nhard fraction (rel_spread > 0.15): "
              f"{int((rs > 0.15).sum())}/{len(rs)} = {(rs > 0.15).mean():.2%}")


if __name__ == '__main__':
    main()

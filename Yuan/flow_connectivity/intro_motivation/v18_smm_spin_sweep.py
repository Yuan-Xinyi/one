"""Sweep tool spin angle for one task seed.

The task constraint is position + pen z-axis direction only — rotation
ABOUT the pen z-axis is irrelevant to the actual line-following task.
But the 6-DOF SMM enumeration locks down the full SO(3), so picking a
specific spin gives a specific 1D SMM. By sweeping the spin angle we
sample multiple 1D SMM "leaves" of the underlying 2D self-motion
manifold (r=2 because the true task is 5-DOF).

For each spin angle θ:
  1. Build R_θ = R_base @ Rotz(θ)  (same pen z-axis as base, rotated about it)
  2. Enumerate 1D SMM branches at (p_tgt, R_θ).
  3. Run 6-DOF strict rollout for n_per_branch q0 per branch, with R_θ
     held fixed throughout the rollout.
  4. Record per-branch L_self.

Output:
  * `task_seed{S}_spin_sweep.png`: 3 panels — branch count / per-branch
    mean L / best–worst gap, all vs spin angle.
  * `task_seed{S}_spin_sweep.jsonl`: per-spin per-branch stats.

Usage:
    python -m Yuan.flow_connectivity.intro_motivation.v18_smm_spin_sweep --seed 118
    python -m Yuan.flow_connectivity.intro_motivation.v18_smm_spin_sweep --seed 42 --free-task --n-spins 24
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

import Yuan.flow_connectivity.config as cfg
from Yuan.flow_connectivity.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.flow_connectivity.batched_rollout import _branch_seed_bank
from Yuan.flow_connectivity.intro_motivation.v18_smm_core import (
    DEDUP_RAD, DEFAULT_H, JOINT_MARGIN,
    as_tensor, enumerate_branches, get_task_target_pose, path_length,
    project_and_filter,
)
from Yuan.flow_connectivity.intro_motivation.v18_smm_rollout_6dof import (
    EPS_ORI_6DOF, EPS_POS_6DOF, V_PATH, _batched_segment_6dof,
)
from Yuan.flow_connectivity.intro_motivation.v18_smm_task import sample_branch_q0s
from Yuan.flow_connectivity.v18_data_prep import _dense_ik_at


def rotz(theta: float) -> np.ndarray:
    """Rotation about the local z-axis by theta (post-multiplied to R_base
    rotates around R_base[:,2], i.e. the tool z-axis)."""
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return np.array([[c, -s, 0.0],
                      [s,  c, 0.0],
                      [0.0, 0.0, 1.0]], dtype=np.float32)


def rollout_lengths_fixed_R(kin, q_batch, track_pts, R_tgt_np,
                              eps_pos=EPS_POS_6DOF, eps_ori=EPS_ORI_6DOF):
    """6-DOF strict rollout that uses a FIXED R_tgt for all segments
    (instead of rebuilding from plane_normal+direction). Required when
    we want to track an arbitrary spin around the pen z-axis."""
    device = kin.device
    B = q_batch.shape[0]
    q = q_batch.clone()
    alive = torch.ones(B, device=device, dtype=torch.bool)
    lengths_m = torch.zeros(B, device=device, dtype=torch.float32)
    R_t = torch.as_tensor(R_tgt_np, device=device,
                            dtype=torch.float32).unsqueeze(0).expand(B, 3, 3)
    for idx in range(track_pts.shape[0] - 1):
        if not bool(alive.any().item()):
            break
        p0 = track_pts[idx]
        seg_vec = track_pts[idx + 1] - p0
        seg_len = float(seg_vec.norm().item())
        if seg_len < 1e-8:
            continue
        direction = seg_vec / seg_vec.norm().clamp_min(1e-12)
        n_steps = max(1, int(round(seg_len / (V_PATH * float(cfg.DT)))))
        v_path = torch.full((B,), V_PATH, device=device, dtype=torch.float32)
        T_total = torch.full((B,), n_steps, device=device, dtype=torch.long)
        alive_entering = alive.clone()

        lengths_step, q, alive = _batched_segment_6dof(
            q_init=q, R_tgt=R_t,
            p0=p0.unsqueeze(0).expand(B, 3),
            d_dir=direction.unsqueeze(0).expand(B, 3),
            v_path=v_path, T_total=T_total, n_steps=n_steps,
            kin=kin, eps_pos=eps_pos, eps_ori=eps_ori,
            alive_mask=alive, enforce_init_pose=(idx == 0),
        )
        completed = lengths_step.float() * (V_PATH * float(cfg.DT))
        lengths_m = torch.where(alive_entering, lengths_m + completed, lengths_m)
    return lengths_m.detach().cpu().numpy()


def analyze_spin(p_tgt, R_theta, kin, rng_seed, track_pts, L_max,
                  n_ik_seeds, n_per_branch, h):
    """For one spin angle, enumerate SMM branches + rollout. Returns per-branch
    dicts with mean L, std L, n_q0, closed flag."""
    rng = np.random.default_rng(rng_seed)
    device = kin.device
    p_t = torch.as_tensor(p_tgt, device=device, dtype=torch.float32)
    R_t = torch.as_tensor(R_theta, device=device, dtype=torch.float32)
    extra = _branch_seed_bank(kin).detach().cpu().numpy()
    Q_seed_t, _ = _dense_ik_at(kin, p_t, R_t, n_ik_seeds, rng, extra_seeds=extra)
    if Q_seed_t.shape[0] == 0:
        return []
    Q = project_and_filter(kin, Q_seed_t.detach().cpu().numpy(), p_tgt, R_theta,
                            kin.lmt_lo.detach().cpu().numpy(),
                            kin.lmt_up.detach().cpu().numpy(),
                            joint_margin=JOINT_MARGIN, dedup_rad=DEDUP_RAD,
                            verbose=False)
    if Q.shape[0] == 0:
        return []
    branches, _ = enumerate_branches(kin, Q, p_tgt, R_theta, h)
    if len(branches) == 0:
        return []

    all_q, all_bid, _ = sample_branch_q0s(branches, n_per_branch)
    q_batch = torch.as_tensor(all_q, device=device, dtype=torch.float32)
    L = rollout_lengths_fixed_R(kin, q_batch, track_pts, R_theta) / L_max

    rows = []
    for bid in range(len(branches)):
        b_L = L[all_bid == bid]
        if len(b_L) == 0:
            continue
        rows.append({
            'branch_id': bid,
            'n_q0': int(len(b_L)),
            'L_mean': float(b_L.mean()),
            'L_std': float(b_L.std()),
            'L_min': float(b_L.min()),
            'L_max': float(b_L.max()),
            'closed': bool(branches[bid]['closed']),
            'arc_rad': float(np.sum(np.linalg.norm(
                np.diff(branches[bid]['traj'], axis=0), axis=1))),
        })
    return rows


def save_plot(out_png: Path, seed: int, spin_angles_deg, results, L_max):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    n_branches = [len(r) for r in results]
    L_best = [max((b['L_mean'] for b in r), default=0.0) for r in results]
    L_worst = [min((b['L_mean'] for b in r), default=0.0) for r in results]
    L_mean_all = [float(np.mean([b['L_mean'] for b in r])) if r else 0.0
                   for r in results]
    gap = [b - w for b, w in zip(L_best, L_worst)]

    # (1) branch count
    ax = axes[0]
    ax.plot(spin_angles_deg, n_branches, '-o', linewidth=2, color='C0')
    ax.set_xlabel('spin angle θ [°]')
    ax.set_ylabel('# SMM branches')
    ax.set_title('branch count vs spin')
    ax.set_xlim(0, 360)
    ax.grid(alpha=0.3)
    yt = sorted(set(n_branches))
    ax.set_yticks(yt)

    # (2) per-branch L scatter + best/worst lines
    ax = axes[1]
    cmap = plt.get_cmap('tab10')
    for theta_deg, branches in zip(spin_angles_deg, results):
        for k, b in enumerate(branches):
            ax.errorbar(theta_deg, b['L_mean'], yerr=b['L_std'],
                         fmt='o', color=cmap(k % 10),
                         alpha=0.7, markersize=5, capsize=2,
                         markeredgecolor='black', markeredgewidth=0.3)
    ax.plot(spin_angles_deg, L_best, '-', color='C0',
             linewidth=2, label='max(L_mean) over branches', alpha=0.7)
    ax.plot(spin_angles_deg, L_worst, '-', color='C3',
             linewidth=2, label='min(L_mean) over branches', alpha=0.7)
    ax.plot(spin_angles_deg, L_mean_all, '--', color='gray',
             linewidth=1.5, label='mean(L_mean) over branches', alpha=0.6)
    ax.set_xlabel('spin angle θ [°]')
    ax.set_ylabel('L_self / L_max')
    ax.set_title('per-branch mean L vs spin')
    ax.set_xlim(0, 360)
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.3)
    ymax = max((max(L_best, default=0.05), max(L_worst, default=0.05))) * 1.15
    ax.set_ylim(-0.02, max(ymax, 0.1))

    # (3) best-worst gap
    ax = axes[2]
    ax.plot(spin_angles_deg, gap, '-o', color='black', linewidth=2)
    ax.set_xlabel('spin angle θ [°]')
    ax.set_ylabel('L_best - L_worst (mean L)')
    ax.set_title('branch differentiation vs spin')
    ax.set_xlim(0, 360)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, max(max(gap, default=0.05) * 1.15, 0.05))

    fig.suptitle(f'Tool spin sweep (seed={seed}, n_spins={len(spin_angles_deg)})',
                  fontsize=12, y=1.02)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=118)
    parser.add_argument('--n-spins', type=int, default=12)
    parser.add_argument('--n-per-branch', type=int, default=100)
    parser.add_argument('--n-ik-seeds', type=int, default=128)
    parser.add_argument('--h', type=float, default=DEFAULT_H)
    parser.add_argument('--free-task', action='store_true')
    parser.add_argument('--out-dir', type=str,
                        default='Yuan/flow_connectivity/intro_motivation/data')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kin = BatchedFR3Kinematics(device=device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    p_tgt, R_base, task = get_task_target_pose(args.seed, kin, rng,
                                                free=args.free_task)
    task_path = task['fine_path_pts']
    L_max = path_length(task_path)
    track_pts = as_tensor(task_path, device)
    print(f'seed={args.seed}, p_tgt={p_tgt}')
    print(f'  base z_tgt = R_base[:,2] = {R_base[:,2]}')
    print(f'  task L_max = {L_max:.3f} m')
    print(f'  sweeping {args.n_spins} spin angles around tool z-axis\n')

    spin_angles = np.linspace(0, 2 * np.pi, args.n_spins, endpoint=False)
    results = []
    for i, theta in enumerate(spin_angles):
        R_theta = (R_base @ rotz(theta)).astype(np.float32)
        # Per-spin rng seed so each sweep is reproducible.
        per_spin_seed = args.seed * 1000 + i
        rows = analyze_spin(p_tgt, R_theta, kin, per_spin_seed, track_pts,
                             L_max, args.n_ik_seeds, args.n_per_branch, args.h)
        n_b = len(rows)
        L_best = max((r['L_mean'] for r in rows), default=0.0)
        L_worst = min((r['L_mean'] for r in rows), default=0.0)
        print(f'  θ={np.rad2deg(theta):6.1f}°: '
              f'{n_b} branches, '
              f'best L={L_best:.3f}, worst L={L_worst:.3f}, '
              f'gap={L_best-L_worst:.3f}')
        results.append(rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f'task_seed{args.seed}_spin_sweep.jsonl'
    with open(out_jsonl, 'w') as f:
        f.write(json.dumps({
            'type': 'meta', 'seed': args.seed, 'free_task': args.free_task,
            'n_spins': args.n_spins, 'p_tgt': p_tgt.tolist(),
            'R_base': R_base.tolist(), 'L_max': float(L_max),
            'spin_angles_rad': spin_angles.tolist(),
        }) + '\n')
        for theta, rows in zip(spin_angles, results):
            f.write(json.dumps({
                'theta_rad': float(theta),
                'theta_deg': float(np.rad2deg(theta)),
                'branches': rows,
            }) + '\n')
    print(f'\nsaved: {out_jsonl}')

    out_png = out_dir / f'task_seed{args.seed}_spin_sweep.png'
    save_plot(out_png, args.seed, np.rad2deg(spin_angles), results, L_max)
    print(f'saved: {out_png}')


if __name__ == '__main__':
    main()

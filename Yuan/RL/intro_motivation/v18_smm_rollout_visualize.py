"""Visualize 6-DOF rollout joint trajectories per SMM branch.

For each SMM branch at the seed's task start, pick a representative q0
(median performer along the arc), run the 6-DOF strict rollout with
recording, and plot 7 joint subplots showing q_j(t) per branch with
JL bands and failure-point stars.

Usage:
    python -m Yuan.RL.intro_motivation.v18_smm_rollout_visualize --seed 118
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import Yuan.RL.config as cfg
from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.RL.batched_rollout import _branch_seed_bank, _dls_pinv, _rotvec_between
from Yuan.RL.intro_motivation.v18_motivation_core import (
    LINE_L_RANGE, TARGET_PATH_M, as_tensor, extend_task_path,
    path_length, sample_line_task,
)
from Yuan.RL.intro_motivation.v18_smm_enumerate import (
    DEFAULT_H,
    enumerate_branches,
    project_and_filter,
)
from Yuan.RL.intro_motivation.v18_smm_rollout_6dof import (
    EPS_ORI_6DOF, EPS_POS_6DOF, JLIMIT_GAIN, JLIMIT_MARGIN, V_PATH,
    rollout_lengths_6dof,
)
from Yuan.RL.v18_data_prep import _build_R_from_normal_direction, _dense_ik_at


def record_6dof_rollout(kin: BatchedFR3Kinematics,
                         q_init: torch.Tensor,
                         track_pts: torch.Tensor,
                         plane_normal_np: np.ndarray,
                         eps_pos: float = EPS_POS_6DOF,
                         eps_ori: float = EPS_ORI_6DOF):
    """Run 6-DOF rollout step-by-step with recording.
    Returns (q_traj (T+1, B, 7), fail_infos list[dict])."""
    device = kin.device
    B = q_init.shape[0]
    eye7 = torch.eye(7, device=device, dtype=torch.float32).expand(B, 7, 7)
    dt = float(cfg.DT)
    lo = kin.lmt_lo
    hi = kin.lmt_up

    q = q_init.clone()
    alive = torch.ones(B, device=device, dtype=torch.bool)
    q_record = [q.clone()]
    fail_info: list[dict | None] = [None] * B
    step_global = 0

    # Init pose check against first segment's target.
    seg_dir0 = (track_pts[1] - track_pts[0])
    seg_dir0 = seg_dir0 / seg_dir0.norm().clamp_min(1e-12)
    rot0_np = _build_R_from_normal_direction(
        plane_normal_np, seg_dir0.detach().cpu().numpy())
    R_tgt0 = torch.as_tensor(rot0_np, device=device, dtype=torch.float32).unsqueeze(0).expand(B, 3, 3)
    p_init, R_init, _, _ = kin.tcp_fk_jac(q)
    init_pos_err = (track_pts[0] - p_init).norm(dim=-1)
    init_ori_err = _rotvec_between(R_init, R_tgt0).norm(dim=-1)
    init_fail = (init_pos_err > eps_pos) | (init_ori_err > eps_ori)
    alive = alive & ~init_fail
    for i in torch.where(init_fail)[0]:
        ii = int(i)
        fail_info[ii] = {
            'reason': 'init_pose_fail', 'fail_step': 0, 'fail_joint': -1,
            'pos_err': float(init_pos_err[ii]),
            'ori_err': float(init_ori_err[ii]),
        }

    for idx in range(track_pts.shape[0] - 1):
        if not bool(alive.any().item()):
            break
        p0 = track_pts[idx]
        seg_vec = track_pts[idx + 1] - p0
        seg_len = float(seg_vec.norm().item())
        if seg_len < 1e-8:
            continue
        direction = seg_vec / seg_vec.norm().clamp_min(1e-12)
        rot_np = _build_R_from_normal_direction(
            plane_normal_np, direction.detach().cpu().numpy())
        R_tgt = torch.as_tensor(rot_np, device=device, dtype=torch.float32).unsqueeze(0).expand(B, 3, 3)
        d_dir = direction.unsqueeze(0).expand(B, 3)
        p0_b = p0.unsqueeze(0).expand(B, 3)
        v_path_v = torch.full((B,), V_PATH, device=device, dtype=torch.float32)
        n_steps = max(1, int(round(seg_len / (V_PATH * dt))))

        for step in range(1, n_steps + 1):
            step_global += 1
            step_alive = alive.clone()
            if not bool(step_alive.any().item()):
                q_record.append(q.clone())
                continue

            p_ref = p0_b + (step * dt) * v_path_v.unsqueeze(-1) * d_dir
            p_dot_ff = v_path_v.unsqueeze(-1) * d_dir
            p_tcp, R_tcp, J, _ = kin.tcp_fk_jac(q)
            omega_err = _rotvec_between(R_tcp, R_tgt)
            x_dot_pos = p_dot_ff + float(cfg.KP_LIN) * (p_ref - p_tcp)
            x_dot_ori = float(cfg.KOMEGA) * omega_err
            x_dot = torch.cat([x_dot_pos, x_dot_ori], dim=-1)

            Jpinv = _dls_pinv(J, float(cfg.DLS_LAMBDA))
            q_dot_primary = (Jpinv @ x_dot.unsqueeze(-1)).squeeze(-1)
            N = eye7 - Jpinv @ J

            dist_lo = q - lo
            dist_hi = hi - q
            danger_lo = (JLIMIT_MARGIN - dist_lo).clamp(min=0.0)
            danger_hi = (JLIMIT_MARGIN - dist_hi).clamp(min=0.0)
            q_dot_jl = JLIMIT_GAIN * (danger_lo - danger_hi)
            q_dot_jl_proj = (N @ q_dot_jl.unsqueeze(-1)).squeeze(-1)

            q_dot = (q_dot_primary + q_dot_jl_proj).clamp(-kin.qdot_max, kin.qdot_max)
            q_new_raw = q + q_dot * dt
            jl_out_lo = (q_new_raw < lo - 1e-6)
            jl_out_hi = (q_new_raw > hi + 1e-6)
            joint_limit_hit = (jl_out_lo | jl_out_hi).any(dim=-1)
            q_new = q_new_raw.clamp(lo, hi)

            p_new, R_new, _, _ = kin.tcp_fk_jac(q_new)
            pos_err = (p_ref - p_new).norm(dim=-1)
            orient_err = _rotvec_between(R_new, R_tgt).norm(dim=-1)

            fail_pos = step_alive & (pos_err > eps_pos)
            fail_ori = step_alive & (orient_err > eps_ori)
            fail_lmt = step_alive & joint_limit_hit
            died = fail_pos | fail_ori | fail_lmt

            for i in torch.where(died)[0]:
                ii = int(i)
                if fail_info[ii] is not None:
                    continue
                if bool(fail_lmt[ii]):
                    joints_out = torch.where(jl_out_lo[ii] | jl_out_hi[ii])[0]
                    fj = int(joints_out[0]) if len(joints_out) > 0 else -1
                    reason = f'joint_limit (j{fj})'
                elif bool(fail_pos[ii]):
                    fj, reason = -1, 'pos_err'
                else:
                    fj, reason = -1, 'ori_err'
                fail_info[ii] = {
                    'reason': reason, 'fail_step': step_global, 'fail_joint': fj,
                    'pos_err': float(pos_err[ii]),
                    'ori_err': float(orient_err[ii]),
                }

            ok = step_alive & ~died
            q = torch.where(ok.unsqueeze(-1), q_new, q)
            alive = alive & ~died
            q_record.append(q.clone())

    for i in range(B):
        if fail_info[i] is None:
            fail_info[i] = {
                'reason': 'completed_path', 'fail_step': step_global,
                'fail_joint': -1, 'pos_err': 0.0, 'ori_err': 0.0,
            }

    q_traj = torch.stack(q_record, dim=0)
    return q_traj, fail_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=118)
    parser.add_argument('--n-ik-seeds', type=int, default=256)
    parser.add_argument('--out-png', type=str,
                        default='Yuan/RL/intro_motivation/data/smm_rollout_curves.png')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kin = BatchedFR3Kinematics(device=device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

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

    p_t = torch.as_tensor(p_tgt, device=device, dtype=torch.float32)
    R_t = torch.as_tensor(R_tgt, device=device, dtype=torch.float32)
    extra = _branch_seed_bank(kin).detach().cpu().numpy()
    Q_seed_t, _ = _dense_ik_at(kin, p_t, R_t, args.n_ik_seeds, rng, extra_seeds=extra)
    Q_seed = Q_seed_t.detach().cpu().numpy()
    lo = kin.lmt_lo.detach().cpu().numpy()
    hi = kin.lmt_up.detach().cpu().numpy()
    Q = project_and_filter(kin, Q_seed, p_tgt, R_tgt, lo, hi)
    branches, _ = enumerate_branches(kin, Q, p_tgt, R_tgt, DEFAULT_H)
    print(f'  → {len(branches)} branches')

    # Pick representative q0 per branch (one whose L matches branch median).
    rep_qs, rep_meta = [], []
    for bid, b in enumerate(branches):
        traj = b['traj']
        n_samp = min(15, traj.shape[0])
        idxs = np.linspace(0, traj.shape[0] - 1, n_samp).astype(int)
        q_samp = torch.as_tensor(traj[idxs], device=device, dtype=torch.float32)
        L_samp = rollout_lengths_6dof(kin, q_samp, track_pts, plane_normal_t)
        med = float(np.median(L_samp))
        pick = int(np.argmin(np.abs(L_samp - med)))
        rep_qs.append(traj[idxs[pick]])
        rep_meta.append({'bid': bid, 'L_self_norm': float(L_samp[pick]) / L_max,
                          'arc_pos': float(idxs[pick]) / max(traj.shape[0] - 1, 1)})
        print(f'  br{bid}: representative q0 at arc {rep_meta[-1]["arc_pos"]:.2f}, '
              f'L_self={rep_meta[-1]["L_self_norm"]:.3f}')

    q_init = torch.as_tensor(np.array(rep_qs), device=device, dtype=torch.float32)
    q_traj, fail_info = record_6dof_rollout(kin, q_init, track_pts, plane_normal_np)
    T = q_traj.shape[0]
    print(f'\n  rollout: T={T} steps ({T * cfg.DT:.2f}s)')
    for bid, info in enumerate(fail_info):
        t_fail = info['fail_step'] * cfg.DT
        L_completed = info['fail_step'] * V_PATH * cfg.DT
        print(f'    br{bid}: dies at t={t_fail:.2f}s, L≈{L_completed:.3f}m '
              f'({L_completed / L_max * 100:.1f}% of path), reason: {info["reason"]}')

    q_traj_np = q_traj.detach().cpu().numpy()

    cmap = plt.get_cmap('tab10')
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flatten()
    t_axis = np.arange(T) * float(cfg.DT)

    for j in range(7):
        ax = axes[j]
        ax.axhspan(lo[j] - 1, lo[j], color='red', alpha=0.10)
        ax.axhspan(hi[j], hi[j] + 1, color='red', alpha=0.10)
        ax.axhline(lo[j], color='red', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.axhline(hi[j], color='red', linestyle='--', linewidth=0.8, alpha=0.7)

        for bid in range(len(branches)):
            n_alive = min(fail_info[bid]['fail_step'] + 1, T)
            ax.plot(t_axis[:n_alive], q_traj_np[:n_alive, bid, j],
                    '-', color=cmap(bid % 10), alpha=0.9, linewidth=1.6,
                    label=f'br{bid}')
            ax.scatter(t_axis[n_alive - 1], q_traj_np[n_alive - 1, bid, j],
                       s=70, c=[cmap(bid % 10)],
                       edgecolors='black', linewidths=1.0, marker='*', zorder=6)

        ax.set_title(f'j{j}  limits [{lo[j]:.2f}, {hi[j]:.2f}]', fontsize=10)
        ax.set_xlabel('t [s]', fontsize=8)
        ax.set_ylabel('q [rad]', fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
        ymin = min(lo[j], q_traj_np[:, :, j].min()) - 0.2
        ymax = max(hi[j], q_traj_np[:, :, j].max()) + 0.2
        ax.set_ylim(ymin, ymax)
        if j == 0:
            ax.legend(fontsize=8)

    ax_sum = axes[7]
    ax_sum.axis('off')
    lines = [f'6-DOF strict rollout',
             f'  eps_pos = {EPS_POS_6DOF*1000:.0f} mm',
             f'  eps_ori = {np.rad2deg(EPS_ORI_6DOF):.1f}°',
             f'  V_path  = {V_PATH:.2f} m/s', '',
             f'seed={args.seed}, task L_max={L_max:.2f} m', '',
             'per-branch outcome:']
    for bid, info in enumerate(fail_info):
        t_fail = info['fail_step'] * cfg.DT
        L_completed = info['fail_step'] * V_PATH * cfg.DT
        lines += [
            f'  br{bid}: t={t_fail:.2f}s, '
            f'L≈{L_completed:.2f}m ({L_completed / L_max * 100:.0f}%)',
            f'        ✗ {info["reason"]}',
        ]
    lines += ['', 'dashed red = FR3 joint limits',
              'star = where rollout died']
    ax_sum.text(0.0, 1.0, '\n'.join(lines), fontsize=9,
                family='monospace', verticalalignment='top')

    fig.suptitle('6-DOF rollout: joint trajectories per SMM branch '
                 f'(seed={args.seed})',
                 fontsize=12, y=1.005)
    fig.tight_layout()
    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'\nsaved: {out_png}')


if __name__ == '__main__':
    main()

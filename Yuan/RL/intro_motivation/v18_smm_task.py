"""End-to-end SMM workflow for one task seed (6-DOF strict throughout).

  1. Re-derive line task path from --seed (--free-task for permissive sampler).
  2. Enumerate SMM branches at the 6-DOF task start pose.
     Print branch count + per-branch stats.
  3. Sample q0 per branch, run 6-DOF strict rollout. Save summary PNG and
     SMM joint-trajectory PNG.
  4. Unless --no-viewer: pick a representative q0 per branch, record
     rollout, launch ONE viewer animating each branch sequentially.

Rollout and SMM enumeration share the same constraint (6-DOF locked pose),
so branches are tested on the exact 1D manifold they were enumerated on.

Usage:
    python -m Yuan.RL.intro_motivation.v18_smm_task --seed 118
    python -m Yuan.RL.intro_motivation.v18_smm_task --seed 42 --free-task
    python -m Yuan.RL.intro_motivation.v18_smm_task --seed 118 --no-viewer
"""
from __future__ import annotations

import argparse
import builtins
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw
from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.RL.batched_rollout import _branch_seed_bank
from Yuan.RL.fr3_with_pen import attach_pen_visual, make_fr3_with_pen
from Yuan.RL.intro_motivation.v18_smm_core import (
    DEDUP_RAD, DEFAULT_H, JOINT_MARGIN,
    as_tensor, enumerate_branches, get_task_target_pose, path_length,
    project_and_filter,
)
from Yuan.RL.intro_motivation.v18_smm_rollout_6dof import (
    record_rollout_6dof, rollout_lengths_6dof,
)
from Yuan.RL.v18_data_prep import _dense_ik_at


PLAYBACK_DT = 0.04
HOLD_AT_END_SEC = 1.5
GHOST_ALPHA = 0.20
ACTIVE_ALPHA = 0.95
HIDDEN_ALPHA = 0.0


def add_task_path(base, task_path: np.ndarray,
                   plane_normal: np.ndarray | None = None,
                   plane_size: float | None = None,
                   plane_rgb: tuple[float, float, float] = (0.82, 0.82, 0.86),
                   plane_alpha: float = 0.25):
    """Draw the task path line + endpoint spheres in the ONE viewer.
    Optional plane (the 'paper') auto-sized to the path."""
    if plane_normal is not None:
        plane_center = task_path.mean(axis=0).astype(np.float32)
        path_len = float(np.linalg.norm(task_path[-1] - task_path[0]))
        if plane_size is None:
            plane_size = max(2.0, path_len + 0.6)
        ossop.plane(pos=tuple(plane_center),
                    normal=tuple(plane_normal.astype(np.float32)),
                    size=(plane_size, plane_size), thickness=2e-3,
                    rgb=plane_rgb, alpha=plane_alpha).attach_to(base.scene)
    segs = np.stack([task_path[:-1], task_path[1:]], axis=1)
    ossop.linsegs(segs=segs, radius=0.0015,
                   srgbs=np.array([0.08, 0.08, 0.08]),
                   alpha=0.75).attach_to(base.scene)
    ossop.sphere(pos=tuple(task_path[0]), radius=0.012,
                  rgb=(0.05, 0.65, 0.20), alpha=0.95).attach_to(base.scene)
    ossop.sphere(pos=tuple(task_path[-1]), radius=0.014,
                  rgb=(0.85, 0.10, 0.10), alpha=0.95).attach_to(base.scene)


def sample_branch_q0s(branches, n_per_branch: int):
    """For each branch, evenly sample n_per_branch q0 along its arc.
    Returns (all_q (N,7), all_bid (N,), all_arc (N,) normalized 0..1)."""
    all_q, all_bid, all_arc = [], [], []
    for bid, b in enumerate(branches):
        traj = b['traj']
        n = min(n_per_branch, traj.shape[0])
        idxs = np.linspace(0, traj.shape[0] - 1, n).astype(int)
        for i in idxs:
            all_q.append(traj[i])
            all_bid.append(bid)
            all_arc.append(float(i) / max(traj.shape[0] - 1, 1))
    return (np.array(all_q, dtype=np.float32),
            np.array(all_bid, dtype=np.int32),
            np.array(all_arc, dtype=np.float32))


def pick_representative_q0(branches, kin, track_pts, plane_normal_t,
                            L_max, mode: str = 'best'):
    """For each branch, sample 15 q0 along its arc and pick one:
       'best'   = highest L_self  (branch potential, default for viewer)
       'median' = closest to median L (typical branch behavior)
       'worst'  = lowest L_self  (branch worst-case)"""
    rep = []
    for bid, b in enumerate(branches):
        traj = b['traj']
        n = min(15, traj.shape[0])
        idxs = np.linspace(0, traj.shape[0] - 1, n).astype(int)
        q_samp = torch.as_tensor(traj[idxs], device=kin.device, dtype=torch.float32)
        L = rollout_lengths_6dof(kin, q_samp, track_pts, plane_normal_t)
        if mode == 'best':
            pick = int(np.argmax(L))
        elif mode == 'worst':
            pick = int(np.argmin(L))
        else:
            med = float(np.median(L))
            pick = int(np.argmin(np.abs(L - med)))
        rep.append({
            'q0': traj[idxs[pick]],
            'L_self_norm': float(L[pick]) / L_max,
            'arc_pos': float(idxs[pick]) / max(traj.shape[0] - 1, 1),
        })
    return rep


def save_smm_joint_curves(out_png: Path, seed: int, kin, branches):
    """7 subplots, one per joint: q_j(arc length along SMM) overlaid per branch,
    with FR3 joint limits as dashed red lines and branch endpoints starred."""
    lo = kin.lmt_lo.detach().cpu().numpy()
    hi = kin.lmt_up.detach().cpu().numpy()
    cmap = plt.get_cmap('tab10')
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    axes = axes.flatten()
    for j in range(7):
        ax = axes[j]
        ax.axhspan(lo[j] - 1, lo[j], color='red', alpha=0.10)
        ax.axhspan(hi[j], hi[j] + 1, color='red', alpha=0.10)
        ax.axhline(lo[j], color='red', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.axhline(hi[j], color='red', linestyle='--', linewidth=0.8, alpha=0.7)
        for bid, b in enumerate(branches):
            traj = b['traj']
            diffs = np.linalg.norm(np.diff(traj, axis=0), axis=1)
            x = np.concatenate([[0.0], np.cumsum(diffs)])
            ax.plot(x, traj[:, j], '-', color=cmap(bid % 10), alpha=0.85,
                     linewidth=1.6,
                     label=f'br{bid} ({"closed" if b["closed"] else "open"})')
            ax.scatter([x[0], x[-1]], [traj[0, j], traj[-1, j]],
                        s=40, c=[cmap(bid % 10)],
                        edgecolors='black', linewidths=0.5, zorder=5)
        ax.set_title(f'j{j}  limits [{lo[j]:.2f}, {hi[j]:.2f}]', fontsize=10)
        ax.set_xlabel('arc length along SMM (rad)', fontsize=8)
        ax.set_ylabel('q [rad]', fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
        all_y = np.concatenate([b['traj'][:, j] for b in branches])
        ymin = min(lo[j], float(all_y.min())) - 0.2
        ymax = max(hi[j], float(all_y.max())) + 0.2
        ax.set_ylim(ymin, ymax)
        if j == 0:
            ax.legend(fontsize=8)
    ax_info = axes[7]
    ax_info.axis('off')
    info = [f'seed={seed}', f'{len(branches)} SMM branches', '']
    for bid, b in enumerate(branches):
        arc = float(np.sum(np.linalg.norm(np.diff(b['traj'], axis=0), axis=1)))
        info.append(f'  br{bid}: T={b["traj"].shape[0]}, arc={arc:.2f} rad, '
                     f'{"closed" if b["closed"] else "open"}')
    info += ['', 'dashed red = FR3 joint limits',
              'star = SMM arc endpoint']
    ax_info.text(0.0, 1.0, '\n'.join(info), fontsize=9,
                  family='monospace', verticalalignment='top')
    fig.suptitle(f'SMM joint trajectories (seed={seed})',
                  fontsize=12, y=1.005)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)


def save_summary_plot(out_png: Path, seed: int,
                       branches, Q, assigned,
                       all_bid, all_arc, L_rel):
    """3-panel summary: SMM PCA scatter + per-branch L violin + intra-branch L vs arc."""
    cmap = plt.get_cmap('tab10')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # (1) SMM PCA scatter
    ax = axes[0]
    all_pts = np.concatenate([b['traj'] for b in branches] + [Q], axis=0)
    mu = all_pts.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(all_pts - mu, full_matrices=False)
    W = Vt[:2].T
    for bid, b in enumerate(branches):
        t2d = (b['traj'] - mu) @ W
        n_m = int((assigned == bid).sum())
        arc = float(np.sum(np.linalg.norm(np.diff(b['traj'], axis=0), axis=1)))
        ax.plot(t2d[:, 0], t2d[:, 1], '-', color=cmap(bid % 10),
                alpha=0.7, linewidth=1.6,
                label=f'br{bid}: {n_m} q0, arc={arc:.1f} rad'
                      + (' (closed)' if b['closed'] else ''))
        ax.scatter(t2d[0, 0], t2d[0, 1], s=80, c=[cmap(bid % 10)],
                   edgecolors='black', linewidth=1.0, marker='*', zorder=6)
    Q_2d = (Q - mu) @ W
    for j in range(Q.shape[0]):
        bid_j = int(assigned[j])
        c = cmap(bid_j % 10) if bid_j >= 0 else 'gray'
        ax.scatter(Q_2d[j, 0], Q_2d[j, 1], s=28, c=[c],
                   edgecolors='black', linewidth=0.3, zorder=5)
    ax.set_title(f'SMM branches in 7-DOF joint space (PCA→2D)\n'
                  f'{len(branches)} branches, {Q.shape[0]} IK candidates')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.legend(fontsize=8, loc='best'); ax.grid(alpha=0.3)

    # (2) violin per branch
    ax = axes[1]
    data, pos, lbls = [], [], []
    for bid in range(len(branches)):
        L = L_rel[all_bid == bid]
        if len(L) > 0:
            data.append(L); pos.append(bid); lbls.append(f'br{bid}')
    parts = ax.violinplot(data, positions=pos, showmeans=True,
                           showmedians=False, widths=0.6)
    for k, pc in enumerate(parts['bodies']):
        pc.set_facecolor(cmap(pos[k] % 10))
        pc.set_alpha(0.55)
        pc.set_edgecolor('black')
    parts['cmeans'].set_color('black'); parts['cmeans'].set_linewidth(2)
    rng_j = np.random.default_rng(0)
    for bid in pos:
        L = L_rel[all_bid == bid]
        jitter = rng_j.uniform(-0.10, 0.10, size=len(L))
        ax.scatter(np.full(len(L), bid) + jitter, L,
                   c=[cmap(bid % 10)], s=18, alpha=0.8,
                   edgecolors='black', linewidths=0.3, zorder=3)
    ax.set_xticks(pos); ax.set_xticklabels(lbls)
    ax.set_ylabel('L_self / L_max')
    ax.set_title('path-following per branch  [6-DOF strict]')
    ax.set_ylim(-0.02, max(float(L_rel.max()) * 1.15, 0.15))
    ax.grid(alpha=0.3, axis='y')

    # (3) intra-branch L vs arc pos
    ax = axes[2]
    for bid in range(len(branches)):
        m = all_bid == bid
        L = L_rel[m]
        if len(L) == 0: continue
        ax.scatter(all_arc[m], L, c=[cmap(bid % 10)],
                   s=30, alpha=0.8, edgecolors='black', linewidths=0.4,
                   label=f'br{bid} (n={len(L)})')
    ax.set_xlabel('normalized arc position within branch')
    ax.set_ylabel('L_self / L_max')
    ax.set_title('intra-branch variation')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_ylim(-0.02, max(float(L_rel.max()) * 1.15, 0.15))

    fig.suptitle(f'task seed={seed},  6-DOF strict rollout '
                 f'(rollout constraint = SMM enumeration constraint)',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)


def launch_viewer(seed: int, kin, branches, rep, q_traj_np, fail_infos,
                   task_path, plane_normal_np):
    """ONE viewer: per branch, a transparent ghost arm sits at q_start
    throughout, and a separate opaque arm animates the rollout. Branches
    play sequentially; ghosts persist."""
    death_step = [info['fail_step'] for info in fail_infos]
    cam_idx = max(0, min(len(task_path) // 4, len(task_path) - 1))
    cam_focus = tuple(float(x) for x in task_path[cam_idx])
    cam_pos = (cam_focus[0] + 1.4, cam_focus[1] - 1.7, cam_focus[2] + 0.9)
    base = ovw.World(cam_pos=cam_pos, cam_lookat_pos=cam_focus,
                     toggle_auto_cam_orbit=False)
    builtins.base = base
    ossop.frame(length_scale=0.18, radius_scale=0.7).attach_to(base.scene)
    add_task_path(base, task_path, plane_normal=plane_normal_np)

    cmap = plt.get_cmap('tab10')
    origin = np.zeros(3, dtype=np.float32)

    def _set_alpha(arm, pen_pair, value: float):
        arm.alpha = value
        if pen_pair is not None:
            shaft, tip = pen_pair
            if shaft is not None:
                shaft.alpha = value
            if tip is not None:
                tip.alpha = value

    # Static transparent ghosts at each branch's q_start. Never updated.
    for bid in range(len(branches)):
        rgb = tuple(float(c) for c in cmap(bid % 10)[:3])
        ghost, _ = make_fr3_with_pen(pos=origin)
        ghost.attach_to(base.scene)
        ghost.rgb = rgb
        ghost.alpha = GHOST_ALPHA
        attach_pen_visual(ghost, rgb=rgb, alpha=GHOST_ALPHA)
        ghost.fk(q_traj_np[0, bid])

    # Opaque animators, one per branch. Only the active one is visible —
    # toggling alpha must also toggle the pen shaft+tip primitives.
    actives = []
    active_pens = []
    for bid in range(len(branches)):
        rgb = tuple(float(c) for c in cmap(bid % 10)[:3])
        arm, _ = make_fr3_with_pen(pos=origin)
        arm.attach_to(base.scene)
        arm.rgb = rgb
        pen_pair = attach_pen_visual(arm, rgb=rgb, alpha=HIDDEN_ALPHA)
        _set_alpha(arm, pen_pair, HIDDEN_ALPHA)
        arm.fk(q_traj_np[0, bid])
        actives.append(arm)
        active_pens.append(pen_pair)

    state = {'t_float': 0.0, 'active_bid': 0, 'just_switched': True,
             'hold': 0.0}
    print(f'\n  sequential viewer: {len(actives)} branches '
          f'(ghosts always visible at q_start)')

    def animate(_dt, *_args, **_kwargs):
        bid = state['active_bid']
        death = death_step[bid]

        if state['just_switched']:
            for i, arm in enumerate(actives):
                target = ACTIVE_ALPHA if i == bid else HIDDEN_ALPHA
                _set_alpha(arm, active_pens[i], target)
            actives[bid].fk(q_traj_np[0, bid])
            state['t_float'] = 0.0
            print(f'  → playing br{bid} '
                  f'(death step {death}, {fail_infos[bid]["reason"]})')
            state['just_switched'] = False

        if state['hold'] > 0.0:
            state['hold'] -= PLAYBACK_DT
            actives[bid].fk(q_traj_np[death, bid])
            if state['hold'] <= 0.0:
                state['active_bid'] = (bid + 1) % len(actives)
                state['just_switched'] = True
            return

        t = state['t_float']
        if t >= death:
            actives[bid].fk(q_traj_np[death, bid])
            state['hold'] = HOLD_AT_END_SEC
            return
        actives[bid].fk(q_traj_np[int(t), bid])
        state['t_float'] += 1.0

    base.schedule_interval(animate, PLAYBACK_DT)
    base.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=118)
    parser.add_argument('--n-per-branch', type=int, default=100)
    parser.add_argument('--n-ik-seeds', type=int, default=256)
    parser.add_argument('--h', type=float, default=DEFAULT_H)
    parser.add_argument('--no-viewer', action='store_true')
    parser.add_argument('--free-task', action='store_true',
                        help='use permissive task sampler (any reachable p0, '
                             'any pen direction) instead of v18 default')
    parser.add_argument('--rep-mode', choices=['best', 'median', 'worst'],
                        default='best',
                        help='which q0 per branch the viewer animates')
    parser.add_argument('--out-dir', type=str,
                        default='Yuan/RL/intro_motivation/data')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kin = BatchedFR3Kinematics(device=device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # --- task pose ---
    p_tgt, R_tgt, task = get_task_target_pose(args.seed, kin, rng,
                                                free=args.free_task)
    if args.free_task:
        print('  (using permissive free-task sampler)')
    task_path = task['fine_path_pts']
    L_max = path_length(task_path)
    track_pts = as_tensor(task_path, device)
    plane_normal_np = task['plane_normal']
    plane_normal_t = as_tensor(plane_normal_np, device)
    print(f'seed={args.seed}, task L_max={L_max:.3f}m, p_tgt={p_tgt}')

    # --- enumerate SMM branches at task start ---
    p_t = torch.as_tensor(p_tgt, device=device, dtype=torch.float32)
    R_t = torch.as_tensor(R_tgt, device=device, dtype=torch.float32)
    extra = _branch_seed_bank(kin).detach().cpu().numpy()
    Q_seed_t, _ = _dense_ik_at(kin, p_t, R_t, args.n_ik_seeds, rng,
                                extra_seeds=extra)
    if Q_seed_t.shape[0] == 0:
        raise RuntimeError('no IK candidates at task start')
    Q_seed = Q_seed_t.detach().cpu().numpy()
    lo = kin.lmt_lo.detach().cpu().numpy()
    hi = kin.lmt_up.detach().cpu().numpy()
    Q = project_and_filter(kin, Q_seed, p_tgt, R_tgt, lo, hi,
                            joint_margin=JOINT_MARGIN, dedup_rad=DEDUP_RAD)
    if Q.shape[0] == 0:
        raise RuntimeError('no valid IK candidates after filter')
    branches, assigned = enumerate_branches(kin, Q, p_tgt, R_tgt, args.h)
    print(f'\n=== {len(branches)} SMM branches at seed {args.seed} task start ===')
    for bid, b in enumerate(branches):
        n_m = int((assigned == bid).sum())
        arc = float(np.sum(np.linalg.norm(np.diff(b['traj'], axis=0), axis=1)))
        print(f'  br{bid}: T={b["traj"].shape[0]}, arc={arc:.2f} rad, '
              f'{"closed" if b["closed"] else "open"}, members={n_m}')

    # --- path-following stats (6-DOF strict) ---
    all_q, all_bid, all_arc = sample_branch_q0s(branches, args.n_per_branch)
    q_batch = torch.as_tensor(all_q, device=device, dtype=torch.float32)
    print(f'\n  6-DOF strict rollout: {q_batch.shape[0]} q0 samples')
    L_abs = rollout_lengths_6dof(kin, q_batch, track_pts, plane_normal_t)
    L_rel = L_abs / L_max

    print('\n  per-branch L_self / L_max:')
    for bid in range(len(branches)):
        L = L_rel[all_bid == bid]
        if len(L) == 0: continue
        print(f'    br{bid}: n={len(L)}, mean={L.mean():.3f}, '
              f'std={L.std():.3f}, range=[{L.min():.3f}, {L.max():.3f}]')

    # --- save PNGs ---
    out_dir = Path(args.out_dir)
    out_png = out_dir / f'task_seed{args.seed}_summary.png'
    save_summary_plot(out_png, args.seed, branches, Q, assigned,
                       all_bid, all_arc, L_rel)
    print(f'\nsaved: {out_png}')
    joints_png = out_dir / f'task_seed{args.seed}_smm_joints.png'
    save_smm_joint_curves(joints_png, args.seed, kin, branches)
    print(f'saved: {joints_png}')

    if args.no_viewer:
        return

    # --- launch ONE viewer with rollout animation ---
    print(f'\n  preparing viewer (q0 selection: {args.rep_mode})...')
    rep = pick_representative_q0(branches, kin, track_pts, plane_normal_t,
                                  L_max, mode=args.rep_mode)
    for bid, r in enumerate(rep):
        print(f'    br{bid}: representative L_self={r["L_self_norm"]:.3f}, '
              f'arc_pos={r["arc_pos"]:.2f}')
    q_init = torch.as_tensor(np.array([r['q0'] for r in rep]),
                              device=device, dtype=torch.float32)
    q_traj, fail_infos = record_rollout_6dof(kin, q_init, track_pts, plane_normal_np)
    q_traj_np = q_traj.detach().cpu().numpy()
    print(f'  recorded {q_traj_np.shape[0]} steps; per-branch death:')
    for bid in range(len(branches)):
        print(f'    br{bid}: step {fail_infos[bid]["fail_step"]} '
              f'({fail_infos[bid]["reason"]})')

    launch_viewer(args.seed, kin, branches, rep, q_traj_np, fail_infos,
                   task_path, plane_normal_np)


if __name__ == '__main__':
    main()

"""End-to-end SMM workflow for one task seed.

  1. Re-derive v18 line task path from --seed.
  2. Enumerate SMM branches at the 6-DOF task start pose.
     Print branch count + per-branch stats.
  3. Sample q0 per branch, run path-following rollout (mode = --task-dof).
     Save a 2-panel summary PNG: SMM PCA scatter + per-branch L violin.
  4. Unless --no-viewer: pick a representative q0 per branch, record
     rollout, launch ONE viewer animating each branch sequentially.

Usage:
    python -m Yuan.RL.intro_motivation.v18_smm_task --seed 118
    python -m Yuan.RL.intro_motivation.v18_smm_task --seed 118 --task-dof 6
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

import Yuan.RL.config as cfg
import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw
from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.RL.batched_rollout import _branch_seed_bank, batched_rollout_segment
from Yuan.RL.fr3_with_pen import attach_pen_visual, make_fr3_with_pen
from Yuan.RL.intro_motivation.v18_smm_core import (
    DEDUP_RAD, DEFAULT_H, JOINT_MARGIN,
    as_tensor, enumerate_branches, get_task_target_pose, path_length,
    project_and_filter,
)
from Yuan.RL.intro_motivation.v18_smm_rollout_5dof_strict import (
    EPS_ORI_5DOF_STRICT, EPS_POS_5DOF_STRICT,
    record_rollout_5dof_strict, rollout_lengths_5dof_strict,
)
from Yuan.RL.intro_motivation.v18_smm_rollout_6dof import (
    EPS_ORI_6DOF, EPS_POS_6DOF, V_PATH,
    record_rollout_6dof, rollout_lengths_6dof,
)
from Yuan.RL.v18_data_prep import _dense_ik_at


PLAYBACK_DT = 0.04
HOLD_AT_END_SEC = 1.5
ROLLOUT_THETA_MAX = np.deg2rad(30.0)
V_PATH_5DOF = 0.10
EPS_P_5DOF = 0.05
CHUNK_SIZE = 1024
_BRANCH_ACTION = (1.0, 0.0, 1.0, 0.0)


def _rollout_chunk_pos_priority(kin, q_init, track_pts, plane_normal_t,
                                 theta_max_rad=ROLLOUT_THETA_MAX,
                                 enforce_init_pose=True):
    """5-DOF pos_priority rollout via batched_rollout_segment. Mirrors the
    behavior used in the original v18 motivation experiments."""
    from Yuan.RL.v18_data_prep import _build_R_from_normal_direction
    device = kin.device
    B = q_init.shape[0]
    q = q_init.clone()
    alive = torch.ones(B, device=device, dtype=torch.bool)
    lengths = torch.zeros(B, device=device, dtype=torch.float32)
    branch_action = torch.tensor(_BRANCH_ACTION, device=device,
                                  dtype=torch.float32).unsqueeze(0).expand(B, 4)
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
            plane_normal_t.detach().cpu().numpy(),
            direction.detach().cpu().numpy())
        n_steps = max(1, int(round(seg_len / (V_PATH_5DOF * float(cfg.DT)))))
        out = batched_rollout_segment(
            q_init=q,
            R_tgt=as_tensor(rot_np, device).unsqueeze(0).expand(B, 3, 3),
            branch_action=branch_action,
            p0=p0.unsqueeze(0).expand(B, 3),
            d_dir=direction.unsqueeze(0).expand(B, 3),
            v_path=torch.full((B,), V_PATH_5DOF, device=device, dtype=torch.float32),
            eps_p=torch.full((B,), EPS_P_5DOF, device=device, dtype=torch.float32),
            T_total=torch.full((B,), n_steps, device=device, dtype=torch.long),
            start_step=0, end_step=n_steps, kin=kin, alive_mask=alive,
            theta_max_rad=theta_max_rad,
            enforce_init_pose=enforce_init_pose,
            pos_priority=True,
        )
        completed = out['lengths'].float() / float(n_steps) * seg_len
        lengths = torch.where(alive, lengths + completed, lengths)
        q = out['q_final']
        alive = out['alive_out']
    return lengths.detach().cpu().numpy()


def rollout_lengths_pos_priority(kin, q_batch, track_pts, plane_normal_t,
                                  theta_max_rad=ROLLOUT_THETA_MAX,
                                  enforce_init_pose=True):
    """Chunked 5-DOF pos_priority rollout. Returns per-q meters travelled."""
    lengths = np.zeros(q_batch.shape[0], dtype=np.float32)
    for start in range(0, q_batch.shape[0], CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, q_batch.shape[0])
        lengths[start:end] = _rollout_chunk_pos_priority(
            kin, q_batch[start:end], track_pts, plane_normal_t,
            theta_max_rad=theta_max_rad,
            enforce_init_pose=enforce_init_pose)
    return lengths


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


def get_rollout_fns(task_dof: str):
    """Returns (rollout_lengths_fn, record_rollout_fn, label).
    record_rollout falls back to 6-DOF for the pos_priority '5' mode
    because the viewer needs the (q_traj, fail_infos) shape produced
    by the strict rollouts."""
    if task_dof == '6':
        def _rl(kin, q, tp, pn):
            return rollout_lengths_6dof(kin, q, tp, pn)
        return _rl, record_rollout_6dof, '6-DOF strict (1D null)'
    if task_dof == '5strict':
        def _rl(kin, q, tp, pn):
            return rollout_lengths_5dof_strict(kin, q, tp, pn)
        return _rl, record_rollout_5dof_strict, '5-DOF strict, spin free (2D null)'
    # '5' = v18 pos_priority (5-DOF + 30° dead zone). Viewer uses 6-DOF for replay.
    def _rl(kin, q, tp, pn):
        return rollout_lengths_pos_priority(kin, q, tp, pn)
    return _rl, record_rollout_6dof, '5-DOF pos_priority (viewer uses 6-DOF replay)'


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
                            rollout_fn, L_max, mode: str = 'best'):
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
        L = rollout_fn(kin, q_samp, track_pts, plane_normal_t)
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


def save_summary_plot(out_png: Path, seed: int, task_dof: str, mode_label: str,
                       branches, Q, assigned,
                       all_bid, all_arc, L_rel):
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
    ax.set_title(f'path-following per branch  [{mode_label}]')
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

    fig.suptitle(f'task seed={seed},  rollout={mode_label}',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)


GHOST_ALPHA = 0.20
ACTIVE_ALPHA = 0.95
HIDDEN_ALPHA = 0.0


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

        # On branch switch: swap alpha (arm + pen primitives), place active
        # arm at start, reset time.
        if state['just_switched']:
            for i, arm in enumerate(actives):
                target = ACTIVE_ALPHA if i == bid else HIDDEN_ALPHA
                _set_alpha(arm, active_pens[i], target)
            actives[bid].fk(q_traj_np[0, bid])
            state['t_float'] = 0.0
            print(f'  → playing br{bid} '
                  f'(death step {death}, {fail_infos[bid]["reason"]})')
            state['just_switched'] = False

        # Hold phase: keep the just-died branch frozen at its death pose.
        # Advance bid only once the hold has fully elapsed (so the *next*
        # frame, via just_switched, swaps the visible arm cleanly).
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
    parser.add_argument('--task-dof', type=str, choices=['5', '5strict', '6'],
                        default='5')
    parser.add_argument('--n-per-branch', type=int, default=30)
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

    rollout_fn, record_fn, mode_label = get_rollout_fns(args.task_dof)

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
    print(f'seed={args.seed}, task L_max={L_max:.3f}m, '
          f'p_tgt={p_tgt}')

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

    # --- path-following stats ---
    all_q, all_bid, all_arc = sample_branch_q0s(branches, args.n_per_branch)
    q_batch = torch.as_tensor(all_q, device=device, dtype=torch.float32)
    print(f'\n  rollout: {q_batch.shape[0]} q0 samples, mode={mode_label}')
    L_abs = rollout_fn(kin, q_batch, track_pts, plane_normal_t)
    L_rel = L_abs / L_max

    print('\n  per-branch L_self / L_max:')
    for bid in range(len(branches)):
        L = L_rel[all_bid == bid]
        if len(L) == 0: continue
        print(f'    br{bid}: n={len(L)}, mean={L.mean():.3f}, '
              f'std={L.std():.3f}, range=[{L.min():.3f}, {L.max():.3f}]')

    # --- save summary PNG ---
    out_dir = Path(args.out_dir)
    out_png = out_dir / f'task_seed{args.seed}_dof{args.task_dof}_summary.png'
    save_summary_plot(out_png, args.seed, args.task_dof, mode_label,
                       branches, Q, assigned, all_bid, all_arc, L_rel)
    print(f'\nsaved: {out_png}')

    if args.no_viewer:
        return

    # --- launch ONE viewer with rollout animation ---
    print(f'\n  preparing viewer (q0 selection: {args.rep_mode})...')
    rep = pick_representative_q0(branches, kin, track_pts, plane_normal_t,
                                  rollout_fn, L_max, mode=args.rep_mode)
    for bid, r in enumerate(rep):
        print(f'    br{bid}: representative L_self={r["L_self_norm"]:.3f}, '
              f'arc_pos={r["arc_pos"]:.2f}')
    q_init = torch.as_tensor(np.array([r['q0'] for r in rep]),
                              device=device, dtype=torch.float32)
    q_traj, fail_infos = record_fn(kin, q_init, track_pts, plane_normal_np)
    q_traj_np = q_traj.detach().cpu().numpy()
    print(f'  recorded {q_traj_np.shape[0]} steps; per-branch death:')
    for bid in range(len(branches)):
        print(f'    br{bid}: step {fail_infos[bid]["fail_step"]} '
              f'({fail_infos[bid]["reason"]})')

    launch_viewer(args.seed, kin, branches, rep, q_traj_np, fail_infos,
                   task_path, plane_normal_np)


if __name__ == '__main__':
    main()

"""ONE viewer animation of 6-DOF rollout per SMM branch.

For each branch at seed's task start, picks a representative q0,
records the 6-DOF strict rollout, and animates all branches in the
ONE viewer following the task path until each one dies. Dead arms
freeze at their last-alive q. The task path is drawn as a line; each
arm tinted a distinct color.

Modes:
  --play-mode sequential (default): one branch plays at a time,
    cycles through. Others are hidden (alpha=0) or parked at start.
  --play-mode simultaneous: all branches play in lockstep, each
    freezes at its own death frame.

Layouts:
  --layout stack (default): all arms share origin (overlap; use
    sequential to disambiguate).
  --layout side: arms separated by --spacing meters; task path is
    duplicated at each arm's offset.

Usage:
    python -m Yuan.RL.intro_motivation.v18_smm_rollout_world --seed 118
    python -m Yuan.RL.intro_motivation.v18_smm_rollout_world --layout side
    python -m Yuan.RL.intro_motivation.v18_smm_rollout_world --play-mode simultaneous
"""
from __future__ import annotations

import argparse
import builtins
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw
from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.RL.batched_rollout import _branch_seed_bank
from Yuan.RL.fr3_with_pen import attach_pen_visual, make_fr3_with_pen
from Yuan.RL.intro_motivation.v18_motivation_core import (
    LINE_L_RANGE, TARGET_PATH_M, add_task_path, as_tensor,
    extend_task_path, path_length, sample_line_task,
)
from Yuan.RL.intro_motivation.v18_smm_enumerate import (
    DEFAULT_H, enumerate_branches, project_and_filter,
)
from Yuan.RL.intro_motivation.v18_smm_rollout_6dof import rollout_lengths_6dof
from Yuan.RL.intro_motivation.v18_smm_rollout_visualize import record_6dof_rollout
from Yuan.RL.v18_data_prep import _build_R_from_normal_direction, _dense_ik_at


PLAYBACK_DT = 0.04
HOLD_AT_END_SEC = 1.5  # how long to freeze on death before restarting


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=118)
    parser.add_argument('--n-ik-seeds', type=int, default=256)
    parser.add_argument('--play-mode', choices=['sequential', 'simultaneous'],
                        default='sequential')
    parser.add_argument('--layout', choices=['stack', 'side'], default='stack')
    parser.add_argument('--spacing', type=float, default=1.4)
    parser.add_argument('--steps-per-tick', type=float, default=1.0,
                        help='rollout steps per viewer frame')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kin = BatchedFR3Kinematics(device=device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # --- task path ---
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
    print(f'seed={args.seed}, task L_max={L_max:.3f}m, p_start={p_tgt}')

    # --- SMM branches ---
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

    # --- representative q0 per branch (median performer) ---
    rep_qs = []
    for bid, b in enumerate(branches):
        traj = b['traj']
        n_samp = min(15, traj.shape[0])
        idxs = np.linspace(0, traj.shape[0] - 1, n_samp).astype(int)
        q_samp = torch.as_tensor(traj[idxs], device=device, dtype=torch.float32)
        L_samp = rollout_lengths_6dof(kin, q_samp, track_pts, plane_normal_t)
        med = float(np.median(L_samp))
        pick = int(np.argmin(np.abs(L_samp - med)))
        rep_qs.append(traj[idxs[pick]])
        print(f'  br{bid}: representative L_self={L_samp[pick] / L_max:.3f}')

    # --- record 6-DOF rollouts ---
    q_init = torch.as_tensor(np.array(rep_qs), device=device, dtype=torch.float32)
    q_traj, fail_info = record_6dof_rollout(kin, q_init, track_pts, plane_normal_np)
    T_total = q_traj.shape[0]
    death_step = [info['fail_step'] for info in fail_info]
    q_traj_np = q_traj.detach().cpu().numpy()
    print(f'  recorded {T_total} steps; per-branch death step:')
    for bid in range(len(branches)):
        print(f'    br{bid}: step {death_step[bid]} '
              f'({death_step[bid] / max(T_total - 1, 1) * 100:.0f}% of max), '
              f'reason: {fail_info[bid]["reason"]}')

    # --- ONE viewer setup ---
    cam_focus = (float(task_path[len(task_path)//4][0]),
                 float(task_path[len(task_path)//4][1]),
                 float(task_path[len(task_path)//4][2]))
    cam_pos = (cam_focus[0] + 1.4, cam_focus[1] - 1.7, cam_focus[2] + 0.9)
    base = ovw.World(cam_pos=cam_pos, cam_lookat_pos=cam_focus,
                     toggle_auto_cam_orbit=False)
    builtins.base = base
    ossop.frame(length_scale=0.18, radius_scale=0.7).attach_to(base.scene)

    cmap = plt.get_cmap('tab10')

    # Layout offsets
    if args.layout == 'side':
        offsets = [np.array([0.0, k * args.spacing, 0.0], dtype=np.float32)
                   for k in range(len(branches))]
        for off in offsets:
            shifted_path = task_path + off[None, :]
            add_task_path(base, shifted_path, plane_normal=plane_normal_np)
    else:
        offsets = [np.zeros(3, dtype=np.float32) for _ in branches]
        add_task_path(base, task_path, plane_normal=plane_normal_np)

    # Arms
    arms = []
    for bid in range(len(branches)):
        rgb = tuple(float(c) for c in cmap(bid % 10)[:3])
        arm, _ = make_fr3_with_pen(pos=offsets[bid])
        arm.attach_to(base.scene)
        arm.rgb = rgb
        arm.alpha = 0.85 if args.layout == 'side' else 0.55
        attach_pen_visual(arm, rgb=rgb, alpha=0.95)
        arm.fk(q_traj_np[0, bid])
        arms.append(arm)
        print(f'  → arm br{bid} attached, color {tuple(round(c, 2) for c in rgb)}')

    # --- animation ---
    if args.play_mode == 'simultaneous':
        state = {'t_float': 0.0, 'hold': 0.0}
        # period = max death + hold; after that, reset
        max_death = max(death_step)

        def animate(_dt, *_args, **_kwargs):
            if state['hold'] > 0.0:
                state['hold'] -= PLAYBACK_DT
                # keep all arms at their last alive q
                for i, arm in enumerate(arms):
                    idx = min(int(state['t_float']), death_step[i])
                    arm.fk(q_traj_np[idx, i])
                return
            t = state['t_float']
            for i, arm in enumerate(arms):
                idx = min(int(t), death_step[i])
                arm.fk(q_traj_np[idx, i])
            state['t_float'] += float(args.steps_per_tick)
            if state['t_float'] >= max_death:
                state['t_float'] = 0.0
                state['hold'] = HOLD_AT_END_SEC
    else:  # sequential
        ACTIVE_ALPHA = 0.95
        HIDDEN_ALPHA = 0.0
        state = {'t_float': 0.0, 'active_bid': 0, 'just_switched': True,
                 'hold': 0.0}
        print(f'\n  sequential playback: cycling through {len(arms)} branches')

        def animate(_dt, *_args, **_kwargs):
            bid = state['active_bid']
            death = death_step[bid]

            if state['just_switched']:
                for i, arm in enumerate(arms):
                    if i != bid:
                        arm.fk(q_traj_np[0, i])
                        arm.alpha = HIDDEN_ALPHA
                    else:
                        arm.alpha = ACTIVE_ALPHA
                print(f'  → playing br{bid} '
                      f'(death step {death}, {fail_info[bid]["reason"]})')
                state['just_switched'] = False

            if state['hold'] > 0.0:
                state['hold'] -= PLAYBACK_DT
                arms[bid].fk(q_traj_np[death, bid])
                return

            t = state['t_float']
            if t >= death:
                # freeze + hold, then switch
                arms[bid].fk(q_traj_np[death, bid])
                state['hold'] = HOLD_AT_END_SEC
                state['active_bid'] = (bid + 1) % len(arms)
                state['t_float'] = 0.0
                state['just_switched'] = True
                return

            idx = int(t)
            arms[bid].fk(q_traj_np[idx, bid])
            state['t_float'] += float(args.steps_per_tick)

    base.schedule_interval(animate, PLAYBACK_DT)
    base.run()


if __name__ == '__main__':
    main()

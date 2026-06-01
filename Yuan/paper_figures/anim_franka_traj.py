"""Animate a (smoothed) captured joint trajectory on a single FR3 arm in
the One viewer -- a quick visual sanity check of what replay_franka_traj.py
will command the real robot to do.

Reads an NPZ written by fig04_joint_trajectories.py or smooth_franka_traj.py
(keys `<mode>_q`, and `<mode>_q_raw` after smoothing, plus the task geometry
`cs_p0`/`cs_line_dir`/`cs_n_target`). One opaque arm sweeps through the
trajectory frames; the dashed reference line and target-normal arrow give
spatial context.

Usage:
    python -m Yuan.paper_figures.anim_franka_traj \\
        --npz Yuan/paper_figures/fig04_traj_task7199_smooth.npz --mode hybrid
    # play the un-smoothed original instead, to compare the motion:
    python -m Yuan.paper_figures.anim_franka_traj --npz ...smooth.npz --raw
"""
from __future__ import annotations

# Conda lib bootstrap (so the One viewer can find shared libraries).
import os, sys
_conda_lib = os.path.join(sys.prefix, 'lib')
if _conda_lib not in os.environ.get('LD_LIBRARY_PATH', ''):
    new_env = dict(os.environ)
    new_env['LD_LIBRARY_PATH'] = _conda_lib + ':' + new_env.get('LD_LIBRARY_PATH', '')
    if __spec__ is not None and __spec__.name != '__main__':
        argv = [sys.executable, '-m', __spec__.name] + sys.argv[1:]
    else:
        argv = [sys.executable] + sys.argv
    os.execvpe(sys.executable, argv, new_env)

import argparse
from pathlib import Path

import numpy as np

from one import ovw, ossop
from one.robots.manipulators.franka.fr3_pen.fr3_with_pen import (
    make_fr3_with_pen, attach_pen_visual,
)

PLAYBACK_DT = 0.04   # 25 fps


def _hex_to_rgb(s):
    s = s.lstrip('#')
    return tuple(int(s[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--npz', default='Yuan/paper_figures/fig04_traj_task7199_smooth.npz')
    p.add_argument('--mode', choices=['classical', 'rl', 'hybrid'], default='hybrid')
    p.add_argument('--raw', action='store_true',
                   help='play <mode>_q_raw (pre-smoothing) instead of the '
                        'smoothed <mode>_q, to compare the motion')
    p.add_argument('--speed', type=float, default=1.0,
                   help='trajectory frames advanced per tick')
    p.add_argument('--n-ghosts', type=int, default=8,
                   help='# of faint static trail ghosts (0 = none)')
    p.add_argument('--ghost-alpha', type=float, default=0.12)
    p.add_argument('--alpha', type=float, default=0.95,
                   help='animator arm alpha')
    p.add_argument('--color', type=str, default=None,
                   help='hex colour for the arm (default: FR3 renderer colour)')
    p.add_argument('--ping-pong', action='store_true',
                   help='sweep forward then backward (default: restart at 0)')
    p.add_argument('--target-distance-m', type=float, default=1.5)
    return p.parse_args()


def _sample_indices(n_frames: int, n: int) -> np.ndarray:
    n = max(1, min(int(n), int(n_frames)))
    return np.linspace(0, n_frames - 1, n).round().astype(int)


def _spawn_arm(base, q, rgb, alpha):
    arm, _ = make_fr3_with_pen(use_pen_tcp=True)
    arm.attach_to(base.scene)
    if rgb is not None:
        attach_pen_visual(arm, rgb=rgb, alpha=alpha)
        arm.rgb = rgb
    else:
        attach_pen_visual(arm, alpha=alpha)
    arm.alpha = alpha
    arm.fk(qs=q.astype(np.float32))
    return arm


def main():
    args = parse_args()
    data = np.load(args.npz, allow_pickle=False)
    task = int(data['task'])

    key = f'{args.mode}_q_raw' if args.raw else f'{args.mode}_q'
    if key not in data:
        raise KeyError(f'{key} not in {args.npz}; available: {list(data.keys())}')
    traj = data[key].astype(np.float32)
    p0 = data['cs_p0'].astype(np.float32)
    d  = data['cs_line_dir'].astype(np.float32)
    nt = data['cs_n_target'].astype(np.float32)
    rgb = _hex_to_rgb(args.color) if args.color else None
    tag = 'raw' if args.raw else 'smoothed'
    print(f'[anim] task={task}  mode={args.mode}  ({tag})  '
          f'frames={len(traj)}  speed={args.speed}')

    base = ovw.World(cam_pos=(2.5, -1.0, 1.5), cam_lookat_pos=(0.5, 0.0, 0.5))
    ossop.frame().attach_to(base.scene)

    line_len = float(args.target_distance_m)
    ossop.dashed_cylinder(spos=p0, epos=p0 + d * line_len,
                          radius=0.003, rgb=(0.4, 0.4, 0.4),
                          alpha=0.85).attach_to(base.scene)
    ossop.arrow(spos=p0, epos=p0 + nt * 0.12, radius=0.005,
                rgb=(0.2, 0.2, 0.2)).attach_to(base.scene)

    # Faint static trail so the swept path stays visible.
    if args.n_ghosts > 0:
        for k in _sample_indices(len(traj), args.n_ghosts):
            _spawn_arm(base, traj[int(k)], rgb, args.ghost_alpha)

    animator = _spawn_arm(base, traj[0], rgb, args.alpha)
    n = len(traj)
    state = {'i': 0.0, 'dir': 1.0}

    if args.ping_pong:
        def animate(_dt, *_a, **_k):
            state['i'] += state['dir'] * args.speed
            if state['i'] >= n - 1:
                state['i'] = n - 1; state['dir'] = -1.0
            elif state['i'] <= 0:
                state['i'] = 0.0; state['dir'] = +1.0
            animator.fk(qs=traj[int(state['i'])])
    else:
        def animate(_dt, *_a, **_k):
            state['i'] = (state['i'] + args.speed) % n
            animator.fk(qs=traj[int(state['i'])])

    base.schedule_interval(animate, PLAYBACK_DT)
    print('[anim] viewer ready. Close the window to exit.')
    base.run()


if __name__ == '__main__':
    main()

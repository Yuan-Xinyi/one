"""Replay the captured joint trajectories on a real Franka via franky.

Loads the NPZ written by `fig04_joint_trajectories.py`
(`fig04_traj_task<T>.npz`), which stores, for each of the three controllers
('classical', 'rl', 'hybrid'), the full 7-joint trajectory `<mode>_q` of
shape (T+1, 7). For each requested mode this script:

  1. moves the arm to the trajectory's start config (= the shared q0 seed),
  2. plays the trajectory as a JointWaypointMotion (Ruckig blends through the
     waypoints under the robot's velocity/accel/jerk limits).

NOTE on timing: franky re-times the motion with Ruckig, so the *shape* in
joint space is reproduced faithfully but the absolute duration will not match
the simulator's fixed-dt rollout. Use `--stride` to thin dense trajectories.

SAFETY: starts at a low relative_dynamics_factor and pauses for a keypress
before every motion. Keep the e-stop in hand. Run with `--dry-run` first to
inspect shapes / limit margins without touching the robot.

Usage (on the robot's control PC, with franky installed):
    python replay_franka_traj.py \
        --npz Yuan/paper_figures/fig04_traj_task7199.npz \
        --host 172.16.0.2 \
        --modes hybrid classical rl \
        --factor 0.05 --stride 1
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--npz', required=True, help='fig04_traj_task<T>.npz path')
    p.add_argument('--host', default='172.16.0.2', help='robot IP / hostname')
    p.add_argument('--modes', nargs='+', default=['hybrid', 'classical', 'rl'],
                   choices=['hybrid', 'classical', 'rl'],
                   help='which trajectories to replay, in order')
    p.add_argument('--factor', type=float, default=0.05,
                   help='relative_dynamics_factor in (0, 1]; keep low for first runs')
    p.add_argument('--stride', type=int, default=1,
                   help='subsample every Nth waypoint (start/end always kept)')
    p.add_argument('--limit-margin', type=float, default=0.01,
                   help='clamp every waypoint into [lo+margin, hi-margin] (rad) '
                        'so sim overshoots never sit on the hard joint limit')
    p.add_argument('--start-factor', type=float, default=0.05,
                   help='relative_dynamics_factor for the move-to-start step')
    p.add_argument('--no-confirm', action='store_true',
                   help='skip the per-motion keypress confirmation')
    p.add_argument('--dry-run', action='store_true',
                   help='load + validate + print, but never command the robot')
    return p.parse_args()


def subsample(q: np.ndarray, stride: int) -> np.ndarray:
    """Keep every `stride`-th row, always including the first and last."""
    if stride <= 1:
        return q
    idx = list(range(0, len(q), stride))
    if idx[-1] != len(q) - 1:
        idx.append(len(q) - 1)
    return q[idx]


def clamp_to_limits(q: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                    margin: float, tag: str) -> np.ndarray:
    """Clamp every waypoint into [lo+margin, hi-margin]; report corrections.

    The on-disk trajectory stays untouched; only the copy commanded to the
    robot is clamped, so sim overshoots beyond the hard joint limits never
    reach the controller.
    """
    safe_lo = lo + margin
    safe_hi = hi - margin
    clamped = np.clip(q, safe_lo[None, :], safe_hi[None, :])
    corr = np.abs(clamped - q)
    moved = corr > 1e-9
    if moved.any():
        n_pts = int(moved.any(axis=1).sum())
        for j in np.where(moved.any(axis=0))[0]:
            print(f'  [clamp] {tag}: joint {j+1} -> {int(moved[:, j].sum())} pts '
                  f'clamped into [{safe_lo[j]:.3f}, {safe_hi[j]:.3f}] '
                  f'(max correction {corr[:, j].max():.4f} rad)')
        print(f'  [clamp] {tag}: {n_pts}/{len(q)} waypoints adjusted')
    margin_after = np.minimum(clamped - lo[None, :], hi[None, :] - clamped).min()
    print(f'  {tag}: within limits after clamp (min margin {margin_after:.3f} rad)')
    return clamped


def confirm(msg: str, skip: bool):
    if skip:
        return
    resp = input(f'{msg}  [Enter = go, q = quit] ')
    if resp.strip().lower() == 'q':
        raise SystemExit('aborted by user')


def main():
    args = parse_args()
    data = np.load(args.npz, allow_pickle=False)
    lo = data['joint_lo'].astype(np.float64)
    hi = data['joint_hi'].astype(np.float64)
    task = int(data['task'])
    print(f'loaded {args.npz}  (task {task})')

    # Pre-load + validate every requested trajectory before touching hardware.
    trajs = {}
    for mode in args.modes:
        key = f'{mode}_q'
        if key not in data:
            raise KeyError(f'{key} not in NPZ; available: {list(data.keys())}')
        q = subsample(data[key].astype(np.float64), args.stride)
        q = clamp_to_limits(q, lo, hi, args.limit_margin, mode)
        trajs[mode] = q
        print(f'{mode:9s}: {len(q)} waypoints (stride {args.stride}), '
              f'start q={np.round(q[0], 3).tolist()}')

    if args.dry_run:
        print('\n[dry-run] no robot commanded.')
        return

    # franky is only imported here so --dry-run works without it installed.
    from franky import Robot, JointMotion, JointWaypointMotion, JointWaypoint

    robot = Robot(args.host)
    robot.relative_dynamics_factor = args.factor
    robot.recover_from_errors()
    cur = np.asarray(robot.current_joint_state.position, dtype=np.float64)
    print(f'\nconnected to {args.host}; current q={np.round(cur, 3).tolist()}')

    for mode in args.modes:
        q = trajs[mode]
        print(f'\n==== mode: {mode}  ({len(q)} waypoints) ====')

        # 1) move to the start config
        confirm(f'move to {mode} start config?', args.no_confirm)
        robot.relative_dynamics_factor = args.start_factor
        robot.recover_from_errors()
        robot.move(JointMotion(q[0].tolist()))

        # 2) play the trajectory through the remaining waypoints
        confirm(f'replay {mode} trajectory?', args.no_confirm)
        robot.relative_dynamics_factor = args.factor
        robot.recover_from_errors()
        waypoints = [JointWaypoint(qi.tolist()) for qi in q[1:]]
        robot.move(JointWaypointMotion(waypoints))
        print(f'  {mode} done.')

    print('\nall requested trajectories replayed.')


if __name__ == '__main__':
    main()

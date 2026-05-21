"""Viewer 4 — Best q0 of a branch walks the straight-line task.

For a selected branch, the animator arm starts at that branch's best
q0 (highest L_self among 50 samples along the SMM arc) and plays the
recorded 6-DOF strict rollout: the EE tracks the moving p_ref along
the task line until the controller dies at fail_step. After dying the
arm holds the death pose for ~1.5 s, then the animation loops.

A faded q_start ghost stays parked at the rollout starting posture so
you can see how far the arm has moved by the time it failed.

Default branch is whichever branch has the highest L_self (most
informative single-branch view).

Usage:
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/viewer_4_best_q0_line.py            # best branch
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/viewer_4_best_q0_line.py --branch 0
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/viewer_4_best_q0_line.py --branch 2 --speed 2
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[3]
for _p in (str(_REPO), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _shared import DEFAULT_SEED, build_or_load  # noqa: E402
from _viewer_common import (  # noqa: E402
    START_GHOST_ALPHA,
    add_task_path, make_animator_arm, make_ghost_arm, make_world,
)


PLAYBACK_DT = 0.04
HOLD_AT_END_SEC = 1.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--branch', type=int, default=2,
                        help='which branch to animate; -1 = pick the best')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='rollout steps advanced per frame')
    parser.add_argument('--no-start-ghost', action='store_true')
    args = parser.parse_args()

    d = build_or_load(seed=args.seed, force=args.force)
    n_branches = int(d['meta']['n_branches'])
    L_self = d['L_self_best']
    bid = int(np.argmax(L_self)) if args.branch < 0 else int(args.branch)
    if not (0 <= bid < n_branches):
        raise SystemExit(f'--branch must be in [0, {n_branches - 1}], '
                          f'got {args.branch}')

    task_path = d['task_path']
    plane_normal = d['plane_normal']
    q_traj_np = d['q_traj_best']               # (T+1, B, 7)
    fail_step = int(d['fail_step'][bid])
    fail_reason = d['meta']['fail_reasons'][bid]

    cmap = plt.get_cmap('tab10')
    rgb = tuple(float(c) for c in cmap(bid % 10)[:3])

    print(f'\nseed={args.seed}, animating br{bid}  '
          f'(best L_self={L_self[bid]:.3f}, die@step={fail_step}, '
          f'reason={fail_reason})')

    base = make_world(task_path)
    add_task_path(base, task_path, plane_normal)

    if not args.no_start_ghost:
        make_ghost_arm(base, q_traj_np[0, bid], rgb, START_GHOST_ALPHA)

    animator, _pen = make_animator_arm(base, q_traj_np[0, bid], rgb)
    state = {'t': 0.0, 'hold': 0.0}

    def animate(_dt, *_args, **_kwargs):
        if state['hold'] > 0.0:
            state['hold'] -= PLAYBACK_DT
            animator.fk(q_traj_np[fail_step, bid])
            if state['hold'] <= 0.0:
                state['t'] = 0.0
            return
        t = state['t']
        if t >= fail_step:
            animator.fk(q_traj_np[fail_step, bid])
            state['hold'] = HOLD_AT_END_SEC
            return
        animator.fk(q_traj_np[int(t), bid])
        state['t'] += args.speed

    base.schedule_interval(animate, PLAYBACK_DT)
    base.run()


if __name__ == '__main__':
    main()

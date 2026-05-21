"""Viewer 3 — Animate one SMM branch along its arc.

For a selected branch, an opaque animator arm sweeps through the
branch's SMM trajectory (q0 → q_end → q0 → ...). Because every q on
the SMM satisfies FK(q) = (p_tgt, R_tgt), the EE stays pinned at the
task start pose — only the elbow / forearm geometry rearranges.

To make the underlying arc visible at a glance, faded transparent
ghosts are sampled along the same arc. They sit motionless while the
opaque arm slides through them.

For closed branches the animator loops at the wrap; for open branches
it ping-pongs (forward, then backward) so you can see both arc tips.

Usage:
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/viewer_3_smm_arc_anim.py --branch 2
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/viewer_3_smm_arc_anim.py --branch 0 --n-ghosts 0
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/viewer_3_smm_arc_anim.py --branch 1 --speed 2
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
    ARC_GHOST_ALPHA,
    add_task_path, make_animator_arm, make_ghost_arm, make_world,
    sample_arc_indices,
)


PLAYBACK_DT = 0.04


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--branch', type=int, default=0,
                        help='which branch to animate')
    parser.add_argument('--n-ghosts', type=int, default=8,
                        help='# of static SMM-arc ghosts (0 = none)')
    parser.add_argument('--alpha', type=float, default=ARC_GHOST_ALPHA)
    parser.add_argument('--speed', type=float, default=1.0,
                        help='SMM-arc steps advanced per frame')
    args = parser.parse_args()

    d = build_or_load(seed=args.seed, force=args.force)
    n_branches = int(d['meta']['n_branches'])
    if not (0 <= args.branch < n_branches):
        raise SystemExit(f'--branch must be in [0, {n_branches - 1}], '
                          f'got {args.branch}')

    task_path = d['task_path']
    plane_normal = d['plane_normal']
    traj = d[f'branch_traj_{args.branch}'].astype(np.float32)
    closed = bool(d['branch_closed'][args.branch])
    arc_rad = float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))

    cmap = plt.get_cmap('tab10')
    rgb = tuple(float(c) for c in cmap(args.branch % 10)[:3])

    print(f'\nseed={args.seed}, br{args.branch}: '
          f'T={traj.shape[0]}, arc={arc_rad:.2f} rad, '
          f'{"closed (looping)" if closed else "open (ping-pong)"}')

    base = make_world(task_path)
    add_task_path(base, task_path, plane_normal)

    if args.n_ghosts > 0:
        for k in sample_arc_indices(traj.shape[0], args.n_ghosts):
            make_ghost_arm(base, traj[int(k)], rgb, args.alpha)

    animator, _pen = make_animator_arm(base, traj[0], rgb)
    T_b = traj.shape[0]
    state = {'i': 0.0, 'dir': 1.0}

    if closed:
        def animate(_dt, *_args, **_kwargs):
            state['i'] = (state['i'] + args.speed) % T_b
            animator.fk(traj[int(state['i'])])
    else:
        def animate(_dt, *_args, **_kwargs):
            state['i'] += state['dir'] * args.speed
            if state['i'] >= T_b - 1:
                state['i'] = T_b - 1
                state['dir'] = -1.0
            elif state['i'] <= 0:
                state['i'] = 0.0
                state['dir'] = +1.0
            animator.fk(traj[int(state['i'])])

    base.schedule_interval(animate, PLAYBACK_DT)
    base.run()


if __name__ == '__main__':
    main()

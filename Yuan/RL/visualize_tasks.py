"""Visualize sampled task conditions (p0, d, n) without any robot/rollout.

Shows several tasks at once in the FR3 base frame so you can see:
  - the spatial distribution of p0
  - the orientations of n (plane normals)
  - the path directions d
  - the TCP target frame R* (= [d, n x d, -n] under the new TCP_z=-n convention)

Usage:
    python -m Yuan.RL.visualize_tasks --n 12 --seed 0 --randomize
    python -m Yuan.RL.visualize_tasks --n 12 --seed 0 --eval-T 80   # in-dist eval
    python -m Yuan.RL.visualize_tasks --n 6  --seed 0 \\
            --tilt-lo 80 --tilt-hi 100        # band test
"""
from __future__ import annotations
import argparse
import builtins

import numpy as np
import torch  # imported first so libstdc++ is loaded

import Yuan.RL.config as cfg
from Yuan.RL.env import FarsightedSeedEnv
from Yuan.RL.rollout import build_target_rotmat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=12,
                        help='number of tasks to draw')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--randomize', action='store_true',
                        help='draw from the training (DR) distribution')
    parser.add_argument('--eval-T', type=int, default=None,
                        help='fixed T for non-randomize mode')
    parser.add_argument('--tilt-lo', type=float, default=None,
                        help='lower tilt bound in DEG (e.g. 80)')
    parser.add_argument('--tilt-hi', type=float, default=None,
                        help='upper tilt bound in DEG (e.g. 100)')
    parser.add_argument('--show-fr3', action='store_true',
                        help='also display one FR3 at home pose for reference')
    parser.add_argument('--show-q', action='store_true',
                        help='draw an FR3 at the joint config that GENERATED '
                             'each task (random_q sampling mode only)')
    args = parser.parse_args()

    n_tilt_range = None
    if args.tilt_lo is not None and args.tilt_hi is not None:
        n_tilt_range = (np.deg2rad(args.tilt_lo), np.deg2rad(args.tilt_hi))

    env = FarsightedSeedEnv(seed=args.seed, randomize=args.randomize,
                            eval_T=args.eval_T,
                            n_tilt_range=n_tilt_range,
                            use_collision=False)

    tasks = []
    for _ in range(args.n):
        env.reset()
        tasks.append(env._cur)

    # report distribution stats
    ns = np.stack([t['c'][6:9] for t in tasks])
    tilts_deg = np.rad2deg(np.arccos(np.clip(ns[:, 2], -1, 1)))
    print(f'sampled {args.n} tasks (randomize={args.randomize})')
    print(f'  n.z   range: [{ns[:, 2].min():+.3f}, {ns[:, 2].max():+.3f}]')
    print(f'  tilt  range: [{tilts_deg.min():.1f}°, {tilts_deg.max():.1f}°]   '
          f'mean={tilts_deg.mean():.1f}°')

    import one.scene.scene_object_primitive as ossop
    import one.viewer.world as ovw

    base = ovw.World(cam_pos=(1.6, 1.4, 1.2),
                     cam_lookat_pos=(0.4, 0.0, 0.4),
                     toggle_auto_cam_orbit=False)
    builtins.base = base

    # world frame at base origin
    ossop.frame(length_scale=0.25, radius_scale=1.0).attach_to(base.scene)

    from one.robots.manipulators.franka.fr3.fr3 import FR3
    if args.show_fr3:
        arm_home = FR3()
        arm_home.fk(arm_home.home_qs)
        arm_home.attach_to(base.scene)

    for i, task in enumerate(tasks):
        c = task['c']
        p0, d, n = c[:3], c[3:6], c[6:9]
        T = task['T']
        v_path = task['v_path']
        path_len = float(T) * cfg.DT * v_path
        R_tgt = build_target_rotmat(d, n)   # TCP frame, z = -n

        # task plane (translucent grey)
        ossop.plane(pos=tuple(p0), normal=tuple(n), size=(0.30, 0.30),
                    rgb=(0.55, 0.55, 0.62), alpha=0.22).attach_to(base.scene)

        # plane normal +n (bright magenta, thick + long so it's visible
        # even with several FR3 instances overlapping in the scene)
        ossop.arrow(spos=tuple(p0),
                    epos=tuple(p0 + 0.28 * n),
                    shaft_radius=0.009, head_radius=0.020, head_length=0.040,
                    rgb=(0.98, 0.10, 0.85), alpha=1.0).attach_to(base.scene)
        # short stub showing -n (where the tool z points; useful contrast)
        ossop.cylinder(spos=tuple(p0),
                       epos=tuple(p0 - 0.08 * n),
                       radius=0.004,
                       rgb=(0.45, 0.05, 0.40), alpha=0.85).attach_to(base.scene)

        # path direction +d (cyan), length = T * dt * v
        ossop.arrow(spos=tuple(p0),
                    epos=tuple(p0 + path_len * d),
                    shaft_radius=0.003, head_radius=0.008, head_length=0.018,
                    rgb=(0.10, 0.80, 0.85), alpha=0.85).attach_to(base.scene)

        # TCP target frame at p0 (R_tgt)
        ossop.frame(pos=p0, rotmat=R_tgt,
                    length_scale=0.10, radius_scale=0.55).attach_to(base.scene)

        # tiny sphere at p0 to mark the point
        ossop.sphere(pos=tuple(p0), radius=0.008,
                     rgb=(0.95, 0.85, 0.10), alpha=0.95).attach_to(base.scene)

        # one FR3 at the joint config that *generated* this task
        if args.show_q and task.get('q_sample') is not None:
            arm_q = FR3()
            full_q = np.zeros(arm_q.qs.shape[0], dtype=np.float32)
            mask = arm_q._chain.active_mask
            full_q[mask] = task['q_sample']
            arm_q.fk(full_q)
            arm_q.attach_to(base.scene)

        # text-style debug print
        q_str = (np.array2string(task.get('q_sample'), precision=2,
                                 suppress_small=True)
                 if task.get('q_sample') is not None else 'None')
        print(f'  task {i:2d}: p0={np.array2string(p0, precision=2)}  '
              f'n.z={n[2]:+.2f}  tilt={tilts_deg[i]:.1f}°  '
              f'T={T}  path_len={path_len:.2f}m  q={q_str}')

    base.run()


if __name__ == '__main__':
    main()

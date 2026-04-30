"""Animate how the 4-D action token (cos phi, sin phi, cos psi, sin psi)
maps to a 7-D FR3 joint configuration via IK projection.

Two display modes:
  --scan phi : fix psi=0, sweep phi from 0 -> 2pi   (elbow swivels around p0)
  --scan psi : fix phi=0, sweep psi from 0 -> 2pi   (TCP rolls about its z axis)
  --scan grid: show 4x3 grid of (phi, psi) combos as static overlapping FR3s

Usage:
  python -m Yuan.RL.visualize_token --seed 14 --eval-T 80 --scan phi
  python -m Yuan.RL.visualize_token --seed 14 --eval-T 80 --scan psi
  python -m Yuan.RL.visualize_token --seed 14 --eval-T 80 --scan grid
"""
from __future__ import annotations
import argparse
import builtins

import numpy as np
import torch  # libstdc++ first

import Yuan.RL.config as cfg
from Yuan.RL.env import FarsightedSeedEnv
from Yuan.RL.batched_rollout import (
    BatchedFR3Kinematics,
    branch_project_multistart,
    build_branch_rotmat_batch,
)
from Yuan.RL.rollout import build_target_rotmat


def project_action(p0: np.ndarray, d: np.ndarray, n: np.ndarray,
                   action: np.ndarray, kin: BatchedFR3Kinematics):
    """Run multistart IK projection: 4D action -> 7D q_0 (or None if fail)."""
    device = kin.device
    a_t  = torch.as_tensor(action[None], dtype=torch.float32, device=device)
    p0_t = torch.as_tensor(p0[None], dtype=torch.float32, device=device)
    d_t  = torch.as_tensor(d[None], dtype=torch.float32, device=device)
    n_t  = torch.as_tensor(n[None], dtype=torch.float32, device=device)
    R_tgt_t = build_branch_rotmat_batch(d_t, n_t, a_t)
    q, ok, _ = branch_project_multistart(kin, p0_t, R_tgt_t, a_t)
    if not bool(ok[0].item()):
        return None
    return q.squeeze(0).cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=14,
                        help='task RNG seed')
    parser.add_argument('--eval-T', type=int, default=80,
                        help='task path length in steps')
    parser.add_argument('--randomize', action='store_true')
    parser.add_argument('--scan', choices=['phi', 'psi', 'grid'], default='phi')
    parser.add_argument('--n-frames', type=int, default=32,
                        help='for phi/psi scan: how many samples per cycle')
    parser.add_argument('--fps', type=float, default=8.0,
                        help='animation rate (frames per second)')
    parser.add_argument('--grid-phi', type=int, default=4)
    parser.add_argument('--grid-psi', type=int, default=3)
    args = parser.parse_args()

    # ---- pick a task ----
    env = FarsightedSeedEnv(seed=args.seed, randomize=args.randomize,
                            eval_T=args.eval_T, use_collision=False)
    env.reset()
    task = env._cur
    c = task['c']
    p0, d, n = c[:3], c[3:6], c[6:9]
    T = task['T']
    v_path = task['v_path']
    R_tgt_base = build_target_rotmat(d, n)            # ψ = 0 case

    print(f'task: p0={p0}  d={d}  n={n}  T={T}')
    tilt_deg = np.rad2deg(np.arccos(np.clip(n[2], -1, 1)))
    print(f'      tilt={tilt_deg:.1f}°  v_path={v_path}')

    # ---- shared geometry: build kin once ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kin = BatchedFR3Kinematics(device=device)

    # ---- scene setup ----
    import one.scene.scene_object_primitive as ossop
    import one.viewer.world as ovw
    from one.robots.manipulators.franka.fr3.fr3 import FR3

    cam_lookat = (p0 + 0.15 * d).tolist()
    base = ovw.World(cam_pos=(1.4, 1.0, 0.9),
                     cam_lookat_pos=cam_lookat,
                     toggle_auto_cam_orbit=False)
    builtins.base = base

    # world frame
    ossop.frame(length_scale=0.20, radius_scale=0.8).attach_to(base.scene)
    # task plane
    ossop.plane(pos=tuple(p0), normal=tuple(n), size=(0.32, 0.32),
                rgb=(0.55, 0.55, 0.62), alpha=0.22).attach_to(base.scene)
    # n arrow (magenta) and -n stub (purple)
    ossop.arrow(spos=tuple(p0), epos=tuple(p0 + 0.20 * n),
                shaft_radius=0.007, head_radius=0.015, head_length=0.030,
                rgb=(0.98, 0.10, 0.85), alpha=1.0).attach_to(base.scene)
    ossop.cylinder(spos=tuple(p0), epos=tuple(p0 - 0.07 * n),
                   radius=0.0035, rgb=(0.45, 0.05, 0.40),
                   alpha=0.85).attach_to(base.scene)
    # path direction (cyan)
    path_len = float(T) * cfg.DT * v_path
    ossop.arrow(spos=tuple(p0), epos=tuple(p0 + path_len * d),
                shaft_radius=0.003, head_radius=0.008, head_length=0.018,
                rgb=(0.10, 0.80, 0.85), alpha=0.85).attach_to(base.scene)
    # p0 marker
    ossop.sphere(pos=tuple(p0), radius=0.010,
                 rgb=(0.95, 0.85, 0.10), alpha=0.95).attach_to(base.scene)
    # base R* (default psi=0) — gold
    ossop.frame(pos=p0, rotmat=R_tgt_base,
                length_scale=0.10, radius_scale=0.55).attach_to(base.scene)

    # shoulder->p0 axis (the elbow swivel rotation axis) as a thin grey rod
    shoulder = np.array([0., 0., 0.333], dtype=np.float32)
    ossop.cylinder(spos=tuple(shoulder), epos=tuple(p0),
                   radius=0.002, rgb=(0.35, 0.35, 0.35),
                   alpha=0.6).attach_to(base.scene)

    if args.scan == 'grid':
        # static: place K_phi x K_psi FR3 robots at their projected q
        K_phi, K_psi = args.grid_phi, args.grid_psi
        phis = np.linspace(0, 2 * np.pi, K_phi, endpoint=False)
        psis = np.linspace(0, 2 * np.pi, K_psi, endpoint=False)
        n_drawn = 0
        for phi in phis:
            for psi in psis:
                action = np.array([np.cos(phi), np.sin(phi),
                                   np.cos(psi), np.sin(psi)], dtype=np.float32)
                q = project_action(p0, d, n, action, kin)
                if q is None:
                    print(f'  phi={np.rad2deg(phi):6.1f}°  psi={np.rad2deg(psi):6.1f}°  IK FAIL')
                    continue
                arm = FR3()
                full_q = np.zeros(arm.qs.shape[0], dtype=np.float32)
                full_q[arm._chain.active_mask] = q
                arm.fk(full_q)
                arm.attach_to(base.scene)
                n_drawn += 1
                print(f'  phi={np.rad2deg(phi):6.1f}°  psi={np.rad2deg(psi):6.1f}°  '
                      f'q={np.array2string(q, precision=2, suppress_small=True)}')
        print(f'\n{n_drawn} / {K_phi * K_psi} branches projected successfully.')
        base.run()
        return

    # ---- animated scan: 1 arm, cycle through tokens ----
    arm = FR3()
    arm.attach_to(base.scene)
    arm.fk(arm.home_qs)

    frames = []
    angles_deg = np.linspace(0, 360, args.n_frames, endpoint=False)
    for ang in angles_deg:
        rad = np.deg2rad(ang)
        if args.scan == 'phi':
            action = np.array([np.cos(rad), np.sin(rad), 1.0, 0.0],
                              dtype=np.float32)
        else:                          # psi scan
            action = np.array([1.0, 0.0, np.cos(rad), np.sin(rad)],
                              dtype=np.float32)
        q = project_action(p0, d, n, action, kin)
        frames.append((ang, action, q))

    n_ok = sum(1 for _, _, q in frames if q is not None)
    print(f'{n_ok}/{len(frames)} frames projected successfully')

    # build a target-frame indicator that we'll update each frame
    # (small rgb axes at p0 showing R_tgt that varies with psi)
    state = {'i': 0, 'tgt_frame': None}
    frame_period = 1.0 / float(args.fps)

    def tick(_dt):
        ang, action, q = frames[state['i']]
        if q is None:
            print(f'  scan {args.scan}={ang:5.1f}°  IK FAIL')
        else:
            full_q = np.zeros(arm.qs.shape[0], dtype=np.float32)
            full_q[arm._chain.active_mask] = q
            arm.fk(full_q)
            tag = f'phi={ang:5.1f}°' if args.scan == 'phi' else f'psi={ang:5.1f}°'
            print(f'  {tag}  q={np.array2string(q, precision=2, suppress_small=True)}')
        state['i'] = (state['i'] + 1) % len(frames)

    base.schedule_interval(tick, interval=frame_period)
    base.run()


if __name__ == '__main__':
    main()

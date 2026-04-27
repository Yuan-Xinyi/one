"""Visualize a trained seed policy with the controller-based rollout.

Usage:
    python -m Yuan.RL.visualize_rollout_world
    python -m Yuan.RL.visualize_rollout_world --ckpt Yuan/RL/checkpoints/ckpt_001000.pt
"""
from __future__ import annotations

import argparse
import glob
import os
import builtins

import numpy as np
import torch

import Yuan.RL.config as cfg
from Yuan.RL.env import FarsightedSeedEnv
from Yuan.RL.policy import GaussianPolicy
from Yuan.RL.rollout import build_target_rotmat, rollout


def latest_ckpt() -> str | None:
    paths = sorted(glob.glob(os.path.join(cfg.CKPT_DIR, 'ckpt_*.pt')))
    return paths[-1] if paths else None


def load_policy(ckpt_path: str, env: FarsightedSeedEnv,
                device: torch.device) -> GaussianPolicy:
    q_mid = torch.as_tensor(env.q_mid, dtype=torch.float32, device=device)
    q_half = torch.as_tensor(env.q_half, dtype=torch.float32, device=device)
    policy = GaussianPolicy(cfg.STATE_DIM, env.ndof, q_mid, q_half).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy.load_state_dict(state['policy'])
    policy.eval()
    return policy


def sample_rollout(ckpt_path: str, seed: int, use_collision: bool,
                   randomize: bool):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = FarsightedSeedEnv(seed=seed, randomize=randomize,
                            use_collision=use_collision)
    policy = load_policy(ckpt_path, env, device)

    state = env.reset()
    task = env._cur
    c = task['c']
    with torch.no_grad():
        st = torch.as_tensor(state[None], dtype=torch.float32, device=device)
        q_seed, _ = policy.act(st, deterministic=True)
    q_seed = q_seed.squeeze(0).cpu().numpy().astype(np.float32)

    info = rollout(env.arm, q_seed, c[:3], c[3:6], c[6:9],
                   mjc=env.mjc,
                   max_steps=task['T'],
                   v_path=task['v_path'],
                   eps_p=task['eps_p'])
    return env, state, task, q_seed, info


def visualize(env, state: np.ndarray, task: dict,
              q_seed: np.ndarray, info: dict,
              fps: float = 30.0):
    import one.scene.scene_object_primitive as ossop
    import one.viewer.world as ovw

    c = task['c']
    p0, d, n = c[:3], c[3:6], c[6:9]
    q_traj = info['q_traj']
    R_tgt = build_target_rotmat(d, n)

    print('state =', np.array2string(state, precision=4, suppress_small=True))
    print('p0 =', np.array2string(p0, precision=4),
          'd =', np.array2string(d, precision=4),
          'n =', np.array2string(n, precision=4))
    print('T =', task['T'], 'v_path =', task['v_path'], 'eps_p =', task['eps_p'])
    print('q_seed =', np.array2string(q_seed, precision=4, suppress_small=True))
    print('qs0 =', None if info['qs0'] is None
          else np.array2string(info['qs0'], precision=4, suppress_small=True))
    print('rollout:', info['reason'], 'length =',
          f"{info['length']}/{task['T']}", 'success =', info['success'])

    base = ovw.World(cam_pos=(1.3, 1.0, 0.9),
                     cam_lookat_pos=(p0 + 0.18 * d).tolist(),
                     toggle_auto_cam_orbit=False)
    # Use the exact same arm instance as rollout. The default env uses bare
    # FR3, whose TCP is the flange frame; fr3_with_hand() adds a gripper TCP
    # offset, which makes the displayed hand tip look off the target line.
    arm = env.arm
    builtins.base = base
    builtins.arm = arm
    arm.attach_to(base.scene)
    arm.toggle_tcp(length_scale=0.15, radius_scale=0.6)
    ossop.frame(length_scale=0.2, radius_scale=0.8).attach_to(base.scene)
    ossop.frame(pos=p0, rotmat=R_tgt, length_scale=0.18,
                radius_scale=0.7).attach_to(base.scene)

    for t in range(task['T'] + 1):
        p = p0 + t * cfg.DT * task['v_path'] * d
        reached = t <= info['length']
        rgb = (0.1, 0.75, 0.25) if reached else (0.85, 0.15, 0.12)
        radius = 0.006 if t % 10 == 0 else 0.0035
        ossop.sphere(pos=tuple(p), radius=radius, rgb=rgb,
                     alpha=0.9).attach_to(base.scene)

    if q_traj:
        arm.fk(q_traj[0])
    else:
        arm.fk(env.arm.home_qs)

    idx = [0]

    def tick(_dt):
        if not q_traj:
            return
        arm.fk(q_traj[idx[0]])
        idx[0] = (idx[0] + 1) % len(q_traj)

    base.schedule_interval(tick, interval=1.0 / fps)
    base.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None,
                        help='Task RNG seed. Omit for a fresh random task.')
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--randomize', action='store_true',
                        help='Also randomize rollout params as in training.')
    parser.add_argument('--collision', action='store_true',
                        help='Enable MJCollider during rollout.')
    args = parser.parse_args()

    ckpt = args.ckpt or latest_ckpt()
    if ckpt is None:
        raise SystemExit(f'No checkpoint found in {cfg.CKPT_DIR}.')
    seed = args.seed
    if seed is None:
        seed = int(np.random.SeedSequence().entropy) % (2 ** 32)
    print('loading', ckpt)
    print('seed', seed)
    env, state, task, q_seed, info = sample_rollout(
        ckpt, seed=seed, use_collision=args.collision,
        randomize=args.randomize)
    visualize(env, state, task, q_seed, info, fps=args.fps)


if __name__ == '__main__':
    main()

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
from Yuan.RL.policy import make_policy
from Yuan.RL.rollout import build_target_rotmat, rollout
from Yuan.RL.controller import DLSController


def latest_ckpt() -> str | None:
    paths = sorted(glob.glob(os.path.join(cfg.CKPT_DIR, 'ckpt_*.pt')))
    return paths[-1] if paths else None


def load_policy(ckpt_path: str, env: FarsightedSeedEnv,
                device: torch.device):
    q_mid = torch.as_tensor(env.action_mid, dtype=torch.float32, device=device)
    q_half = torch.as_tensor(env.action_half, dtype=torch.float32, device=device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy = make_policy(cfg.STATE_DIM, env.action_dim, q_mid, q_half,
                         policy_type=state.get('policy_type', 'gaussian')).to(device)
    policy.load_state_dict(state['policy'])
    policy.eval()
    return policy


def sample_rollout(ckpt_path: str, seed: int, use_collision: bool,
                   randomize: bool, best_components: bool,
                   eval_T: int | None = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = FarsightedSeedEnv(seed=seed, randomize=randomize,
                            use_collision=use_collision,
                            eval_T=eval_T)
    policy = load_policy(ckpt_path, env, device)

    state = env.reset()
    task = env._cur
    c = task['c']
    with torch.no_grad():
        st = torch.as_tensor(state[None], dtype=torch.float32, device=device)
        if best_components and hasattr(policy, 'component_actions'):
            q_cand = policy.component_actions(st).squeeze(0)
        else:
            q_seed, _ = policy.act(st, deterministic=True)
            q_cand = q_seed
    best_seed = None
    best_info = None
    for q_i in q_cand.reshape(-1, env.action_dim):
        q_seed = q_i.cpu().numpy().astype(np.float32)
        info = rollout(env.arm, q_seed, c[:3], c[3:6], c[6:9],
                       mjc=env.mjc,
                       max_steps=task['T'],
                       v_path=task['v_path'],
                       eps_p=task['eps_p'])
        if best_info is None or info['length'] > best_info['length']:
            best_seed = q_seed
            best_info = info
    q_seed = best_seed
    info = best_info
    return env, state, task, q_seed, info


def tcp_trace_stats(arm, q_traj: list[np.ndarray], task: dict,
                    R_tgt: np.ndarray):
    if not q_traj:
        return None
    ctrl = DLSController(arm)
    p0, d = task['c'][:3], task['c'][3:6]
    positions = []
    pos_errs = []
    orient_errs = []
    for t, q_full in enumerate(q_traj):
        q_active = q_full[arm._chain.active_mask].astype(np.float32)
        p_tcp, R_tcp, _ = ctrl.fk_with_jac(q_active)
        p_ref = p0 + t * cfg.DT * task['v_path'] * d
        z_tcp = R_tcp[:, 2]
        z_tgt = R_tgt[:, 2]
        positions.append(p_tcp.astype(np.float32))
        pos_errs.append(float(np.linalg.norm(p_ref - p_tcp)))
        orient_errs.append(float(np.arccos(np.clip(z_tcp @ z_tgt, -1.0, 1.0))))
    return {
        'positions': np.asarray(positions, dtype=np.float32),
        'pos_errs': np.asarray(pos_errs, dtype=np.float32),
        'orient_errs': np.asarray(orient_errs, dtype=np.float32),
    }


def tcp_pose_error(arm, q_full: np.ndarray, p_ref: np.ndarray,
                   R_tgt: np.ndarray) -> tuple[np.ndarray, float, float]:
    ctrl = DLSController(arm)
    q_active = q_full[arm._chain.active_mask].astype(np.float32)
    p_tcp, R_tcp, _ = ctrl.fk_with_jac(q_active)
    pos_err = float(np.linalg.norm(p_ref - p_tcp))
    orient_err = float(np.arccos(np.clip(R_tcp[:, 2] @ R_tgt[:, 2],
                                         -1.0, 1.0)))
    return p_tcp.astype(np.float32), pos_err, orient_err


def build_branch_rotmat(d: np.ndarray, n: np.ndarray,
                        branch_action: np.ndarray) -> np.ndarray:
    R0 = build_target_rotmat(d, n)
    psi_vec = branch_action[2:4].astype(np.float32)
    psi_norm = float(np.linalg.norm(psi_vec))
    if psi_norm < 1e-6:
        psi_vec = np.array([1.0, 0.0], dtype=np.float32)
    else:
        psi_vec = psi_vec / psi_norm
    x = psi_vec[0] * R0[:, 0] + psi_vec[1] * R0[:, 1]
    x = x / (np.linalg.norm(x) + 1e-12)
    z = R0[:, 2]
    y = np.cross(z, x)
    R = np.empty((3, 3), dtype=np.float32)
    R[:, 0] = x
    R[:, 1] = y
    R[:, 2] = z
    return R


def visualize(env, state: np.ndarray, task: dict,
              q_seed: np.ndarray, info: dict,
              fps: float = 30.0):
    import one.scene.scene_object_primitive as ossop
    import one.viewer.world as ovw

    c = task['c']
    p0, d, n = c[:3], c[3:6], c[6:9]
    q_traj = info['q_traj']
    if cfg.ACTION_MODE == 'branch_descriptor':
        R_tgt = build_branch_rotmat(d, n, q_seed)
    else:
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
    seed_tcp = None
    if cfg.ACTION_MODE == "joint_seed":
        seed_tcp, seed_pos_err, seed_orient_err = tcp_pose_error(
            env.arm, q_seed, p0, R_tgt)
        print('q_seed tcp pos =',
              np.array2string(seed_tcp, precision=4, suppress_small=True))
        print('q_seed tcp err: '
              f'pos={seed_pos_err * 1000.0:.3f} mm  '
              f'orient={np.rad2deg(seed_orient_err):.3f} deg')
    else:
        print('branch descriptor =',
              np.array2string(q_seed, precision=4, suppress_small=True))
    tcp_stats = tcp_trace_stats(env.arm, q_traj, task, R_tgt)
    if tcp_stats is not None:
        pos_errs = tcp_stats['pos_errs']
        orient_errs = tcp_stats['orient_errs']
        print('tcp pos err [mm]: '
              f'init={pos_errs[0] * 1000.0:.3f}  '
              f'mean={pos_errs.mean() * 1000.0:.3f}  '
              f'max={pos_errs.max() * 1000.0:.3f}  '
              f'final={pos_errs[-1] * 1000.0:.3f}')
        print('tcp orient err [deg]: '
              f'init={np.rad2deg(orient_errs[0]):.3f}  '
              f'mean={np.rad2deg(orient_errs.mean()):.3f}  '
              f'max={np.rad2deg(orient_errs.max()):.3f}  '
              f'final={np.rad2deg(orient_errs[-1]):.3f}')

    base = ovw.World(cam_pos=(1.3, 1.0, 0.9),
                     cam_lookat_pos=(p0 + 0.18 * d).tolist(),
                     toggle_auto_cam_orbit=False)
    # Use the exact same arm instance as rollout. env.arm is FR3 + Franka
    # hand + pen (TCP at pen tip per cfg.USE_PEN_TCP), so the displayed TCP
    # frame matches the controlled point.
    from Yuan.RL.fr3_with_pen import attach_pen_visual
    arm = env.arm
    builtins.base = base
    builtins.arm = arm
    arm.attach_to(base.scene)
    attach_pen_visual(arm)
    arm.toggle_tcp(length_scale=0.15, radius_scale=0.6)
    ossop.frame(length_scale=0.2, radius_scale=0.8).attach_to(base.scene)
    ossop.frame(pos=p0, rotmat=R_tgt, length_scale=0.18,
                radius_scale=0.7).attach_to(base.scene)
    # task plane at p0 with normal n (translucent grey disc/box)
    ossop.plane(pos=tuple(p0), normal=tuple(n),
                size=(0.4, 0.4),
                rgb=(0.55, 0.55, 0.6), alpha=0.25).attach_to(base.scene)
    # plane-normal arrow from p0 along +n (magenta, ~0.18 m long)
    ossop.arrow(spos=tuple(p0),
                epos=tuple(p0 + 0.18 * n),
                shaft_radius=0.005, head_radius=0.012, head_length=0.025,
                rgb=(0.95, 0.20, 0.85), alpha=0.95).attach_to(base.scene)
    # path direction arrow from p0 along +d (cyan, length scaled by T*v*dt)
    path_len = float(task['T']) * cfg.DT * task['v_path']
    ossop.arrow(spos=tuple(p0),
                epos=tuple(p0 + path_len * d),
                shaft_radius=0.003, head_radius=0.008, head_length=0.018,
                rgb=(0.10, 0.80, 0.85), alpha=0.85).attach_to(base.scene)
    if seed_tcp is not None:
        ossop.sphere(pos=tuple(seed_tcp), radius=0.012,
                     rgb=(1.0, 0.65, 0.05), alpha=0.9).attach_to(base.scene)
        ossop.frame(pos=seed_tcp, rotmat=np.eye(3, dtype=np.float32),
                    length_scale=0.12, radius_scale=0.55).attach_to(base.scene)

    for t in range(task['T'] + 1):
        p = p0 + t * cfg.DT * task['v_path'] * d
        reached = t <= info['length']
        rgb = (0.1, 0.75, 0.25) if reached else (0.85, 0.15, 0.12)
        radius = 0.006 if t % 10 == 0 else 0.0035
        ossop.sphere(pos=tuple(p), radius=radius, rgb=rgb,
                     alpha=0.9).attach_to(base.scene)
    if tcp_stats is not None:
        for t, p_tcp in enumerate(tcp_stats['positions']):
            radius = 0.005 if t % 10 == 0 else 0.003
            ossop.sphere(pos=tuple(p_tcp), radius=radius,
                         rgb=(0.08, 0.35, 0.95),
                         alpha=0.85).attach_to(base.scene)

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
    parser.add_argument('--best-components', action='store_true',
                        help='For mixture policies, visualize the best component mean.')
    parser.add_argument('--eval-T', type=int, default=None,
                        help='Override task path length (in steps). '
                             'Only used in non-randomize mode. '
                             'e.g. --eval-T 80 for 0.4 m, 240 for 1.2 m')
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
        randomize=args.randomize, best_components=args.best_components,
        eval_T=args.eval_T)
    visualize(env, state, task, q_seed, info, fps=args.fps)


if __name__ == '__main__':
    main()

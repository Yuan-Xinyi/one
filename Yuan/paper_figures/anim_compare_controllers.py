"""Side-by-side animated comparison of the three controllers on ONE task.

Spawns three FR3+pen arms in a single One-viewer scene, offset along y, each
replaying the same task under a different controller and stepping forward in
lock-step:

    left   classical  (yellow)
    middle rl         (red)
    right  hybrid      (per-frame: red while RL is active, yellow while
                        Classical is active)

Each arm gets its own copy of the task line (dashed reference + solid black
progress stick = distance actually reached) and the target-normal arrow, so
the three are directly comparable. Shorter trajectories hold their final pose
while the longer ones keep moving; the whole thing loops.

Trajectories are captured on the fly (works for any --task), identical to
fig03/fig04. The same data is what `fig04_traj_task<T>.npz` stores.

Usage:
    python -m Yuan.paper_figures.anim_compare_controllers --task 1546
    python -m Yuan.paper_figures.anim_compare_controllers --task 7199 --fps 30 --spacing 1.2
"""
from __future__ import annotations

# Conda lib bootstrap (so the One viewer finds its shared libraries).
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
import builtins
import dataclasses
from pathlib import Path

import numpy as np
import torch
import yaml

from one import ovw, ossop
from one.robots.manipulators.franka.fr3_pen.fr3_with_pen import (
    make_fr3_with_pen, attach_pen_visual,
)
from Yuan.RL_controller.env.env import (
    NSRLBatchedEnv, EnvConfig, build_task_aligned_basis,
)
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.RL_controller.env.line_distribution import ScriptedLineDistribution
from Yuan.RL_controller.algorithms.ppo import Agent

COLOR_CLASSICAL = (1.0, 190/255, 122/255)       # #FFBE7A
COLOR_RL        = (250/255, 127/255, 111/255)   # #FA7F6F

DEFAULT_EVAL_NPZ = 'Yuan/system_eval/runs/eval_10k_systematic/eval_set_10k.npz'
DEFAULT_DP_NPZ   = 'Yuan/system_eval/runs/eval_10k_systematic/sweeps/cfg_only_w1.5.npz'

MODES = ('classical', 'rl', 'hybrid')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--task', type=int, default=1546, help='eval-set task index')
    p.add_argument('--fps', type=float, default=30.0)
    p.add_argument('--spacing', type=float, default=1.2,
                   help='lateral (y) gap between the three arms, metres')
    p.add_argument('--hold-frames', type=int, default=20,
                   help='frames to hold the final poses before looping')
    p.add_argument('--alpha', type=float, default=0.95)
    p.add_argument('--config', default='Yuan/system_eval/config.yaml')
    p.add_argument('--eval-set', default=DEFAULT_EVAL_NPZ)
    p.add_argument('--dp-seeds', default=DEFAULT_DP_NPZ)
    p.add_argument('--target-distance-m', type=float, default=1.5)
    return p.parse_args()


# ---- capture (n_envs=1), identical logic to fig03/fig04 -----------------
def _build_single_env(env_yaml, device):
    with open(env_yaml) as f:
        cfg = yaml.safe_load(f)
    valid = {f.name for f in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in cfg['env'].items() if k in valid}
    return NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': 1}), line_dist=None,
                          device=device)


def _load_agent(ckpt_dir, env, device):
    with open(Path(ckpt_dir) / 'config.yaml') as f:
        rl_cfg = yaml.safe_load(f)
    agent = Agent(env.obs_dim, env.act_dim,
                  hidden_dim=rl_cfg['ppo']['hidden_dim'],
                  init_log_std=rl_cfg['ppo']['init_log_std']).to(device)
    state = torch.load(Path(ckpt_dir) / 'agent.pt', map_location=device,
                       weights_only=False)
    agent.load_state_dict(state['model'] if isinstance(state, dict)
                          and 'model' in state else state)
    agent.eval()
    return agent


def _classical_action(env, classical):
    q_dot = classical.q_dot_null(env.q, env.line_dir, env.n_target)
    B_basis, _ = build_task_aligned_basis(
        env.kin, env.q, env.line_dir, env.n_target,
        env.kin.q_mid, env.q_half, env.cfg.manip_damping)
    a = (B_basis.transpose(-1, -2) @ q_dot.unsqueeze(-1)).squeeze(-1)
    return (a / env.a_max).clamp(-1.0, 1.0)


def capture(env, classical, agent, q0, p0, d, n, mode, te, tx, device):
    spec = {'q0': torch.as_tensor(q0[None], device=device, dtype=env.kin.dtype),
            'line_dir': torch.as_tensor(d[None], device=device, dtype=env.kin.dtype),
            'n_target': torch.as_tensor(n[None], device=device, dtype=env.kin.dtype)}
    env.line_dist = ScriptedLineDistribution(spec)
    env.reset()
    env.p_start[:] = torch.as_tensor(p0[None], device=device, dtype=env.kin.dtype)
    q_mid, q_half = env.q_mid, env.q_half

    def _max_qn(q):
        return ((q - q_mid).abs() / q_half).max(dim=-1).values

    using_rl = bool((_max_qn(env.q) < te).item())
    q_traj = [env.q[0].cpu().numpy().copy()]
    url = []
    for _ in range(env.max_steps + 1):
        cur = float(_max_qn(env.q).item())
        if mode == 'hybrid':
            if using_rl and cur >= te:
                using_rl = False
            elif (not using_rl) and cur < tx:
                using_rl = True
        else:
            using_rl = (mode == 'rl')
        url.append(using_rl)
        with torch.no_grad():
            a = (agent.actor_mean(env.current_obs()).clamp(-1, 1) if using_rl
                 else _classical_action(env, classical))
        _, _, _, _, info = env.step(a, auto_reset=False)
        q_traj.append(env.q[0].cpu().numpy().copy())
        if bool(info['episode_done'][0].item()):
            break
    url.append(url[-1])
    q_traj = np.stack(q_traj).astype(np.float32)
    p_now, _, _, _ = env.kin.tcp_fk_jac(env.q)
    prog = float(((p_now - env.p_start) * env.line_dir).sum(-1)[0].item())
    return q_traj, np.array(url, bool), prog


# ---- scene + animation --------------------------------------------------
def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    es = np.load(args.eval_set, allow_pickle=False)
    T = int(args.task)
    p0 = es['cs_p0'][T].astype(np.float32)
    d  = es['cs_line_dir'][T].astype(np.float32)
    nt = es['cs_n_target'][T].astype(np.float32)
    dp = np.load(args.dp_seeds)['seeds'][T].astype(np.float32)
    bucket = str(es['bucket'][T])

    env = _build_single_env(cfg['env']['config_yaml'], dev)
    classical = ClassicalNullspaceController(env.kin)
    agent = _load_agent(cfg['rl_controller']['ckpt_dir'], env, dev)
    te = float(cfg['rl_controller']['tau_enter'])
    tx = float(cfg['rl_controller']['tau_exit'])

    traj, url, prog = {}, {}, {}
    for m in MODES:
        traj[m], url[m], prog[m] = capture(env, classical, agent, dp, p0, d, nt,
                                           m, te, tx, dev)
    print(f'[anim] task={T} bucket={bucket}')
    for m in MODES:
        print(f'  {m:9s}: {len(traj[m])-1:>3d} steps  progress={prog[m]:.3f} m')

    # lateral offsets: classical | rl | hybrid, left→right along +y
    offsets = {m: np.array([0.0, (k - 1) * args.spacing, 0.0], np.float32)
               for k, m in enumerate(MODES)}

    centre = p0 + d * (0.5 * args.target_distance_m)
    base = ovw.World(cam_pos=(2.8, 0.0, 1.8),
                     cam_lookat_pos=tuple(centre.tolist()))
    builtins.base = base
    ossop.frame().attach_to(base.scene)

    arms = {}
    for m in MODES:
        off = offsets[m]
        # per-arm task line + target arrow + progress stick (translated)
        ossop.dashed_cylinder(spos=p0 + off, epos=p0 + d * args.target_distance_m + off,
                              radius=0.003, rgb=(0.4, 0.4, 0.4), alpha=0.85
                              ).attach_to(base.scene)
        if prog[m] > 1e-4:
            ossop.cylinder(spos=p0 + off, epos=p0 + d * float(prog[m]) + off,
                           radius=0.003, rgb=(0.0, 0.0, 0.0), alpha=1.0
                           ).attach_to(base.scene)
        ossop.arrow(spos=p0 + off, epos=p0 + nt * 0.12 + off,
                    radius=0.005, rgb=(0.2, 0.2, 0.2)).attach_to(base.scene)
        # legend marker: a colored sphere above each base
        base_rgb = COLOR_RL if m == 'rl' else COLOR_CLASSICAL
        ossop.sphere(pos=tuple((off + np.array([0, 0, 0.9])).tolist()),
                     radius=0.04, rgb=base_rgb, alpha=0.95).attach_to(base.scene)

        arm, _ = make_fr3_with_pen(pos=off.astype(float), use_pen_tcp=True)
        arm.attach_to(base.scene)
        init_rgb = COLOR_RL if m == 'rl' else COLOR_CLASSICAL
        arm.rgb, arm.alpha = init_rgb, args.alpha
        attach_pen_visual(arm, rgb=(0.15, 0.15, 0.15), alpha=args.alpha)
        arm.fk(traj[m][0])
        arms[m] = arm

    print(f'[anim] left→right: classical (yellow) | rl (red) | '
          f'hybrid (red=RL / yellow=Cls). close window to exit.')

    n_max = max(len(traj[m]) for m in MODES)
    frame = [0]
    loop_len = n_max + args.hold_frames

    def tick(_dt):
        f = frame[0]
        for m in MODES:
            i = min(f, len(traj[m]) - 1)
            arms[m].fk(traj[m][i])
            if m == 'hybrid':                       # recolour by active sub-ctrl
                arms[m].rgb = COLOR_RL if url[m][i] else COLOR_CLASSICAL
        frame[0] = (f + 1) % loop_len

    base.schedule_interval(tick, interval=1.0 / args.fps)
    base.run()


if __name__ == '__main__':
    main()

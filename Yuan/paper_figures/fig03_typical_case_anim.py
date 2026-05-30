"""Figure 3: static ghost-overlay of one task's trajectory under a chosen
controller mode. Renders N evenly-spaced arm poses with transparency in the
One viewer.

Modes
-----
  --mode classical : the whole trajectory drawn in yellow (#FFBE7A).
  --mode rl        : the whole trajectory drawn in red    (#FA7F6F).
  --mode hybrid    : each ghost coloured by the sub-controller active at
                     that step -- RL segments red, classical segments yellow.

Spacing is controlled by --n-samples (default 10); the start and end poses
are always included. Transparency by --alpha (default 0.35).

Usage:
    python -m Yuan.paper_figures.fig03_typical_case_anim --mode hybrid
    python -m Yuan.paper_figures.fig03_typical_case_anim --mode rl --task 9580
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
    TERM_ALIVE, TERM_COLLISION, TERM_CONE, TERM_JL, TERM_TRUNCATED, TERM_LATERAL,
)
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.RL_controller.env.line_distribution import ScriptedLineDistribution
from Yuan.RL_controller.algorithms.ppo import Agent


COLOR_CLASSICAL = (1.0, 190/255, 122/255)       # #FFBE7A (yellow)
COLOR_RL        = (250/255, 127/255, 111/255)   # #FA7F6F (red)

DEFAULT_EVAL_NPZ = 'Yuan/system_eval/runs/eval_10k_systematic/eval_set_10k.npz'
DEFAULT_DP_NPZ   = 'Yuan/system_eval/runs/eval_10k_systematic/sweeps/cfg_only_w1.5.npz'

TERM_NAMES = {TERM_ALIVE: 'alive', TERM_COLLISION: 'collision', TERM_CONE: 'cone',
              TERM_JL: 'jl', TERM_TRUNCATED: 'truncated', TERM_LATERAL: 'lateral'}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['classical', 'rl', 'hybrid'], default='hybrid',
                   help='Which controller to visualise.')
    p.add_argument('--task', type=int, default=7199, help='eval-set task index')
    p.add_argument('--n-samples', type=int, default=6,
                   help='Number of ghost arm poses to overlay (>=2). '
                        'Start and end frames are always included.')
    p.add_argument('--alpha', type=float, default=0.8,
                   help='Per-ghost transparency in [0, 1].')
    p.add_argument('--config', default='Yuan/system_eval/config.yaml')
    p.add_argument('--eval-set', default=DEFAULT_EVAL_NPZ)
    p.add_argument('--dp-seeds', default=DEFAULT_DP_NPZ)
    p.add_argument('--target-distance-m', type=float, default=1.5)
    return p.parse_args()


# -----------------------------------------------------------------------
# Trajectory capture (n_envs = 1, single task)
# -----------------------------------------------------------------------
def _build_single_env(env_yaml: str, device: torch.device) -> NSRLBatchedEnv:
    with open(env_yaml, 'r') as f:
        cfg = yaml.safe_load(f)
    valid_keys = {f.name for f in dataclasses.fields(EnvConfig)}
    env_kw = {k: v for k, v in cfg['env'].items() if k in valid_keys}
    env_cfg = EnvConfig(**{**env_kw, 'n_envs': 1})
    return NSRLBatchedEnv(env_cfg, line_dist=None, device=device)


def _load_agent(ckpt_dir: str, env: NSRLBatchedEnv, device: torch.device) -> Agent:
    with open(Path(ckpt_dir) / 'config.yaml', 'r') as f:
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
        env.kin.q_mid, env.q_half, env.cfg.manip_damping,
    )
    cls_act = (B_basis.transpose(-1, -2) @ q_dot.unsqueeze(-1)).squeeze(-1)
    return (cls_act / env.a_max).clamp(-1.0, 1.0)


def _rl_action(env, agent):
    with torch.no_grad():
        return agent.actor_mean(env.current_obs()).clamp(-1.0, 1.0)


def capture_trajectory(env, classical, agent, q0, p0, d, n_target,
                       mode, tau_enter, tau_exit, device):
    """Roll one task under ``mode`` controller; capture per-step q (T+1, 7)
    and a per-step using_rl flag (T+1,)."""
    spec = {'q0': torch.as_tensor(q0[None], device=device, dtype=env.kin.dtype),
            'line_dir': torch.as_tensor(d[None], device=device, dtype=env.kin.dtype),
            'n_target': torch.as_tensor(n_target[None], device=device, dtype=env.kin.dtype)}
    env.line_dist = ScriptedLineDistribution(spec)
    env.reset()
    env.p_start[:] = torch.as_tensor(p0[None], device=device, dtype=env.kin.dtype)

    q_mid = env.q_mid; q_half = env.q_half

    def _max_abs_qn(q):
        return ((q - q_mid).abs() / q_half).max(dim=-1).values

    q_traj = [env.q[0].detach().cpu().numpy().copy()]
    using_rl_per_step = []
    using_rl = bool((_max_abs_qn(env.q) < tau_enter).item())
    term_reason = TERM_ALIVE

    for _ in range(env.max_steps + 1):
        cur = float(_max_abs_qn(env.q).item())
        if mode == 'hybrid':
            if using_rl and cur >= tau_enter:
                using_rl = False
            elif (not using_rl) and cur < tau_exit:
                using_rl = True
        elif mode == 'rl':
            using_rl = True
        else:  # classical
            using_rl = False
        using_rl_per_step.append(using_rl)

        a = _rl_action(env, agent) if using_rl else _classical_action(env, classical)
        _, _, _, _, info = env.step(a, auto_reset=False)
        q_traj.append(env.q[0].detach().cpu().numpy().copy())
        if bool(info['episode_done'][0].item()):
            term_reason = int(info['term_reason'][0].item())
            break

    q_traj = np.stack(q_traj, axis=0).astype(np.float32)
    # Per-frame using_rl aligned with q_traj (the last frame holds the
    # using_rl flag that was active when the terminating step ran).
    using_rl_per_step.append(using_rl_per_step[-1])
    using_rl_per_step = np.array(using_rl_per_step, dtype=bool)

    p_now, _, _, _ = env.kin.tcp_fk_jac(env.q)
    prog_m = float(((p_now - env.p_start) * env.line_dir).sum(-1)[0].item())
    return q_traj, using_rl_per_step, prog_m, term_reason


# -----------------------------------------------------------------------
# Static ghost-overlay rendering
# -----------------------------------------------------------------------
def _pick_sample_indices(n_frames: int, n_samples: int) -> np.ndarray:
    """Evenly-spaced indices in [0, n_frames-1] of length min(n_samples, n_frames),
    always including 0 and n_frames-1."""
    if n_samples >= n_frames:
        return np.arange(n_frames)
    return np.linspace(0, n_frames - 1, n_samples).round().astype(int)


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

    print(f'[fig03] task={T}  mode={args.mode}  bucket={bucket}')

    env = _build_single_env(cfg['env']['config_yaml'], dev)
    classical = ClassicalNullspaceController(env.kin)
    agent = _load_agent(cfg['rl_controller']['ckpt_dir'], env, dev)
    te = float(cfg['rl_controller']['tau_enter'])
    tx = float(cfg['rl_controller']['tau_exit'])

    q_traj, using_rl_traj, prog, term = capture_trajectory(
        env, classical, agent, dp, p0, d, nt, args.mode, te, tx, dev)
    n_frames = len(q_traj)
    print(f'[fig03] T_steps={n_frames-1}  progress={prog:.3f} m  '
          f'term={TERM_NAMES.get(term, term)}')

    idxs = _pick_sample_indices(n_frames, max(2, args.n_samples))
    print(f'[fig03] rendering {len(idxs)} ghost poses at steps: {idxs.tolist()}')

    # ---- Scene ---------------------------------------------------------
    base = ovw.World(cam_pos=(2.5, -1.0, 1.5),
                     cam_lookat_pos=(0.5, 0.0, 0.5))
    ossop.frame().attach_to(base.scene)

    # Task elements: dashed reference line + target-normal arrow.
    line_len = float(args.target_distance_m)
    ossop.dashed_cylinder(
        spos=p0, epos=p0 + d * line_len,
        radius=0.003, rgb=(0.4, 0.4, 0.4), alpha=0.85,
    ).attach_to(base.scene)
    # Solid black stick over the dashed reference, length = actually achieved
    # progress under this controller.
    if prog > 1e-4:
        ossop.cylinder(
            spos=p0, epos=p0 + d * float(prog),
            radius=0.003, rgb=(0.0, 0.0, 0.0), alpha=1.0,
        ).attach_to(base.scene)
    ossop.arrow(spos=p0, epos=p0 + nt * 0.12,
                radius=0.005, rgb=(0.2, 0.2, 0.2)).attach_to(base.scene)

    # Per-ghost colour
    if args.mode == 'classical':
        per_idx_color = lambda i: COLOR_CLASSICAL
    elif args.mode == 'rl':
        per_idx_color = lambda i: COLOR_RL
    else:  # hybrid: colour by sub-controller active at this step
        per_idx_color = lambda i: COLOR_RL if using_rl_traj[i] else COLOR_CLASSICAL

    for i in idxs:
        color = per_idx_color(int(i))
        arm, _ = make_fr3_with_pen(use_pen_tcp=True)
        arm.attach_to(base.scene)
        attach_pen_visual(arm, rgb=color, alpha=args.alpha)
        arm.rgb = color
        arm.alpha = args.alpha
        arm.fk(qs=q_traj[int(i)])
        # TCP z-axis arrow at this ghost's TCP, same style as the step-0 arrow.
        tcp_tf = np.asarray(arm.gl_tcp_tf)
        tcp_pos = tcp_tf[:3, 3].astype(np.float32)
        tcp_z   = tcp_tf[:3, 2].astype(np.float32)
        ossop.arrow(spos=tcp_pos, epos=tcp_pos + tcp_z * 0.12,
                    radius=0.005, rgb=(0.2, 0.2, 0.2)).attach_to(base.scene)

    print(f'[fig03] viewer ready. Close the window to exit.')
    base.run()


if __name__ == '__main__':
    main()

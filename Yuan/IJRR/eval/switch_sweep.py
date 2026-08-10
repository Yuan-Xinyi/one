"""Are the hybrid switching thresholds sensitive to the arm's morphology?

The hybrid switches on rho(q) = max_i |q_i - q_mid,i| / (range_i / 2), which
is normalized per joint, so one pair of thresholds denotes the same relative
proximity to a limit on any arm. Whether the *optimum* of that pair moves
across arms is the empirical question this sweeps: the single-threshold grid
and the two hysteresis pairs of the paper's threshold ablation, on the same
2048 tasks per arm, reporting the paired hybrid/classical ratio and the mean
number of switches per stroke.

Usage:
    python -m Yuan.IJRR.eval.switch_sweep --robot fr3 --n-tasks 1024
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.classical_nullspace import (
    ClassicalNullspaceController, cn_action_fn)
from Yuan.IJRR.env.line_distribution import (
    LineDistribution, ScriptedLineDistribution)
from Yuan.IJRR.eval.eval_curve import _agent

REPO = Path(__file__).resolve().parents[3]

CONFIGS = {
    'fr3': ('Yuan/IJRR/stage2_traj/config_vertex_line.yaml',
            'Yuan/IJRR/runs/rl_vertex_line_30M'),
    'xarm7': ('Yuan/IJRR/stage2_traj/config_vertex_line_xarm7.yaml',
              'Yuan/IJRR/runs/rl_vertex_line_xarm7_30M'),
    'cobotta': ('Yuan/IJRR/stage2_traj/config_vertex_line_cobotta.yaml',
                'Yuan/IJRR/runs/rl_vertex_line_cobotta_30M'),
}

SETTINGS = [(t, t) for t in (0.85, 0.90, 0.95, 0.97, 0.98, 0.99)] \
         + [(0.98, 0.94), (0.99, 0.93)]


@torch.no_grad()
def rollout_hybrid(env, agent, classical, te, tx):
    env.reset()
    p0, u = env.p_start.clone(), env.line_dir.clone()
    n = env.n_envs
    fn_cl = cn_action_fn(classical)
    using = None
    switches = torch.zeros(n, device=env.device)
    done = torch.zeros(n, dtype=torch.bool, device=env.device)
    for _ in range(env.max_steps):
        qn = ((env.q - env.q_mid) / env.q_half).abs().max(dim=-1).values
        if using is None:
            using = qn < te
        else:
            new = torch.where(using, qn < te, qn < tx)
            switches += ((new != using) & ~done).float()
            using = new
        a = torch.where(using.unsqueeze(-1),
                        agent.actor_mean(env.current_obs()).clamp(-1, 1),
                        fn_cl(env))
        env.step(a, auto_reset=False)
        done = env.done_persistent.clone()
        if bool(done.all()):
            break
    p, _, _, _ = env.kin.tcp_fk_jac(env.q)
    prog = ((p - p0) * u).sum(-1)
    return prog.cpu().numpy().copy(), switches.cpu().numpy().copy()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--robot', required=True, choices=list(CONFIGS))
    ap.add_argument('--n-tasks', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=4242)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()

    cfg_path, ckpt = CONFIGS[a.robot]
    y = yaml.safe_load(open(REPO / cfg_path))
    keys = {f.name for f in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in y['env'].items() if k in keys}
    N = a.n_tasks
    dev = torch.device(a.device)

    env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': N}), None, dev)
    pool = LineDistribution.load_or_build(
        kin=env.kin, collision=env.collision, n_pool=max(3 * N, 20000),
        n_target_noise_deg=5.0, seed=a.seed, env_cfg=env.cfg,
        feasibility_threshold_m=0.1, verbose=False)
    idx = torch.nonzero(pool.valid_mask, as_tuple=False).squeeze(-1)[:N]
    spec = {'q0': pool.q_pool[idx], 'line_dir': pool.line_dir_pool[idx],
            'n_target': pool.n_target_pool[idx]}
    fresh = lambda: setattr(env, 'line_dist', ScriptedLineDistribution(
        {k: v.clone() for k, v in spec.items()}))

    agent = _agent(REPO / ckpt, env.obs_dim, dev, act_dim=env.act_dim)
    cl = ClassicalNullspaceController(env.kin)

    fresh()
    env.reset()
    fn_cl = cn_action_fn(cl)
    done = torch.zeros(N, dtype=torch.bool, device=dev)
    for _ in range(env.max_steps):
        env.step(fn_cl(env), auto_reset=False)
        if bool(env.done_persistent.all()):
            break
    p, _, _, _ = env.kin.tcp_fk_jac(env.q)
    base = ((p - env.p_start) * env.line_dir).sum(-1).cpu().numpy().copy()
    ok = base > 1e-6

    print(f'{a.robot}: {N} tasks, {int(ok.sum())} with nonzero classical '
          f'baseline\n')
    print(f'{"(te, tx)":>14s}{"hybrid/classical":>18s}{"switches/stroke":>17s}')
    rows = {}
    for te, tx in SETTINGS:
        fresh()
        prog, sw = rollout_hybrid(env, agent, cl, te, tx)
        r = float((prog[ok] / base[ok]).mean())
        s = float(sw[ok].mean())
        rows[f'{te:.2f},{tx:.2f}'] = {'ratio': r, 'switches': s}
        print(f'({te:.2f}, {tx:.2f})'.rjust(14)
              + f'{r:>18.4f}{s:>17.2f}')

    dst = REPO / ckpt / 'switch_sweep.json'
    dst.write_text(json.dumps({'robot': a.robot, 'n_tasks': int(N),
                               'rows': rows}, indent=1))
    print(f'\nwrote {dst}')


if __name__ == '__main__':
    main()

"""Does the vertex command set stay lossless as the control rate falls?

The relaxation theorem buys the vertex parameterisation nothing for free: it
says vertex-valued commands can approximate box-valued ones, at the price of
switching fast enough. The learned resolution already switches at 16.6 Hz
against a 20 Hz control rate, so the budget is nearly spent, which predicts that
lengthening the control period should start to cost the vertex-valued command
more than the continuous one -- an intermediate magnitude being exactly what a
single command can offer in place of a switch it is no longer allowed to make.

The decisive comparison is within each control rate, between the same policy
executed as-is and executed with its command projected onto the nearest vertex.
Both suffer equally from being run away from the rate they were trained at, so
the difference between them isolates the command set.

The integration step is held at 25 ms, which the convergence check showed to be
converged, so refining it further changes nothing.

Usage:
    python -m Yuan.IJRR.eval.control_rate_sweep --n-tasks 1024
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import torch
import yaml

from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.classical_nullspace import (
    ClassicalNullspaceController, cn_action_fn)
from Yuan.IJRR.env.line_distribution import (
    LineDistribution, ScriptedLineDistribution)
from Yuan.IJRR.stage2_traj.ppo import Agent

REPO = Path(__file__).resolve().parents[3]
SIM_DT = 0.025          # converged integration step
CONT = 'Yuan/RL_controller/runs/rl_smmstart_30M'
VTX = 'Yuan/IJRR/runs/rl_vertex_line_30M'


def load(path, obs_dim, dev):
    ck = torch.load(REPO / path / 'agent.pt', map_location=dev,
                    weights_only=False)
    sd = ck['agent'] if isinstance(ck, dict) and 'agent' in ck else ck
    if any(k.startswith('_logits_head') for k in sd):
        from Yuan.IJRR.stage2_traj.vertex_agent import VertexAgent
        m = VertexAgent(obs_dim=obs_dim, act_dim=4, hidden_dim=512).to(dev)
    else:
        m = Agent(obs_dim=obs_dim, act_dim=4, hidden_dim=512).to(dev)
    m.load_state_dict(sd)
    return m.eval()


@torch.no_grad()
def rollout_zoh(env, action_fn, substeps, max_blocks):
    env.reset()
    p0, u = env.p_start.clone(), env.line_dir.clone()
    prog = torch.zeros(env.n_envs, device=env.device)
    done = torch.zeros(env.n_envs, dtype=torch.bool, device=env.device)
    for _ in range(max_blocks):
        a = action_fn(env)
        for _ in range(substeps):
            env.step(a, auto_reset=False)
        p, _, _, _ = env.kin.tcp_fk_jac(env.q)
        prog = torch.where(done, prog, ((p - p0) * u).sum(-1))
        done = env.done_persistent.clone()
        if bool(done.all()):
            break
    return prog.cpu().numpy().copy()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-tasks', type=int, default=1024)
    ap.add_argument('--periods', default='25,50,100,200',
                    help='control period in ms; must be multiples of 25')
    ap.add_argument('--seed', type=int, default=4242)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()

    dev = torch.device(a.device)
    y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/config.yaml'))
    keys = {f.name for f in dataclasses.fields(EnvConfig)}
    base = {k: v for k, v in y['env'].items() if k in keys}
    N = a.n_tasks
    periods = [int(p) for p in a.periods.split(',')]

    kw = dict(base); kw['dt'] = SIM_DT
    env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': N}), None, dev)
    pool = LineDistribution.load_or_build(
        kin=env.kin, collision=env.collision, n_pool=20000,
        n_target_noise_deg=5.0, seed=a.seed, env_cfg=env.cfg,
        feasibility_threshold_m=0.1, verbose=False)
    idx = torch.nonzero(pool.valid_mask, as_tuple=False).squeeze(-1)[:N]
    spec = {'q0': pool.q_pool[idx], 'line_dir': pool.line_dir_pool[idx],
            'n_target': pool.n_target_pool[idx]}
    fresh = lambda: setattr(env, 'line_dist', ScriptedLineDistribution(
        {k: v.clone() for k, v in spec.items()}))

    cont = load(CONT, env.obs_dim, dev)
    vtx = load(VTX, env.obs_dim, dev) if (REPO / VTX / 'agent.pt').exists() else None
    cl = ClassicalNullspaceController(env.kin)

    arms = {
        'classical': cn_action_fn(cl),
        'continuous': lambda e: cont.actor_mean(e.current_obs()).clamp(-1, 1),
        'projected to vertex': lambda e: torch.sign(
            cont.actor_mean(e.current_obs())).clamp(-1, 1),
    }
    if vtx is not None:
        arms['trained on vertices'] = lambda e: vtx.actor_mean(e.current_obs())

    res = {}
    for ms in periods:
        sub = int(round(ms / 1000 / SIM_DT))
        blocks = int(base['max_steps'] * base['dt'] / (ms / 1000)) + 1
        for nm, fn in arms.items():
            fresh()
            res[(ms, nm)] = rollout_zoh(env, fn, sub, blocks)
        print(f'  control period {ms} ms ({sub} substeps of {SIM_DT*1000:.0f} ms) done')

    rng = np.random.default_rng(0)
    print(f'\n{"period":>8s}{"rate":>7s}' +
          ''.join(f'{n:>21s}' for n in list(arms)[1:]))
    print(f'{"(ms)":>8s}{"(Hz)":>7s}' +
          ''.join(f'{"ratio to classical":>21s}' for _ in list(arms)[1:]))
    for ms in periods:
        b = res[(ms, 'classical')]
        ok = b > 1e-6
        cells = ''.join(f'{(res[(ms, n)][ok] / b[ok]).mean():>21.4f}'
                        for n in list(arms)[1:])
        print(f'{ms:>8d}{1000/ms:>7.1f}{cells}')

    print(f'\nprojected minus continuous, same policy, paired')
    for ms in periods:
        b = res[(ms, 'classical')]
        ok = b > 1e-6
        d = (res[(ms, 'projected to vertex')][ok]
             - res[(ms, 'continuous')][ok]) / b[ok]
        i = rng.integers(0, len(d), size=(20000, len(d)))
        m = d[i].mean(1)
        print(f'  {ms:>4d} ms{d.mean():>+10.4f}  '
              f'[{np.percentile(m, 2.5):+.4f}, {np.percentile(m, 97.5):+.4f}]  '
              f'win {(d > 0).mean():.1%}')


if __name__ == '__main__':
    main()

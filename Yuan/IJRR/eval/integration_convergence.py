"""Is the 50 ms evaluation numerically converged, and does it change the ranking?

A saved open-loop command sequence that reaches 1.76x the classical law at the
default 50 ms integration step reaches only 1.17x once the same commands are
integrated at 25 ms, and 12.5 ms and 6.25 ms agree with 25 ms. The default
evaluation is therefore not converged with respect to this substepping scheme,
and every quantity measured under it has to be re-checked.

The question that matters for the main results is narrower than "are the
absolute numbers right": it is whether the ordering and the relative gaps
between the three resolutions survive. This script answers exactly that. The
control rate is held at 50 ms in every condition — each controller is queried
once per control block and its command is held across the substeps — so the
only thing that varies is the fidelity of the integration, not the bandwidth of
the controller.

Usage:
    python -m Yuan.IJRR.eval.integration_convergence --n-tasks 1024
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import torch
import yaml

from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig, TERM_NAMES
from Yuan.IJRR.env.classical_nullspace import (
    ClassicalNullspaceController, cn_action_fn)
from Yuan.IJRR.env.line_distribution import (
    LineDistribution, ScriptedLineDistribution)
from Yuan.IJRR.stage2_traj.ppo import Agent

REPO = Path(__file__).resolve().parents[3]
CKPT = 'Yuan/RL_controller/runs/rl_smmstart_30M'
TE, TX = 0.98, 0.94


@torch.no_grad()
def rollout_zoh(env, action_fn, substeps, max_blocks):
    """One episode, one command per control block held across `substeps`."""
    env.reset()
    p0 = env.p_start.clone()
    u = env.line_dir.clone()
    n = env.n_envs
    prog = torch.zeros(n, device=env.device)
    done = torch.zeros(n, dtype=torch.bool, device=env.device)
    term = torch.full((n,), -1, dtype=torch.long, device=env.device)
    for _ in range(max_blocks):
        a = action_fn(env)
        for _ in range(substeps):
            _, _, _, _, info = env.step(a, auto_reset=False)
            new = info['episode_done']
            if new.any():
                term[new] = info['term_reason'][new]
        p, _, _, _ = env.kin.tcp_fk_jac(env.q)
        prog = torch.where(done, prog, ((p - p0) * u).sum(-1))
        done = env.done_persistent.clone()
        if bool(done.all()):
            break
    return prog, term


def hybrid_fn(agent, classical, te, tx):
    st = {}

    @torch.no_grad()
    def fn(env):
        qn = ((env.q - env.q_mid) / env.q_half).abs().max(dim=-1).values
        u = st.get('u')
        u = (qn < te) if (u is None or u.shape[0] != qn.shape[0]) \
            else torch.where(u, qn < te, qn < tx)
        st['u'] = u
        return torch.where(u.unsqueeze(-1),
                           agent.actor_mean(env.current_obs()).clamp(-1, 1),
                           cn_action_fn(classical)(env))
    return fn


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-tasks', type=int, default=1024)
    ap.add_argument('--substeps', default='1,2,4')
    ap.add_argument('--seed', type=int, default=4242)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()

    dev = torch.device(a.device)
    y = yaml.safe_load(open(REPO / 'Yuan/IJRR/stage2_traj/config.yaml'))
    keys = {f.name for f in dataclasses.fields(EnvConfig)}
    base = {k: v for k, v in y['env'].items() if k in keys}
    N = a.n_tasks
    subs = [int(s) for s in a.substeps.split(',')]

    # one task set, drawn once, shared by every condition
    e0 = NSRLBatchedEnv(EnvConfig(**{**base, 'n_envs': N}), None, dev)
    pool = LineDistribution.load_or_build(
        kin=e0.kin, collision=e0.collision, n_pool=20000,
        n_target_noise_deg=5.0, seed=a.seed, env_cfg=e0.cfg,
        feasibility_threshold_m=0.1, verbose=False)
    idx = torch.nonzero(pool.valid_mask, as_tuple=False).squeeze(-1)[:N]
    spec = {'q0': pool.q_pool[idx], 'line_dir': pool.line_dir_pool[idx],
            'n_target': pool.n_target_pool[idx]}

    ck = torch.load(REPO / CKPT / 'agent.pt', map_location=dev,
                    weights_only=False)
    sd = ck['agent'] if isinstance(ck, dict) and 'agent' in ck else ck

    res, terms = {}, {}
    for m in subs:
        kw = dict(base)
        kw['dt'] = base['dt'] / m
        kw['max_steps'] = int(base['max_steps'] * m)
        env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': N}), None, dev)
        ag = Agent(obs_dim=env.obs_dim, act_dim=4, hidden_dim=512).to(dev)
        ag.load_state_dict(sd); ag.eval()
        cl = ClassicalNullspaceController(env.kin)
        arms = {
            'classical': cn_action_fn(cl),
            'rl': lambda e: ag.actor_mean(e.current_obs()).clamp(-1, 1),
            'hybrid': hybrid_fn(ag, cl, TE, TX),
        }
        for nm, fn in arms.items():
            env.line_dist = ScriptedLineDistribution(
                {k: v.clone() for k, v in spec.items()})
            p, t = rollout_zoh(env, fn, m, base['max_steps'])
            res[(m, nm)] = p.cpu().numpy().copy()
            terms[(m, nm)] = t.cpu().numpy().copy()
        print(f'  substeps={m} (dt={kw["dt"]*1000:.2f} ms) done')

    print(f'\n{"substeps":>9s}{"dt (ms)":>9s}{"classical (m)":>15s}'
          f'{"RL / cls":>10s}{"Hybrid / cls":>14s}{"Hybrid / RL":>13s}')
    for m in subs:
        b = res[(m, 'classical')]
        ok = b > 1e-6
        r_rl = (res[(m, 'rl')][ok] / b[ok]).mean()
        r_hy = (res[(m, 'hybrid')][ok] / b[ok]).mean()
        r_hr = (res[(m, 'hybrid')][ok] / np.maximum(res[(m, 'rl')][ok], 1e-6)).mean()
        print(f'{m:>9d}{base["dt"]*1000/m:>9.2f}{b.mean():>15.4f}'
              f'{r_rl:>10.4f}{r_hy:>14.4f}{r_hr:>13.4f}')

    print(f'\nabsolute mean path length (m), same tasks')
    print(f'{"substeps":>9s}' + ''.join(f'{n:>12s}' for n in
                                        ('classical', 'rl', 'hybrid')))
    for m in subs:
        print(f'{m:>9d}' + ''.join(f'{res[(m, n)].mean():>12.4f}'
                                   for n in ('classical', 'rl', 'hybrid')))

    print(f'\ntermination reasons (%), hybrid arm')
    for m in subs:
        t = terms[(m, 'hybrid')]
        print(f'  substeps={m}  ' + '  '.join(
            f'{TERM_NAMES[c]}={100*(t == c).mean():.1f}'
            for c in TERM_NAMES if (t == c).any()))


if __name__ == '__main__':
    main()

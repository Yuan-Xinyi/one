"""Per-control-period decision time of the tab:mainresult baselines (zero
null-space, classical gradient, continuous PPO, PPO+classical hybrid) on an
idle GPU, same protocol as probes/horizon_latency.py: median over 20
periods, batch 1 (single-task latency) and batch 2500 (amortized per task).
Only the action-selection call is timed; the null-space projection inside
the environment step is shared by every method and excluded.

Writes runs/paper_fill/horizon/latency_base_{robot}.json.
usage: python -m Yuan.IJRR.probes.baseline_latency [--robot fr3]
"""
import argparse
import dataclasses
import json
import time
from pathlib import Path

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot  # noqa: F401,E402
import numpy as np
import torch
import yaml

import Yuan.IJRR.eval.horizon_ladder as hl
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.classical_nullspace import (ClassicalNullspaceController,
                                               cn_action_fn)
from Yuan.IJRR.eval.eval_curve import _agent
from Yuan.IJRR.eval.mpc_mppi_baselines import load_tasks, scripted, SUB, OUT, MAIN
from Yuan.IJRR.stage2_traj.ppo import Agent

WT = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
CONT = {'fr3': 'rl_cont_sqent_30M', 'xarm7': 'rl_cont_sqent_xarm7_30M',
        'cobotta': 'rl_cont_sqent_cobotta_30M'}


def ladder_env(robot, batch):
    y = yaml.safe_load(open(Path('/home/lqin/one') / hl.ROBOTS[robot][0]))
    keys = {f.name for f in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in y['env'].items() if k in keys}
    kw['dt'] = kw['dt'] / SUB
    kw['max_steps'] = int(y['env']['max_steps'] * SUB)
    if robot == 'cobotta':
        kw['v'], kw['max_steps'] = 0.05, 2000 * SUB
    return NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': batch}), None, dev)


def arms(env, robot):
    classical = ClassicalNullspaceController(env.kin)
    fcl = cn_action_fn(classical)
    ag = Agent(env.obs_dim, env.act_dim).to(dev)
    ag.load_state_dict(torch.load(WT / 'Yuan/IJRR/runs' / CONT[robot]
                                  / 'agent.pt', map_location=dev))
    ag.eval()
    vag = _agent(WT / hl.ROBOTS[robot][1], env.obs_dim, dev,
                 act_dim=env.act_dim)
    # obs is computed once per period outside the timed call (the policy
    # rows of horizon_latency.py are timed the same way); the classical law
    # computes its own quantities from env.q inside the call.
    state = {}

    def hybrid(e, dn, obs):
        qn = ((e.q - e.q_mid) / e.q_half).abs().max(dim=-1).values
        using_rl = state.get('using_rl')
        if using_rl is None or using_rl.shape[0] != qn.shape[0]:
            using_rl = qn < 0.98
        stay = torch.where(using_rl, qn < 0.98, qn < 0.94)
        state['using_rl'] = stay
        return torch.where(stay.unsqueeze(-1),
                           vag.actor_mean(obs).clamp(-1.0, 1.0), fcl(e))
    return {'zero': lambda e, dn, obs: torch.zeros((e.n_envs, e.act_dim),
                                                   device=e.device),
            'classical': lambda e, dn, obs: fcl(e),
            'cont': lambda e, dn, obs: ag.actor_mean(obs),
            'hybrid': hybrid}


@torch.no_grad()
def timeit(env, fn, spec, N, periods):
    env.line_dist = scripted(env, spec, 0, N)
    env.reset()
    done = torch.zeros(env.n_envs, dtype=torch.bool, device=dev)
    ts = []
    for p in range(periods + 2):
        obs = env.current_obs()
        torch.cuda.synchronize()
        t0 = time.time()
        a = fn(env, done, obs)
        torch.cuda.synchronize()
        if p >= 2:
            ts.append((time.time() - t0) * 1e3)
        for _ in range(SUB):
            env.step(a, auto_reset=False)
        done = env.done_persistent.clone()
    return float(np.median(ts))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--robot', default='fr3')
    ap.add_argument('--periods', type=int, default=20)
    a = ap.parse_args()
    dev = torch.device('cuda')
    spec, _ = load_tasks(a.robot, 'straight')
    rows = {}
    for N in (1, 2500):
        env = ladder_env(a.robot, N)
        for name, fn in arms(env, a.robot).items():
            ms = timeit(env, fn, spec, N, a.periods)
            rows.setdefault(name, {})[f'batch{N}_ms'] = ms
            rows[name][f'batch{N}_ms_per_task'] = ms / N
            print(f'{name:10s} batch {N:5d}: {ms:8.3f} ms per period '
                  f'({ms / N:.5f} ms per task)', flush=True)
        del env
        torch.cuda.empty_cache()
    json.dump(rows, open(OUT / f'latency_base_{a.robot}.json', 'w'),
              indent=1)

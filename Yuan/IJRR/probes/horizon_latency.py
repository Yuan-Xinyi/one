"""Per-control-period decision latency of the horizon baselines vs the
reactive policy, measured on an otherwise idle GPU.

Two numbers per method: single-task latency (batch 1, what a real-time
controller pays every 50 ms period) and amortized cost at batch 2500 (the
evaluation batch, ms per task-period). The reactive policy is timed as its
deployed computation: one forward pass of the actor (the null-space
projection and amplitude bound are shared by all methods inside the
environment step and are excluded from every row alike).

usage: python -m Yuan.IJRR.probes.horizon_latency [--robot fr3] [--periods 20]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot  # noqa: F401,E402
import numpy as np
import torch
import yaml

from Yuan.IJRR.eval.mpc_mppi_baselines import (
    build_env, load_tasks, scripted, DirFracModel, MPPI, MPC, SUB, OUT,
    MAIN, CFG)
from Yuan.IJRR.stage2_traj.ppo import Agent

WT = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
CKPT = {'fr3': WT / 'Yuan/IJRR/runs/rl_dirfrac_e8kXXL_rm/agent.pt',
        'xarm7': WT / 'Yuan/IJRR/runs/rl_dirfrac_xarm7_e8kXXL_rm/agent.pt'}


def time_planner(env, planner, spec, N, periods):
    env.line_dist = scripted(env, spec, 0, N)
    env.reset()
    planner.reset(env.n_envs)
    ts = []
    for p in range(periods + 2):
        env.current_obs()
        live = (~env.done_persistent).nonzero(as_tuple=False).squeeze(-1)
        if live.numel() == 0:
            break
        torch.cuda.synchronize()
        t0 = time.time()
        a_live = planner.plan(env, live)
        torch.cuda.synchronize()
        if p >= 2:                       # warm-up excluded
            ts.append((time.time() - t0) * 1e3)
        a = torch.zeros((env.n_envs, env.act_dim_policy), device=env.device,
                        dtype=env.q.dtype)
        a[live] = a_live
        for _ in range(SUB):
            env.step(a, auto_reset=False)
    return float(np.median(ts)), int(live.numel())


def time_policy(env, ag, spec, N, periods):
    env.line_dist = scripted(env, spec, 0, N)
    obs = env.reset()
    ts = []
    with torch.no_grad():
        for p in range(periods + 2):
            obs = env.current_obs()
            torch.cuda.synchronize()
            t0 = time.time()
            a = ag.actor_mean(obs)
            torch.cuda.synchronize()
            if p >= 2:
                ts.append((time.time() - t0) * 1e3)
            for _ in range(SUB):
                env.step(a, auto_reset=False)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--robot', default='fr3')
    ap.add_argument('--periods', type=int, default=20)
    ap.add_argument('--big', type=int, default=2500)
    a = ap.parse_args()
    dev = torch.device('cuda')
    spec, _ = load_tasks(a.robot, 'straight')
    rows = {}
    for N in (1, a.big):
        env = build_env(a.robot, 'straight', N, dev)
        model = DirFracModel(env)
        planners = [MPC(model, H, chunk=min(4096, max(256, 122880 // (H * SUB))))
                    for H in (10, 20, 30)] + \
                   [MPPI(model, H, seed=0) for H in (16, 32, 64)]
        for pl in planners:
            ms, live = time_planner(env, pl, spec, N, a.periods)
            rows.setdefault(pl.name, {})[f'batch{N}_ms'] = ms
            rows[pl.name][f'batch{N}_ms_per_task'] = ms / max(live, 1)
            print(f'{pl.name:8s} batch {N:5d}: {ms:9.2f} ms per period '
                  f'({ms / max(live, 1):.4f} ms per task)', flush=True)
        y = yaml.safe_load(open(MAIN / 'stage2_traj' / CFG[a.robot]))
        ag = Agent(env.obs_dim, env.act_dim_policy,
                   hidden_dim=y['ppo']['hidden_dim']).to(dev)
        ag.load_state_dict(torch.load(CKPT[a.robot], map_location=dev))
        ag.eval()
        ms = time_policy(env, ag, spec, N, a.periods)
        rows.setdefault('ours', {})[f'batch{N}_ms'] = ms
        rows['ours'][f'batch{N}_ms_per_task'] = ms / N
        print(f'{"ours":8s} batch {N:5d}: {ms:9.2f} ms per period '
              f'({ms / N:.4f} ms per task)', flush=True)
        del env, model, planners, ag
        torch.cuda.empty_cache()
    json.dump(rows, open(OUT / f'latency_{a.robot}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()

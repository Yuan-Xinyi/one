"""Do the three arms learn the same redundancy strategy?

The task-aligned basis gives the action coordinates one meaning on every
arm: e0 is the projected directional-manipulability gradient, e1 the
projected cone gradient, e2 the projected joint-centering gradient (the
7-DoF arms add one orthogonal complement). A vertex action is therefore a
signed vote on each of the classical objectives, and the learned strategies
of different arms become directly comparable: what fraction of steps pushes
each coordinate positive, and how does that profile move over the stroke.

Reported per arm and per coordinate: mean sign over all live steps, and the
mean sign in the first / middle / last third of each stroke's own duration.
+1 means the policy pushes up the corresponding classical objective at every
step; 0 means it uses both signs equally.

Usage:
    python -m Yuan.IJRR.eval.action_semantics --robot fr3
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import torch
import yaml

from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.line_distribution import (
    LineDistribution, ScriptedLineDistribution)
from Yuan.IJRR.eval.eval_curve import _agent
from Yuan.IJRR.eval.switch_sweep import CONFIGS

COORD = ['manip', 'cone', 'center', 'compl']


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--robot', required=True, choices=list(CONFIGS))
    ap.add_argument('--n-tasks', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=4242)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()

    REPO = Path(__file__).resolve().parents[3]
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
    env.line_dist = ScriptedLineDistribution(
        {'q0': pool.q_pool[idx], 'line_dir': pool.line_dir_pool[idx],
         'n_target': pool.n_target_pool[idx]})
    agent = _agent(REPO / ckpt, env.obs_dim, dev, act_dim=env.act_dim)

    env.reset()
    acts, alive = [], []
    done = torch.zeros(N, dtype=torch.bool, device=dev)
    for _ in range(env.max_steps):
        a_t = agent.actor_mean(env.current_obs())
        acts.append(a_t.cpu())
        alive.append((~done).cpu())
        env.step(a_t, auto_reset=False)
        done = env.done_persistent.clone()
        if bool(done.all()):
            break
    A = torch.stack(acts).numpy()          # (T, N, m)
    L = torch.stack(alive).numpy()         # (T, N)
    T, n, m = A.shape

    length = L.sum(axis=0)                  # steps per episode
    t_idx = np.arange(T)[:, None]
    phase = np.where(length[None, :] > 0, t_idx / np.maximum(length, 1)[None, :], 2.0)
    live = L.astype(bool)

    print(f'{a.robot}: {n} tasks, median length {np.median(length):.0f} steps, '
          f'm = {m}\n')
    hdr = f'{"coordinate":<12s}{"mean sign":>10s}' + ''.join(
        f'{seg:>12s}' for seg in ('early', 'mid', 'late'))
    print(hdr)
    for j in range(m):
        aj = A[..., j]
        row = f'{COORD[j]:<12s}{aj[live].mean():>+10.3f}'
        for lo, hi in ((0, 1/3), (1/3, 2/3), (2/3, 1.0001)):
            sel = live & (phase >= lo) & (phase < hi)
            row += f'{aj[sel].mean():>+12.3f}'
        print(row)


if __name__ == '__main__':
    main()

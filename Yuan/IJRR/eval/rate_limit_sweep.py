"""Performance-smoothness trade-off of the vertex policy under rate limiting.

At deployment a rate limiter |a_k - a_{k-1}| <= da_max clamps how far the
applied command may move per control period; the applied command is fed back
into the observation, so the policy sees what was executed, not what it
asked. da_max = 2 is the unrestricted vertex policy (any vertex can follow
any other); smaller values force the applied command through the interior.

Reports, per da_max: stroke retained (paired ratio to the unrestricted
policy), ratio to the classical law, and the acceleration/jerk percentiles
of the executed joint trajectory. If the stroke survives moderate limiting
while the acceleration tail collapses, the high-frequency switching is not
itself physically essential.

Usage:
    python -m Yuan.IJRR.eval.rate_limit_sweep --n-tasks 1024
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
from Yuan.IJRR.eval.eval_curve import _agent

REPO = Path(__file__).resolve().parents[3]
CKPT = 'Yuan/IJRR/runs/rl_vertex_line_30M'
CFG = 'Yuan/IJRR/stage2_traj/config_vertex_line.yaml'


@torch.no_grad()
def rollout(env, action_fn):
    env.reset()
    p0, u = env.p_start.clone(), env.line_dir.clone()
    prog = torch.zeros(env.n_envs, device=env.device)
    done = torch.zeros(env.n_envs, dtype=torch.bool, device=env.device)
    qd, live = [], []
    for _ in range(env.max_steps):
        q_before = env.q.clone()
        a = action_fn(env)
        env.step(a, auto_reset=False)
        qd.append(((env.q - q_before) / env.cfg.dt).cpu())
        live.append((~done).cpu())
        p, _, _, _ = env.kin.tcp_fk_jac(env.q)
        prog = torch.where(done, prog, ((p - p0) * u).sum(-1))
        done = env.done_persistent.clone()
        if bool(done.all()):
            break
    return (prog.cpu().numpy().copy(), torch.stack(qd).numpy(),
            torch.stack(live).numpy())


def accel_jerk(qd, live, dt):
    a = np.diff(qd, axis=0) / dt
    j = np.diff(qd, n=2, axis=0) / dt ** 2
    lv = live.astype(bool)
    la = lv[:-1] & lv[1:]
    lj = lv[:-2] & lv[1:-1] & lv[2:]
    aa = np.abs(a[la]).reshape(-1)
    jj = np.abs(j[lj]).reshape(-1)
    return (np.percentile(aa, 95), np.percentile(aa, 99),
            np.percentile(jj, 95), np.percentile(jj, 99))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-tasks', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=4242)
    ap.add_argument('--limits', default='2,1,0.5,0.25,0.125')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()

    dev = torch.device(a.device)
    y = yaml.safe_load(open(REPO / CFG))
    keys = {f.name for f in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in y['env'].items() if k in keys}
    N = a.n_tasks

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

    agent = _agent(REPO / CKPT, env.obs_dim, dev, act_dim=env.act_dim)
    classical = ClassicalNullspaceController(env.kin)

    fresh()
    base_cl, _, _ = rollout(env, cn_action_fn(classical))
    ok = base_cl > 1e-6
    dt = env.cfg.dt

    print(f'{a.n_tasks} tasks; classical baseline computed\n')
    print(f'{"da_max":>7s}{"vs unrestricted":>17s}{"vs classical":>14s}'
          f'{"|qddot| p95":>13s}{"p99":>9s}{"|qdddot| p95":>14s}{"p99":>10s}')
    ref = None
    for lim in [float(x) for x in a.limits.split(',')]:
        state = {}

        def fn(e, lim=lim, state=state):
            want = agent.actor_mean(e.current_obs())
            prev = state.get('a')
            if prev is None:
                applied = want
            else:
                applied = torch.clamp(want, prev - lim, prev + lim)
            state['a'] = applied
            return applied

        fresh()
        prog, qd, live = rollout(env, fn)
        if ref is None:
            ref = prog
        keep = (prog[ok] / np.maximum(ref[ok], 1e-6)).mean()
        r_cl = (prog[ok] / base_cl[ok]).mean()
        a95, a99, j95, j99 = accel_jerk(qd, live, dt)
        print(f'{lim:>7.3f}{keep:>17.4f}{r_cl:>14.4f}'
              f'{a95:>13.2f}{a99:>9.2f}{j95:>14.0f}{j99:>10.0f}')


if __name__ == '__main__':
    main()

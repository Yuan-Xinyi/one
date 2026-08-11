"""Report card for the prior-anchored residual sweep.

For each checkpoint: stroke ratio to the classical law, disagreement rate
with the prior's argmax (overall and by the prior's confidence gap
Delta_prior = s(v_top1) - s(v_top2), in quartiles), and the paired net
effect of all deviations (same checkpoint executed with the prior's argmax
forced, on the same tasks).

Usage:
    python -m Yuan.IJRR.eval.prior_anchor_report --ckpts run1,run2,...
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
CFG = 'Yuan/IJRR/stage2_traj/config_vertex_line_prior.yaml'


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', required=True,
                    help='comma-separated run dirs under Yuan/IJRR/runs')
    ap.add_argument('--n-tasks', type=int, default=2048)
    ap.add_argument('--seed', type=int, default=4242)
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
    n_prior = 2 ** env.act_dim

    def rollout(actor_fn):
        fresh(); env.reset()
        p0, u = env.p_start.clone(), env.line_dir.clone()
        done = torch.zeros(N, dtype=torch.bool, device=dev)
        dev_cnt = torch.zeros(N, device=dev)
        steps = torch.zeros(N, device=dev)
        gaps, devs = [], []
        for _ in range(env.max_steps):
            obs = env.current_obs()
            prior = obs[:, -n_prior:]
            top2 = prior.topk(2, dim=-1).values
            gap = top2[:, 0] - top2[:, 1]
            a_idx, act = actor_fn(obs)
            live = ~done
            dv = (a_idx != prior.argmax(-1)) & live
            dev_cnt += dv.float(); steps += live.float()
            if live.any():
                gaps.append(gap[live].cpu()); devs.append(dv[live].cpu())
            env.step(act, auto_reset=False)
            done = env.done_persistent.clone()
            if bool(done.all()):
                break
        p, _, _, _ = env.kin.tcp_fk_jac(env.q)
        prog = ((p - p0) * u).sum(-1).cpu().numpy().copy()
        return (prog, (dev_cnt / steps.clamp_min(1)).cpu().numpy(),
                torch.cat(gaps).numpy(), torch.cat(devs).numpy())

    cl = ClassicalNullspaceController(env.kin)
    fn_cl = cn_action_fn(cl)
    fresh(); env.reset()
    p0, u = env.p_start.clone(), env.line_dir.clone()
    for _ in range(env.max_steps):
        env.step(fn_cl(env), auto_reset=False)
        if bool(env.done_persistent.all()):
            break
    p, _, _, _ = env.kin.tcp_fk_jac(env.q)
    base = ((p - p0) * u).sum(-1).cpu().numpy().copy()
    ok = base > 1e-6
    rng = np.random.default_rng(0)

    print(f'{"run":<26s}{"ratio":>7s}{"prior":>7s}{"net dev [95% CI]":>22s}'
          f"{'dev%':>6s}{'Q1':>6s}{'Q2':>6s}{'Q3':>6s}{'Q4':>6s}")
    for run in a.ckpts.split(','):
        ag = _agent(REPO / 'Yuan/IJRR/runs' / run, env.obs_dim, dev,
                    act_dim=env.act_dim)

        def pol(obs, ag=ag):
            idx_ = ag._logits(obs).argmax(-1)
            return idx_, ag.vertices[idx_]

        def pri(obs, ag=ag):
            idx_ = obs[:, -n_prior:].argmax(-1)
            return idx_, ag.vertices[idx_]

        prog_p, devrate, gaps, devs = rollout(pol)
        prog_0, _, _, _ = rollout(pri)
        r_p = (prog_p[ok] / base[ok]).mean()
        r_0 = (prog_0[ok] / base[ok]).mean()
        d = (prog_p[ok] - prog_0[ok]) / base[ok]
        i = rng.integers(0, len(d), (20000, len(d)))
        m = d[i].mean(1)
        q = np.quantile(gaps, [0, .25, .5, .75, 1.0])
        cells = ''.join(
            f'{devs[(gaps >= q[j]) & (gaps <= q[j + 1])].mean():>6.2f}'
            for j in range(4))
        print(f'{run:<26s}{r_p:>7.3f}{r_0:>7.3f}'
              f'{d.mean():>+9.3f} [{np.percentile(m, 2.5):+.3f},'
              f'{np.percentile(m, 97.5):+.3f}]'
              f'{devrate.mean():>6.1%}{cells}')


if __name__ == '__main__':
    main()

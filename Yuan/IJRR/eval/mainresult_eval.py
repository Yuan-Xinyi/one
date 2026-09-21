"""Ladder rollouts on the fixed FR3 evaluation set, cached per task.

Runs each requested arm over the 10k evaluation tasks (task-generating
configuration as the shared start, task-defined p0 as the anchor) in chunks,
and saves one npz per arm with the executed length, the termination reason
and the mean per-decision wall time. The realized-fraction table, the
termination table and the survival curves are all cut from these caches by
fill_report.py; nothing here needs to be re-run for follow-up analysis.

Usage:
    python -m Yuan.IJRR.eval.mainresult_eval --arms zero,classical,myopic \
        --n-tasks 10000 --out-dir Yuan/IJRR/runs/paper_fill
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import torch
import yaml

import Yuan.IJRR.eval.horizon_ladder as hl
from Yuan.IJRR.eval.horizon_ladder import (
    StraightModel, rollout_env, make_myopic, make_sgngrad, make_cem,
    make_beam)
from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.IJRR.env.classical_nullspace import (
    ClassicalNullspaceController, cn_action_fn)
from Yuan.IJRR.env.line_distribution import ScriptedLineDistribution
from Yuan.IJRR.eval.eval_curve import _agent

REPO = Path(__file__).resolve().parents[3]
TASKS = 'Yuan/IJRR/runs/eval_10k_systematic/eval_set_10k.npz'


def build(batch, dev):
    cfg_path, ckpt = hl.ROBOTS['fr3']
    y = yaml.safe_load(open(REPO / cfg_path))
    keys = {f.name for f in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in y['env'].items() if k in keys}
    kw['dt'] = kw['dt'] / hl.SUB
    kw['max_steps'] = int(y['env']['max_steps'] * hl.SUB)
    env = NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': batch}), None, dev)
    model = StraightModel(env)
    model.cfg = dataclasses.replace(env.cfg, dt=y['env']['dt'])
    model.terms = [0, 1]
    return env, model, ckpt


def make_arm(name, env, model, classical, ckpt, dev):
    if name == 'zero':
        return lambda e, dn: torch.zeros((e.n_envs, e.act_dim),
                                         device=e.device)
    if name == 'classical':
        f = cn_action_fn(classical)
        return lambda e, dn, f=f: f(e)
    if name == 'myopic':
        return make_myopic(model)
    if name == 'sgngrad':
        return make_sgngrad(model)
    if name.startswith('mcem'):
        return make_cem(model, H=int(name[4:]), objective='margin')
    if name.startswith('cem'):
        return make_cem(model, H=int(name[3:]))
    if name.startswith('beam'):
        w, h = name[4:].split('x')
        return make_beam(model, width=int(w), H=int(h))
    if name == 'vertex':
        ag = _agent(REPO / ckpt, env.obs_dim, dev, act_dim=env.act_dim)

        @torch.no_grad()
        def fn(e, dn, ag=ag):
            return ag.actor_mean(e.current_obs())
        return fn
    raise ValueError(name)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', required=True)
    ap.add_argument('--n-tasks', type=int, default=10000)
    ap.add_argument('--batch', type=int, default=2048)
    ap.add_argument('--out-dir', default='Yuan/IJRR/runs/paper_fill')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    dev = torch.device(a.device)
    out_dir = REPO / a.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    t = np.load(REPO / TASKS)
    N = min(a.n_tasks, len(t['cs_p0']))
    batch = min(a.batch, N)
    env, model, ckpt = build(batch, dev)
    classical = ClassicalNullspaceController(env.kin)
    dt_k = env.kin.dtype

    spec_all = {
        'q0': torch.tensor(t['q0_seed'][:N], dtype=dt_k),
        'line_dir': torch.tensor(t['cs_line_dir'][:N], dtype=dt_k),
        'n_target': torch.tensor(t['cs_n_target'][:N], dtype=dt_k),
        'p0': torch.tensor(t['cs_p0'][:N], dtype=torch.float32),
    }
    spec_all['line_dir'] /= spec_all['line_dir'].norm(dim=-1, keepdim=True)
    spec_all['n_target'] /= spec_all['n_target'].norm(dim=-1, keepdim=True)

    for name in a.arms.split(','):
        arm = make_arm(name, env, model, classical, ckpt, dev)
        prog = np.zeros(N, np.float32)
        term = np.zeros(N, np.int64)
        times = []
        for lo in range(0, N, batch):
            hi = min(lo + batch, N)
            sub = {k: v[lo:hi].clone() for k, v in spec_all.items()}
            if hi - lo < batch:
                pad = batch - (hi - lo)
                sub = {k: torch.cat([v, v[-1:].expand(pad, *v.shape[1:])])
                       for k, v in sub.items()}
            for k in ('q0', 'line_dir', 'n_target'):
                sub[k] = sub[k].to(device=dev, dtype=dt_k)
            env.line_dist = ScriptedLineDistribution(sub)
            timer = []
            p, tm = rollout_env(env, arm, timer=timer)
            prog[lo:hi] = p[:hi - lo]
            term[lo:hi] = tm[:hi - lo]
            times.extend(timer)
            print(f'[{name}] {hi}/{N} mean {prog[:hi].mean():.3f} m',
                  flush=True)
        np.savez(out_dir / f'arm_{name}_{N}.npz', prog=prog, term=term,
                 ms=float(np.mean(times) * 1000) if times else 0.0)
        print(f'[{name}] saved: mean {prog.mean():.4f} m, '
              f'{np.mean(times) * 1000 if times else 0:.1f} ms/decision',
              flush=True)


if __name__ == '__main__':
    main()

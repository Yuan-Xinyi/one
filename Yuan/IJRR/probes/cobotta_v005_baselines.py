"""Cobotta rows of tab:mainresult at the reduced task speed v = 0.05 m/s.

Re-runs the four redundancy-resolution baselines of the main table (zero
null-space, classical gradient, continuous PPO, PPO+classical hybrid) with
the fam_unify protocol (config_vertex_line_cobotta env, dt/2 integration,
max_steps x2, k_lateral = 5.0 on curved families) but with v = 0.05 and
max_steps = 2000, i.e. the same speed and time budget as the Cobotta
DirFrac controller (rl_dirfrac_cobotta_e8kXXL_v005) and the Cobotta MPC /
MPPI cells. Same tasks and references as tab:horizon.

Writes runs/paper_fill/horizon/cobotta_v005_{arm}_{fam}.npz and a summary
json per family.
"""
import json
import sys
import time
import dataclasses
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
from Yuan.IJRR.eval.mpc_mppi_baselines import (load_tasks, scripted, stat,
                                               SUB, OUT, MAIN)
from Yuan.IJRR.stage2_traj.ppo import Agent

WT = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
CONT_CKPT = WT / 'Yuan/IJRR/runs/rl_cont_sqent_cobotta_30M/agent.pt'
VERTEX_CKPT = WT / 'Yuan/IJRR/runs/rl_vertex_line_cobotta_30M'
V, MAX_STEPS = 0.05, 2000
dev = torch.device('cuda')
ARMS = sys.argv[1].split(',') if len(sys.argv) > 1 else \
    ['zero', 'classical', 'cont', 'hybrid']


def ladder_env(batch, k_lateral):
    y = yaml.safe_load(open(Path('/home/lqin/one') / hl.ROBOTS['cobotta'][0]))
    keys = {f.name for f in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in y['env'].items() if k in keys}
    kw['v'] = V
    kw['dt'] = kw['dt'] / SUB
    kw['max_steps'] = int(MAX_STEPS * SUB)
    if k_lateral:
        kw['k_lateral'] = k_lateral
    return NSRLBatchedEnv(EnvConfig(**{**kw, 'n_envs': batch}), None, dev)


def build_arms(env):
    classical = ClassicalNullspaceController(env.kin)
    fcl = cn_action_fn(classical)
    arms = {'zero': lambda e, dn: torch.zeros((e.n_envs, e.act_dim),
                                              device=e.device),
            'classical': lambda e, dn: fcl(e)}
    if 'cont' in ARMS:
        ag = Agent(env.obs_dim, env.act_dim).to(dev)
        ag.load_state_dict(torch.load(CONT_CKPT, map_location=dev))
        ag.eval()
        arms['cont'] = lambda e, dn, g_=ag: g_.actor_mean(e.current_obs())
    if 'hybrid' in ARMS:
        vag = _agent(VERTEX_CKPT, env.obs_dim, dev, act_dim=env.act_dim)
        arms['hybrid'] = hl.make_hybrid(env, vag, classical, 0.98, 0.94)
    return {k: v for k, v in arms.items() if k in ARMS}


@torch.no_grad()
def run(env, afn, spec, N):
    out = np.zeros(N, np.float32)
    B = env.n_envs
    for lo in range(0, N, B):
        hi = min(lo + B, N)
        env.line_dist = scripted(env, spec, lo, hi)
        env.reset()
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        for _ in range(env.cfg.max_steps // SUB):
            a = afn(env, done)
            for _ in range(SUB):
                env.step(a, auto_reset=False)
            done = env.done_persistent.clone()
            if bool(done.all()):
                break
        out[lo:hi] = env.arc_progress.float().cpu().numpy()[:hi - lo]
    return out


for fam in ('straight', 'serpentine', 'nonplanar'):
    spec, ref = load_tasks('cobotta', fam)
    N = spec['q0'].shape[0]
    env = ladder_env(min(2500, N), 5.0 if fam != 'straight' else 0.0)
    assert abs(env.v - V) < 1e-9 and env.cfg.max_steps == MAX_STEPS * SUB
    summ = {}
    for name, fn in build_arms(env).items():
        f = OUT / f'cobotta_v005_{name}_{fam}.npz'
        if f.exists():
            prog = np.load(f)['prog']
        else:
            t0 = time.time()
            prog = run(env, fn, spec, N)
            np.savez_compressed(f, prog=prog)
            print(f'  ({time.time() - t0:.0f}s)', flush=True)
        mean, rm, p10 = stat(prog, ref)
        summ[name] = {'stroke': mean, 'ratio_mean': rm, 'ratio_p10': p10}
        print(f'[cobotta/{fam}] {name:10s} v=0.05: stroke {mean:.3f}  '
              f'ratio {rm:.1f} / {p10:.1f}', flush=True)
    old = OUT / f'cobotta_v005_baselines_{fam}.json'
    if old.exists():
        summ = {**json.load(open(old)), **summ}
    json.dump(summ, open(old, 'w'), indent=1)
    del env
    torch.cuda.empty_cache()

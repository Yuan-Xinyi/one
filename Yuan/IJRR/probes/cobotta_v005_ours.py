"""Cobotta DirFrac rows at the reduced task speed v = 0.05 m/s (the
rl_dirfrac_cobotta_e8kXXL_v005 checkpoint), on the same tasks and with the
same references as the Cobotta MPC/MPPI cells of tab:horizon: straight 10k
(tasks_pool_cobotta) and the two curved families (tasks_selx_*_cobotta,
k_lateral = 5.0). Writes runs/paper_fill/horizon/cobotta_{fam}_ours.json.
"""
import json
import time
from pathlib import Path

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot  # noqa: F401,E402
import numpy as np
import torch
import yaml

from Yuan.IJRR.eval.mpc_mppi_baselines import (
    build_env, load_tasks, scripted, stat, SUB, OUT, MAIN, CFG)
from Yuan.IJRR.stage2_traj.ppo import Agent

WT = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
CKPT = WT / 'Yuan/IJRR/runs/rl_dirfrac_cobotta_e8kXXL_v005/agent.pt'
dev = torch.device('cuda')


@torch.no_grad()
def run(env, ag, spec, N):
    out = np.zeros(N, np.float32)
    B = env.n_envs
    for lo in range(0, N, B):
        hi = min(lo + B, N)
        env.line_dist = scripted(env, spec, lo, hi)
        env.reset()
        for _ in range(env.cfg.max_steps // SUB):
            a = ag.actor_mean(env.current_obs())
            for _ in range(SUB):
                env.step(a, auto_reset=False)
            if bool(env.done_persistent.all()):
                break
        out[lo:hi] = env.arc_progress.float().cpu().numpy()[:hi - lo]
        print(f'  {hi}/{N} mean {out[:hi].mean():.4f}', flush=True)
    return out


y = yaml.safe_load(open(MAIN / 'stage2_traj' / CFG['cobotta']))
assert abs(y['env']['v'] - 0.05) < 1e-9
for fam in ('straight', 'serpentine', 'nonplanar'):
    spec, ref = load_tasks('cobotta', fam)
    N = spec['q0'].shape[0]
    env = build_env('cobotta', fam, min(2500, N), dev)
    ag = Agent(env.obs_dim, env.act_dim_policy,
               hidden_dim=y['ppo']['hidden_dim']).to(dev)
    ag.load_state_dict(torch.load(CKPT, map_location=dev))
    ag.eval()
    t0 = time.time()
    prog = run(env, ag, spec, N)
    mean, rm, p10 = stat(prog, ref)
    print(f'[cobotta/{fam}] ours(v=0.05): stroke {mean:.3f}  ratio {rm:.1f} / '
          f'{p10:.1f}  ({time.time() - t0:.0f}s)', flush=True)
    np.savez_compressed(OUT / f'cobotta_{fam}_ours.npz', prog=prog)
    json.dump({'robot': 'cobotta', 'family': fam, 'method': 'ours',
               'v': 0.05, 'N': N, 'stroke': mean, 'ratio_mean': rm,
               'ratio_p10': p10, 'ckpt': str(CKPT)},
              open(OUT / f'cobotta_{fam}_ours.json', 'w'), indent=1)
    del env, ag
    torch.cuda.empty_cache()

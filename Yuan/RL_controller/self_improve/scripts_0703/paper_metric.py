"""Paper-metric refresh (DP seeds x pct of oracle_hyb) for the new finals:
  - single net soup3_s2_b975 (pure)
  - system r12m_b0.965_soup2 + switch @0.985/0.96
Mirrors run_main_table.py; caches per-task npz in the sweeps dir.
"""
import os, sys
_conda_lib = os.path.join(sys.prefix, "lib")
if _conda_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    new_env = dict(os.environ)
    new_env["LD_LIBRARY_PATH"] = _conda_lib + ":" + new_env.get("LD_LIBRARY_PATH", "")
    os.execvpe(sys.executable, [sys.executable] + sys.argv, new_env)

from pathlib import Path
import numpy as np
import torch, yaml

sys.path.insert(0, "/home/lqin/one")
os.chdir("/home/lqin/one")
from Yuan.system_eval.rollout_controllers import (
    build_env, rollout_seeds_batched, load_rl_agent)
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController

cfg = yaml.safe_load(open('Yuan/system_eval/config.yaml'))
dev = torch.device('cuda')
tdm = float(cfg['env']['target_distance_m'])
SW = Path('Yuan/system_eval/runs/eval_10k_systematic/sweeps')

seeds = np.load(SW / 'cfg_only_w1.5.npz')['seeds'].astype(np.float32)
T = seeds.shape[0]
z = np.load('Yuan/system_eval/runs/eval_10k_systematic/eval_set_10k.npz')
p0 = z['cs_p0'][:T].astype(np.float32)
d = z['cs_line_dir'][:T].astype(np.float32)
n = z['cs_n_target'][:T].astype(np.float32)
oh = np.load('Yuan/system_eval/runs/eval_10k_systematic/'
             'cell_oracle_hyb_results.npz')['L_best'][:T].astype(np.float32) * tdm
bucket = np.where(oh >= 0.80, 'Easy', np.where(oh >= 0.45, 'Medium', 'Difficult'))

env = build_env(cfg['env']['config_yaml'], n_envs=int(cfg['env']['n_envs']),
                device=dev)
classical = ClassicalNullspaceController(env.kin)

JOBS = [
    ('soup3_pure', 'Yuan/RL_controller/runs/distill_soup3_s2_b975', 1.0, 1.0),
    ('soup3+switch0.985-0.965', 'Yuan/RL_controller/runs/distill_soup3_s2_b975',
     0.985, 0.965),
    ('r12m965+switch0.985-0.96',
     'Yuan/RL_controller/runs/distill_r12m_b0.965_soup2', 0.985, 0.96),
]
for name, ckpt, te, tx in JOBS:
    out = SW / f'main_{name}.npz'
    if out.exists():
        pct = np.load(out)['pct']
    else:
        agent = load_rl_agent(Path(ckpt), env, dev)
        res = rollout_seeds_batched(
            seeds, p0, d, n, env=env, controller='hybrid_variantB',
            classical=classical, agent=agent, tau_enter=te, tau_exit=tx,
            target_distance_m=tdm, progress_prefix=f'[{name}] ')
        L = res['L'].astype(np.float32)
        with np.errstate(invalid='ignore', divide='ignore'):
            pct = 100.0 * (L * tdm) / np.maximum(oh, 1e-9)
        np.savez_compressed(out, L=L, pct=pct, bucket=bucket)
    fin = np.isfinite(pct)
    print(f'[{name}] All {pct[fin].mean():.1f}%  ' +
          '  '.join(f'{b} {pct[fin & (bucket == b)].mean():.1f}%'
                    for b in ('Easy', 'Medium', 'Difficult')), flush=True)
print('[paper] done', flush=True)

"""Hysteresis-threshold sweep for the journal paper's ablation: roll the
mixed-selector pick on a fixed 1,024-task subset of eval10k under the
hybrid controller for a grid of (tau_enter, tau_exit).
Output: runs/iksel_final_n48/tau_sweep.json
"""
import json, os
from pathlib import Path
import numpy as np
import torch

from Yuan.unified_rl.candidate_batch import CachedSeedCandidateDataset
from Yuan.unified_rl.checkpoint import (
    build_env_from_run, load_controller_agent, load_run_config,
    ppo_config_from_run, resolve_controller_dir)
from Yuan.unified_rl.controller_rollout import (
    FrozenHybridController, rollout_selected_seeds)
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.unified_rl.iksel_campaign import _load_pool_env, _load_sel, C0_DIR
from Yuan.unified_rl.ikpool_bidir import _picks

G = Path('Yuan/unified_rl/runs/iksel_final_n48')
ROLL_CHUNK = int(os.environ.get('ROLL_CHUNK', '128'))
N_SUB = 1024
GRID = [(0.95, 0.95), (0.96, 0.96), (0.97, 0.97), (0.98, 0.98),
        (0.985, 0.985), (0.99, 0.99), (0.985, 0.96), (0.985, 0.97),
        (0.99, 0.96)]

dev = torch.device('cuda:0')
X, P, V = _load_pool_env(G / 'iksel_eval10k_candidates.npz',
                         G / 'iksel_eval10k_returns_hybrid.npz', dev)
sel = _load_sel(G / 'sel_mixed_run0.pt', dev)
pick = _picks(*sel, X, V).cpu().numpy()
lref = torch.where(V, P, torch.tensor(-1e9, device=dev)) \
    .max(1).values.cpu().numpy()

c = np.load(G / 'iksel_eval10k_candidates.npz')
rng = np.random.default_rng(20260803)
sub = np.sort(rng.choice(len(c['p0']), N_SUB, replace=False))
K = c['seeds'].shape[1]
seeds = np.where((pick[sub] < K)[:, None],
                 c['seeds'][sub, np.minimum(pick[sub], K - 1)],
                 c['q0_pilot'][sub]).astype(np.float32)
tmp = G / '_tau_sweep_tmp.npz'
np.savez(tmp, seeds=seeds[:, None], ik_ok=np.ones((N_SUB, 1), bool),
         p0=c['p0'][sub], line_dir=c['line_dir'][sub],
         n_target=c['n_target'][sub], q0_pilot=c['q0_pilot'][sub],
         task_indices=np.arange(N_SUB, dtype=np.int64))

env = build_env_from_run(resolve_controller_dir(C0_DIR), ROLL_CHUNK, dev)
gamma = float(ppo_config_from_run(load_run_config(
    resolve_controller_dir(C0_DIR))).gamma)
agent = load_controller_agent(resolve_controller_dir(C0_DIR), env, dev).eval()
ds = CachedSeedCandidateDataset.from_npz(tmp, include_fallback=False)

res = {}
for te, tx in GRID:
    ctl = FrozenHybridController(agent, ClassicalNullspaceController(env.kin),
                                 te, tx)
    prog = np.zeros(N_SUB, np.float32)
    sw = np.zeros(N_SUB, np.float32)
    for s in range(0, N_SUB, ROLL_CHUNK):
        r = torch.arange(s, min(s + ROLL_CHUNK, N_SUB))
        nr = len(r)
        if nr < ROLL_CHUNK:
            r = torch.cat([r, r[-1:].expand(ROLL_CHUNK - nr)])
        cb = ds.batch.index_select(r).to(device=dev, dtype=env.kin.dtype)
        out = rollout_selected_seeds(
            env, cb, torch.zeros(ROLL_CHUNK, dtype=torch.long, device=dev),
            ctl, gamma=gamma)
        prog[s:s + nr] = out.progress_m[:nr].cpu().numpy()
        sw[s:s + nr] = out.switch_count[:nr].float().cpu().numpy()
    ratio = float((prog / np.maximum(lref[sub], 1e-6)).mean() * 100)
    res[f'{te}/{tx}'] = {'len': round(float(prog.mean()), 4),
                         'ratio': round(ratio, 2),
                         'switches': round(float(sw.mean()), 2)}
    print(f'[tau {te}/{tx}] len={prog.mean():.4f} ratio={ratio:.2f} '
          f'sw={sw.mean():.2f}', flush=True)
(G / 'tau_sweep.json').write_text(json.dumps(res, indent=1))
print('TAU SWEEP DONE', flush=True)

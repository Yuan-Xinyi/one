"""Re-measure the full pipeline with the simplified (search-free) student.

Deployed controller: hybrid(distill_simple_exit_final, tau 0.985/0.96).
1. Paper metric, DP seeds: standalone + hybrid rows.
2. Ranked seeding: re-roll the 25 cached candidates per task under the new
   controller (candidates and ranker picks are controller-independent),
   then report first-valid / ranked(v4 picks) / best-of-25 on frozen oracle'.
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

CKPT = Path('Yuan/RL_controller/runs/exit_rounds7plus/final_avg')
TAU = (0.985, 0.96)
P0DIR = Path('Yuan/seed_selection/runs/rank_phase0')
OUT = P0DIR / 'final_ctrl'
OUT.mkdir(exist_ok=True)
SW = Path('Yuan/system_eval/runs/eval_10k_systematic/sweeps')

dev = torch.device('cuda')
cfg = yaml.safe_load(open('Yuan/system_eval/config.yaml'))
tdm = float(cfg['env']['target_distance_m'])
z = np.load('Yuan/system_eval/runs/eval_10k_systematic/eval_set_10k.npz')
oh = np.load('Yuan/system_eval/runs/eval_10k_systematic/'
             'cell_oracle_hyb_results.npz')['L_best'].astype(np.float32) * tdm
bucket = np.where(oh >= 0.80, 'Easy', np.where(oh >= 0.45, 'Medium', 'Difficult'))

env = build_env(CKPT / 'config.yaml', 4096, dev)
classical = ClassicalNullspaceController(env.kin)
agent = load_rl_agent(CKPT, env, dev)

# ---- 1. paper metric on DP seeds ----
seeds_dp = np.load(SW / 'cfg_only_w1.5.npz')['seeds'].astype(np.float32)
T = seeds_dp.shape[0]
p0 = z['cs_p0'][:T].astype(np.float32)
d_ = z['cs_line_dir'][:T].astype(np.float32)
n_ = z['cs_n_target'][:T].astype(np.float32)
for name, te, tx in [('final_pure', 1.0, 1.0),
                     ('final+switch0.985-0.96', TAU[0], TAU[1])]:
    out = SW / f'main_{name}.npz'
    if out.exists():
        pct = np.load(out)['pct']
    else:
        res = rollout_seeds_batched(seeds_dp, p0, d_, n_, env=env,
                                    controller='hybrid_variantB',
                                    classical=classical, agent=agent,
                                    tau_enter=te, tau_exit=tx,
                                    target_distance_m=tdm,
                                    progress_prefix=f'[{name}] ')
        with np.errstate(invalid='ignore', divide='ignore'):
            pct = 100.0 * (res['L'] * tdm) / np.maximum(oh[:T], 1e-9)
        np.savez_compressed(out, L=res['L'], pct=pct)
    fin = np.isfinite(pct)
    print(f'[{name}] All {pct[fin].mean():.1f}%  ' +
          '  '.join(f'{b} {pct[fin & (bucket[:T] == b)].mean():.1f}%'
                    for b in ('Easy', 'Medium', 'Difficult')), flush=True)

# ---- 2. re-roll the 25 candidate slots on the eval set ----
pd0 = np.load(P0DIR / 'candidates_K8.npz')
pde = np.load(P0DIR / 'candidates_ext8.npz')
pdw = np.load(P0DIR / 'candidates_extw1.npz')
seeds25 = np.concatenate([pd0['seeds'], pde['seeds'],
                          pdw['seeds'], z['q0_seed'][:, None, :]], 1).astype(np.float32)
ok25 = np.concatenate([pd0['ik_ok'], pde['ik_ok'], pdw['ik_ok'],
                       np.ones((len(oh), 1), bool)], 1)
p0s = z['cs_p0'].astype(np.float32)
lds = z['cs_line_dir'].astype(np.float32)
nts = z['cs_n_target'].astype(np.float32)
L25 = np.zeros((len(oh), 25), np.float32)
for si in range(25):
    f = OUT / f'L_slot{si}.npz'
    if f.exists():
        L25[:, si] = np.load(f)['L']
        continue
    r = rollout_seeds_batched(seeds25[:, si], p0s, lds, nts, env=env,
                              controller='hybrid_variantB', classical=classical,
                              agent=agent, tau_enter=TAU[0], tau_exit=TAU[1],
                              progress_prefix=f'slot{si} ')
    L25[:, si] = r['L']
    np.savez_compressed(f, L=r['L'])
    print(f'[reroll] slot {si} done', flush=True)
L25m = L25 * tdm

# ---- 3. rows on frozen oracle' with the v4 picks ----
pick = np.load(P0DIR / 'phase3_ranked_v4.npz')['pick']   # controller-independent
first_idx = np.argmax(ok25[:, :16], 1)
has = ok25[:, :16].any(1)
L_first = np.where(has, L25m[np.arange(len(oh)), first_idx], L25m[:, 24])
L_rank = L25m[np.arange(len(oh)), pick]
L_best = np.where(ok25, L25m, -np.inf).max(1)
fin = oh > 1e-9
def pct(Lm_):
    return 100.0 * (Lm_[fin] / oh[fin]).mean()
print('\n==== FINAL (round-12) CONTROLLER, RANKED SEEDING (frozen oracle\') ====')
print(f'  first-valid   {pct(L_first):.2f}%   [with search: 91.41]')
print(f'  ranked (v4 picks) {pct(L_rank):.2f}%   [with search: 98.38]')
print(f'  best-of-25    {pct(L_best):.2f}%   [with search: 103.22]')
for b in ('Easy', 'Medium', 'Difficult'):
    m = fin & (bucket == b)
    print(f'  {b:9s}: ranked {100*(L_rank[m]/oh[m]).mean():.1f}%')
np.savez_compressed(OUT / 'summary.npz', L_first=L_first, L_rank=L_rank,
                    L_best=L_best, oh=oh)
print('[simple-reroll] done', flush=True)

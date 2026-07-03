"""Seed-ranking Phase 0: real-sampler headroom on the 10k eval set.

Generate K=8 IK-valid-or-not DP candidates per task (deployment params:
w=1.5, DDIM 50, EMA), roll EVERY candidate with the adopted controller
(r12m_b0.965_soup2 hybrid @0.985/0.96), and report first-valid (deployment
status quo) vs best-of-K prefixes vs mean. Caches everything for Phase 3.
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
from Yuan.system_eval.seed_sources import diffusion_seeds
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController

OUT = Path('Yuan/seed_selection/runs/rank_phase0')
OUT.mkdir(parents=True, exist_ok=True)
CKPT = Path('Yuan/RL_controller/runs/distill_r12m_b0.965_soup2')
TAU = (0.985, 0.96)
K = 8

dev = torch.device('cuda')
cfg = yaml.safe_load(open('Yuan/system_eval/config.yaml'))
dc = cfg['diffusion']
z = np.load('Yuan/system_eval/runs/eval_10k_systematic/eval_set_10k.npz')
eval_set = {k: z[k] for k in z.keys()}
n_tasks = eval_set['cs_p0'].shape[0]
oh = np.load('Yuan/system_eval/runs/eval_10k_systematic/'
             'cell_oracle_hyb_results.npz')['L_best'][:n_tasks].astype(np.float32) * 1.5

env = build_env(CKPT / 'config.yaml', 4096, dev)
classical = ClassicalNullspaceController(env.kin)
agent = load_rl_agent(CKPT, env, dev)

# ---- 1. candidates ----
cand_npz = OUT / 'candidates_K8.npz'
if cand_npz.exists():
    d = np.load(cand_npz)
    seeds, ik_ok = d['seeds'], d['ik_ok']
    print(f'[p0] candidates cached: IK ok {100*ik_ok.mean():.1f}%', flush=True)
else:
    seeds, ik_ok = diffusion_seeds(
        eval_set, dc['ckpt'], n_samples=K, ddim_steps=int(dc['ddim_steps']),
        cfg_w=1.5, sample_seed=9300, kin=env.kin, device=dev,
        use_ema=bool(dc['use_ema']))
    np.savez_compressed(cand_npz, seeds=seeds, ik_ok=ik_ok)
    print(f'[p0] candidates saved: IK ok {100*ik_ok.mean():.1f}%', flush=True)

# ---- 2. roll every slot ----
p0s = eval_set['cs_p0'].astype(np.float32)
lds = eval_set['cs_line_dir'].astype(np.float32)
nts = eval_set['cs_n_target'].astype(np.float32)
L_slots = np.full((n_tasks, K), np.nan, dtype=np.float32)
for si in range(K):
    slot_npz = OUT / f'L_slot{si}.npz'
    if slot_npz.exists():
        L_slots[:, si] = np.load(slot_npz)['L']
        continue
    r = rollout_seeds_batched(
        seeds[:, si].astype(np.float32), p0s, lds, nts, env=env,
        controller='hybrid_variantB', classical=classical, agent=agent,
        tau_enter=TAU[0], tau_exit=TAU[1], progress_prefix=f'slot{si} ')
    L_slots[:, si] = r['L']
    np.savez_compressed(slot_npz, L=r['L'], episode_len=r['episode_len'],
                        term_reason=r['term_reason'])
    print(f'[p0] slot {si} done', flush=True)

# ---- 3. fallback for tasks with no IK-valid slot: pilot q0_seed ----
none_ok = ~ik_ok.any(axis=1)
L_fb = np.zeros(n_tasks, dtype=np.float32)
if none_ok.any():
    idx = np.nonzero(none_ok)[0]
    r = rollout_seeds_batched(
        eval_set['q0_seed'][idx].astype(np.float32), p0s[idx], lds[idx],
        nts[idx], env=env, controller='hybrid_variantB', classical=classical,
        agent=agent, tau_enter=TAU[0], tau_exit=TAU[1],
        progress_prefix='fallback ')
    L_fb[idx] = r['L']
    print(f'[p0] fallback rolled for {len(idx)} tasks', flush=True)

# ---- 4. metrics ----
Lm = L_slots * 1.5
valid_mask = ik_ok.copy()
Lm_valid = np.where(valid_mask, Lm, -np.inf)

def pct_of(L_m):
    fin = oh > 1e-9
    return 100.0 * (L_m[fin] / oh[fin])

# first-valid (deployment status quo)
first_idx = np.argmax(valid_mask, axis=1)          # first True (0 if none)
L_first = np.where(none_ok, L_fb * 1.5,
                   Lm[np.arange(n_tasks), first_idx])
# best-of-K prefixes
res = {'first_valid': pct_of(L_first).mean()}
for kk in (1, 2, 4, 8):
    Lbk = Lm_valid[:, :kk].max(axis=1)
    Lbk = np.where(np.isfinite(Lbk), Lbk, L_fb * 1.5)
    res[f'best_of_{kk}'] = pct_of(Lbk).mean()
# mean over valid
with np.errstate(invalid='ignore'):
    Lmean = np.where(valid_mask, Lm, np.nan)
    Lmean = np.nanmean(Lmean, axis=1)
    Lmean = np.where(np.isnan(Lmean), L_fb * 1.5, Lmean)
res['mean_of_valid'] = pct_of(Lmean).mean()

np.savez_compressed(OUT / 'phase0_results.npz',
                    L_slots=L_slots, ik_ok=ik_ok, L_fallback=L_fb,
                    L_first=L_first, oh=oh,
                    **{k: np.float64(v) for k, v in res.items()})
print('\n==== PHASE 0 RESULTS (paper pct of oracle_hyb) ====')
for k, v in res.items():
    print(f'  {k:14s} {v:.2f}%')
print(f'  (anchor: adopted first-valid main-table row = 91.4%)')
print(f'  GATE: best_of_8 - first_valid = '
      f'{res["best_of_8"] - res["first_valid"]:+.2f}pp '
      f'({"PASS" if res["best_of_8"] - res["first_valid"] >= 1.0 else "FAIL"})',
      flush=True)

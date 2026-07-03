"""Seed-ranking v3: 40k train tasks (rank_train + rank_train_b), ens10-pair,
applied to the 17-way eval choice set (16 DP candidates + pilot).

Prereqs: rank_phase1b done (train_b slots), rank_k16_ext done (eval slots
8-15 + features). Computes train_b pilot labels/features if missing.
"""
import os, sys
_conda_lib = os.path.join(sys.prefix, "lib")
if _conda_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    new_env = dict(os.environ)
    new_env["LD_LIBRARY_PATH"] = _conda_lib + ":" + new_env.get("LD_LIBRARY_PATH", "")
    os.execvpe(sys.executable, [sys.executable] + sys.argv, new_env)

from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/home/lqin/one")
os.chdir("/home/lqin/one")
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.system_eval.rollout_controllers import (
    build_env, rollout_seeds_batched, load_rl_agent)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rank_v2 import obs_and_manip, Rank, fit_one, capture_eval

TRA = Path('Yuan/seed_selection/runs/rank_train')
TRB = Path('Yuan/seed_selection/runs/rank_train_b')
P0DIR = Path('Yuan/seed_selection/runs/rank_phase0')
CKPT = Path('Yuan/RL_controller/runs/distill_r12m_b0.965_soup2')
TAU = (0.985, 0.96)
K = 8
dev = torch.device('cuda')

# ---- ensure train_b pilot labels + features ----
env = build_env(CKPT / 'config.yaml', 4096, dev)
classical = ClassicalNullspaceController(env.kin)
agent = load_rl_agent(CKPT, env, dev)
cdb = np.load(TRB / 'candidates_K8.npz')
if not (TRB / 'L_pilot.npz').exists():
    r = rollout_seeds_batched(cdb['q0_pilot'], cdb['p0'], cdb['line_dir'],
                              cdb['n_target'], env=env,
                              controller='hybrid_variantB',
                              classical=classical, agent=agent,
                              tau_enter=TAU[0], tau_exit=TAU[1],
                              progress_prefix='pilot-b ')
    np.savez_compressed(TRB / 'L_pilot.npz', L=r['L'])
    print('[v3] train_b pilot labeled', flush=True)
if not (TRB / 'feat_v2.npz').exists():
    seeds = cdb['seeds']
    N = seeds.shape[0]
    obs_p, mu_p = obs_and_manip(env, cdb['q0_pilot'], cdb['p0'],
                                cdb['line_dir'], cdb['n_target'])
    obs_s = np.zeros((N, K, 31), np.float32)
    mu_s = np.zeros((N, K), np.float32)
    for si in range(K):
        obs_s[:, si], mu_s[:, si] = obs_and_manip(
            env, seeds[:, si], cdb['p0'], cdb['line_dir'], cdb['n_target'])
        print(f'[v3] train_b feat slot {si}', flush=True)
    np.savez_compressed(TRB / 'feat_v2.npz', obs_pilot=obs_p, mu_pilot=mu_p,
                        obs_slots=obs_s, mu_slots=mu_s)


def load_train(d):
    cd = np.load(d / 'candidates_K8.npz')
    fv = np.load(d / 'feat_v2.npz')
    L = np.stack([np.load(d / f'L_slot{si}.npz')['L'] for si in range(K)], 1)
    Lp = np.load(d / 'L_pilot.npz')['L']
    obs = np.concatenate([fv['obs_slots'], fv['obs_pilot'][:, None, :]], 1)
    mu = np.concatenate([fv['mu_slots'], fv['mu_pilot'][:, None]], 1)
    X = np.concatenate([obs, np.log(mu[..., None] + 1e-9)], -1)
    y = np.concatenate([L, Lp[:, None]], 1) * 1.5
    ok = np.concatenate([cd['ik_ok'], np.ones((len(L), 1), bool)], 1)
    return X.astype(np.float32), y.astype(np.float32), ok

Xa, ya, oka = load_train(TRA)
Xb, yb, okb = load_train(TRB)
X = np.concatenate([Xa, Xb]); y = np.concatenate([ya, yb])
ok = np.concatenate([oka, okb])
N = X.shape[0]
print(f'[v3] merged train: {N} tasks', flush=True)

rng = np.random.default_rng(0)
perm = rng.permutation(N)
n_val = N // 10
va_t, tr_t = perm[:n_val], perm[n_val:]
mean = X[tr_t][ok[tr_t]].mean(0)
std = X[tr_t][ok[tr_t]].std(0) + 1e-6
Xn = (X - mean) / std
ysd = float(y[tr_t][ok[tr_t]].std())
Xd = torch.from_numpy(Xn[tr_t]).to(dev)
yd = torch.from_numpy(y[tr_t] / ysd).to(dev)
okd = torch.from_numpy(ok[tr_t]).to(dev)

ik_dp = ok[va_t][:, :K]
first_idx = np.argmax(ik_dp, 1)
y_first = np.where(ik_dp.any(1), y[va_t][np.arange(len(va_t)), first_idx],
                   y[va_t][:, K])
Xva = torch.from_numpy(Xn[va_t]).float().to(dev)


@torch.no_grad()
def score_val(net):
    return net(Xva.reshape(-1, X.shape[-1])).view(len(va_t), 9).cpu().numpy()

ens, sc_sum = [], None
for sd_i in range(10):
    n_i = fit_one(Xd, yd, None, okd, 'pair', seed=sd_i)
    ens.append(n_i)
    sc = score_val(n_i)
    sc_sum = sc if sc_sum is None else sc_sum + sc
    Lm, cap = capture_eval(sc_sum / (sd_i + 1), y[va_t], ok[va_t], y_first)
    print(f'[v3] ens{sd_i+1}-pair val L {Lm:.4f} capture {cap:.1f}%', flush=True)

torch.save({'nets': [n.state_dict() for n in ens], 'mean': mean, 'std': std,
            'ysd': ysd, 'kind': 'pair', 'feat': 'obs31+logmu'},
           TRA / 'ranker_v3.pt')

# ---- apply: 17-way on 10k ----
z = np.load('Yuan/system_eval/runs/eval_10k_systematic/eval_set_10k.npz')
pd0 = np.load(P0DIR / 'candidates_K8.npz')
pde = np.load(P0DIR / 'candidates_ext8.npz')
fv0 = np.load(P0DIR / 'feat_v2.npz')
fve = np.load(P0DIR / 'feat_v2_ext8.npz')
L16 = np.stack([np.load(P0DIR / f'L_slot{si}.npz')['L'] for si in range(16)], 1) * 1.5
Lp10 = np.load(CKPT / 'eval_10k.npz')['L_hyb'] * 1.5
ok10 = np.concatenate([pd0['ik_ok'], pde['ik_ok'],
                       np.ones((len(L16), 1), bool)], 1)
y10 = np.concatenate([L16, Lp10[:, None]], 1)
obs10 = np.concatenate([fv0['obs_slots'], fve['obs_slots'],
                        fv0['obs_pilot'][:, None, :]], 1)
mu10 = np.concatenate([fv0['mu_slots'], fve['mu_slots'],
                       fv0['mu_pilot'][:, None]], 1)
X10 = np.concatenate([obs10, np.log(mu10[..., None] + 1e-9)], -1)
X10n = torch.from_numpy((X10 - mean) / std).float()

sc10 = np.zeros(y10.shape, np.float32)
with torch.no_grad():
    flat = X10n.reshape(-1, X10.shape[-1])
    for n_i in ens:
        out = []
        for s in range(0, len(flat), 65536):
            out.append(n_i(flat[s:s + 65536].to(dev)).cpu())
        sc10 += torch.cat(out).view(y10.shape).numpy()

res0 = np.load(P0DIR / 'phase0_results.npz')
oh, L_first10 = res0['oh'], res0['L_first']
fin = oh > 1e-9
def pct(Lm_):
    return 100.0 * (Lm_[fin] / oh[fin]).mean()

for tag, kk in (('K=8+pilot (9-way)', list(range(8)) + [16]),
                ('K=16+pilot (17-way)', list(range(17)))):
    sel = np.array(kk)
    s = np.where(ok10[:, sel], sc10[:, sel], -np.inf)
    pick = s.argmax(1)
    L_pick = y10[:, sel][np.arange(len(y10)), pick]
    Lbest = np.where(ok10[:, sel], y10[:, sel], -np.inf).max(1)
    d10 = (L_pick[fin] - L_first10[fin]) / oh[fin] * 100
    se = d10.std(ddof=1) / np.sqrt(len(d10))
    print(f'\n[v3] {tag}: ranked {pct(L_pick):.2f}%  best {pct(Lbest):.2f}%  '
          f'(+{d10.mean():.2f}pp ± {se:.2f} vs first-valid {pct(L_first10):.2f}%)')
    if len(sel) == 17:
        bucket = np.where(oh >= 0.80, 'Easy',
                          np.where(oh >= 0.45, 'Medium', 'Difficult'))
        for b in ('Easy', 'Medium', 'Difficult'):
            m = fin & (bucket == b)
            print(f'  {b:9s}: {100*(L_pick[m]/oh[m]).mean():.1f}%')
        tail = fin & (bucket == 'Easy') & (
            100 * L_first10 / np.maximum(oh, 1e-9) < 50)
        print(f'  Easy tail: {100*(L_pick[tail]/oh[tail]).mean():.1f}%  '
              f'still<50%: {(100*L_pick[tail]/oh[tail] < 50).sum()}/{tail.sum()}')
        np.savez_compressed(P0DIR / 'phase3_ranked_v3.npz',
                            L_ranked=L_pick, pick=pick)
print('[v3] done', flush=True)

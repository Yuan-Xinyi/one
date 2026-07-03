"""Seed-ranking v4: 60k train (a+b+c) x pair+list compound loss x ens10,
applied to the 25-way eval choice set (16 w1.5 + 8 w1.0 + pilot).

Prereqs: rank_phase1c (train_c) and rank_extw1 (eval slots 16-23) done.
Computes train_c pilot labels/features if missing.
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
import torch.nn as nn

sys.path.insert(0, "/home/lqin/one")
os.chdir("/home/lqin/one")
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.system_eval.rollout_controllers import (
    build_env, rollout_seeds_batched, load_rl_agent)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rank_v2 import obs_and_manip

TRDIRS = [Path('Yuan/seed_selection/runs/rank_train'),
          Path('Yuan/seed_selection/runs/rank_train_b'),
          Path('Yuan/seed_selection/runs/rank_train_c')]
P0DIR = Path('Yuan/seed_selection/runs/rank_phase0')
CKPT = Path('Yuan/RL_controller/runs/distill_r12m_b0.965_soup2')
TAU = (0.985, 0.96)
K = 8
dev = torch.device('cuda')

# ---- ensure train_c pilot + features ----
TRC = TRDIRS[2]
cdc = np.load(TRC / 'candidates_K8.npz')
if not (TRC / 'L_pilot.npz').exists() or not (TRC / 'feat_v2.npz').exists():
    env = build_env(CKPT / 'config.yaml', 4096, dev)
    classical = ClassicalNullspaceController(env.kin)
    agent = load_rl_agent(CKPT, env, dev)
    if not (TRC / 'L_pilot.npz').exists():
        r = rollout_seeds_batched(cdc['q0_pilot'], cdc['p0'], cdc['line_dir'],
                                  cdc['n_target'], env=env,
                                  controller='hybrid_variantB',
                                  classical=classical, agent=agent,
                                  tau_enter=TAU[0], tau_exit=TAU[1],
                                  progress_prefix='pilot-c ')
        np.savez_compressed(TRC / 'L_pilot.npz', L=r['L'])
        print('[v4] train_c pilot labeled', flush=True)
    if not (TRC / 'feat_v2.npz').exists():
        seeds = cdc['seeds']
        Nc = seeds.shape[0]
        obs_p, mu_p = obs_and_manip(env, cdc['q0_pilot'], cdc['p0'],
                                    cdc['line_dir'], cdc['n_target'])
        obs_s = np.zeros((Nc, K, 31), np.float32)
        mu_s = np.zeros((Nc, K), np.float32)
        for si in range(K):
            obs_s[:, si], mu_s[:, si] = obs_and_manip(
                env, seeds[:, si], cdc['p0'], cdc['line_dir'], cdc['n_target'])
        np.savez_compressed(TRC / 'feat_v2.npz', obs_pilot=obs_p,
                            mu_pilot=mu_p, obs_slots=obs_s, mu_slots=mu_s)
        print('[v4] train_c features done', flush=True)


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

parts = [load_train(d) for d in TRDIRS]
X = np.concatenate([p[0] for p in parts])
y = np.concatenate([p[1] for p in parts])
ok = np.concatenate([p[2] for p in parts])
N = X.shape[0]
print(f'[v4] merged train: {N} tasks', flush=True)

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
Xva = torch.from_numpy(Xn[va_t]).float().to(dev)

ik_dp = ok[va_t][:, :K]
first_idx = np.argmax(ik_dp, 1)
y_first = np.where(ik_dp.any(1), y[va_t][np.arange(len(va_t)), first_idx],
                   y[va_t][:, K])
y_best = np.where(ok[va_t], y[va_t], -np.inf).max(1)


class Rank(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 512), nn.ReLU(),
                                 nn.Linear(512, 512), nn.ReLU(),
                                 nn.Linear(512, 256), nn.ReLU(),
                                 nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit(seed, epochs=60):
    torch.manual_seed(seed)
    net = Rank().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Ntr = Xd.shape[0]
    for ep in range(epochs):
        order = torch.randperm(Ntr, device=dev)
        for s in range(0, Ntr, 1024):
            ti = order[s:s + 1024]
            pred = net(Xd[ti].reshape(-1, 32)).view(len(ti), 9)
            yy, okk = yd[ti], okd[ti]
            dy = yy.unsqueeze(2) - yy.unsqueeze(1)
            dp = pred.unsqueeze(2) - pred.unsqueeze(1)
            pv = okk.unsqueeze(2) & okk.unsqueeze(1) & (dy > 0.01)
            pl = (torch.relu(0.05 - dp[pv]).mean() if pv.sum() > 0
                  else pred.sum() * 0.0)
            logp = torch.log_softmax(pred.masked_fill(~okk, -1e9), -1)
            tgt = torch.softmax((yy / 0.05).masked_fill(~okk, -1e9), -1)
            loss = pl + 0.5 * (-(tgt * logp).sum(-1).mean())
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return net


@torch.no_grad()
def score(net, Xt, shape):
    flat = Xt.reshape(-1, 32)
    out = []
    for s in range(0, len(flat), 65536):
        out.append(net(flat[s:s + 65536].to(dev)).cpu())
    return torch.cat(out).view(shape).numpy()

ens, sc_sum = [], None
for sd in range(10):
    net = fit(sd)
    ens.append(net)
    sc = score(net, Xva, (len(va_t), 9))
    sc_sum = sc if sc_sum is None else sc_sum + sc
    s_ = np.where(ok[va_t], sc_sum, -np.inf)
    Lp = y[va_t][np.arange(len(va_t)), s_.argmax(1)]
    cap = 100 * (Lp - y_first).sum() / (y_best - y_first).sum()
    print(f'[v4] ens{sd+1} val capture {cap:.1f}%', flush=True)

torch.save({'nets': [n.state_dict() for n in ens], 'mean': mean, 'std': std,
            'ysd': ysd, 'kind': 'pair+list', 'feat': 'obs31+logmu'},
           TRDIRS[0] / 'ranker_v4.pt')

# ---- apply: 25-way ----
z = np.load('Yuan/system_eval/runs/eval_10k_systematic/eval_set_10k.npz')
pd0 = np.load(P0DIR / 'candidates_K8.npz')
pde = np.load(P0DIR / 'candidates_ext8.npz')
pdw = np.load(P0DIR / 'candidates_extw1.npz')
fv0 = np.load(P0DIR / 'feat_v2.npz')
fve = np.load(P0DIR / 'feat_v2_ext8.npz')
fvw = np.load(P0DIR / 'feat_v2_extw1.npz')
L24 = np.stack([np.load(P0DIR / f'L_slot{si}.npz')['L'] for si in range(24)], 1) * 1.5
Lp10 = np.load(CKPT / 'eval_10k.npz')['L_hyb'] * 1.5
ok10 = np.concatenate([pd0['ik_ok'], pde['ik_ok'], pdw['ik_ok'],
                       np.ones((len(L24), 1), bool)], 1)
y10 = np.concatenate([L24, Lp10[:, None]], 1)
obs10 = np.concatenate([fv0['obs_slots'], fve['obs_slots'], fvw['obs_slots'],
                        fv0['obs_pilot'][:, None, :]], 1)
mu10 = np.concatenate([fv0['mu_slots'], fve['mu_slots'], fvw['mu_slots'],
                       fv0['mu_pilot'][:, None]], 1)
X10 = np.concatenate([obs10, np.log(mu10[..., None] + 1e-9)], -1)
X10n = torch.from_numpy((X10 - mean) / std).float()

sc10 = np.zeros(y10.shape, np.float32)
for n_i in ens:
    sc10 += score(n_i, X10n, y10.shape)

res0 = np.load(P0DIR / 'phase0_results.npz')
oh, L_first10 = res0['oh'], res0['L_first']
fin = oh > 1e-9
def pct(Lm_):
    return 100.0 * (Lm_[fin] / oh[fin]).mean()

SETS = {'17-way (w1.5 only)': list(range(16)) + [24],
        'w1.0-only 9-way': list(range(16, 25)),
        '25-way (mixed w)': list(range(25))}
for tag, sel in SETS.items():
    sel = np.array(sel)
    s_ = np.where(ok10[:, sel], sc10[:, sel], -np.inf)
    pick = s_.argmax(1)
    L_pick = y10[:, sel][np.arange(len(y10)), pick]
    Lbest = np.where(ok10[:, sel], y10[:, sel], -np.inf).max(1)
    Lbest = np.where(np.isfinite(Lbest), Lbest, L_first10)
    L_pick = np.where(np.isfinite(L_pick), L_pick, L_first10)
    d10 = (L_pick[fin] - L_first10[fin]) / oh[fin] * 100
    se = d10.std(ddof=1) / np.sqrt(len(d10))
    print(f'[v4] {tag}: ranked {pct(L_pick):.2f}%  best {pct(Lbest):.2f}%  '
          f'(+{d10.mean():.2f}pp ± {se:.2f})', flush=True)
    if tag.startswith('25'):
        bucket = np.where(oh >= 0.80, 'Easy',
                          np.where(oh >= 0.45, 'Medium', 'Difficult'))
        for b in ('Easy', 'Medium', 'Difficult'):
            m = fin & (bucket == b)
            print(f'  {b:9s}: {100*(L_pick[m]/oh[m]).mean():.1f}%')
        tail = fin & (bucket == 'Easy') & (
            100 * L_first10 / np.maximum(oh, 1e-9) < 50)
        print(f'  Easy tail: {100*(L_pick[tail]/oh[tail]).mean():.1f}%  '
              f'still<50%: {(100*L_pick[tail]/oh[tail] < 50).sum()}/{tail.sum()}')
        np.savez_compressed(P0DIR / 'phase3_ranked_v4.npz',
                            L_ranked=L_pick, pick=pick)
print('[v4] done', flush=True)

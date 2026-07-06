"""Regenerate scorer training labels under the simplified controller and
retrain the ranking scorer (pair+list, ens10) for full paper consistency.

Candidates and features for the 3 x 20480 training tasks are cached; only the
per-candidate rollout labels change (hybrid(distill_simple_exit_final,
0.985/0.96)). New labels go to <dir>/simple/L_slot*.npz + L_pilot.npz.
Then fit and apply to the re-rolled eval caches (rank_phase0/simple_ctrl).
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
from Yuan.system_eval.rollout_controllers import (
    build_env, rollout_seeds_batched, load_rl_agent)
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController

CKPT = Path('Yuan/RL_controller/runs/distill_simple_exit_final')
TAU = (0.985, 0.96)
TRDIRS = [Path('Yuan/seed_selection/runs/rank_train'),
          Path('Yuan/seed_selection/runs/rank_train_b'),
          Path('Yuan/seed_selection/runs/rank_train_c')]
P0DIR = Path('Yuan/seed_selection/runs/rank_phase0')
K = 8
dev = torch.device('cuda')

env = build_env(CKPT / 'config.yaml', 4096, dev)
classical = ClassicalNullspaceController(env.kin)
agent = load_rl_agent(CKPT, env, dev)

# ---- 1. relabel: 3 dirs x (8 slots + pilot) ----
for d in TRDIRS:
    cd = np.load(d / 'candidates_K8.npz')
    p0, ld, nt = cd['p0'], cd['line_dir'], cd['n_target']
    sd = d / 'simple'
    sd.mkdir(exist_ok=True)
    for si in list(range(K)) + ['pilot']:
        f = sd / (f'L_slot{si}.npz' if si != 'pilot' else 'L_pilot.npz')
        if f.exists():
            continue
        qs = cd['seeds'][:, si] if si != 'pilot' else cd['q0_pilot']
        r = rollout_seeds_batched(qs.astype(np.float32), p0, ld, nt, env=env,
                                  controller='hybrid_variantB',
                                  classical=classical, agent=agent,
                                  tau_enter=TAU[0], tau_exit=TAU[1],
                                  progress_prefix=f'{d.name}-{si} ')
        np.savez_compressed(f, L=r['L'])
        print(f'[relabel] {d.name} slot {si} done', flush=True)


def load_train(d):
    cd = np.load(d / 'candidates_K8.npz')
    fv = np.load(d / 'feat_v2.npz')
    sd = d / 'simple'
    L = np.stack([np.load(sd / f'L_slot{si}.npz')['L'] for si in range(K)], 1)
    Lp = np.load(sd / 'L_pilot.npz')['L']
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
print(f'[fit] {N} tasks with simplified-controller labels', flush=True)

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
for sd_i in range(10):
    net = fit(sd_i)
    ens.append(net)
    sc = score(net, Xva, (len(va_t), 9))
    sc_sum = sc if sc_sum is None else sc_sum + sc
    s_ = np.where(ok[va_t], sc_sum, -np.inf)
    Lp = y[va_t][np.arange(len(va_t)), s_.argmax(1)]
    cap = 100 * (Lp - y_first).sum() / (y_best - y_first).sum()
    print(f'[fit] ens{sd_i+1} val capture {cap:.1f}%', flush=True)
torch.save({'nets': [n.state_dict() for n in ens], 'mean': mean, 'std': std,
            'ysd': ysd, 'kind': 'pair+list',
            'labels': 'hybrid(simple_exit_final,0.985/0.96)'},
           TRDIRS[0] / 'ranker_simple.pt')

# ---- 3. apply to the eval set (simplified-controller slot caches) ----
z = np.load('Yuan/system_eval/runs/eval_10k_systematic/eval_set_10k.npz')
fv0 = np.load(P0DIR / 'feat_v2.npz')
fve = np.load(P0DIR / 'feat_v2_ext8.npz')
fvw = np.load(P0DIR / 'feat_v2_extw1.npz')
pd0 = np.load(P0DIR / 'candidates_K8.npz')
pde = np.load(P0DIR / 'candidates_ext8.npz')
pdw = np.load(P0DIR / 'candidates_extw1.npz')
SC = P0DIR / 'simple_ctrl'
L25 = np.stack([np.load(SC / f'L_slot{si}.npz')['L'] for si in range(25)], 1) * 1.5
ok25 = np.concatenate([pd0['ik_ok'], pde['ik_ok'], pdw['ik_ok'],
                       np.ones((len(L25), 1), bool)], 1)
obs10 = np.concatenate([fv0['obs_slots'], fve['obs_slots'], fvw['obs_slots'],
                        fv0['obs_pilot'][:, None, :]], 1)
mu10 = np.concatenate([fv0['mu_slots'], fve['mu_slots'], fvw['mu_slots'],
                       fv0['mu_pilot'][:, None]], 1)
X10 = np.concatenate([obs10, np.log(mu10[..., None] + 1e-9)], -1)
X10n = torch.from_numpy((X10 - mean) / std).float()
sc10 = np.zeros((len(L25), 25), np.float32)
for n_i in ens:
    sc10 += score(n_i, X10n, (len(L25), 25))
pick = np.where(ok25, sc10, -np.inf).argmax(1)
L_rank = L25[np.arange(len(L25)), pick]
oh = np.load('Yuan/system_eval/runs/eval_10k_systematic/'
             'cell_oracle_hyb_results.npz')['L_best'].astype(np.float32) * 1.5
fin = oh > 1e-9
first_idx = np.argmax(ok25[:, :16], 1)
has = ok25[:, :16].any(1)
L_first = np.where(has, L25[np.arange(len(L25)), first_idx], L25[:, 24])
L_best = np.where(ok25, L25, -np.inf).max(1)
def pct(Lm_):
    return 100.0 * (Lm_[fin] / oh[fin]).mean()
print('\n==== FINAL: simplified controller + retrained scorer ====')
print(f'  first-valid   {pct(L_first):.2f}%')
print(f'  ranked        {pct(L_rank):.2f}%   [old scorer on same controller: 97.91]')
print(f'  best-of-25    {pct(L_best):.2f}%')
np.savez_compressed(SC / 'ranked_retrained.npz', L_ranked=L_rank, pick=pick)
print('[retrain] done', flush=True)

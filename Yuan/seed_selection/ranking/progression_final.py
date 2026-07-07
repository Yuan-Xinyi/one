"""Ablation rows for the simplified framework's ranked-initialization table:
ranking objective (pointwise vs pair+list, both ens10, 60k simplified labels)
and candidate-set size (K=8 w1.5 / K=16 w1.5 / 24 mixed, + pilot), evaluated
by slicing the simplified-controller slot caches. No new rollouts.
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

TRDIRS = [Path('Yuan/seed_selection/runs/rank_train'),
          Path('Yuan/seed_selection/runs/rank_train_b'),
          Path('Yuan/seed_selection/runs/rank_train_c')]
P0DIR = Path('Yuan/seed_selection/runs/rank_phase0')
SC = P0DIR / 'final_ctrl'
K = 8
dev = torch.device('cuda')


def load_train(d):
    cd = np.load(d / 'candidates_K8.npz')
    fv = np.load(d / 'feat_v2.npz')
    sd = d / 'final'
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


class Rank(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 512), nn.ReLU(),
                                 nn.Linear(512, 512), nn.ReLU(),
                                 nn.Linear(512, 256), nn.ReLU(),
                                 nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit(kind, seed, epochs=60):
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
            if kind == 'point':
                loss = (((pred - yy) ** 2) * okk).sum() / okk.sum()
            else:
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

# ---- eval-set tensors ----
fv0 = np.load(P0DIR / 'feat_v2.npz')
fve = np.load(P0DIR / 'feat_v2_ext8.npz')
fvw = np.load(P0DIR / 'feat_v2_extw1.npz')
pd0 = np.load(P0DIR / 'candidates_K8.npz')
pde = np.load(P0DIR / 'candidates_ext8.npz')
pdw = np.load(P0DIR / 'candidates_extw1.npz')
L25 = np.stack([np.load(SC / f'L_slot{si}.npz')['L'] for si in range(25)], 1) * 1.5
ok25 = np.concatenate([pd0['ik_ok'], pde['ik_ok'], pdw['ik_ok'],
                       np.ones((len(L25), 1), bool)], 1)
obs10 = np.concatenate([fv0['obs_slots'], fve['obs_slots'], fvw['obs_slots'],
                        fv0['obs_pilot'][:, None, :]], 1)
mu10 = np.concatenate([fv0['mu_slots'], fve['mu_slots'], fvw['mu_slots'],
                       fv0['mu_pilot'][:, None]], 1)
X10 = np.concatenate([obs10, np.log(mu10[..., None] + 1e-9)], -1)
X10n = torch.from_numpy((X10 - mean) / std).float()
oh = np.load('Yuan/system_eval/runs/eval_10k_systematic/'
             'cell_oracle_hyb_results.npz')['L_best'].astype(np.float32) * 1.5
fin = oh > 1e-9
first_idx = np.argmax(ok25[:, :16], 1)
has = ok25[:, :16].any(1)
L_first = np.where(has, L25[np.arange(len(L25)), first_idx], L25[:, 24])
def pct(Lm_):
    return 100.0 * (Lm_[fin] / oh[fin]).mean()


def apply_subset(sc10, cols, tag):
    sel = np.array(cols)
    s_ = np.where(ok25[:, sel], sc10[:, sel], -np.inf)
    pick = s_.argmax(1)
    Lp = L25[:, sel][np.arange(len(L25)), pick]
    Lp = np.where(np.isfinite(Lp), Lp, L_first)
    print(f'  {tag:28s} {pct(Lp):.2f}%', flush=True)

# ---- objective ablation: pointwise ens10 ----
sc_point = np.zeros((len(L25), 25), np.float32)
for sd_i in range(10):
    net = fit('point', sd_i)
    sc_point += score(net, X10n, (len(L25), 25))
    print(f'[prog] point ens{sd_i+1} fitted', flush=True)

# pair+list ens10 = the deployed scorer
ck = torch.load(TRDIRS[0] / 'ranker_final.pt', map_location=dev,
                weights_only=False)
nets = []
for sd_ in ck['nets']:
    n = Rank().to(dev); n.load_state_dict(sd_); n.eval(); nets.append(n)
sc_pl = np.zeros((len(L25), 25), np.float32)
for n_i in nets:
    sc_pl += score(n_i, X10n, (len(L25), 25))

print('\n==== ABLATION ROWS (final controller, frozen oracle\') ====')
print(f'  {"first-valid (no selection)":28s} {pct(L_first):.2f}%')
print('  -- ranking objective (all 25 candidates, ens10, 60k tasks) --')
apply_subset(sc_point, list(range(25)), 'pointwise regression')
apply_subset(sc_pl, list(range(25)), 'pairwise + listwise')
print('  -- candidate set (deployed scorer) --')
apply_subset(sc_pl, list(range(8)) + [24], 'K=8 (w=1.5) + fallback')
apply_subset(sc_pl, list(range(16)) + [24], 'K=16 (w=1.5) + fallback')
apply_subset(sc_pl, list(range(25)), 'K=24 mixed w + fallback')
Lb = np.where(ok25, L25, -np.inf).max(1)
print(f'  {"best-of-25 (oracle)":28s} {pct(Lb):.2f}%')
np.savez_compressed(SC / 'ablation_rows.npz', sc_point=sc_point, sc_pl=sc_pl)
print('[prog] done', flush=True)

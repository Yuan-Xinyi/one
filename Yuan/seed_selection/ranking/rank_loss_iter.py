"""C: offline loss iteration on the existing 40k train set (no rollouts).

Variants: pair margin {0.02, 0.05, 0.10}, pair+point, pair+list compound.
Single-seed nets compared on held-out capture; winner noted for v4.
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

K = 8
dev = torch.device('cuda')


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


TRA = Path('Yuan/seed_selection/runs/rank_train')
TRB = Path('Yuan/seed_selection/runs/rank_train_b')
Xa, ya, oka = load_train(TRA)
Xb, yb, okb = load_train(TRB)
X = np.concatenate([Xa, Xb]); y = np.concatenate([ya, yb])
ok = np.concatenate([oka, okb])
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


def loss_fn(kind, pred, yy, okk):
    dy = yy.unsqueeze(2) - yy.unsqueeze(1)
    dp = pred.unsqueeze(2) - pred.unsqueeze(1)
    def pair(m_y, m_s):
        pv = okk.unsqueeze(2) & okk.unsqueeze(1) & (dy > m_y)
        if pv.sum() == 0:
            return pred.sum() * 0.0
        return torch.relu(m_s - dp[pv]).mean()
    if kind == 'pair05':
        return pair(0.01, 0.05)
    if kind == 'pair02':
        return pair(0.01, 0.02)
    if kind == 'pair10':
        return pair(0.01, 0.10)
    if kind == 'pair+point':
        pl = (((pred - yy) ** 2) * okk).sum() / okk.sum()
        return pair(0.01, 0.05) + 0.5 * pl
    if kind == 'pair+list':
        logp = torch.log_softmax(pred.masked_fill(~okk, -1e9), -1)
        tgt = torch.softmax((yy / 0.05).masked_fill(~okk, -1e9), -1)
        ll = -(tgt * logp).sum(-1).mean()
        return pair(0.01, 0.05) + 0.5 * ll
    raise ValueError(kind)


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
            loss = loss_fn(kind, pred, yd[ti], okd[ti])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return net


@torch.no_grad()
def capture(net_or_sc):
    sc = (net_or_sc if isinstance(net_or_sc, np.ndarray) else
          net_or_sc(Xva.reshape(-1, 32)).view(len(va_t), 9).cpu().numpy())
    s = np.where(ok[va_t], sc, -np.inf)
    pick = s.argmax(1)
    Lp = y[va_t][np.arange(len(va_t)), pick]
    return Lp.mean(), 100 * (Lp - y_first).sum() / (y_best - y_first).sum(), sc

print(f'[C] 40k loaded, val first {y_first.mean():.4f} best {y_best.mean():.4f}',
      flush=True)
results = {}
for kind in ('pair05', 'pair02', 'pair10', 'pair+point', 'pair+list'):
    net = fit(kind, seed=0)
    Lm, cap, _ = capture(net)
    results[kind] = cap
    print(f'[C] {kind:11s} val L {Lm:.4f}  capture {cap:.1f}%', flush=True)
winner = max(results, key=results.get)
print(f'[C] winner: {winner} — ens5:', flush=True)
sc_sum = None
for sd in range(5):
    net = fit(winner, seed=sd)
    _, _, sc = capture(net)
    sc_sum = sc if sc_sum is None else sc_sum + sc
    Lm, cap, _ = capture(sc_sum / (sd + 1))
    print(f'[C]   ens{sd+1}: capture {cap:.1f}%', flush=True)
print('[C] done', flush=True)

"""Selector capacity probe on the E2 internal split (never touches val/external).

Question: is the backward direction limited by selector capacity/training
budget? Compare configs at full training size on the SAME geometry-disjoint
internal test split used by E2 (seed 20260724, n=3000).
"""
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from Yuan.unified_rl.candidate_batch import CachedSeedCandidateDataset
from Yuan.unified_rl.checkpoint import build_env_from_run, resolve_controller_dir
from Yuan.unified_rl.offline_seed_ensemble_train import _build_features

D = Path('Yuan/unified_rl/runs/ikpool_full_v1')
dev = torch.device('cuda:0')
N_TEST = 3000

env = build_env_from_run(resolve_controller_dir('Yuan/unified_rl/runs/r2_grouped_best'), 1, dev)
ds = CachedSeedCandidateDataset.from_npz(D / 'ikpool_candidates.npz')
ret = np.load(D / 'ikpool_returns.npz')
X = _build_features(env.kin, ds, 4096).to(dev)
P = torch.nan_to_num(torch.as_tensor(ret['progress_m']), nan=0.0).to(dev)
V = torch.as_tensor(ret['valid']).to(dev)
N = len(P)

# identical split construction to ikpool_e2_scaling.py
sig = np.concatenate([ds.batch.p0.numpy(), ds.batch.line_dir.numpy(),
                      ds.batch.n_target.numpy()], axis=1)
_, gid = np.unique(np.round(sig, 6), axis=0, return_inverse=True)
rng0 = np.random.default_rng(20260724)
order = rng0.permutation(N)
is_test = np.zeros(N, bool); seen = set(); cnt = 0
for r in order:
    if cnt >= N_TEST:
        break
    g = gid[r]
    if g in seen:
        continue
    seen.add(g); is_test[gid == g] = True; cnt = is_test.sum()
test = torch.as_tensor(np.nonzero(is_test)[0], device=dev)
fit = torch.as_tensor(np.nonzero(~is_test)[0], device=dev)


class SetSel(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(45, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.score = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 1))
        self.feas = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, X, Vm):
        e = self.enc(X)
        vf = Vm.unsqueeze(-1).float()
        ctx = (e * vf).sum(1) / vf.sum(1).clamp_min(1)
        h = torch.cat([e, ctx.unsqueeze(1).expand(-1, e.shape[1], -1)], -1)
        return self.score(h).squeeze(-1), self.feas(h).squeeze(-1)


mu, sd = X[fit][V[fit]].mean(0), X[fit][V[fit]].std(0).clamp_min(1e-6)
Xz = ((X - mu) / sd).masked_fill(~V.unsqueeze(-1), 0.0)
lo = torch.where(V, P, torch.tensor(1e9, device=dev)).min(1, keepdim=True).values
hi = torch.where(V, P, torch.tensor(-1e9, device=dev)).max(1, keepdim=True).values
T = ((P - lo) / (hi - lo).clamp_min(1e-6)).masked_fill(~V, 0.0)
hub = nn.HuberLoss(delta=0.05, reduction='none')


def capture(sel, rows):
    first = V[rows].float().argmax(1)
    pr = P[rows]
    ora = torch.where(V[rows], pr, torch.tensor(-1e9, device=dev)).max(1).values
    i = torch.arange(len(rows), device=dev)
    s, f = pr[i, sel], pr[i, first]
    return float((s - f).sum() / (ora - f).sum() * 100), float(s.mean())


def train_one(h, epochs, lr, seed, batch_tasks=4096):
    torch.manual_seed(seed)
    net = SetSel(h).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    g = torch.Generator(device=dev).manual_seed(seed)
    for _ in range(epochs):
        rows = fit[torch.randint(0, len(fit), (batch_tasks,), generator=g, device=dev)]
        opt.zero_grad()
        s, f = net(Xz[rows], V[rows])
        s = s.masked_fill(~V[rows], -1e9)
        tgt = torch.softmax((T[rows] / 0.1).masked_fill(~V[rows], -1e9), 1)
        rank = -(tgt * torch.log_softmax(s, 1).clamp_min(-30)).sum(1).mean()
        feas = (hub(f, P[rows]) * V[rows].float()).sum() / V[rows].float().sum()
        (rank + feas).backward(); opt.step(); sched.step()
    return net


CONFIGS = [
    {'name': 'base_h256_e300', 'h': 256, 'epochs': 300, 'lr': 1e-3},
    {'name': 'h512_e300', 'h': 512, 'epochs': 300, 'lr': 1e-3},
    {'name': 'h256_e1000', 'h': 256, 'epochs': 1000, 'lr': 1e-3},
    {'name': 'h512_e1000', 'h': 512, 'epochs': 1000, 'lr': 1e-3},
    {'name': 'h512_e2000_lr5e4', 'h': 512, 'epochs': 2000, 'lr': 5e-4},
]
out = {}
for cfg in CONFIGS:
    logits = []
    caps = []
    for seed in (0, 1, 2):
        net = train_one(cfg['h'], cfg['epochs'], cfg['lr'], seed)
        with torch.no_grad():
            s, _ = net(Xz[test], V[test])
            s = s.masked_fill(~V[test], -1e9)
        logits.append(s)
        c, _ = capture(s.argmax(1), test)
        caps.append(c)
    ens = torch.stack(logits).mean(0)
    ec, em = capture(ens.argmax(1), test)
    out[cfg['name']] = {'per_seed_capture': [round(c, 2) for c in caps],
                        'mean': float(np.mean(caps)), 'std': float(np.std(caps)),
                        'ens3_capture': ec, 'ens3_mean_m': em}
    print(f"{cfg['name']:>20}: {np.mean(caps):5.1f}%±{np.std(caps):.1f}  ens3={ec:5.1f}%", flush=True)
(D / 'ikpool_capacity.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))

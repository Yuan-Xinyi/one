"""Definitive signal test: retrain selectors ON the IK pool (not transfer).

400 train / 100 test tasks, 45-D features -> progress. If even a retrained
model cannot beat first-valid on held-out tasks, the static-feature signal is
absent, not merely shifted.
"""
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from scipy.stats import spearmanr

from Yuan.unified_rl.candidate_batch import CachedSeedCandidateDataset
from Yuan.unified_rl.checkpoint import build_env_from_run, resolve_controller_dir
from Yuan.unified_rl.validity import validate_cached_dataset
from Yuan.unified_rl.offline_seed_ensemble_train import _build_features

OUT = Path('Yuan/unified_rl/runs/_ikpool_pilot_v1')
device = torch.device('cuda:0')
torch.manual_seed(0)

env = build_env_from_run(resolve_controller_dir('Yuan/unified_rl/runs/r2_grouped_best'), 1, device)
ds = CachedSeedCandidateDataset.from_npz(OUT / 'ikpool_candidates.npz')
ds, _ = validate_cached_dataset(ds, env.kin, env.collision, chunk_size=4096,
                                cone_deg=env.cfg.cone_deg)
ret = np.load(OUT / 'ikpool_returns.npz')
assert np.array_equal(ds.task_indices.numpy(), ret['task_indices'])
P = torch.as_tensor(ret['progress_m'])          # (N,K) nan on invalid
V = torch.as_tensor(ret['valid'])               # (N,K) bool
X = _build_features(env.kin, ds, 4096)          # (N,K,45) raw

n = len(ds)
g = torch.Generator().manual_seed(123)
perm = torch.randperm(n, generator=g)
tr, te = perm[:400], perm[400:]

# valid-only standardization fitted on train tasks
Xtr_valid = X[tr][V[tr]]
mu, sd = Xtr_valid.mean(0), Xtr_valid.std(0).clamp_min(1e-6)
Xz = ((X - mu) / sd).masked_fill(~V.unsqueeze(-1), 0.0).to(device)
Pd = P.to(device); Vd = V.to(device)

def capture(sel_idx, rows):
    r = rows.to(device)
    first = Vd[r].float().argmax(1)
    pr = Pd[r]
    ora = torch.where(Vd[r], pr, torch.tensor(-torch.inf, device=device)).max(1).values
    s = pr[torch.arange(len(r), device=device), sel_idx]
    f = pr[torch.arange(len(r), device=device), first]
    return float((s - f).sum() / (ora - f).sum() * 100), float(s.mean())

# within-task range-normalized target (their listwise construction)
def norm_target(rows):
    pr, v = Pd[rows.to(device)], Vd[rows.to(device)]
    lo = torch.where(v, pr, torch.tensor(torch.inf, device=device)).min(1, keepdim=True).values
    hi = torch.where(v, pr, torch.tensor(-torch.inf, device=device)).max(1, keepdim=True).values
    t = (pr - lo) / (hi - lo).clamp_min(1e-6)
    return t.masked_fill(~v, 0.0)

results = {}

# ---- model 1: ridge (least squares w/ L2) on raw progress
A = Xz[tr][Vd[tr].cpu()] if False else Xz[tr.to(device)][Vd[tr.to(device)]]
y = Pd[tr.to(device)][Vd[tr.to(device)]]
lam = 1e-3
W = torch.linalg.solve(A.T @ A + lam * torch.eye(45, device=device), A.T @ y)
scores = (Xz @ W).masked_fill(~Vd, -torch.inf)
ridge_cap, ridge_mean = capture(scores[te.to(device)].argmax(1), te)
results['ridge'] = {'test_capture_pct': ridge_cap, 'test_mean_m': ridge_mean}

# ---- model 2: listwise MLP (softmax CE over candidates, temperature 0.1)
mlp = nn.Sequential(nn.Linear(45, 256), nn.ReLU(), nn.Linear(256, 256),
                    nn.ReLU(), nn.Linear(256, 1)).to(device)
opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-5)
T = norm_target(tr)
trd = tr.to(device)
for epoch in range(300):
    opt.zero_grad()
    logits = mlp(Xz[trd]).squeeze(-1).masked_fill(~Vd[trd], -torch.inf)
    target = torch.softmax((T / 0.1).masked_fill(~Vd[trd], -torch.inf), dim=1)
    loss = -(target * torch.log_softmax(logits, dim=1).clamp_min(-30)).sum(1).mean()
    loss.backward(); opt.step()
with torch.no_grad():
    s_te = mlp(Xz[te.to(device)]).squeeze(-1).masked_fill(~Vd[te.to(device)], -torch.inf)
    s_tr = mlp(Xz[trd]).squeeze(-1).masked_fill(~Vd[trd], -torch.inf)
mlp_cap, mlp_mean = capture(s_te.argmax(1), te)
mlp_cap_train, _ = capture(s_tr.argmax(1), tr)
# held-out within-task spearman
rho = []
for i in te.tolist():
    v = V[i].numpy()
    if v.sum() >= 3:
        with torch.no_grad():
            sc = mlp(Xz[i:i+1]).squeeze().cpu().numpy()
        s = spearmanr(sc[v], P[i].numpy()[v]).statistic
        if np.isfinite(s):
            rho.append(s)
results['mlp_listwise'] = {
    'train_capture_pct': mlp_cap_train, 'test_capture_pct': mlp_cap,
    'test_mean_m': mlp_mean, 'test_spearman_median': float(np.median(rho))}

# references on the SAME test tasks
ted = te.to(device)
first_cap, first_mean = capture(Vd[ted].float().argmax(1), te)
results['first_valid'] = {'test_capture_pct': first_cap, 'test_mean_m': first_mean}
ora_sel = torch.where(Vd[ted], Pd[ted], torch.tensor(-torch.inf, device=device)).argmax(1)
_, ora_mean = capture(ora_sel, te)
results['oracle_mean_m'] = ora_mean
print(json.dumps(results, indent=1))
(OUT / 'ikpool_retrain.json').write_text(json.dumps(results, indent=1))

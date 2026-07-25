"""E2: capture vs training-size scaling curve on the full IK pool (decision expt).

Fixed geometry-disjoint held-out TEST set; vary the number of FIT tasks the
selector sees. Held-out capture = fraction of (oracle - first_valid) headroom
recovered on TEST. If it climbs to >=45% by 15k tasks -> the IK-pool line is
real; <35% -> line downgraded.

Model: listwise MLP (softmax CE over candidates, range-normalized target),
same family as the production selector; ridge as a cheap reference. Feature
normalization is fit on each training subset's valid-only moments (the diffusion
normalization is what made transfer fail).
"""
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from scipy.stats import spearmanr

from Yuan.unified_rl.candidate_batch import CachedSeedCandidateDataset
from Yuan.unified_rl.checkpoint import build_env_from_run, resolve_controller_dir
from Yuan.unified_rl.offline_seed_ensemble_train import _build_features

D = Path('Yuan/unified_rl/runs/ikpool_full_v1')
OUT = D / 'ikpool_e2_scaling.json'
device = torch.device('cuda:0')
N_TEST = 3000
SIZES = [1000, 2000, 5000, 10000, 15000]
SEEDS = [0, 1, 2]
EPOCHS, TEMP, WD = 250, 0.1, 1e-4

env = build_env_from_run(resolve_controller_dir('Yuan/unified_rl/runs/r2_grouped_best'), 1, device)
ds = CachedSeedCandidateDataset.from_npz(D / 'ikpool_candidates.npz')
ret = np.load(D / 'ikpool_returns.npz')
assert np.array_equal(ds.task_indices.numpy(), ret['task_indices'])
X = _build_features(env.kin, ds, 4096).to(device)          # (N,K,45)
P = torch.as_tensor(ret['progress_m']).to(device)          # (N,K) nan invalid
V = torch.as_tensor(ret['valid']).to(device)               # (N,K) physical valid
N, K = P.shape
P = torch.nan_to_num(P, nan=0.0)

# geometry-disjoint split by exact float32 (p0,line_dir,n_target) signature
sig = np.concatenate([ds.batch.p0.numpy(), ds.batch.line_dir.numpy(),
                      ds.batch.n_target.numpy()], axis=1)
_, gid = np.unique(np.round(sig, 6), axis=0, return_inverse=True)
rng0 = np.random.default_rng(20260724)
groups = rng0.permutation(gid.max() + 1)
test_groups = set(groups[:0].tolist())  # placeholder
# assign whole geometry groups to test until N_TEST rows reached
order = rng0.permutation(N)
is_test = np.zeros(N, bool)
seen_g, cnt = set(), 0
for r in order:
    if cnt >= N_TEST:
        break
    g = gid[r]
    if g in seen_g:
        continue
    seen_g.add(g)
    is_test[gid == g] = True
    cnt = is_test.sum()
test = torch.as_tensor(np.nonzero(is_test)[0], device=device)
fitpool = np.nonzero(~is_test)[0]
assert not (set(gid[is_test]) & set(gid[~is_test])), 'geometry leak'
print(f'N={N} test={len(test)} fitpool={len(fitpool)} geometries={gid.max()+1}', flush=True)


def capture(sel, rows):
    r = rows
    first = V[r].float().argmax(1)
    pr = P[r]
    ora = torch.where(V[r], pr, torch.tensor(-1e9, device=device)).max(1).values
    idx = torch.arange(len(r), device=device)
    s, f = pr[idx, sel], pr[idx, first]
    return float((s - f).sum() / (ora - f).sum() * 100), float(s.mean())


def norm_target(rows):
    pr, v = P[rows], V[rows]
    lo = torch.where(v, pr, torch.tensor(1e9, device=device)).min(1, keepdim=True).values
    hi = torch.where(v, pr, torch.tensor(-1e9, device=device)).max(1, keepdim=True).values
    return ((pr - lo) / (hi - lo).clamp_min(1e-6)).masked_fill(~v, 0.0)


def train_mlp(trows, seed):
    torch.manual_seed(seed)
    mlp = nn.Sequential(nn.Linear(45, 256), nn.ReLU(), nn.Linear(256, 256),
                        nn.ReLU(), nn.Linear(256, 1)).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=WD)
    mu = X[trows][V[trows]].mean(0)
    sd = X[trows][V[trows]].std(0).clamp_min(1e-6)
    Xz = ((X - mu) / sd).masked_fill(~V.unsqueeze(-1), 0.0)
    T = norm_target(trows)
    for _ in range(EPOCHS):
        opt.zero_grad()
        logit = mlp(Xz[trows]).squeeze(-1).masked_fill(~V[trows], -1e9)
        tgt = torch.softmax((T / TEMP).masked_fill(~V[trows], -1e9), 1)
        loss = -(tgt * torch.log_softmax(logit, 1).clamp_min(-30)).sum(1).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        s_te = mlp(Xz[test]).squeeze(-1).masked_fill(~V[test], -1e9)
        s_tr = mlp(Xz[trows]).squeeze(-1).masked_fill(~V[trows], -1e9)
    return s_te, s_tr, mu, sd


# references on TEST
first_cap, first_mean = capture(V[test].float().argmax(1), test)
ora_sel = torch.where(V[test], P[test], torch.tensor(-1e9, device=device)).argmax(1)
_, ora_mean = capture(ora_sel, test)
results = {'n_test': int(len(test)), 'first_valid_mean_m': first_mean,
           'oracle_mean_m': ora_mean, 'headroom_mm': (ora_mean - first_mean) * 1e3,
           'sizes': {}}
print(f'TEST first={first_mean:.4f} oracle={ora_mean:.4f} headroom={(ora_mean-first_mean)*1e3:.1f}mm', flush=True)

rng = np.random.default_rng(999)
for size in SIZES:
    per_seed, per_seed_m, train_caps, logits_te = [], [], [], []
    for seed in SEEDS:
        sub = torch.as_tensor(
            rng.choice(fitpool, size=min(size, len(fitpool)), replace=False), device=device)
        s_te, s_tr, _, _ = train_mlp(sub, seed)
        c_te, m_te = capture(s_te.argmax(1), test)
        c_tr, _ = capture(s_tr.argmax(1), sub)
        per_seed.append(c_te); per_seed_m.append(m_te); train_caps.append(c_tr)
        logits_te.append(s_te)
    ens = torch.stack(logits_te).mean(0)
    ens_cap, ens_m = capture(ens.argmax(1), test)
    results['sizes'][str(size)] = {
        'test_capture_mean': float(np.mean(per_seed)),
        'test_capture_std': float(np.std(per_seed)),
        'test_capture_per_seed': [round(x, 2) for x in per_seed],
        'test_mean_m': float(np.mean(per_seed_m)),
        'ensemble3_capture': ens_cap, 'ensemble3_mean_m': ens_m,
        'train_capture_mean': float(np.mean(train_caps)),
    }
    print(f'size={size:>5}: test_cap={np.mean(per_seed):5.1f}%±{np.std(per_seed):.1f} '
          f'ens3={ens_cap:5.1f}% mean={ens_m:.4f}m train_cap={np.mean(train_caps):.1f}%', flush=True)

OUT.write_text(json.dumps(results, indent=1))
print('\n' + json.dumps(results['sizes'], indent=1))

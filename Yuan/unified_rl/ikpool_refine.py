"""E3 refinement (steps 1+2): stronger selector + fail-closed margin gate.

Step 1: permutation-equivariant 5-member set-selector (mean-pool context +
        per-candidate score + feasibility-metre head), trained on the full
        18,432-task IK pool. Push capture above the plain-MLP 66.6%.
Step 2: fail-closed margin gate. Fallback = classical IK seed (q0_pilot slot,
        else first-valid). Deploy the selector pick only if its predicted
        progress margin over the fallback exceeds a threshold tuned on ONE
        held-out set (validation) and applied to the OTHER (external), and
        vice-versa -- no tuning on the reported set.

Pure IK, no diffusion at any point. Old-system numbers reused from cache.
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
OLD = {'validation': 'Yuan/unified_rl/runs/r2_seed_ensemble_v1_seed31000/eval_validation_cmp1024.npz',
       'external': 'Yuan/unified_rl/runs/r2_seed_ensemble_v1_seed31000/eval_external_dev_cmp1024.npz'}
MEMBERS, EPOCHS, TEMP, WD = 5, 300, 0.1, 1e-4
kin = build_env_from_run(resolve_controller_dir('Yuan/unified_rl/runs/r2_grouped_best'), 1, dev).kin


class SetSel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(45, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.score = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1))
        self.feas = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, X, V):
        e = self.enc(X)
        vf = V.unsqueeze(-1).float()
        ctx = (e * vf).sum(1) / vf.sum(1).clamp_min(1)
        h = torch.cat([e, ctx.unsqueeze(1).expand(-1, e.shape[1], -1)], -1)
        return self.score(h).squeeze(-1), self.feas(h).squeeze(-1)


def load_pool(which):
    ds = CachedSeedCandidateDataset.from_npz(D / f'ikpool_{which}_candidates.npz'
                                             if which != 'train' else D / 'ikpool_candidates.npz')
    ret = np.load(D / (f'ikpool_{which}_returns.npz' if which != 'train' else 'ikpool_returns.npz'))
    X = _build_features(kin, ds, 4096).to(dev)
    P = torch.nan_to_num(torch.as_tensor(ret['progress_m']), nan=0.0).to(dev)
    V = torch.as_tensor(ret['valid']).to(dev)
    fb = ds.fallback_index if ds.fallback_index is not None else None
    return X, P, V, ret['task_indices'], fb


Xtr, Ptr, Vtr, _, _ = load_pool('train')
mu, sd = Xtr[Vtr].mean(0), Xtr[Vtr].std(0).clamp_min(1e-6)


def znorm(X, V):
    return ((X - mu) / sd).masked_fill(~V.unsqueeze(-1), 0.0)


def train_member(seed):
    g = torch.Generator(device='cpu').manual_seed(seed)
    boot = torch.randint(0, len(Xtr), (len(Xtr),), generator=g).to(dev)  # bootstrap tasks
    Xz = znorm(Xtr, Vtr)[boot]; V = Vtr[boot]; P = Ptr[boot]
    lo = torch.where(V, P, torch.tensor(1e9, device=dev)).min(1, keepdim=True).values
    hi = torch.where(V, P, torch.tensor(-1e9, device=dev)).max(1, keepdim=True).values
    T = ((P - lo) / (hi - lo).clamp_min(1e-6)).masked_fill(~V, 0.0)
    torch.manual_seed(seed)
    net = SetSel().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=WD)
    hub = nn.HuberLoss(delta=0.05, reduction='none')
    for _ in range(EPOCHS):
        opt.zero_grad()
        s, f = net(Xz, V)
        s = s.masked_fill(~V, -1e9)
        tgt = torch.softmax((T / TEMP).masked_fill(~V, -1e9), 1)
        rank = -(tgt * torch.log_softmax(s, 1).clamp_min(-30)).sum(1).mean()
        feas = (hub(f, P) * V.float()).sum() / V.float().sum()
        (rank + feas).backward(); opt.step()
    return net


nets = [train_member(1000 * (m + 1)) for m in range(MEMBERS)]


@torch.no_grad()
def infer(which):
    X, P, V, tids, fb = load_pool(which)
    Xz = znorm(X, V)
    ss, ff = [], []
    for net in nets:
        s, f = net(Xz, V)
        ss.append(s.masked_fill(~V, -1e9)); ff.append(f)
    score = torch.stack(ss).mean(0); feas = torch.stack(ff).mean(0)
    pick = score.argmax(1)
    idx = torch.arange(len(P), device=dev)
    ik_oracle = torch.where(V, P, torch.tensor(-1e9, device=dev)).max(1).values
    first = V.float().argmax(1)
    # fallback = classical IK seed slot if valid else first-valid
    fb_idx = torch.full((len(P),), fb if fb is not None else 0, device=dev, dtype=torch.long)
    fb_valid = V[idx, fb_idx] if fb is not None else torch.zeros(len(P), dtype=torch.bool, device=dev)
    fb_idx = torch.where(fb_valid, fb_idx, first)
    margin = feas[idx, pick] - feas[idx, fb_idx]
    o = np.load(OLD[which], allow_pickle=True)
    order = {int(t): i for i, t in enumerate(o['task_indices'])}
    perm = np.array([order[int(t)] for t in tids])
    old_pol = np.nan_to_num(o['policy_progress_m'])[perm]
    return {'P': P, 'idx': idx, 'pick': pick, 'fb_idx': fb_idx, 'margin': margin,
            'first': P[idx, first].cpu().numpy(), 'ik_oracle': ik_oracle.cpu().numpy(),
            'old': old_pol, 'new_nogate': P[idx, pick].cpu().numpy(), 'tids': tids}


def stats(new, old, first, ik_oracle):
    d = new - old
    tr = np.sort(d); k = int(0.05 * len(d)); trm = tr[k:-k] if k else tr
    return dict(new_m=float(new.mean()), old_m=float(old.mean()),
               delta_mm=float(d.mean() * 1e3), trimmed_mm=float(trm.mean() * 1e3),
               harm_pct=float((d < -1e-3).mean() * 100), win_pct=float((d > 1e-3).mean() * 100),
               capture_pct=float((new - first).sum() / (ik_oracle - first).sum() * 100))


R = {w: infer(w) for w in ('validation', 'external')}
report = {'step1_no_gate': {}, 'step2_gated': {}}
for w in ('validation', 'external'):
    r = R[w]
    report['step1_no_gate'][w] = stats(r['new_nogate'], r['old'], r['first'], r['ik_oracle'])

# gate: tune threshold on the OTHER set, apply here
def apply_gate(r, thr):
    use = (r['margin'] >= thr)
    sel = torch.where(use, r['pick'], r['fb_idx'])
    return r['P'][r['idx'], sel].cpu().numpy()

thr_grid = np.quantile(np.concatenate([R['validation']['margin'].cpu().numpy(),
                                        R['external']['margin'].cpu().numpy()]),
                       np.linspace(0, 0.6, 25))
for report_set, tune_set in [('validation', 'external'), ('external', 'validation')]:
    # tune on tune_set: max net delta s.t. harm <= 12%
    best = None
    for thr in thr_grid:
        ng = apply_gate(R[tune_set], thr)
        s = stats(ng, R[tune_set]['old'], R[tune_set]['first'], R[tune_set]['ik_oracle'])
        if s['harm_pct'] <= 12.0 and (best is None or s['delta_mm'] > best[1]):
            best = (float(thr), s['delta_mm'])
    thr = best[0] if best else float(thr_grid[-1])
    gated = apply_gate(R[report_set], thr)
    st = stats(gated, R[report_set]['old'], R[report_set]['first'], R[report_set]['ik_oracle'])
    st['gate_threshold'] = thr; st['tuned_on'] = tune_set
    report['step2_gated'][report_set] = st

print(json.dumps(report, indent=1))
(D / 'ikpool_refine.json').write_text(json.dumps(report, indent=1))

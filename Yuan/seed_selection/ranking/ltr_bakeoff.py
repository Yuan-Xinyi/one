"""Learning-to-rank bake-off for the seed ranker, all textbook methods with
citations, single model each, identical conditions (no fallback, IK-fixed,
24 candidates, 60k training tasks). Plus an ensemble ablation on the winner.

Methods (each = a named loss from the LTR literature):
  pointwise   : L2 regression on length                       (baseline)
  ranknet     : pairwise logistic                             Burges et al. 2005
  listnet     : top-1 softmax cross-entropy (listwise)        Cao  et al. 2007
  listmle     : Plackett-Luce permutation likelihood          Xia  et al. 2008
  lambdarank  : RankNet gradient weighted by |Delta NDCG|     Burges     2010
  softmax_ce  : multi-positive InfoNCE / SupCon               Oord 2018 / Khosla 2020

Selection metric = held-out capture on tasks with >=2 valid candidates.
Deployment metric = % of reference length on the 10k eval set (IK-fixed,
no fallback).
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
P0 = Path('Yuan/seed_selection/runs/rank_phase0')
TRDIRS = [Path('Yuan/seed_selection/runs/rank_train'),
          Path('Yuan/seed_selection/runs/rank_train_b'),
          Path('Yuan/seed_selection/runs/rank_train_c')]
K = 8
dev = torch.device('cuda')

oh = np.load('Yuan/system_eval/runs/eval_10k_systematic/'
             'cell_oracle_hyb_results.npz')['L_best'].astype(np.float32) * 1.5
fin = oh > 1e-6
bucket = np.where(oh >= 0.80, 'Easy', np.where(oh >= 0.45, 'Medium', 'Difficult'))


def load_train(d):
    cd = np.load(d / 'candidates_K8.npz')
    fv = np.load(d / 'feat_v2.npz')
    sd = d / 'final'
    L = np.stack([np.load(sd / f'L_slot{si}.npz')['L'] for si in range(K)], 1) * 1.5
    obs = fv['obs_slots']
    mu = fv['mu_slots']
    X = np.concatenate([obs, np.log(mu[..., None] + 1e-9)], -1)
    return X.astype(np.float32), L.astype(np.float32), cd['ik_ok']


P = [load_train(d) for d in TRDIRS]
X = np.concatenate([p[0] for p in P])
y = np.concatenate([p[1] for p in P])
ok = np.concatenate([p[2] for p in P])
N = len(X)
rng = np.random.default_rng(0)
perm = rng.permutation(N)
nv = N // 10
va, tr = perm[:nv], perm[nv:]
mean = X[tr][ok[tr]].mean(0)
std = X[tr][ok[tr]].std(0) + 1e-6
Xn = (X - mean) / std
ysd = float(y[tr][ok[tr]].std())
Xd = torch.from_numpy(Xn[tr]).to(dev)
yd = torch.from_numpy(y[tr]).to(dev)               # raw meters (for NDCG gains)
ydn = yd / ysd
okd = torch.from_numpy(ok[tr]).to(dev)

# IK-fixed, no-fallback eval arrays
c = np.load(P0 / 'ikfix_nofallback.npz')
ok24, L24 = c['ok24'], c['L24']
obs24 = c['obs24']
mu24 = c['mu24']
X10 = np.concatenate([obs24, np.log(mu24[..., None] + 1e-9)], -1)
X10n = torch.from_numpy((X10 - mean) / std).float()


class Rank(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 512), nn.ReLU(),
                                 nn.Linear(512, 512), nn.ReLU(),
                                 nn.Linear(512, 256), nn.ReLU(),
                                 nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


NEG = -1e9


def loss_fn(kind, pred, yy, okk):
    """pred (B,K) scores; yy (B,K) lengths in meters; okk (B,K) valid mask."""
    yv = yy.masked_fill(~okk, NEG)
    if kind == 'pointwise':
        return (((pred - yy / ysd) ** 2) * okk).sum() / okk.sum().clamp(min=1)
    if kind == 'listnet':
        logp = torch.log_softmax(pred.masked_fill(~okk, NEG), -1)
        tgt = torch.softmax(yv / 0.05, -1)
        return -(tgt * logp).sum(-1).mean()
    if kind == 'softmax_ce':          # multi-positive InfoNCE / SupCon
        best = yv.max(-1, keepdim=True).values
        pos = okk & (yy >= best * 0.97)
        logp = torch.log_softmax(pred.masked_fill(~okk, NEG), -1)
        return -((logp * pos).sum(-1) / pos.sum(-1).clamp(min=1)).mean()
    if kind == 'listmle':             # Plackett-Luce (Xia 2008)
        # sort by length desc; likelihood of that permutation under scores
        B, Kk = pred.shape
        order = yv.argsort(dim=-1, descending=True)
        ps = torch.gather(pred, 1, order)
        vs = torch.gather(okk.float(), 1, order)
        ps = ps.masked_fill(vs == 0, NEG)
        # log P = sum_i [ s_i - logsumexp(s_i..s_K) ] over valid prefix
        rev_lse = torch.flip(torch.logcumsumexp(torch.flip(ps, [-1]), -1), [-1])
        ll = (ps - rev_lse) * vs
        return -(ll.sum(-1) / vs.sum(-1).clamp(min=1)).mean()
    # pairwise family: RankNet / LambdaRank
    dy = yy.unsqueeze(2) - yy.unsqueeze(1)          # (B,K,K) length diff i-j
    dp = pred.unsqueeze(2) - pred.unsqueeze(1)      # score diff i-j
    valid_pair = okk.unsqueeze(2) & okk.unsqueeze(1) & (dy > 0)   # i better than j
    if valid_pair.sum() == 0:
        return pred.sum() * 0
    # RankNet target P_ij = 1 (i ranked above j); loss = softplus(-(s_i-s_j))
    base = torch.nn.functional.softplus(-dp)
    if kind == 'ranknet':
        w = torch.ones_like(base)
    else:  # lambdarank: weight by |Delta NDCG| from swapping i,j
        yn = yy.masked_fill(~okk, 0.0)
        gain = (2 ** (yn / (yn.max() + 1e-6) * 5) - 1)          # graded gain
        # rank positions by current scores
        ranks = pred.masked_fill(~okk, NEG).argsort(dim=-1, descending=True).argsort(-1).float()
        disc = 1.0 / torch.log2(ranks + 2.0)
        dcg_i = (gain.unsqueeze(2) * disc.unsqueeze(2))
        dcg_j = (gain.unsqueeze(1) * disc.unsqueeze(1))
        w = (dcg_i - dcg_j).abs() + 1e-3
    return (base * w * valid_pair).sum() / valid_pair.sum()


def fit(kind, seed, epochs=60):
    torch.manual_seed(seed)
    net = Rank().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Ntr = Xd.shape[0]
    for ep in range(epochs):
        order = torch.randperm(Ntr, device=dev)
        for s in range(0, Ntr, 1024):
            ti = order[s:s + 1024]
            pred = net(Xd[ti].reshape(-1, 32)).view(len(ti), K)
            loss = loss_fn(kind, pred, yd[ti], okd[ti])
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    return net


okv, yv_ = ok[va], y[va]
first_v = yv_[np.arange(len(va)), np.argmax(okv, 1)]
best_v = np.where(okv, yv_, -np.inf).max(1)
m2 = okv.sum(1) >= 2
Xva = torch.from_numpy(Xn[va]).float().to(dev)


@torch.no_grad()
def capture(net):
    sc = net(Xva.reshape(-1, 32)).view(len(va), K).cpu().numpy()
    pick = np.where(okv, sc, -np.inf).argmax(1)
    Lp = yv_[np.arange(len(va)), pick]
    return 100 * (Lp[m2] - first_v[m2]).sum() / (best_v[m2] - first_v[m2]).sum()


@torch.no_grad()
def eval10k(nets):
    sc = np.zeros((len(L24), 24), np.float32)
    flat = X10n.reshape(-1, 32)
    for net in nets:
        out = [net(flat[s:s + 65536].to(dev)).cpu() for s in range(0, len(flat), 65536)]
        sc += torch.cat(out).view(len(L24), 24).numpy()
    pick = np.where(ok24, sc, -np.inf).argmax(1)
    Lp = np.where(ok24.any(1), L24[np.arange(len(L24)), pick], 0.0)
    p = lambda mm: 100 * (Lp[mm] / oh[mm]).mean()
    return p(fin), p(fin & (bucket == 'Easy')), p(fin & (bucket == 'Medium')), p(fin & (bucket == 'Difficult'))


METHODS = ['pointwise', 'ranknet', 'listnet', 'listmle', 'lambdarank', 'softmax_ce']
print(f"{'method':12s} {'capture':>8s}  {'All':>5s} {'Easy':>5s} {'Med':>5s} {'Diff':>6s}", flush=True)
results = {}
for kind in METHODS:
    net = fit(kind, 0)
    cap = capture(net)
    a, e, m, dd = eval10k([net])
    results[kind] = (cap, a, net)
    print(f"{kind:12s} {cap:7.1f}%  {a:5.1f} {e:5.1f} {m:5.1f} {dd:6.1f}", flush=True)

# ensemble ablation on the best single method (answers: is the drop from
# switching models, or from removing the fallback?)
winner = max(results, key=lambda k: results[k][1])
print(f"\n[ablation] ensembles of the best method ({winner}):", flush=True)
nets = [results[winner][2]]
for sd in range(1, 10):
    nets.append(fit(winner, sd))
    if len(nets) in (1, 3, 5, 10):
        a, e, m, dd = eval10k(nets)
        print(f"  ens{len(nets):2d}: All {a:.1f}  E {e:.1f} M {m:.1f} D {dd:.1f}", flush=True)
print("[done]", flush=True)

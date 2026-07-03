"""Seed-ranking Phase 2+3: fit rankers, select by top-1 regret, apply to 10k.

Phase 2 (train data from rank_train/): pointwise L-regression vs within-task
pairwise ranking, vs baselines (first-valid / random / min-qn / V_cls / V_pi).
Phase 3: best ranker picks among the Phase-0 candidates on the 10k eval set;
score by slicing cached per-slot L (deterministic env). Reports paper pct.
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
from Yuan.RL_controller.env.line_distribution import ScriptedLineDistribution
from Yuan.system_eval.rollout_controllers import build_env
from Yuan.RL_controller.self_improve.vcls import VClsNet

TRAIN = Path('Yuan/seed_selection/runs/rank_train')
P0DIR = Path('Yuan/seed_selection/runs/rank_phase0')
CKPT = Path('Yuan/RL_controller/runs/distill_r12m_b0.965_soup2')
VSW = Path('Yuan/RL_controller/runs/distill_r11_belt0.965/vswitch')
dev = torch.device('cuda')
K = 8

# ---------- load train data ----------
cd = np.load(TRAIN / 'candidates_K8.npz')
ik_ok = cd['ik_ok']
L = np.stack([np.load(TRAIN / f'L_slot{si}.npz')['L'] for si in range(K)], 1)
obs0 = np.load(TRAIN / 'obs0_K8.npz')['obs0']          # (N, K, 31)
N = L.shape[0]
qn = np.abs(obs0[..., :7]).max(-1)                      # (N, K)
print(f'[p2] train {N} tasks, IK ok {100*ik_ok.mean():.1f}%')

rng = np.random.default_rng(0)
perm = rng.permutation(N)
n_val = N // 5
va_t, tr_t = perm[:n_val], perm[n_val:]


def flat(x, tidx):
    sel = x[tidx]
    return sel.reshape(-1, *x.shape[2:])

X_tr = torch.from_numpy(flat(obs0, tr_t)).float()
y_tr = torch.from_numpy(flat(L, tr_t)).float()
m_tr = torch.from_numpy(flat(ik_ok, tr_t))
task_tr = torch.arange(len(tr_t)).repeat_interleave(K)
X_tr, y_tr, task_tr = X_tr[m_tr], y_tr[m_tr], task_tr[m_tr]
sd = float(y_tr.std())


class Rank(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(31, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU(),
                                 nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit(loss_kind):
    net = Rank().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xd, yd, td = X_tr.to(dev), (y_tr / sd).to(dev), task_tr.to(dev)
    for ep in range(50):
        order = torch.randperm(len(Xd), device=dev)
        for s in range(0, len(order), 8192):
            i = order[s:s + 8192]
            pred = net(Xd[i])
            if loss_kind == 'point':
                loss = ((pred - yd[i]) ** 2).mean()
            else:  # pairwise within batch, same task
                ti = td[i]
                same = ti.unsqueeze(0) == ti.unsqueeze(1)
                dy = yd[i].unsqueeze(0) - yd[i].unsqueeze(1)
                dp = pred.unsqueeze(0) - pred.unsqueeze(1)
                pair = same & (dy > 0.02)   # y_j > y_i by margin (scaled units)
                if pair.sum() == 0:
                    continue
                loss = torch.relu(0.1 - dp[pair.T]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return net


def load_vnet(path):
    ck = torch.load(path, map_location=dev, weights_only=False)
    net = VClsNet().to(dev)
    net.load_state_dict(ck['state_dict'])
    net.eval()
    return lambda o: net(o) * float(ck['target_std']) + float(ck['target_mean'])


@torch.no_grad()
def score_all(fn, obs):
    out = np.zeros(obs.shape[:2], dtype=np.float32)
    for si in range(obs.shape[1]):
        for s in range(0, obs.shape[0], 65536):
            e = min(s + 65536, obs.shape[0])
            out[s:e, si] = fn(torch.from_numpy(obs[s:e, si]).float().to(dev)
                              ).float().cpu().numpy()
    return out


def evaluate(scores, L_, ok_, name, L_first, L_best):
    s = np.where(ok_, scores, -np.inf)
    pick = s.argmax(1)
    has = ok_.any(1)
    Lp = np.where(has, L_[np.arange(len(L_)), pick], L_first)
    regret = (L_best[has] - Lp[has]).mean()
    denom = (L_best - L_first)[has]
    capt = ((Lp - L_first)[has].sum() / max(denom.sum(), 1e-9))
    print(f'  {name:14s} L {Lp.mean():.4f}  top1-regret {regret:.4f}  '
          f'capture {100*capt:.1f}%')
    return Lp


# ---------- Phase 2 evaluation on held-out train tasks ----------
L_va, ok_va, obs_va, qn_va = L[va_t], ik_ok[va_t], obs0[va_t], qn[va_t]
Lm = np.where(ok_va, L_va, -np.inf)
L_best = Lm.max(1)
first_idx = np.argmax(ok_va, 1)
L_first = L_va[np.arange(len(va_t)), first_idx]
has = ok_va.any(1)
print(f'\n[p2] held-out ({len(va_t)} tasks): first {L_first[has].mean():.4f}  '
      f'best-of-8 {L_best[has].mean():.4f}')

nets = {}
for kind in ('point', 'pair'):
    nets[kind] = fit(kind)
    print(f'[p2] fitted {kind}')

results = {}
for kind, net in nets.items():
    sc = score_all(lambda o, n=net: n(o), obs_va)
    results[kind] = evaluate(sc, L_va, ok_va, f'rank-{kind}', L_first, L_best)
evaluate(-qn_va, L_va, ok_va, 'min-qn', L_first, L_best)
evaluate(rng.random(qn_va.shape).astype(np.float32), L_va, ok_va, 'random',
         L_first, L_best)
for nm, pth in (('V_cls', VSW / 'vcls_sym.pt'), ('V_pi', VSW / 'vpi_sym.pt')):
    fn = load_vnet(pth)
    sc = score_all(fn, obs_va)
    evaluate(sc, L_va, ok_va, nm, L_first, L_best)

# pick winner by held-out mean L
winner = max(nets, key=lambda k: results[k][ok_va.any(1)].mean())
print(f'[p2] winner: rank-{winner}')
torch.save({'state_dict': nets[winner].net.state_dict(), 'scale': sd,
            'kind': winner}, TRAIN / 'ranker.pt')

# ---------- Phase 3: apply to 10k ----------
pd = np.load(P0DIR / 'candidates_K8.npz')
seeds10, ok10 = pd['seeds'], pd['ik_ok']
L10 = np.stack([np.load(P0DIR / f'L_slot{si}.npz')['L'] for si in range(K)], 1)
res0 = np.load(P0DIR / 'phase0_results.npz')
oh, L_first10 = res0['oh'], res0['L_first']
z = np.load('Yuan/system_eval/runs/eval_10k_systematic/eval_set_10k.npz')

# initial obs for eval candidates
obs10_npz = P0DIR / 'obs0_K8.npz'
if obs10_npz.exists():
    obs10 = np.load(obs10_npz)['obs0']
else:
    env = build_env(CKPT / 'config.yaml', 4096, dev)
    dtype = env.kin.dtype
    n10 = seeds10.shape[0]
    obs10 = np.zeros((n10, K, 31), dtype=np.float32)
    lds = z['cs_line_dir'].astype(np.float32)
    nts = z['cs_n_target'].astype(np.float32)
    p0s = z['cs_p0'].astype(np.float32)
    for si in range(K):
        outs = []
        for s in range(0, n10, 4096):
            e = min(s + 4096, n10)
            pad = 4096 - (e - s)
            def _t(x, w):
                t = torch.as_tensor(x[s:e], device=dev, dtype=dtype)
                return torch.cat([t, t[-1:].expand(pad, w)]) if pad else t
            env.line_dist = ScriptedLineDistribution(
                {'q0': _t(seeds10[:, si], 7), 'line_dir': _t(lds, 3),
                 'n_target': _t(nts, 3)})
            env.reset()
            env.p_start[:] = _t(p0s, 3)
            outs.append(env.current_obs()[:e - s].float().cpu())
        obs10[:, si] = torch.cat(outs).numpy()
    np.savez_compressed(obs10_npz, obs0=obs10)
    print('[p3] eval obs cached', flush=True)

net = nets[winner]
sc10 = score_all(lambda o: net(o), obs10)
s = np.where(ok10, sc10, -np.inf)
pick = s.argmax(1)
has10 = ok10.any(1)
L_pick = np.where(has10, L10[np.arange(len(L10)), pick] * 1.5, L_first10)
L_pick_m = np.where(has10, L_pick, L_first10)

fin = oh > 1e-9
def pct(Lm_):
    return 100.0 * (Lm_[fin] / oh[fin]).mean()

Lbest8 = np.where(ok10, L10 * 1.5, -np.inf).max(1)
Lbest8 = np.where(np.isfinite(Lbest8), Lbest8, L_first10)
print(f'\n==== PHASE 3: 10k paper metric ====')
print(f'  first-valid (status quo) {pct(L_first10):.2f}%')
print(f'  ranked (rank-{winner})    {pct(L_pick_m):.2f}%')
print(f'  best-of-8 (ceiling)      {pct(Lbest8):.2f}%')
d10 = (L_pick_m[fin] - L_first10[fin]) / oh[fin] * 100
se = d10.std(ddof=1) / np.sqrt(len(d10))
print(f'  ranked - first: {d10.mean():+.2f}pp ± {se:.2f} '
      f'({"SIGNIFICANT" if abs(d10.mean()) > 2*se else "tie"})')
# tail check
bucket = np.where(oh >= 0.80, 'Easy', np.where(oh >= 0.45, 'Medium', 'Difficult'))
for b in ('Easy', 'Medium', 'Difficult'):
    m = fin & (bucket == b)
    print(f'  {b:9s}: first {100*(L_first10[m]/oh[m]).mean():.1f}%  '
          f'ranked {100*(L_pick_m[m]/oh[m]).mean():.1f}%')
tail = fin & (bucket == 'Easy') & (100*L_first10/np.maximum(oh,1e-9) < 50)
print(f'  Easy tail (<50%): n={tail.sum()}  '
      f'first {100*(L_first10[tail]/oh[tail]).mean():.1f}%  '
      f'ranked {100*(L_pick_m[tail]/oh[tail]).mean():.1f}%')
np.savez_compressed(P0DIR / 'phase3_ranked.npz', L_ranked=L_pick_m, pick=pick,
                    winner=np.str_(winner))
print('[p3] done', flush=True)

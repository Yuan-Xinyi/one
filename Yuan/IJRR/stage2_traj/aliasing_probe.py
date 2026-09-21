"""Does cross-task interference explain the learned controllers' gap?

Three probes on the cached (obs27, one-step margins) dataset, no training
environment needed:

  P1  Neighbor aliasing: for state pairs at matched observation distance,
      does the myopic-optimal vertex disagree more often when the two states
      come from different tasks than from the same task? If cross-task
      aliasing is the failure mode, disagreement must be systematically
      higher across tasks at the same distance, with a margin cost attached.
  P2  Split test: train the same classifier twice -- held-out STATES of seen
      tasks vs held-out TASKS -- and compare top-1 and margin regret. A large
      gap says the map is learnable within tasks but interferes across them;
      no gap says the difficulty is per-state resolution, not task mixing.
  P3  Single-task memorization: train one small net per task on a handful of
      tasks. If even memorizing one task's state->vertex map fails on
      held-out states of that same task, no amount of task separation can
      help a shared policy.

Usage:
    python -m Yuan.IJRR.stage2_traj.aliasing_probe
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[3]
DATA = 'Yuan/IJRR/runs/qsbe/dataset.pt'
D_SLICE = slice(14, 17)   # obs27 layout: qn(7) qn^2(7) d(3) z(3) n(3) z.n(1) zxn(3)
N_SLICE = slice(20, 23)


def best_action(SL, SD):
    m = SL.clone()
    m[SD.bool()] = -1e9
    return m.argmax(-1), m


def margin_regret(m, pred, best):
    """Regret among predictions that pick an alive action; dead picks are
    counted separately by the caller (m holds -1e9 on dead actions)."""
    r = (m.gather(1, best[:, None]) - m.gather(1, pred[:, None])).squeeze(1)
    alive = m.gather(1, pred[:, None]).squeeze(1) > -1e8
    return r[alive], (~alive).float().mean().item()


class Head(nn.Module):
    def __init__(self, hid=256, n_out=16):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(27, hid), nn.ReLU(),
                               nn.Linear(hid, hid), nn.ReLU(),
                               nn.Linear(hid, n_out))

    def forward(self, x):
        return self.f(x)


def train_cls(Xtr, Ytr, Xte, Yte, Mte, dev, epochs=8, hid=256, lr=1e-3,
              seed=0):
    torch.manual_seed(seed)
    net = Head(hid).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(Xtr)
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, 8192):
            b = perm[i:i + 8192]
            loss = nn.functional.cross_entropy(net(Xtr[b].to(dev)),
                                               Ytr[b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = net(Xte.to(dev)).argmax(-1).cpu()
    acc = (pred == Yte).float().mean().item()
    reg, dead_frac = margin_regret(Mte, pred, Yte)
    return acc, reg.mean().item(), dead_frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--pairs', type=int, default=200000)
    a = ap.parse_args()
    dev = torch.device(a.device)
    S, SO, SL, SD = torch.load(REPO / DATA, weights_only=False,
                               map_location='cpu')
    S = S.float()
    Y, M = best_action(SL.float(), SD)
    n_alive = (~SD.bool()).sum(-1)
    multi = n_alive >= 2
    top2 = M[multi].topk(2, -1).values
    print(f'states {len(S)}, actions 16; >=2 alive on {multi.float().mean():.1%}; '
          f'top-1/top-2 margin gap (those states): '
          f'median {(top2[:, 0] - top2[:, 1]).median():.4f}, '
          f'mean {(top2[:, 0] - top2[:, 1]).mean():.4f}')

    # task identity from the (d, n) constants
    dn = torch.cat([S[:, D_SLICE], S[:, N_SLICE]], 1)
    _, task_id = torch.unique(torch.round(dn * 1e5) / 1e5, dim=0,
                              return_inverse=True)
    n_tasks = int(task_id.max()) + 1
    print(f'tasks recovered from (d,n): {n_tasks}')

    # standardized obs for distance computations
    mu, sd = S.mean(0), S.std(0).clamp_min(1e-6)
    Z = ((S - mu) / sd)

    # ---- P1: matched-distance neighbor aliasing --------------------------
    g = torch.Generator().manual_seed(0)
    Zg = Z.to(dev)
    q_idx = torch.randperm(len(S), generator=g)[:4096]
    res = {'same': ([], []), 'cross': ([], [])}
    for lo in range(0, len(q_idx), 256):
        qi = q_idx[lo:lo + 256]
        dmat = torch.cdist(Zg[qi], Zg)                     # (256, N)
        dmat[torch.arange(len(qi)), qi] = 1e9
        same = task_id[qi][:, None] == task_id[None, :]
        for kind, mask in (('same', same), ('cross', ~same)):
            dm = dmat.masked_fill(~mask.to(dev), 1e9)
            dist, nb = dm.min(-1)
            nb = nb.cpu()
            res[kind][0].append(dist.cpu())
            res[kind][1].append((Y[qi] != Y[nb]).float())
    print('\nP1: nearest-neighbor disagreement (raw, unmatched distance)')
    for kind in ('same', 'cross'):
        d_ = torch.cat(res[kind][0]); dis = torch.cat(res[kind][1])
        print(f'  {kind:5s}: mean dist {d_.mean():.3f}, disagree '
              f'{dis.mean():.1%}')
    # matched-distance curve
    d_s, x_s = torch.cat(res['same'][0]), torch.cat(res['same'][1])
    d_c, x_c = torch.cat(res['cross'][0]), torch.cat(res['cross'][1])
    print('  distance-bin | same-task disagree | cross-task disagree')
    for lo_, hi_ in [(0, .1), (.1, .2), (.2, .4), (.4, .8), (.8, 1.6)]:
        ms = (d_s >= lo_) & (d_s < hi_); mc = (d_c >= lo_) & (d_c < hi_)
        ss = f'{x_s[ms].mean():.1%} (n={int(ms.sum())})' if ms.any() else '--'
        cc = f'{x_c[mc].mean():.1%} (n={int(mc.sum())})' if mc.any() else '--'
        print(f'  [{lo_:.1f},{hi_:.1f}) | {ss} | {cc}')

    # ---- P2: state-split vs task-split classification --------------------
    print('\nP2: identical classifier, two splits')
    g2 = torch.Generator().manual_seed(1)
    perm_t = torch.randperm(n_tasks, generator=g2)
    te_tasks = torch.zeros(n_tasks, dtype=torch.bool)
    te_tasks[perm_t[:n_tasks // 5]] = True
    is_te_task = te_tasks[task_id]
    perm_s = torch.randperm(len(S), generator=g2)
    te_states = torch.zeros(len(S), dtype=torch.bool)
    te_states[perm_s[:len(S) // 5]] = True
    for name, te in (('held-out states (seen tasks)', te_states),
                     ('held-out tasks', is_te_task)):
        acc, reg, df = train_cls(Z[~te], Y[~te], Z[te], Y[te], M[te], dev)
        print(f'  {name:30s}: top-1 {acc:.1%}, margin regret {reg:.4f}, '
              f'picked-dead {df:.1%}')

    # ---- P3: single-task memorization ------------------------------------
    print('\nP3: one net per task (memorization ceiling)')
    counts = torch.bincount(task_id, minlength=n_tasks)
    big = torch.nonzero(counts > 400).squeeze(-1)[:8]
    accs, regs, base = [], [], []
    if not len(big):
        print('  no task with >400 states; lowering threshold')
        big = counts.argsort(descending=True)[:8]
    for t in big.tolist():
        idx = torch.nonzero(task_id == t).squeeze(-1)
        g3 = torch.Generator().manual_seed(t)
        pp = idx[torch.randperm(len(idx), generator=g3)]
        n_te = max(len(idx) // 5, 1)
        te, tr = pp[:n_te], pp[n_te:]
        acc, reg, _ = train_cls(Z[tr], Y[tr], Z[te], Y[te], M[te], dev,
                                epochs=60, hid=128)
        maj = torch.bincount(Y[tr], minlength=16).argmax()
        accs.append(acc); regs.append(reg)
        base.append((Y[te] == maj).float().mean().item())
    print(f'  task sizes used: {counts[big].tolist()}')
    print(f'  per-task held-out top-1: mean {np.mean(accs):.1%} '
          f'(range {min(accs):.1%}-{max(accs):.1%}); '
          f'majority-class baseline {np.mean(base):.1%}; '
          f'margin regret {np.mean(regs):.4f}')


if __name__ == '__main__':
    main()

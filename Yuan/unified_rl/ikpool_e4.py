"""E4: formal multi-training-seed table + preregistered publish rule.

Six runs (run_seed 0 = the already-published S0, plus 41000..45000), each a
5-member SetSel ensemble on the full 18,432-task C0 return cache. Every run is
evaluated by cache lookup on validation/external vs the old-SOTA deployed
policy. The published selector for the sealed set is the run with the MEDIAN
mean delta (rule frozen in runs/ikpool_sealed_v1_freeze.json).
"""
import json
import numpy as np
import torch
from pathlib import Path

from Yuan.unified_rl.ikpool_bidir import (
    SetSel, _load_pool, _picks, _paired, D, OLD)
import torch.nn as nn

OUT = Path('Yuan/unified_rl/runs/_multiseed_final')
RUN_SEEDS = [0, 41000, 42000, 43000, 44000, 45000]
MEMBERS, EPOCHS, TEMP, WD = 5, 300, 0.1, 1e-4
dev = torch.device('cuda:0')


def train_run(run_seed, X, P, V):
    hub = nn.HuberLoss(delta=0.05, reduction='none')
    mu, sd = X[V].mean(0), X[V].std(0).clamp_min(1e-6)
    Xz_all = ((X - mu) / sd).masked_fill(~V.unsqueeze(-1), 0.0)
    nets = []
    for m in range(MEMBERS):
        seed = run_seed + 1000 * (m + 1)
        g = torch.Generator().manual_seed(seed)
        boot = torch.randint(0, len(X), (len(X),), generator=g).to(dev)
        Xz, Vb, Pb = Xz_all[boot], V[boot], P[boot]
        lo = torch.where(Vb, Pb, torch.tensor(1e9, device=dev)).min(1, keepdim=True).values
        hi = torch.where(Vb, Pb, torch.tensor(-1e9, device=dev)).max(1, keepdim=True).values
        T = ((Pb - lo) / (hi - lo).clamp_min(1e-6)).masked_fill(~Vb, 0.0)
        torch.manual_seed(seed)
        net = SetSel().to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=WD)
        for _ in range(EPOCHS):
            opt.zero_grad()
            s, f = net(Xz, Vb)
            s = s.masked_fill(~Vb, -1e9)
            tgt = torch.softmax((T / TEMP).masked_fill(~Vb, -1e9), 1)
            rank = -(tgt * torch.log_softmax(s, 1).clamp_min(-30)).sum(1).mean()
            feas = (hub(f, Pb) * Vb.float()).sum() / Vb.float().sum()
            (rank + feas).backward(); opt.step()
        nets.append(net)
    return nets, mu, sd


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _, Xtr, Ptr, Vtr, _ = _load_pool('train', dev)
    pools = {w: _load_pool(w, dev) for w in ('validation', 'external')}
    olds = {}
    for w in pools:
        o = np.load(OLD[w], allow_pickle=True)
        order = {int(t): i for i, t in enumerate(o['task_indices'])}
        tids = pools[w][4]
        olds[w] = np.nan_to_num(o['policy_progress_m'])[
            np.array([order[int(t)] for t in tids])]
    results = {}
    for rs in RUN_SEEDS:
        ck_path = OUT / f'sel_run{rs}.pt'
        if rs == 0:
            src = D / 'ikpool_selector_s0.pt'
            ck = torch.load(src, map_location=dev, weights_only=False)
            nets = []
            for st in ck['members']:
                n = SetSel().to(dev); n.load_state_dict(st); n.eval(); nets.append(n)
            mu, sd = ck['mu'].to(dev), ck['sd'].to(dev)
        elif ck_path.exists():
            ck = torch.load(ck_path, map_location=dev, weights_only=False)
            nets = []
            for st in ck['members']:
                n = SetSel().to(dev); n.load_state_dict(st); n.eval(); nets.append(n)
            mu, sd = ck['mu'].to(dev), ck['sd'].to(dev)
        else:
            nets, mu, sd = train_run(rs, Xtr, Ptr, Vtr)
            torch.save({'members': [n.state_dict() for n in nets],
                        'mu': mu.cpu(), 'sd': sd.cpu()}, ck_path)
        entry = {}
        deltas = []
        for w, (ds, X, P, V, tids) in pools.items():
            pick = _picks(nets, mu, sd, X, V)
            idx = torch.arange(len(P), device=dev)
            new = P[idx, pick].cpu().numpy()
            st = _paired(new, olds[w])
            st['new_mean_m'] = float(new.mean())
            entry[w] = st
            deltas.append(st['delta_mm'])
        entry['mean_delta_mm'] = float(np.mean(deltas))
        results[str(rs)] = entry
        print(f'[e4] run {rs}: val {entry["validation"]["delta_mm"]:+.1f} '
              f'ext {entry["external"]["delta_mm"]:+.1f} '
              f'mean {entry["mean_delta_mm"]:+.1f}', flush=True)
    means = {rs: results[rs]['mean_delta_mm'] for rs in results}
    med = sorted(means.values())[len(means) // 2 - (1 - len(means) % 2)]
    # median of 6 = lower-middle per tie rule toward lower run_seed
    ordered = sorted(means.items(), key=lambda kv: (kv[1], int(kv[0])))
    publish = ordered[(len(ordered) - 1) // 2][0]
    agg = {
        'per_run': results,
        'across_runs': {
            w: {'mean_mm': float(np.mean([results[r][w]['delta_mm'] for r in results])),
                'std_mm': float(np.std([results[r][w]['delta_mm'] for r in results], ddof=1))}
            for w in ('validation', 'external')},
        'publish_rule': 'median mean-delta, ties toward lower run_seed',
        'published_run_seed': publish,
    }
    (OUT / 'ikpool_e4.json').write_text(json.dumps(agg, indent=1))
    print(json.dumps(agg['across_runs'], indent=1))
    print('published run:', publish, flush=True)


if __name__ == '__main__':
    main()

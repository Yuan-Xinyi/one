"""Full system on the fixed FR3 evaluation set: candidates, selector, rollout.

For every evaluation task, generate the K-candidate pool at the task anchor,
roll every candidate to termination under the analytic margin law (this both
labels the pool and gives the oracle/random/first references), then score the
pool with the trained road-table-conditioned ranker and record the stroke of
its top-1 pick. Saves one npz carrying the full per-task candidate matrix so
every selection row of the paper is a slice of this cache.

Usage:
    python -m Yuan.IJRR.eval.fullsys_eval --n-tasks 10000 \
        --ranker Yuan/IJRR/runs/selector_ood/v1/rankers.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from Yuan.IJRR.stage1_seed.selector_ood import (
    build_base_env, gen_candidates, label_family, cand_features,
    road_table, Ranker, K_CAND)
from Yuan.IJRR.eval.mainresult_eval import TASKS

REPO = Path(__file__).resolve().parents[3]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-tasks', type=int, default=10000)
    ap.add_argument('--batch', type=int, default=2048)
    ap.add_argument('--ranker',
                    default='Yuan/IJRR/runs/selector_ood/v1/rankers.pt')
    ap.add_argument('--out',
                    default='Yuan/IJRR/runs/paper_fill/fullsys_10k.npz')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    dev = torch.device(a.device)
    env, model = build_base_env(a.batch, dev)

    t = np.load(REPO / TASKS)
    N = min(a.n_tasks, len(t['cs_p0']))
    p0 = t['cs_p0'][:N].astype(np.float32)
    d = t['cs_line_dir'][:N].astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    n0 = t['cs_n_target'][:N].astype(np.float32)
    n0 /= np.linalg.norm(n0, axis=1, keepdims=True)
    qg = t['q0_seed'][:N].astype(np.float32)

    rng = np.random.default_rng(7)
    cands, n_found = gen_candidates(env, p0, n0, qg, rng)
    print(f'[fullsys] candidates: mean found '
          f'{n_found.astype(float).mean():.2f} of {K_CAND}', flush=True)

    spec = {'q0': torch.tensor(qg), 'line_dir': torch.tensor(d),
            'n_target': torch.tensor(n0), 'p0': torch.tensor(p0)}
    L = label_family(env, model, spec, cands, a.batch)     # (N, K)

    sd = torch.load(REPO / a.ranker, weights_only=False)
    Xc = cand_features(env, cands, spec)
    rt = road_table(spec)
    Xr = torch.tensor(rt.reshape(rt.shape[0], -1))
    M = torch.arange(K_CAND)[None, :] < torch.tensor(n_found)[:, None]
    picks = {}
    for key, cond in (('cond', True), ('nocond', False)):
        net = Ranker(Xc.shape[-1], Xr.shape[-1], conditioned=cond).to(dev)
        net.load_state_dict(sd[key])
        s = net(Xc.float().to(dev), Xr.float().to(dev)).cpu()
        s = torch.where(M, s, torch.full_like(s, -1e9))
        picks[key] = s.argmax(1).numpy()

    np.savez(REPO / a.out, L=L, n_found=n_found, pick_cond=picks['cond'],
             pick_nocond=picks['nocond'], cands=cands)
    sel = L[np.arange(N), picks['cond']]
    best = np.where(M.numpy(), L, -1e9).max(1)
    print(f'[fullsys] selector {sel.mean():.4f} m | generator '
          f'{L[:, 0].mean():.4f} m | oracle {best.mean():.4f} m', flush=True)


if __name__ == '__main__':
    main()

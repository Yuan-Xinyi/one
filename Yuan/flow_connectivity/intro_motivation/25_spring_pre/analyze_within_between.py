"""Re-rank catalog seeds by `between-branch / within-branch` variance ratio.

The initial search (search_clean_tasks.py) ranks seeds by
`L_best.max() / L_best.min()` — i.e. ratio of best-q0 across branches.
That favors seeds with high CEILINGS but ignores within-branch spread.
A seed like 17 has a 13x ceiling ratio but ALSO huge within-branch
variance on its strongest branch, so the ANOVA decomposition ends up
within-dominated (81% within, 19% between).

This script does a focused second pass on the top-K seeds by ceiling
ratio and computes the ANOVA-style decomposition with MORE q0 samples
per branch (default 50). Output: a sorted re-rank by
`frac_between = SS_between / SS_total`. Big `frac_between` means
within-branch is tight relative to the between-branch jumps — i.e. the
"q0 branch is decisive" narrative is cleanest.

Usage:
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/analyze_within_between.py
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/analyze_within_between.py --top-k 30 --n-per-branch 50
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[3]
for _p in (str(_REPO), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from Yuan.flow_connectivity.batched_fr3_kin import BatchedFR3Kinematics  # noqa: E402
from Yuan.flow_connectivity.batched_rollout import _branch_seed_bank  # noqa: E402
from Yuan.flow_connectivity.intro_motivation.v18_smm_core import (  # noqa: E402
    DEDUP_RAD, DEFAULT_H, JOINT_MARGIN,
    as_tensor, enumerate_branches, path_length, project_and_filter,
)
from Yuan.flow_connectivity.intro_motivation.v18_smm_rollout_6dof import (  # noqa: E402
    rollout_lengths_6dof,
)
from Yuan.flow_connectivity.v18_data_prep import _dense_ik_at  # noqa: E402

from task_sampler import CleanFilter, get_clean_task_target_pose  # noqa: E402


CATALOG_DIR = _HERE / 'catalog'


def evaluate_with_anova(seed: int, kin: BatchedFR3Kinematics,
                          cfilter: CleanFilter, n_per_branch: int) -> dict | None:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    try:
        p_tgt, R_tgt, task = get_clean_task_target_pose(seed, kin, rng, cfilter)
    except RuntimeError:
        return None
    path = task['fine_path_pts']
    L_max = float(path_length(path))
    track = as_tensor(path, kin.device)
    pn = as_tensor(task['plane_normal'].astype(np.float32), kin.device)
    p_t = torch.as_tensor(p_tgt, device=kin.device, dtype=torch.float32)
    R_t = torch.as_tensor(R_tgt, device=kin.device, dtype=torch.float32)
    extra = _branch_seed_bank(kin).detach().cpu().numpy()
    Q_seed_t, _ = _dense_ik_at(kin, p_t, R_t, 128, rng, extra_seeds=extra)
    if Q_seed_t.shape[0] == 0:
        return None
    Q = project_and_filter(
        kin, Q_seed_t.detach().cpu().numpy(), p_tgt, R_tgt,
        kin.lmt_lo.detach().cpu().numpy(),
        kin.lmt_up.detach().cpu().numpy(),
        joint_margin=JOINT_MARGIN, dedup_rad=DEDUP_RAD, verbose=False)
    if Q.shape[0] == 0:
        return None
    branches, _ = enumerate_branches(kin, Q, p_tgt, R_tgt, DEFAULT_H)
    if len(branches) < 2:
        return None

    all_q, all_bid = [], []
    for bid, b in enumerate(branches):
        T_b = b['traj'].shape[0]
        idxs = np.linspace(0, T_b - 1, min(n_per_branch, T_b)).astype(int)
        for k in idxs:
            all_q.append(b['traj'][int(k)])
            all_bid.append(bid)
    Q_batch = torch.as_tensor(np.array(all_q, dtype=np.float32),
                                device=kin.device, dtype=torch.float32)
    L_abs = rollout_lengths_6dof(kin, Q_batch, track, pn)
    L_rel = (L_abs / L_max).astype(np.float32)
    bid_arr = np.array(all_bid, dtype=np.int32)

    grand = float(L_rel.mean())
    total_var = float(((L_rel - grand) ** 2).sum())
    between = 0.0; within = 0.0
    means = []; stds = []; bests = []
    for bid in range(len(branches)):
        L = L_rel[bid_arr == bid]
        if L.size == 0:
            continue
        mu = float(L.mean()); sd = float(L.std()); best = float(L.max())
        between += L.size * (mu - grand) ** 2
        within += float(((L - mu) ** 2).sum())
        means.append(mu); stds.append(sd); bests.append(best)
    frac_between = between / max(total_var, 1e-12)
    frac_within = within / max(total_var, 1e-12)
    ceiling_ratio = max(bests) / max(min(bests), 1e-6)
    mean_ratio = max(means) / max(min(means), 1e-6)
    return {
        'seed': seed,
        'nb': len(branches),
        'means': means,
        'stds': stds,
        'bests': bests,
        'frac_between': frac_between,
        'frac_within': frac_within,
        'ceiling_ratio': ceiling_ratio,
        'mean_ratio': mean_ratio,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--catalog', type=str,
                        default=str(CATALOG_DIR / 'clean_tasks.json'))
    parser.add_argument('--top-k', type=int, default=30,
                        help='re-rank the top-K seeds by ceiling ratio')
    parser.add_argument('--n-per-branch', type=int, default=50)
    parser.add_argument('--out', type=str,
                        default=str(CATALOG_DIR / 'within_between_rerank.json'))
    args = parser.parse_args()

    with open(args.catalog) as f:
        cat = json.load(f)
    rows0 = sorted(cat['rows'], key=lambda r: -r['ratio_best_worst'])[:args.top_k]
    seeds = [int(r['seed']) for r in rows0]
    print(f'Re-ranking {len(seeds)} top-ceiling seeds with '
          f'n_per_branch={args.n_per_branch}\n')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kin = BatchedFR3Kinematics(device=device)
    cfilter = CleanFilter(min_base_clearance=cat['meta']['min_base_clearance'],
                           min_z=cat['meta']['min_z'],
                           require_outward_dir=cat['meta']['require_outward_dir'])

    t0 = time.time()
    rows = []
    for k, seed in enumerate(seeds):
        try:
            r = evaluate_with_anova(seed, kin, cfilter, args.n_per_branch)
        except Exception as e:
            print(f'  seed {seed:4d}: error: {e}'); continue
        if r is None:
            continue
        rows.append(r)
        means_s = ' '.join(f'{m:.2f}' for m in r['means'])
        stds_s = ' '.join(f'{s:.2f}' for s in r['stds'])
        elapsed = time.time() - t0
        print(f'  seed {seed:4d}: nb={r["nb"]} '
              f'mean=[{means_s}]  std=[{stds_s}]  '
              f'between={100*r["frac_between"]:.0f}%  '
              f'ceiling_ratio={r["ceiling_ratio"]:.2f}  '
              f'({elapsed:.0f}s, {k+1}/{len(seeds)})')

    rows.sort(key=lambda r: -r['frac_between'])
    with open(args.out, 'w') as f:
        json.dump({'rows': rows, 'n_per_branch': args.n_per_branch}, f, indent=2)
    print(f'\nsaved: {args.out}')
    print(f'\nTop 10 by frac_between (high = tight within, wide between):')
    hdr = f'  {"seed":<6}{"nb":<4}{"between%":<10}{"ceiling":<10}{"means":<28}{"stds":<22}'
    print(hdr)
    for r in rows[:10]:
        ms = ' '.join(f'{m:.2f}' for m in r['means'])
        ss = ' '.join(f'{s:.2f}' for s in r['stds'])
        print(f'  {r["seed"]:<6}{r["nb"]:<4}{100*r["frac_between"]:<10.1f}'
              f'{r["ceiling_ratio"]:<10.2f}[{ms:<26}][{ss:<20}]')


if __name__ == '__main__':
    main()

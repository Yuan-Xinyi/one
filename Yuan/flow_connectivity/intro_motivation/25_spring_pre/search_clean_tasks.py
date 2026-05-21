"""Search seeds that satisfy the geometric clean filter AND give large
SMM-rollout L spread across branches.

For each seed in [start, start + n_seeds), do a lightweight pipeline:
  1. Sample a clean line task (rejection-filtered by task_sampler).
  2. Enumerate SMM branches at the task start pose (6-DOF locked).
  3. For each branch, sample 15 q0 along the arc, run 6-DOF strict
     rollout, take the BEST L_self per branch.
  4. Score the seed by:
        nb       — number of branches (want >= 2)
        ratio    — L_best.max() / max(L_best.min(), eps)
        L_max_b  — strongest branch's L_self
        d_base   — line clearance from origin
        min_z    — lowest z on the extended line

Record everything to `catalog/clean_tasks.json`. Print top candidates
sorted by `ratio`. The intent: pick a seed whose narrative (one branch
clearly dominates) is visually convincing for figs 3 / 4 and whose
geometry is clean enough for the ONE viewer animations.

Usage:
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/search_clean_tasks.py
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/search_clean_tasks.py --start 0 --n-seeds 40
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/search_clean_tasks.py --min-clear 0.40 --min-z 0.25
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
N_Q0_PER_BRANCH_PROBE = 15
N_IK_SEEDS_PROBE = 128


def evaluate_seed(seed: int, kin: BatchedFR3Kinematics,
                   cfilter: CleanFilter) -> dict | None:
    """Run the lightweight pipeline on one seed. Returns a row dict or None
    if the seed is rejected (filter fails, no IK candidates, < 2 branches)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    try:
        p_tgt, R_tgt, task = get_clean_task_target_pose(seed, kin, rng,
                                                         cfilter=cfilter)
    except RuntimeError:
        return None
    path = task['fine_path_pts']
    L_max = float(path_length(path))
    plane_normal = task['plane_normal'].astype(np.float32)
    track = as_tensor(path, kin.device)
    pn = as_tensor(plane_normal, kin.device)

    p_t = torch.as_tensor(p_tgt, device=kin.device, dtype=torch.float32)
    R_t = torch.as_tensor(R_tgt, device=kin.device, dtype=torch.float32)
    extra = _branch_seed_bank(kin).detach().cpu().numpy()
    Q_seed_t, _ = _dense_ik_at(kin, p_t, R_t, N_IK_SEEDS_PROBE, rng,
                                extra_seeds=extra)
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

    # Sample N_Q0_PER_BRANCH_PROBE q0 per branch, run 6-DOF rollout, take best.
    all_q, all_bid = [], []
    for bid, b in enumerate(branches):
        T_b = b['traj'].shape[0]
        idxs = np.linspace(0, T_b - 1, min(N_Q0_PER_BRANCH_PROBE, T_b)).astype(int)
        for k in idxs:
            all_q.append(b['traj'][int(k)])
            all_bid.append(bid)
    Q_batch = torch.as_tensor(np.array(all_q, dtype=np.float32),
                                device=kin.device, dtype=torch.float32)
    L_abs = rollout_lengths_6dof(kin, Q_batch, track, pn)
    L_rel = (L_abs / L_max).astype(np.float32)
    bid_arr = np.array(all_bid, dtype=np.int32)
    L_best_per_branch = np.array([
        float(L_rel[bid_arr == bid].max()) for bid in range(len(branches))
    ], dtype=np.float32)

    L_max_b = float(L_best_per_branch.max())
    L_min_b = float(L_best_per_branch.min())
    ratio = L_max_b / max(L_min_b, 1e-6)
    closed = [bool(b['closed']) for b in branches]

    # Geometry stats (re-compute from the chosen task path).
    pts = (np.linspace(0, 1, 100)[:, None] *
            (path[-1] - path[0])[None, :] + path[0][None, :])
    d_base = float(np.linalg.norm(pts, axis=-1).min())
    z_min = float(pts[:, 2].min())

    return {
        'seed': int(seed),
        'n_branches': int(len(branches)),
        'L_best_per_branch': L_best_per_branch.tolist(),
        'L_max_branch': L_max_b,
        'L_min_branch': L_min_b,
        'ratio_best_worst': float(ratio),
        'closed_per_branch': closed,
        'd_base': d_base,
        'min_z': z_min,
        'p_tgt': p_tgt.tolist(),
        'L_max_path': L_max,
        'plane_normal': plane_normal.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--n-seeds', type=int, default=40)
    parser.add_argument('--min-clear', type=float, default=0.35,
                        help='min line-to-base distance')
    parser.add_argument('--min-z', type=float, default=0.20,
                        help='min z anywhere on the extended line')
    parser.add_argument('--require-outward', action='store_true', default=True)
    parser.add_argument('--no-require-outward', action='store_false',
                        dest='require_outward')
    parser.add_argument('--out', type=str, default=None,
                        help='catalog json path; default catalog/clean_tasks.json')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kin = BatchedFR3Kinematics(device=device)
    cfilter = CleanFilter(min_base_clearance=args.min_clear,
                           min_z=args.min_z,
                           require_outward_dir=args.require_outward)

    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else CATALOG_DIR / 'clean_tasks.json'

    rows = []
    t0 = time.time()
    print(f'searching seeds [{args.start}, {args.start + args.n_seeds})  '
          f'filter: min_clear={args.min_clear:.2f}, min_z={args.min_z:.2f}, '
          f'outward={args.require_outward}')
    for k, seed in enumerate(range(args.start, args.start + args.n_seeds)):
        try:
            row = evaluate_seed(seed, kin, cfilter)
        except Exception as e:
            print(f'  seed {seed:4d}: error: {e}')
            continue
        if row is None:
            continue
        rows.append(row)
        Lb = row['L_best_per_branch']
        Lb_str = ' '.join(f'{x:.3f}' for x in Lb)
        elapsed = time.time() - t0
        print(f'  seed {seed:4d}: nb={row["n_branches"]} '
              f'L=[{Lb_str}] ratio={row["ratio_best_worst"]:.2f} '
              f'd_base={row["d_base"]:.2f} min_z={row["min_z"]:.2f}  '
              f'({elapsed:.0f}s elapsed, {k+1}/{args.n_seeds} done)')

    rows.sort(key=lambda r: -r['ratio_best_worst'])
    with open(out_path, 'w') as f:
        json.dump({
            'meta': {
                'min_base_clearance': args.min_clear,
                'min_z': args.min_z,
                'require_outward_dir': args.require_outward,
                'start': args.start,
                'n_seeds_scanned': args.n_seeds,
                'n_accepted': len(rows),
            },
            'rows': rows,
        }, f, indent=2)
    print(f'\nsaved: {out_path}  ({len(rows)} accepted out of {args.n_seeds})')
    print(f'total time: {time.time() - t0:.0f}s')

    if rows:
        print('\nTop 10 by best/worst ratio:')
        hdr = f'  {"seed":<6}{"nb":<4}{"L_max":<7}{"ratio":<7}{"d_base":<8}{"min_z":<8}'
        print(hdr)
        for r in rows[:10]:
            print(f'  {r["seed"]:<6}{r["n_branches"]:<4}'
                  f'{r["L_max_branch"]:<7.3f}{r["ratio_best_worst"]:<7.2f}'
                  f'{r["d_base"]:<8.2f}{r["min_z"]:<8.2f}')


if __name__ == '__main__':
    main()

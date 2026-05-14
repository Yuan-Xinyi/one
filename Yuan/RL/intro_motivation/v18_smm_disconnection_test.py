"""Are the 4 SMM branches really disconnected, or just JL-cut artifacts?

Direct empirical test (per Lück & Lee 1993): re-walk null(J) from each
WITH-FR3-JL branch's start point, but this time with joint limits
replaced by ±5π (effectively no JL). If two branches' no-JL walks
visit the same closed loop, the original "branches" were one connected
component cut by JL. If their walks stay in disjoint sets, the
branches are topologically distinct (different IK polynomial root
families).

Output:
  * For each WITH-JL branch: closure status, arc length under no-JL walk.
  * Pairwise: minimum 7D distance between no-JL walks.
  * Verdict per pair: SAME loop (JL artifact) vs DIFFERENT (topology).

Usage:
    python -m Yuan.RL.intro_motivation.v18_smm_disconnection_test --seed 118
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.RL.batched_rollout import _branch_seed_bank
from Yuan.RL.intro_motivation.v18_motivation_core import (
    LINE_L_RANGE, TARGET_PATH_M, extend_task_path, sample_line_task,
)
from Yuan.RL.intro_motivation.v18_smm_enumerate import (
    DEDUP_RAD, DEFAULT_H, JOINT_MARGIN,
    enumerate_branches, project_and_filter, walk_branch,
)
from Yuan.RL.v18_data_prep import _build_R_from_normal_direction, _dense_ik_at


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=118)
    parser.add_argument('--n-seeds', type=int, default=256)
    parser.add_argument('--h', type=float, default=DEFAULT_H)
    parser.add_argument('--big-limit-mult', type=float, default=5.0,
                        help='no-JL test uses ±this * pi as fake limits')
    parser.add_argument('--out-png', type=str,
                        default='Yuan/RL/intro_motivation/data/smm_disconnection.png')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kin = BatchedFR3Kinematics(device=device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # Derive target pose from seed (same as branch_comparison / smm_enumerate task mode)
    task = sample_line_task(rng, kin, l_range=LINE_L_RANGE)
    task = extend_task_path(task, TARGET_PATH_M)
    task_path = task['fine_path_pts']
    plane_normal_np = task['plane_normal']
    seg_dir = task_path[1] - task_path[0]
    seg_dir = seg_dir / max(np.linalg.norm(seg_dir), 1e-12)
    R_tgt = _build_R_from_normal_direction(plane_normal_np, seg_dir).astype(np.float32)
    p_tgt = task_path[0].astype(np.float32)
    print(f'seed={args.seed}, p_tgt={p_tgt}')

    p_t = torch.as_tensor(p_tgt, device=device, dtype=torch.float32)
    R_t = torch.as_tensor(R_tgt, device=device, dtype=torch.float32)

    # === Phase 1: enumerate branches WITH FR3 JL (control) ===
    orig_lo = kin.lmt_lo.clone()
    orig_hi = kin.lmt_up.clone()
    extra = _branch_seed_bank(kin).detach().cpu().numpy()
    Q_seed_t, _ = _dense_ik_at(kin, p_t, R_t, args.n_seeds, rng, extra_seeds=extra)
    Q_seed = Q_seed_t.detach().cpu().numpy()
    Q_jl = project_and_filter(
        kin, Q_seed, p_tgt, R_tgt,
        orig_lo.detach().cpu().numpy(),
        orig_hi.detach().cpu().numpy(),
        joint_margin=JOINT_MARGIN, dedup_rad=DEDUP_RAD)
    branches_jl, _ = enumerate_branches(kin, Q_jl, p_tgt, R_tgt, args.h)
    print(f'\n=== WITH FR3 JL: {len(branches_jl)} branches ===')
    for bid, b in enumerate(branches_jl):
        arc = float(np.sum(np.linalg.norm(np.diff(b['traj'], axis=0), axis=1)))
        print(f'  br{bid}: T={b["traj"].shape[0]}, arc={arc:.2f} rad, '
              f'closed={b["closed"]}')

    # === Phase 2: walk each branch's start without JL ===
    print(f'\n=== Re-walking each branch start with NO JL (±{args.big_limit_mult:.1f}π) ===')
    big = float(args.big_limit_mult) * np.pi
    no_jl_lo = np.full(7, -big, dtype=np.float32)
    no_jl_hi = np.full(7,  big, dtype=np.float32)
    kin.lmt_lo = torch.as_tensor(no_jl_lo, device=device, dtype=torch.float32)
    kin.lmt_up = torch.as_tensor(no_jl_hi, device=device, dtype=torch.float32)

    no_jl_trajs = []
    for bid, b in enumerate(branches_jl):
        q0 = b['traj'][0]
        traj_nojl, closed_nojl, why = walk_branch(
            kin, q0, p_tgt, R_tgt, no_jl_lo, no_jl_hi, args.h)
        arc_nojl = float(np.sum(np.linalg.norm(np.diff(traj_nojl, axis=0), axis=1)))
        print(f'  br{bid}_start under no-JL: T={traj_nojl.shape[0]}, '
              f'arc={arc_nojl:.2f} rad, '
              f'{"CLOSED" if closed_nojl else "OPEN (" + why + ")"}')
        no_jl_trajs.append({
            'traj': traj_nojl, 'closed': closed_nojl, 'reason': why,
            'arc': arc_nojl, 'q0': q0,
        })

    # Restore FR3 JL
    kin.lmt_lo = orig_lo
    kin.lmt_up = orig_hi

    # === Phase 3: pairwise distance between no-JL walks ===
    eps = 2.0 * args.h
    print(f'\n=== Pairwise minimum distance between no-JL walks '
          f'(eps={eps:.2f} = 2*h) ===')
    verdict_matrix = np.zeros((len(branches_jl), len(branches_jl)))
    for i in range(len(branches_jl)):
        for j in range(len(branches_jl)):
            if i == j:
                verdict_matrix[i, j] = 0.0
                continue
            traj_i = no_jl_trajs[i]['traj']
            traj_j = no_jl_trajs[j]['traj']
            d_min = float('inf')
            for q in traj_i:
                d = np.linalg.norm(traj_j - q[None, :], axis=1).min()
                d_min = min(d_min, d)
            verdict_matrix[i, j] = d_min
    print(f'  {"":<6}' + ' '.join(f'br{j:<3}' for j in range(len(branches_jl))))
    for i in range(len(branches_jl)):
        row = []
        for j in range(len(branches_jl)):
            if i == j:
                row.append('  -  ')
            else:
                d = verdict_matrix[i, j]
                tag = 'SAME' if d < eps else 'DIFF'
                row.append(f'{d:5.2f}')
        print(f'  br{i:<4}' + ' '.join(row))
    print()

    # Connected-component analysis: cluster branches by "any-pair distance < eps"
    n = len(branches_jl)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for i in range(n):
        for j in range(i + 1, n):
            if verdict_matrix[i, j] < eps:
                union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    print(f'=== Verdict ===')
    n_true_components = len(groups)
    print(f'  FR3-JL branch count:   {n}')
    print(f'  no-JL component count: {n_true_components}')
    if n_true_components < n:
        print(f'  → {n - n_true_components} branch pairs were JL-cut artifacts.')
    else:
        print(f'  → all branches are topologically distinct (IK root families).')
    for gid, members in groups.items():
        if len(members) > 1:
            print(f'    • JL-cut group: branches {members} → one underlying loop')
        else:
            print(f'    • standalone:   branch {members[0]}')

    # === Phase 4: visualization ===
    all_pts = np.concatenate(
        [b['traj'] for b in branches_jl] +
        [d['traj'] for d in no_jl_trajs], axis=0)
    mu = all_pts.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(all_pts - mu, full_matrices=False)
    W = Vt[:2].T

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    cmap = plt.get_cmap('tab10')

    for bid, b in enumerate(branches_jl):
        t2d = (b['traj'] - mu) @ W
        ax1.plot(t2d[:, 0], t2d[:, 1], '-',
                 color=cmap(bid % 10), alpha=0.85, linewidth=1.8,
                 label=f'br{bid} ({b["traj"].shape[0]} steps, '
                       f'{"closed" if b["closed"] else "open"})')
        ax1.scatter(t2d[0, 0], t2d[0, 1], s=80, c=[cmap(bid % 10)],
                    edgecolors='black', linewidth=1.0, marker='*', zorder=6)
    ax1.set_title(f'WITH FR3 JL: {len(branches_jl)} branches (PCA → 2D)')
    ax1.set_xlabel('PC1'); ax1.set_ylabel('PC2')
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    for bid, d in enumerate(no_jl_trajs):
        t2d = (d['traj'] - mu) @ W
        ax2.plot(t2d[:, 0], t2d[:, 1], '-',
                 color=cmap(bid % 10), alpha=0.7, linewidth=1.6,
                 label=f"br{bid} no-JL ({d['traj'].shape[0]} steps, "
                       f"{'closed' if d['closed'] else 'open'})")
        ax2.scatter(t2d[0, 0], t2d[0, 1], s=80, c=[cmap(bid % 10)],
                    edgecolors='black', linewidth=1.0, marker='*', zorder=6)
    ax2.set_title(f'NO JL (±{args.big_limit_mult:.0f}π): '
                  f'{n_true_components} component(s)')
    ax2.set_xlabel('PC1'); ax2.set_ylabel('PC2')
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    fig.suptitle(f'SMM disconnection test (seed={args.seed}): '
                 f'JL-cut artifact vs topological separation',
                 fontsize=12, y=1.01)
    fig.tight_layout()
    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'\nsaved: {out_png}')


if __name__ == '__main__':
    main()

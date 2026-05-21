"""Fig 6 — Why the SMM is disconnected: two superimposed reasons.

Same representative task as the other Part-1 figures (seed 118). We
implement the Lück & Lee 1993 disconnection probe: re-walk null(J) from
each WITH-JL branch's start point but with FR3 joint limits replaced
by ±5pi (effectively no JL). Two outcomes:

  * Two branches' no-JL walks meet (min 7-D distance < 2·h): they were
    one topologically connected curve that FR3's joint limits split.
  * They stay apart: the branches are different IK polynomial root
    families. Removing joint limits cannot connect them.

Panels:

  (A) PCA WITH FR3 JL: the {n_branches} disjoint arcs we have been
      working with.

  (B) PCA with NO JL (joint limits replaced by ±5pi): same arcs,
      re-walked. Compare to (A) — if any of (A)'s arcs would have
      merged here, JL cut them.

  (C) Pairwise minimum 7-D distance heatmap across the no-JL walks
      (lower triangle), with SAME / DIFF labels at the eps = 2·h
      threshold. Diagonal blanked.

  (D) Verdict: number of true topological components vs FR3-JL branch
      count, and the per-pair breakdown.

Usage:
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/fig_6_disconnection_reasons.py
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

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _shared import (  # noqa: E402
    DEFAULT_SEED, DISCONNECT_EPS_MULT, FIG_DIR, NO_JL_LIMIT_MULT,
    build_or_load,
)


def _connected_components(D, eps):
    n = D.shape[0]
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
            if D[i, j] < eps:
                union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    d = build_or_load(seed=args.seed, force=args.force)
    n_branches = int(d['meta']['n_branches'])
    branches_jl = [{'traj': d[f'branch_traj_{b}'].astype(np.float32),
                     'closed': bool(d['branch_closed'][b])}
                    for b in range(n_branches)]
    no_jl_trajs = [d[f'no_jl_traj_{b}'].astype(np.float32)
                    for b in range(n_branches)]
    no_jl_closed = d['no_jl_closed']
    D = d['pairwise_min_d']
    h = float(d['h'])
    eps = float(DISCONNECT_EPS_MULT) * h

    cmap = plt.get_cmap('tab10')
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)

    # Shared PCA basis across WITH-JL and NO-JL trajectories.
    all_pts = np.concatenate(
        [b['traj'] for b in branches_jl] + no_jl_trajs, axis=0)
    mu = all_pts.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(all_pts - mu, full_matrices=False)
    W = Vt[:2].T

    # --- Panel A: with FR3 JL ---
    ax_a = fig.add_subplot(gs[0, 0])
    for bid, b in enumerate(branches_jl):
        t2d = (b['traj'] - mu) @ W
        ax_a.plot(t2d[:, 0], t2d[:, 1], '-', color=cmap(bid % 10),
                  alpha=0.9, linewidth=1.9,
                  label=f'br{bid} ({b["traj"].shape[0]} steps, '
                        f'{"closed" if b["closed"] else "open"})')
        ax_a.scatter(t2d[0, 0], t2d[0, 1], s=80, c=[cmap(bid % 10)],
                     edgecolors='black', linewidth=1.0, marker='*', zorder=6)
    ax_a.set_title(f'A) WITH FR3 joint limits  — {n_branches} arcs',
                   fontsize=11)
    ax_a.set_xlabel('PC1'); ax_a.set_ylabel('PC2')
    ax_a.legend(fontsize=8); ax_a.grid(alpha=0.3)

    # --- Panel B: no JL ---
    ax_b = fig.add_subplot(gs[0, 1])
    components = _connected_components(D, eps)
    n_comp = len(components)
    for bid, traj in enumerate(no_jl_trajs):
        t2d = (traj - mu) @ W
        ax_b.plot(t2d[:, 0], t2d[:, 1], '-', color=cmap(bid % 10),
                  alpha=0.75, linewidth=1.6,
                  label=f'br{bid} no-JL ({traj.shape[0]} steps, '
                        f'{"closed" if no_jl_closed[bid] else "open"})')
        ax_b.scatter(t2d[0, 0], t2d[0, 1], s=80, c=[cmap(bid % 10)],
                     edgecolors='black', linewidth=1.0, marker='*', zorder=6)
    ax_b.set_title(
        f'B) NO joint limits (limits replaced by ±{NO_JL_LIMIT_MULT:.0f}pi) '
        f'— {n_comp} component(s)', fontsize=11)
    ax_b.set_xlabel('PC1'); ax_b.set_ylabel('PC2')
    ax_b.legend(fontsize=8); ax_b.grid(alpha=0.3)

    # --- Panel C: pairwise distance heatmap ---
    ax_c = fig.add_subplot(gs[1, 0])
    D_show = np.where(np.eye(n_branches, dtype=bool), np.nan, D)
    im = ax_c.imshow(D_show, cmap='viridis', vmin=0.0,
                     vmax=max(eps * 4, float(np.nanmax(D_show))))
    for i in range(n_branches):
        for j in range(n_branches):
            if i == j:
                ax_c.text(j, i, '—', ha='center', va='center',
                          color='white', fontsize=10)
            else:
                v = float(D[i, j])
                tag = 'SAME' if v < eps else 'DIFF'
                color = 'white' if v < (eps * 4) * 0.5 else 'black'
                ax_c.text(j, i, f'{v:.2f}\n{tag}',
                          ha='center', va='center',
                          color=color, fontsize=8,
                          fontweight='bold' if tag == 'SAME' else 'normal')
    ax_c.set_xticks(range(n_branches))
    ax_c.set_yticks(range(n_branches))
    ax_c.set_xticklabels([f'br{b}' for b in range(n_branches)])
    ax_c.set_yticklabels([f'br{b}' for b in range(n_branches)])
    ax_c.set_title(f'C) pairwise min 7-D distance between no-JL walks\n'
                   f'SAME ⇔ d < 2·h = {eps:.2f} rad (JL-cut artifact)',
                   fontsize=11)
    plt.colorbar(im, ax=ax_c, shrink=0.85, label='min 7-D distance')

    # --- Panel D: verdict ---
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_axis_off()
    ax_d.set_xlim(0, 10); ax_d.set_ylim(0, 10)
    lines = [
        f'Two reasons SMM is disconnected on FR3:',
        '',
        '  1. IK solution space itself is not connected.',
        '     Six task constraints in 7-D joint space give a finite set',
        '     of disjoint 1-D curves (different IK polynomial root families).',
        '',
        '  2. Joint limits cut each curve into smaller arcs.',
        '     FR3 j3 and j5 carry asymmetric limits that truncate the',
        '     1-D curves where they would otherwise be reachable.',
        '',
        f'For seed {args.seed}:',
        f'  FR3-JL branch count:   {n_branches}',
        f'  no-JL component count: {n_comp}',
    ]
    if n_comp < n_branches:
        lines.append(f'  ⇒ {n_branches - n_comp} branch pair(s) were JL-cut artifacts:')
    else:
        lines.append('  ⇒ all branches are topologically distinct')
        lines.append('     (different IK polynomial root families)')
    for cid, members in enumerate(components):
        if len(members) > 1:
            lines.append(f'      JL-cut group: branches {members} share one underlying curve')
        else:
            lines.append(f'      standalone:   branch {members[0]} '
                         f'(distinct IK root family)')
    y0 = 9.5
    for line in lines:
        ax_d.text(0.2, y0, line, fontsize=10, family='monospace',
                  ha='left', va='top')
        y0 -= 0.55
    ax_d.set_title('D) Verdict for this task', fontsize=11)

    fig.suptitle(
        f'Disconnection probe (Lück & Lee 1993 style) — seed {args.seed}',
        fontsize=12, y=1.00,
    )

    out_path = Path(args.out) if args.out else FIG_DIR / f'fig6_seed{args.seed}.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved: {out_path}')


if __name__ == '__main__':
    main()

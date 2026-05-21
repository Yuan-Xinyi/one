"""Fig 3 — Length-to-failure splits into two factors.

   L_total = f(q0 branch choice [discrete],  controller within branch [continuous])

Three panels on the SAME representative task (seed 118 by default):

  (A) Branch ceiling bars: max L_self / L_max per branch, with L_max line
      at 1.0. Shows that the discrete choice of which SMM branch q0 lives
      on sets a hard ceiling on what any controller can achieve.

  (B) Within-branch spread: per-branch violin of L_self / L_max across all
      sampled q0 along the branch, plus the best-q0 marker. Visualizes the
      continuous variation the controller is responsible for once the
      branch is fixed.

  (C) Variance decomposition: ANOVA-style breakdown of total variance of
      L across all q0 samples into between-branch (factor 1) vs
      within-branch (factor 2). Shows numerically which factor dominates.

Usage (folder name starts with a digit so -m invocation is not supported):
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/fig_3_two_factor_decomposition.py
    python Yuan/flow_connectivity/intro_motivation/25_spring_pre/fig_3_two_factor_decomposition.py --seed 118 --force
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[3]   # /home/lqin/one
for _p in (str(_REPO), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _shared import DEFAULT_SEED, FIG_DIR, build_or_load  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--force', action='store_true',
                        help='rebuild the cache even if it exists')
    parser.add_argument('--out', type=str, default=None,
                        help='output PNG path (default: figs/fig3_seed{S}.png)')
    args = parser.parse_args()

    d = build_or_load(seed=args.seed, force=args.force)
    n_branches = int(d['meta']['n_branches'])
    L_self_best = d['L_self_best']
    all_bid = d['all_q0_bid']
    all_L = d['all_q0_L_rel']

    cmap = plt.get_cmap('tab10')
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Panel A: branch ceilings (discrete factor) ---
    ax = axes[0]
    x = np.arange(n_branches)
    colors = [cmap(b % 10) for b in range(n_branches)]
    ax.bar(x, L_self_best, color=colors, edgecolor='black', alpha=0.85)
    ax.axhline(1.0, color='k', linestyle='--', linewidth=1.0, alpha=0.6,
               label='L_max (full path)')
    for xi, v in zip(x, L_self_best):
        ax.text(xi, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f'br{b}' for b in range(n_branches)])
    ax.set_ylabel('L_self / L_max  (best q0 in branch)')
    ax.set_ylim(0.0, max(1.05, float(L_self_best.max()) * 1.3))
    ax.set_title('A) Factor 1 — discrete: branch choice sets the ceiling')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3, axis='y')

    # --- Panel B: within-branch spread (continuous factor) ---
    ax = axes[1]
    data, pos = [], []
    for bid in range(n_branches):
        L = all_L[all_bid == bid]
        if L.size:
            data.append(L); pos.append(bid)
    parts = ax.violinplot(data, positions=pos, showmeans=True,
                          showmedians=False, widths=0.7)
    for k, pc in enumerate(parts['bodies']):
        pc.set_facecolor(cmap(pos[k] % 10))
        pc.set_alpha(0.55); pc.set_edgecolor('black')
    parts['cmeans'].set_color('black'); parts['cmeans'].set_linewidth(2)
    rng_j = np.random.default_rng(0)
    for bid in pos:
        L = all_L[all_bid == bid]
        jitter = rng_j.uniform(-0.10, 0.10, size=len(L))
        ax.scatter(np.full(len(L), bid) + jitter, L,
                   c=[cmap(bid % 10)], s=14, alpha=0.65,
                   edgecolors='black', linewidths=0.2, zorder=3)
    ax.scatter(pos, [L_self_best[b] for b in pos], marker='*',
               s=200, c=[cmap(b % 10) for b in pos],
               edgecolors='black', linewidths=0.9, zorder=6,
               label='best q0 in branch')
    ax.set_xticks(pos)
    ax.set_xticklabels([f'br{b}' for b in pos])
    ax.set_ylabel('L_self / L_max')
    ax.set_title('B) Factor 2 — continuous: spread within a fixed branch')
    ax.set_ylim(-0.02, max(float(all_L.max()) * 1.15, 0.15))
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(alpha=0.3, axis='y')

    # --- Panel C: variance decomposition ---
    ax = axes[2]
    grand = float(all_L.mean())
    total_var = float(((all_L - grand) ** 2).sum())
    between = 0.0
    within = 0.0
    for bid in range(n_branches):
        L = all_L[all_bid == bid]
        if L.size == 0:
            continue
        mu_b = float(L.mean())
        between += L.size * (mu_b - grand) ** 2
        within += float(((L - mu_b) ** 2).sum())
    frac_between = between / max(total_var, 1e-12)
    frac_within = within / max(total_var, 1e-12)
    bars_y = [frac_between, frac_within]
    bar_colors = ['C0', 'C3']
    labels = ['between-branch\n(Factor 1: q0 choice)',
              'within-branch\n(Factor 2: controller x arc pos)']
    ax.bar([0, 1], bars_y, color=bar_colors, edgecolor='black', alpha=0.85)
    for i, v in enumerate(bars_y):
        ax.text(i, v + 0.02, f'{100*v:.1f}%', ha='center', fontsize=11,
                fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.0, 1.10)
    ax.set_ylabel('fraction of total variance in L_self')
    ax.set_title('C) Variance decomposition of L across all q0 samples')
    ax.grid(alpha=0.3, axis='y')

    fig.suptitle(
        f'L_total = f(q0,  pi)  —  two-factor decomposition '
        f'(seed={args.seed}, {n_branches} branches, '
        f'{all_L.size} q0 samples)',
        fontsize=12, y=1.02,
    )
    fig.tight_layout()

    out_path = Path(args.out) if args.out else FIG_DIR / f'fig3_seed{args.seed}.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved: {out_path}')


if __name__ == '__main__':
    main()

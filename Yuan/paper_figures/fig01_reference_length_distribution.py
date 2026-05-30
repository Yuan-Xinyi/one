"""Figure 1: distribution of the reference length on the evaluation set.

Background: histogram of l_ref over the 10,000 evaluation tasks.
Foreground: fitted density curve (KDE), linewidth = 3.

Reference length per task is computed by `run_oracle_prime.py`:
    l_ref_i = max over k in SMM-pool of L_k * d_target  (d_target = 1.5 m).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


DEFAULT_ORACLE_NPZ = (
    'Yuan/system_eval/runs/eval_10k_systematic/cell_oracle_hyb_results.npz'
)
DEFAULT_OUT_PATH = 'Yuan/ISRR2026_Xinyi_new/imgs/task_distribution.png'
TARGET_DISTANCE_M = 1.5


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--oracle-npz', default=DEFAULT_ORACLE_NPZ,
                   help='cell_oracle_hyb_results.npz with L_best (T,).')
    p.add_argument('--out', default=DEFAULT_OUT_PATH,
                   help='Output figure path.')
    p.add_argument('--bins', type=int, default=80)
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--figsize', nargs=2, type=float, default=(12, 6))
    return p.parse_args()


def main():
    args = parse_args()
    z = np.load(args.oracle_npz, allow_pickle=False)
    l_ref = z['L_best'].astype(np.float32) * TARGET_DISTANCE_M
    l_ref = l_ref[np.isfinite(l_ref)]

    print(f'[fig01] n_tasks={len(l_ref)}  '
          f'min={l_ref.min():.3f}  median={np.median(l_ref):.3f}  '
          f'mean={l_ref.mean():.3f}  max={l_ref.max():.3f}')

    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    # ---- Background: histogram (light gray) ---------------------------
    ax.hist(l_ref, bins=args.bins, density=True, color='#BFBFBF',
            edgecolor='#BFBFBF', alpha=0.85, linewidth=0.6,
            label='Histogram')

    # ---- Bucket thresholds at 0.45 and 0.80 m -------------------------
    for xv in (0.45, 0.80):
        ax.axvline(xv, color='gray', linestyle='--', linewidth=1.2,
                   alpha=0.7, zorder=1.5)

    # ---- Foreground: KDE-fitted density curve (black), linewidth 3 ----
    kde = gaussian_kde(l_ref, bw_method='scott')
    x = np.linspace(l_ref.min(), l_ref.max(), 400)
    ax.plot(x, kde(x), color='black', linewidth=3.0,
            label='Fitted density (KDE)')

    # Axes
    ax.set_xlabel(r'Reference length $\ell^{\mathrm{ref}}$ (m)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_xlim(0, l_ref.max() * 1.02)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis='both', labelsize=10)
    ax.legend(frameon=False, fontsize=10, loc='upper right')

    # Clean look (no top/right spines)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    fig.tight_layout()

    # Show the figure first (blocks until the window is closed), then save.
    plt.show()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
    print(f'[fig01] wrote {out_path}')

    # Also keep a local copy next to this script.
    local_path = Path(__file__).with_suffix('.png')
    fig.savefig(local_path, dpi=args.dpi, bbox_inches='tight')
    print(f'[fig01] wrote {local_path}')


if __name__ == '__main__':
    main()

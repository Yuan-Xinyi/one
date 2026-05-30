"""Figure 2: grouped bar chart of Table 1 (System Integration).

Three difficulty buckets (Easy / Medium / Difficult) on the x-axis,
three bars per bucket for the three controllers (Classical / RL / Hybrid).
Style matches fig01: figsize 12x6, 300 DPI, no top/right spines, palette
colours from the user-supplied set (teal, orange, red).

Data is read from the main-table run results:
    Yuan/system_eval/runs/eval_10k_systematic/sweeps/main_table_results.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


DEFAULT_RESULTS_JSON = (
    'Yuan/system_eval/runs/eval_10k_systematic/sweeps/main_table_results.json'
)
DEFAULT_OUT_PATH = 'Yuan/ISRR2026_Xinyi_new/imgs/main_result.png'

# Palette: Hybrid (our method) = blue, RL = red, Classical = yellow/orange.
COLOR_CLASSICAL = '#FFBE7A'   # yellow/orange
COLOR_RL        = '#FA7F6F'   # red
COLOR_HYBRID    = '#82B0D2'   # blue  -- our method

BUCKETS     = ('Easy', 'Medium', 'Difficult')
CONTROLLERS = ('Classical', 'RL', 'Hybrid')
COLOR_MAP   = {'Classical': COLOR_CLASSICAL, 'RL': COLOR_RL, 'Hybrid': COLOR_HYBRID}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--results-json', default=DEFAULT_RESULTS_JSON,
                   help='main_table_results.json with rows[ctrl][bucket][pct].')
    p.add_argument('--out', default=DEFAULT_OUT_PATH,
                   help='Output figure path.')
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--figsize', nargs=2, type=float, default=(12, 6))
    p.add_argument('--bar-labels', action='store_true',
                   help='Draw the value text on top of each bar. '
                        'Outputs are suffixed with "_labeled" so the default '
                        '(unlabeled) version is preserved.')
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.results_json) as f:
        data = json.load(f)
    rows = data['rows']

    # Extract mean % per (controller, bucket); pct field is [mean, std, min, max]
    means = {c: [rows[c][b]['pct'][0] for b in BUCKETS] for c in CONTROLLERS}

    print(f'[fig02] data:')
    for c in CONTROLLERS:
        print(f'  {c:>10s}: ' +
              ' '.join(f'{b}={means[c][i]:.2f}' for i, b in enumerate(BUCKETS)))

    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    x = np.arange(len(BUCKETS), dtype=np.float64)
    n_ctrl = len(CONTROLLERS)
    bar_w = 0.22                              # bar width (slightly smaller)
    gap   = 0.03                              # gap between adjacent bars
    step  = bar_w + gap                       # centre-to-centre distance
    offsets = np.linspace(-(n_ctrl - 1) / 2, (n_ctrl - 1) / 2, n_ctrl) * step

    for i, ctrl in enumerate(CONTROLLERS):
        bars = ax.bar(x + offsets[i], means[ctrl], bar_w,
                      color=COLOR_MAP[ctrl], edgecolor=COLOR_MAP[ctrl],
                      label=ctrl, zorder=3)
        if args.bar_labels:
            for rect, v in zip(bars, means[ctrl]):
                ax.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_height() + 1.0, f'{v:.1f}',
                        ha='center', va='bottom', fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(BUCKETS, fontsize=11)
    ax.set_ylabel(r'$l\,/\,\ell^{\mathrm{ref}}$  (\%)', fontsize=11)
    ax.set_ylim(35, 105)
    ax.tick_params(axis='both', labelsize=10)
    ax.legend(frameon=False, fontsize=10, loc='upper left', ncol=3)

    # Clean look (no top/right spines)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    fig.tight_layout()

    # Show the figure first (blocks until the window is closed), then save.
    plt.show()

    suffix = '_labeled' if args.bar_labels else ''
    out_path = Path(args.out)
    out_path = out_path.with_name(out_path.stem + suffix + out_path.suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
    print(f'[fig02] wrote {out_path}')

    # Also keep a local copy next to this script.
    local_base = Path(__file__).with_suffix('')
    local_path = local_base.with_name(local_base.name + suffix + '.png')
    fig.savefig(local_path, dpi=args.dpi, bbox_inches='tight')
    print(f'[fig02] wrote {local_path}')


if __name__ == '__main__':
    main()

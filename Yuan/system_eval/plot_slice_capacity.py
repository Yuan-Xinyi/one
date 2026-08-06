"""Render a two-panel pointwise/continuous reachability slice from NPZ."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def _sample_fine(field, fine_axis, x_grid, y_grid):
    xx, yy = np.meshgrid(x_grid, y_grid, indexing='ij')
    step = fine_axis[1] - fine_axis[0]
    ix = np.clip(np.rint((xx - fine_axis[0]) / step).astype(int),
                 0, len(fine_axis) - 1)
    iy = np.clip(np.rint((yy - fine_axis[0]) / step).astype(int),
                 0, len(fine_axis) - 1)
    return field[ix, iy]


def render(npz_path: Path, out_path: Path, dpi: int = 300,
           terminal_step_m: float = 0.0) -> dict:
    data = np.load(npz_path)
    xs, ys = data['xs'], data['ys']
    reach = _sample_fine(data['L_cone'], data['mxs'], xs, ys)
    continuous = data['L_max'].copy()
    if terminal_step_m > 0:
        continuous = np.where(
            np.isfinite(continuous),
            np.maximum(continuous - terminal_step_m, 0.0), continuous)
    shared = (np.isfinite(reach) & np.isfinite(continuous) & (reach > 0.05))
    reach = np.where(shared, reach, np.nan)
    continuous = np.where(shared, continuous, np.nan)

    tol = 1e-3
    violations = shared & (continuous > reach + tol)
    vmax = float(max(np.nanmax(reach), np.nanmax(continuous)))
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    extent = [xs[0] - dx / 2, xs[-1] + dx / 2,
              ys[0] - dy / 2, ys[-1] + dy / 2]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.65),
                             constrained_layout=True)
    image = None
    for ax, field, title in zip(
            axes, (reach, continuous),
            ('(a) Pointwise reachability',
             '(b) Continuous reachability')):
        image = ax.imshow(field.T, origin='lower', extent=extent,
                          cmap='viridis', vmin=0.0, vmax=vmax,
                          interpolation='nearest')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.plot(0, 0, 'w^', ms=9, mec='k')
        ax.set_aspect('equal')
    fig.colorbar(image, ax=axes, label='reachable length [m]',
                 shrink=0.88, pad=0.03)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    return {
        'n_visible': int(shared.sum()),
        'n_bound_violations': int(violations.sum()),
        'max_bound_violation_m': (float(np.max(continuous[violations]
                                               - reach[violations]))
                                  if violations.any() else 0.0),
        'reach_median_m': float(np.nanmedian(reach)),
        'continuous_median_m': float(np.nanmedian(continuous)),
        'continuity_ratio_median': float(np.nanmedian(continuous / reach)),
        'vmax_m': vmax,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument(
        '--terminal-step-m', type=float, default=0.0,
        help='subtract a legacy rollout terminal step (use 0.01 for old runs)')
    args = parser.parse_args()
    stats = render(Path(args.npz), Path(args.out), args.dpi,
                   args.terminal_step_m)
    for key, value in stats.items():
        print(f'{key}: {value}')
    print(f'saved: {args.out}')


if __name__ == '__main__':
    main()

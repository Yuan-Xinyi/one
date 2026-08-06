"""Figures and summary statistics for a reachability map produced by reach_map.py.

Produces the two views that a reachability analysis is normally reported with:

  * vertical section through the shoulder (x-z plane, |y| <= res/2) and a
    horizontal section at the shoulder height, both coloured by the
    reachability index D;
  * D as a function of distance from the shoulder, which is where the
    reachable and the dexterous workspace radii can be read off.

Usage:
    python -m Yuan.reachability.plot_reach_map --npz Yuan/reachability/runs/reach_5cm_50dir.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_CENTER = np.array([0.0, 0.0, 0.333])


def load(npz_path: str | Path) -> dict:
    z = np.load(npz_path, allow_pickle=False)
    d = {k: z[k] for k in z.files if k != 'meta'}
    d['meta'] = json.loads(str(z['meta']))
    return d


def summarise(m: dict) -> dict:
    res = float(m['meta']['res'])
    D, xyz = m['D'], m['voxel_xyz']
    vol = res ** 3
    r = np.linalg.norm(xyz - BASE_CENTER, axis=1)
    reach = D > 0
    s = {
        'voxels_total': int(len(D)),
        'voxels_reachable': int(reach.sum()),
        'reachable_volume_m3': float(reach.sum() * vol),
        # "dexterous" in the usual sense: every sampled tool axis is attainable.
        'dexterous_volume_m3': float((D >= 0.999).sum() * vol),
        'mean_D_over_reachable': float(D[reach].mean()),
        'max_reach_radius_m': float(r[reach].max()) if reach.any() else 0.0,
        'p95_reach_radius_m': float(np.quantile(r[reach], 0.95)) if reach.any() else 0.0,
        'max_dexterous_radius_m': (float(r[D >= 0.999].max())
                                   if (D >= 0.999).any() else 0.0),
        'z_span_m': [float(xyz[reach, 2].min()), float(xyz[reach, 2].max())],
    }
    return s


def _slice(m: dict, axis: int, value: float, tol: float):
    xyz, D = m['voxel_xyz'], m['D']
    sel = np.abs(xyz[:, axis] - value) <= tol
    return xyz[sel], D[sel]


def plot(m: dict, out_png: str | Path) -> None:
    res = float(m['meta']['res'])
    tol = res / 2.0
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    # --- vertical section y = 0 ---
    p, d = _slice(m, 1, 0.0, tol)
    sc = axes[0].scatter(p[:, 0], p[:, 2], c=d, s=res * 1600, marker='s',
                         cmap='viridis', vmin=0, vmax=1)
    axes[0].set(xlabel='x [m]', ylabel='z [m]',
                title=f'vertical section (y=0)')
    axes[0].plot(0, BASE_CENTER[2], 'r+', ms=12, mew=2)
    axes[0].set_aspect('equal')
    fig.colorbar(sc, ax=axes[0], label='reachability index D')

    # --- horizontal section at shoulder height ---
    p, d = _slice(m, 2, BASE_CENTER[2], tol)
    sc = axes[1].scatter(p[:, 0], p[:, 1], c=d, s=res * 1600, marker='s',
                         cmap='viridis', vmin=0, vmax=1)
    axes[1].set(xlabel='x [m]', ylabel='y [m]',
                title=f'horizontal section (z={BASE_CENTER[2]:.3f})')
    axes[1].plot(0, 0, 'r+', ms=12, mew=2)
    axes[1].set_aspect('equal')
    fig.colorbar(sc, ax=axes[1], label='reachability index D')

    # --- radial profile ---
    r = np.linalg.norm(m['voxel_xyz'] - BASE_CENTER, axis=1)
    edges = np.arange(0.0, r.max() + res, res)
    mid = 0.5 * (edges[:-1] + edges[1:])
    mean_D = np.array([m['D'][(r >= a) & (r < b)].mean() if ((r >= a) & (r < b)).any()
                       else np.nan for a, b in zip(edges[:-1], edges[1:])])
    frac = np.array([(m['D'][(r >= a) & (r < b)] > 0).mean()
                     if ((r >= a) & (r < b)).any() else np.nan
                     for a, b in zip(edges[:-1], edges[1:])])
    axes[2].plot(mid, mean_D, 'o-', label='mean D')
    axes[2].plot(mid, frac, 's--', label='fraction of voxels with D>0')
    axes[2].set(xlabel='distance from shoulder [m]', ylabel='',
                title='radial profile', ylim=(-0.02, 1.02))
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    fig.suptitle(f"FR3 + pen reachability  |  res={res} m, "
                 f"{m['meta']['n_dirs']} tool-axis directions, "
                 f"ang_tol={m['meta']['ang_tol_deg']:.1f} deg, "
                 f"self-collision={'on' if m['meta']['self_collision'] else 'off'}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f'[plot] wrote {out_png}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default='Yuan/reachability/runs/reach_5cm_50dir.npz')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    m = load(a.npz)
    s = summarise(m)
    print(json.dumps({'meta': m['meta'], 'summary': s}, indent=2))
    out = a.out or str(Path(a.npz).with_suffix('.png'))
    plot(m, out)
    Path(a.npz).with_suffix('.summary.json').write_text(
        json.dumps({'meta': m['meta'], 'summary': s}, indent=2))


if __name__ == '__main__':
    main()

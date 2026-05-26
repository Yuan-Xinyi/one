"""ONE-world viewer for the perturbation scan around seed=17.

Loads the pickle saved by perturb_scan_seed17.py and draws each
perturbation point as a colored line segment in 3D:
  * the thick colored portion = achieved length L_max * L_task (in metres),
    coloured by L_max / L_task via the viridis colormap;
  * a faint gray segment continues to the full 1.5 m theoretical line,
    so the "missing" part is visible.

For Scan A and B all lines share the same p0 (a single sphere). For Scan
C, p0 changes — each direction's family of p0 points is a row of small
spheres, colored by direction (red=d0, green=n0, blue=d0xn0).

Run:
    python -m Yuan.flow_connectivity.intro_motivation.25_spring_pre.viewer_perturb_seed17 --scan A
"""
from __future__ import annotations

import argparse
import builtins
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw


L_TASK_M = 1.5  # full theoretical line length (matches TARGET_PATH_M)
ACHIEVED_RADIUS = 0.0030
REMAINING_RADIUS = 0.0008
P0_RADIUS = 0.012
DIR_COLOR = {  # for Scan C only
    'd0':    np.array([0.85, 0.20, 0.20], dtype=np.float32),
    'n0':    np.array([0.20, 0.75, 0.30], dtype=np.float32),
    'd0xn0': np.array([0.20, 0.40, 0.85], dtype=np.float32),
}
CMAP = plt.get_cmap('viridis')


def top_branch_Lrel(pt: dict) -> float:
    """Top branch's normalized rollout length at this perturbation point."""
    if pt['n_branches'] == 0 or pt['branch_L_max'].size == 0:
        return 0.0
    return float(pt['branch_L_max'].max())


def color_from_Lrel(L_rel: float) -> np.ndarray:
    return np.array(CMAP(float(np.clip(L_rel, 0.0, 1.0)))[:3], dtype=np.float32)


def draw_one_line(achieved_segs, remaining_segs, achieved_colors,
                   p0, d, L_rel):
    """Append one perturbation point's two segments to the global buffers."""
    achieved_end = p0 + (L_rel * L_TASK_M) * d
    full_end = p0 + L_TASK_M * d
    achieved_segs.append(np.stack([p0, achieved_end]))
    remaining_segs.append(np.stack([achieved_end, full_end]))
    achieved_colors.append(color_from_Lrel(L_rel))


def render_scan_AB(base, bundle):
    """All lines share the same p0 (rotation of d or n only)."""
    achieved, remaining, colors = [], [], []
    p0_drawn = False
    for pt, xv in zip(bundle['points'], bundle['x_values']):
        L_rel = top_branch_Lrel(pt)
        p0 = pt['p_tgt']
        # The direction d isn't stored explicitly; recover from R_tgt (col 0
        # is x_local = direction projected onto the surface plane).
        R = pt['R_tgt']
        d = R[:, 0].astype(np.float32)
        d = d / (np.linalg.norm(d) + 1e-12)
        draw_one_line(achieved, remaining, colors, p0, d, L_rel)
        if not p0_drawn:
            ossop.sphere(pos=tuple(p0), radius=P0_RADIUS,
                          rgb=(0.10, 0.10, 0.10), alpha=0.95).attach_to(base.scene)
            p0_drawn = True
    achieved = np.stack(achieved)
    remaining = np.stack(remaining)
    colors = np.stack(colors)
    ossop.linsegs(segs=achieved, radius=ACHIEVED_RADIUS,
                   srgbs=colors, alpha=0.92).attach_to(base.scene)
    ossop.linsegs(segs=remaining, radius=REMAINING_RADIUS,
                   srgbs=np.array([0.7, 0.7, 0.7], dtype=np.float32),
                   alpha=0.35).attach_to(base.scene)
    # Surface plane: for Scan A n is constant, for Scan B it tilts. We use
    # the baseline (zero-perturb) plane only for orientation reference.
    i_zero = int(np.argmin(np.abs(np.asarray(bundle['x_values']))))
    n0 = bundle['points'][i_zero]['plane_normal']
    p0_ref = bundle['points'][i_zero]['p_tgt']
    plane_center = p0_ref + 0.5 * L_TASK_M * bundle['points'][i_zero]['R_tgt'][:, 0]
    ossop.plane(pos=tuple(plane_center.astype(np.float32)),
                normal=tuple(n0.astype(np.float32)),
                size=(2.5, 2.5), thickness=2e-3,
                rgb=(0.85, 0.85, 0.88), alpha=0.15).attach_to(base.scene)


def render_scan_C(base, bundle):
    """Lines have different p0; direction d is constant across all points."""
    achieved, remaining, colors = [], [], []
    dir_col = bundle['extra_cols']['direction']
    for pt, dname in zip(bundle['points'], dir_col):
        L_rel = top_branch_Lrel(pt)
        p0 = pt['p_tgt']
        d = pt['R_tgt'][:, 0].astype(np.float32)
        d = d / (np.linalg.norm(d) + 1e-12)
        draw_one_line(achieved, remaining, colors, p0, d, L_rel)
        # p0 sphere, color by direction.
        ossop.sphere(pos=tuple(p0), radius=P0_RADIUS * 0.6,
                      rgb=tuple(DIR_COLOR[dname]), alpha=0.95).attach_to(base.scene)
    achieved = np.stack(achieved)
    remaining = np.stack(remaining)
    colors = np.stack(colors)
    ossop.linsegs(segs=achieved, radius=ACHIEVED_RADIUS,
                   srgbs=colors, alpha=0.92).attach_to(base.scene)
    ossop.linsegs(segs=remaining, radius=REMAINING_RADIUS,
                   srgbs=np.array([0.7, 0.7, 0.7], dtype=np.float32),
                   alpha=0.30).attach_to(base.scene)
    # Plane at the centroid of all p0's.
    p0s = np.stack([pt['p_tgt'] for pt in bundle['points']], axis=0)
    n0 = bundle['points'][0]['plane_normal']
    plane_center = p0s.mean(axis=0)
    ossop.plane(pos=tuple(plane_center.astype(np.float32)),
                normal=tuple(n0.astype(np.float32)),
                size=(2.5, 2.5), thickness=2e-3,
                rgb=(0.85, 0.85, 0.88), alpha=0.15).attach_to(base.scene)


def print_legend(bundle, scan_label):
    """Print a textual legend to the terminal — the ONE viewer has no
    built-in colorbar, so this is how the user reads the colors."""
    L_rels = [top_branch_Lrel(pt) for pt in bundle['points']]
    print()
    print(f'=== scan {scan_label} viewer legend ===')
    print(f'  colormap = viridis: L_rel 0.0 -> dark purple,'
          f' 0.5 -> teal, 1.0 -> yellow')
    print(f'  thick colored = achieved length (= L_rel * 1.5 m)')
    print(f'  faint gray    = missing length (to full 1.5 m line)')
    if scan_label == 'C':
        print(f'  p0 sphere color: red=d0, green=n0, blue=d0xn0')
    print(f'  per-point L_rel of top branch:')
    if scan_label == 'C':
        dir_col = bundle['extra_cols']['direction']
        for x, L, dname in zip(bundle['x_values'], L_rels, dir_col):
            print(f'    {dname:>6s}  {bundle["x_label"]}={x:+6.2f}: L_rel={L:.3f}')
    else:
        for x, L in zip(bundle['x_values'], L_rels):
            print(f'    {bundle["x_label"]}={x:+6.2f}: L_rel={L:.3f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scan', choices=['A', 'B', 'C'], required=True)
    ap.add_argument('--out-dir', type=str,
                    default='Yuan/flow_connectivity/intro_motivation/25_spring_pre/perturb_out')
    args = ap.parse_args()

    pkl = Path(args.out_dir) / f'scan_{args.scan}.pkl'
    if not pkl.exists():
        raise FileNotFoundError(f'no scan pickle at {pkl}; '
                                 f'run perturb_scan_seed17.py first')
    with open(pkl, 'rb') as f:
        bundle = pickle.load(f)

    # Pick camera focus near the lines.
    p0_ref = bundle['points'][0]['p_tgt']
    d_ref = bundle['points'][0]['R_tgt'][:, 0].astype(np.float32)
    d_ref = d_ref / (np.linalg.norm(d_ref) + 1e-12)
    focus = (p0_ref + 0.5 * L_TASK_M * d_ref).astype(np.float32)
    cam_focus = tuple(float(x) for x in focus)
    cam_pos = (cam_focus[0] + 1.6, cam_focus[1] - 1.8, cam_focus[2] + 1.0)
    base = ovw.World(cam_pos=cam_pos, cam_lookat_pos=cam_focus,
                      toggle_auto_cam_orbit=False)
    builtins.base = base
    ossop.frame(length_scale=0.18, radius_scale=0.7).attach_to(base.scene)

    if args.scan in ('A', 'B'):
        render_scan_AB(base, bundle)
    else:
        render_scan_C(base, bundle)

    print_legend(bundle, args.scan)
    print('\n  launching ONE viewer; close window to exit.')
    base.run()


if __name__ == '__main__':
    main()

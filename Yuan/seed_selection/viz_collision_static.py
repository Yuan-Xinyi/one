"""Static PNG viz that makes plane straddling obvious.

For a chosen task, renders the FR3 collision spheres at each label and sample
q, color-coded by their signed_dist to the task plane. Two views:
  Left:  3D scatter (signed-dist colored spheres + task plane patch + line_dir
         + n_target arrows).
  Right: Per-sphere signed_dist bar chart (sorted by link index). Spheres on
         opposite sides of zero are the proof that the arm straddles the plane.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

from one.robots.manipulators.franka.fr3.sphere_collision import FR3SphereCollision
from one.robots.manipulators.franka.fr3_pen.batched_fr3_kin import BatchedFR3Kinematics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path,
                   default=Path('Yuan/seed_selection/runs/pilot_day5/pilot_20k.npz'))
    p.add_argument('--task', type=int, required=True)
    p.add_argument('--which-q', default='label_0',
                   choices=['q0_seed', 'label_0', 'label_1', 'label_2'])
    p.add_argument('--exclude-links', type=int, nargs='*', default=[0, 1])
    p.add_argument('--out', type=Path, default=None)
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    z = np.load(args.data, allow_pickle=False)
    p0 = z['cs_p0'][args.task].astype(np.float32)
    line_dir = z['cs_line_dir'][args.task].astype(np.float32)
    n_target = z['cs_n_target'][args.task].astype(np.float32)
    n_target = n_target / np.linalg.norm(n_target)

    if args.which_q == 'q0_seed':
        q = z['q0_seeds'][args.task].astype(np.float32)
    else:
        j = int(args.which_q.split('_')[1])
        q = z['labels_q0'][args.task, j].astype(np.float32)

    kin = BatchedFR3Kinematics(device=device)
    coll = FR3SphereCollision(device=device, dtype=kin.dtype)
    radii = coll.radii.detach().cpu().numpy().astype(np.float32)
    link_idx = coll.link_indices.detach().cpu().numpy().astype(np.int32)

    qt = torch.as_tensor(q[None], device=device, dtype=kin.dtype)
    sp = coll.sphere_positions(kin.link_transforms(qt))[0].detach().cpu().numpy().astype(np.float32)
    signed = (sp - p0).dot(n_target)
    line_dir_n = line_dir / (np.linalg.norm(line_dir) + 1e-12)
    proj_d = (sp - p0).dot(line_dir_n)   # along +line_dir from p0
    # "Over the plane" iff projection in [0, plane_extent_m]
    plane_extent_m = 1.5
    over_plane = (proj_d >= 0.0) & (proj_d <= plane_extent_m)
    keep = (~np.isin(link_idx, args.exclude_links)) & over_plane
    sp_k = sp[keep]; signed_k = signed[keep]; radii_k = radii[keep]; link_k = link_idx[keep]
    has_pos = (signed_k > 0).any()
    has_neg = (signed_k < 0).any()
    straddles = bool(has_pos and has_neg)

    # --- Plot ---
    fig = plt.figure(figsize=(15, 7))

    # Left: 3D scatter
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    # Plane patch: 1.5m square starting at p0 and extending in +line_dir direction
    # (not centered on p0; matches the bounded plane used by the filter).
    y_axis = np.cross(n_target, line_dir); y_axis /= (np.linalg.norm(y_axis) + 1e-12)
    line_dir_proj = np.cross(y_axis, n_target)
    line_dir_proj = line_dir_proj / (np.linalg.norm(line_dir_proj) + 1e-12)
    fwd_len = 1.5  # matches default plane_extent_m
    side = 0.75
    corners = np.array([
        p0 + line_dir_proj * fwd_len + y_axis * side,
        p0 + line_dir_proj * fwd_len - y_axis * side,
        p0 + line_dir_proj * 0.0     - y_axis * side,
        p0 + line_dir_proj * 0.0     + y_axis * side,
    ])
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    plane_poly = Poly3DCollection([corners], alpha=0.18, facecolor='gold', edgecolor='goldenrod')
    ax.add_collection3d(plane_poly)
    # p0
    ax.scatter(*p0, c='red', s=120, marker='*', label='p0', zorder=5)
    # Origin (base)
    ax.scatter(0, 0, 0, c='black', s=60, marker='s', label='base (world)')
    # n_target arrow
    ax.quiver(*p0, *(n_target * 0.25), color='green', linewidth=2, label='n_target (+side)')
    ax.quiver(*p0, *(line_dir * 0.25), color='blue', linewidth=2, label='line_dir')

    # Spheres: split by side
    plus_mask = signed_k > 0
    neg_mask = signed_k < 0
    if plus_mask.any():
        ax.scatter(sp_k[plus_mask, 0], sp_k[plus_mask, 1], sp_k[plus_mask, 2],
                   c='red', s=80, alpha=0.85, edgecolors='darkred',
                   label=f'sphere on +n_target side ({int(plus_mask.sum())})')
    if neg_mask.any():
        ax.scatter(sp_k[neg_mask, 0], sp_k[neg_mask, 1], sp_k[neg_mask, 2],
                   c='green', s=80, alpha=0.85, edgecolors='darkgreen',
                   label=f'sphere on -n_target side ({int(neg_mask.sum())})')
    # Excluded spheres (base/shoulder) drawn faint grey
    if (~keep).any():
        ax.scatter(sp[~keep, 0], sp[~keep, 1], sp[~keep, 2], c='lightgrey',
                   s=30, alpha=0.4, label=f'excluded link {args.exclude_links} ({int((~keep).sum())})')
    # Connect spheres in link-order (chain along arm)
    sp_sorted = sp[np.argsort(np.arange(len(sp)))]  # already in order
    ax.plot(sp_sorted[:, 0], sp_sorted[:, 1], sp_sorted[:, 2],
            'k-', lw=0.5, alpha=0.4)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    straddle_str = '⚠ STRADDLES' if straddles else '✓ entirely one side'
    ax.set_title(f'task {args.task}  q={args.which_q}  {straddle_str}\n'
                 f'+side spheres: {int(plus_mask.sum())}, -side: {int(neg_mask.sum())}')
    ax.legend(fontsize=8, loc='upper left')
    # Center the view
    all_pts = np.vstack([sp, p0[None], np.array([[0,0,0]])])
    ctr = all_pts.mean(axis=0); rng = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() * 0.6
    rng = max(rng, 0.5)
    ax.set_xlim(ctr[0]-rng, ctr[0]+rng); ax.set_ylim(ctr[1]-rng, ctr[1]+rng); ax.set_zlim(ctr[2]-rng, ctr[2]+rng)

    # Right: signed_dist bar chart, sorted by link index
    ax = fig.add_subplot(1, 2, 2)
    order = np.argsort(link_idx)
    sp_o = sp[order]; signed_o = signed[order]; link_o = link_idx[order]; rad_o = radii[order]
    keep_o = keep[order]
    colors = ['red' if s > 0 else 'green' for s in signed_o]
    for i, k in enumerate(keep_o):
        if not k:
            colors[i] = 'lightgrey'
    ax.bar(np.arange(len(signed_o)), signed_o, color=colors, edgecolor='k', linewidth=0.4)
    ax.axhline(0, color='k', lw=1.5)
    # Link boundaries
    prev_link = -1
    for i, li in enumerate(link_o):
        if li != prev_link:
            ax.axvline(i - 0.5, color='lightgrey', lw=0.5, ls='--')
            ax.text(i, ax.get_ylim()[1] * 0.95, f'link{li}',
                    fontsize=7, color='darkblue', ha='left', va='top')
            prev_link = li
    ax.set_xlabel('sphere index (sorted by link)')
    ax.set_ylabel('signed_dist (m) — positive = +n_target side')
    ax.set_title(f'Per-sphere signed_dist  ({"STRADDLE" if straddles else "no straddle"})')
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    out = args.out or (args.data.parent /
                       f'viz_collision_task{args.task}_{args.which_q}.png')
    plt.savefig(out, dpi=140, bbox_inches='tight')
    print(f'Saved: {out}')
    print(f'  signed_dist range: [{signed.min():+.3f}, {signed.max():+.3f}]')
    print(f'  (excluding link {args.exclude_links}): [{signed_k.min():+.3f}, {signed_k.max():+.3f}]')
    print(f'  straddles: {straddles}')


if __name__ == '__main__':
    main()

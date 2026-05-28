"""Check arm-vs-task-plane collision for each task's q0_seed and labels.

For each task in the NPZ:
  - signed_dist(sphere) = (sphere_center - p0) · n_target
  - sphere penetrates plane iff signed_dist < sphere_radius (with safety margin)
  - task is "plane-colliding" iff any sphere of any (q0_seed | label) penetrates

Reports overall counts + cross-tabulates with status / bucket.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from one.robots.manipulators.franka.fr3.sphere_collision import FR3SphereCollision
from one.robots.manipulators.franka.fr3_pen.batched_fr3_kin import BatchedFR3Kinematics


_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_NPZ = _REPO_ROOT / 'Yuan/seed_selection/runs/pilot_20k/pilot_20k.npz'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=DEFAULT_NPZ)
    p.add_argument('--mode', choices=['thin', 'halfspace', 'straddle'], default='straddle',
                   help="'straddle' (default, topological): arm crosses plane iff its sphere set "
                        "has some on +n_target side AND some on -n_target side. Doesn't depend on "
                        "radius or which side is 'obstacle'. "
                        "'thin': sphere intersects plane iff |signed_dist| < radius. "
                        "'halfspace': sphere on +n_target side (assumes obstacle there).")
    p.add_argument('--margin', type=float, default=0.0,
                   help='extra tolerance added to radius (m). Default 0.')
    p.add_argument('--exclude-links', type=int, nargs='*', default=[],
                   help='link indices to exclude from collision check. Default: [] (no exclusion). '
                        'For halfspace mode, [0, 1] was previously used to skip the fixed base/shoulder.')
    p.add_argument('--plane-extent-m', type=float, default=1.5,
                   help='plane extent in +line_dir direction starting from p0 (m). '
                        'Spheres projecting outside [0, plane-extent-m] along line_dir are '
                        'IGNORED (they are "off the plane"). Default: 1.5m matches pen travel. '
                        'Use a large number (e.g. 1e9) for unbounded forward.')
    p.add_argument('--device', default='cuda')
    p.add_argument('--save-flags', type=Path, default=None,
                   help='if given, save per-task collision flags as NPZ here.')
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    z = np.load(args.data, allow_pickle=False)
    N = int(z['L_seed'].shape[0])
    p0_all = z['cs_p0'].astype(np.float32)              # (N, 3)
    n_all  = z['cs_n_target'].astype(np.float32)        # (N, 3)
    n_all  = n_all / (np.linalg.norm(n_all, axis=-1, keepdims=True) + 1e-12)
    q0_seeds = z['q0_seeds'].astype(np.float32)         # (N, 7)
    labels_q0 = z['labels_q0'].astype(np.float32)       # (N, k, 7)
    n_labels = z['n_labels'].astype(np.int32)           # (N,)
    k = labels_q0.shape[1]

    # Build kin + collision.
    kin = BatchedFR3Kinematics(device=device)
    coll = FR3SphereCollision(device=device, dtype=kin.dtype)
    sphere_radii = coll.radii.detach().cpu().numpy().astype(np.float32)  # (S,)
    sphere_link_idx = coll.link_indices.detach().cpu().numpy().astype(np.int32)  # (S,)
    n_spheres = sphere_radii.shape[0]
    excluded = set(int(li) for li in args.exclude_links)
    if excluded:
        keep_sphere = np.array([int(li) not in excluded for li in sphere_link_idx], dtype=bool)
        keep_sphere_t = torch.from_numpy(keep_sphere).to(device)
        n_kept = int(keep_sphere.sum())
    else:
        keep_sphere_t = None
        n_kept = n_spheres
    print(f'[plane-coll] mode={args.mode}, FR3 sphere model: {n_spheres} spheres total, '
          f'{n_kept} after excluding link_indices={sorted(excluded)}')
    print(f'              spheres per link: '
          f'{[(int(li), int((sphere_link_idx==li).sum())) for li in sorted(set(sphere_link_idx.tolist()))]}')
    print(f'[plane-coll] N tasks: {N}')

    # Vectorized: for each task and each q (seed + k labels), FK then check spheres.
    # We process tasks in batches to fit GPU.
    BATCH = 256
    seed_coll  = np.zeros(N, dtype=bool)                 # (N,)
    seed_pen_max = np.full(N, 0.0, dtype=np.float32)     # how deep does the worst sphere go below the plane?
    label_coll = np.zeros((N, k), dtype=bool)
    label_pen_max = np.full((N, k), 0.0, dtype=np.float32)

    def fk_and_check(q_batch_np: np.ndarray, p0_b: np.ndarray, n_b: np.ndarray,
                     d_b: np.ndarray):
        """q_batch: (B, 7); p0/n/d: (B, 3). Returns (B,) collided, (B,) max penetration in meters.

        Plane is bounded: extends from p0 along +line_dir for `plane_extent_m`
        meters. Spheres whose projection along line_dir from p0 falls outside
        [0, plane_extent_m] are ignored ("off the plane").
        """
        q_t = torch.as_tensor(q_batch_np, device=device, dtype=kin.dtype)
        link_tfs = kin.link_transforms(q_t)              # (B, 8, 4, 4)
        sphere_pos = coll.sphere_positions(link_tfs)     # (B, S, 3)
        p0_t = torch.as_tensor(p0_b, device=device, dtype=kin.dtype)   # (B, 3)
        n_t  = torch.as_tensor(n_b,  device=device, dtype=kin.dtype)   # (B, 3)
        d_t  = torch.as_tensor(d_b,  device=device, dtype=kin.dtype)   # (B, 3) line_dir
        # signed_dist along the plane normal (n_target)
        signed = ((sphere_pos - p0_t[:, None, :]) * n_t[:, None, :]).sum(dim=-1)   # (B, S)
        # projection along line_dir from p0 (which sphere is "over the plane")
        proj_d = ((sphere_pos - p0_t[:, None, :]) * d_t[:, None, :]).sum(dim=-1)   # (B, S)
        radii_t = coll.radii.unsqueeze(0)                # (1, S)

        # "Over the plane" iff projection along line_dir is in [0, plane_extent_m].
        over_plane = (proj_d >= 0.0) & (proj_d <= float(args.plane_extent_m))
        # Combine with link exclusion.
        if keep_sphere_t is not None:
            valid = over_plane & keep_sphere_t.bool().unsqueeze(0)
        else:
            valid = over_plane

        # For straddle mode, apply both filters to signed via NaN.
        if args.mode == 'straddle':
            signed_m = signed.masked_fill(~valid, float('nan'))
            has_pos = (signed_m > float(args.margin)).any(dim=-1)
            has_neg = (signed_m < -float(args.margin)).any(dim=-1)
            collided_mask = (has_pos & has_neg).cpu().numpy()
            # "Penetration depth" for straddle: min(|signed|) over VALID spheres
            # — distance of the closest sphere (that's over the plane) to it.
            pen_proxy = signed_m.abs().nan_to_num(nan=float('inf')).min(dim=-1).values
            pen_proxy = pen_proxy.where(torch.isfinite(pen_proxy), torch.zeros_like(pen_proxy))
            return collided_mask, pen_proxy.cpu().numpy()

        if args.mode == 'thin':
            penetration = (radii_t - signed.abs()).clamp_min(0.0)
        else:  # halfspace
            penetration = (signed + radii_t).clamp_min(0.0)
        if keep_sphere_t is not None:
            penetration = penetration * keep_sphere_t.to(penetration.dtype).unsqueeze(0)
        collided = (penetration > args.margin).any(dim=-1)
        max_pen = penetration.amax(dim=-1)
        return collided.detach().cpu().numpy(), max_pen.detach().cpu().numpy()

    line_dir_all = z['cs_line_dir'].astype(np.float32)
    line_dir_all = line_dir_all / (np.linalg.norm(line_dir_all, axis=-1, keepdims=True) + 1e-12)

    for lo in range(0, N, BATCH):
        hi = min(lo + BATCH, N)
        idx = np.arange(lo, hi)
        # q0_seed pass
        seed_c, seed_p = fk_and_check(q0_seeds[idx], p0_all[idx], n_all[idx], line_dir_all[idx])
        seed_coll[idx] = seed_c
        seed_pen_max[idx] = seed_p
        # labels pass — only valid label slots
        for j in range(k):
            mask = n_labels[idx] > j
            sub = idx[mask]
            if len(sub) == 0: continue
            c, pp = fk_and_check(labels_q0[sub, j], p0_all[sub], n_all[sub], line_dir_all[sub])
            label_coll[sub, j] = c
            label_pen_max[sub, j] = pp

    any_label_coll = label_coll.any(axis=1)
    any_q_coll = seed_coll | any_label_coll

    # Aggregate report
    status = z['status']

    print('\n' + '=' * 70)
    print(f'Plane-collision summary (margin={args.margin}m):')
    print('=' * 70)
    print(f'  q0_seed collides       : {int(seed_coll.sum()):>5} / {N} ({100*seed_coll.mean():.1f}%)')
    print(f'  any label collides     : {int(any_label_coll.sum()):>5} / {N} ({100*any_label_coll.mean():.1f}%)')
    print(f'  any q (seed|label) coll: {int(any_q_coll.sum()):>5} / {N} ({100*any_q_coll.mean():.1f}%)')
    print()
    print('  By status (any q collides):')
    for s in ['kept', 'edge', 'low_quality', 'infeasible']:
        m = (status == s)
        if m.sum() == 0: continue
        coll = any_q_coll[m]
        print(f'    {s:<14} N={int(m.sum()):>5} | collides {int(coll.sum())} ({100*coll.mean():.1f}%)')

    # Penetration depth distribution (for tasks that collide)
    if any_q_coll.any():
        all_pens = np.concatenate([seed_pen_max[any_q_coll],
                                    label_pen_max[any_q_coll].max(axis=1)])
        all_pens = all_pens[all_pens > 0]
        print()
        print(f'  Penetration depth (m) of colliding spheres (worst sphere per task):')
        for q in [10, 25, 50, 75, 90, 99]:
            print(f'    p{q:>2d} = {np.percentile(all_pens, q):.3f}m')
        print(f'    max  = {all_pens.max():.3f}m')

    # Save flags
    if args.save_flags:
        np.savez(args.save_flags,
                 seed_collides=seed_coll,
                 label_collides=label_coll,
                 any_label_collides=any_label_coll,
                 any_q_collides=any_q_coll,
                 seed_max_penetration_m=seed_pen_max,
                 label_max_penetration_m=label_pen_max,
                 margin_m=float(args.margin))
        print(f'\nSaved per-task flags: {args.save_flags}')


if __name__ == '__main__':
    main()

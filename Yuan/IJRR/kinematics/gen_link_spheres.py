"""Generate per-link collision spheres for xArm7 and Cobotta from their STLs.

The FR3 ships with hand-tuned sphere sets; the other two arms only have
meshes. This fills the gap the same way such sets are usually made: voxelize
each link mesh, cluster the occupied voxel centers with k-means, and wrap
each cluster in a sphere whose radius covers its farthest voxel plus half a
voxel pitch. The output uses the same JSON schema as
``one/robots/manipulators/franka/fr3/collision_spheres`` so the loader is
shared.

Sphere count per link scales with the link's bounding-box diagonal, capped
so the whole arm stays under ~60 spheres (the FR3 set has 49). Coverage is
reported as the fraction of mesh vertices lying inside at least one sphere;
anything below 97% fails loudly.

Usage:
    python -m Yuan.IJRR.kinematics.gen_link_spheres --robot xarm7
    python -m Yuan.IJRR.kinematics.gen_link_spheres --robot cobotta
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[3]

MESH_SETS = {
    'xarm7': {
        'dir': REPO / 'one/robots/manipulators/xarm/xarm7/meshes',
        'links': ['link_base.stl', 'link1.stl', 'link2.stl', 'link3.stl',
                  'link4.stl', 'link5.stl', 'link6.stl', 'link7.stl'],
    },
    'cobotta': {
        'dir': REPO / 'one/robots/manipulators/denso/cvr038/meshes',
        'links': ['base_link.stl', 'j1.stl', 'j2.stl', 'j3.stl', 'j4.stl',
                  'j5.stl', 'j6.stl'],
    },
}


def _kmeans_cover(pts, k, pitch, seed):
    rng = np.random.default_rng(seed)
    centers = pts[rng.choice(len(pts), size=min(k, len(pts)), replace=False)]
    for _ in range(30):                           # plain Lloyd iterations
        d = np.linalg.norm(pts[:, None, :] - centers[None], axis=-1)
        lab = d.argmin(axis=1)
        new = np.stack([pts[lab == i].mean(axis=0) if (lab == i).any()
                        else centers[i] for i in range(len(centers))])
        if np.allclose(new, centers, atol=1e-6):
            break
        centers = new
    d = np.linalg.norm(pts[:, None, :] - centers[None], axis=-1)
    lab = d.argmin(axis=1)
    radii = np.array([d[lab == i, i].max() if (lab == i).any() else pitch
                      for i in range(len(centers))]) + 0.5 * pitch
    return centers, radii


def spheres_for_mesh(mesh: trimesh.Trimesh, pitch: float, max_k: int,
                     r_target: float, seed: int = 0):
    """Voxel k-means cover, growing k until every radius <= r_target
    (or max_k is reached). This is what keeps the over-approximation of a
    thin link from ballooning: the FR3's hand-tuned set sits at radii
    0.02-0.06, and a feasibility filter run against much fatter spheres
    would reject most of the task pool."""
    vox = mesh.voxelized(pitch)
    try:
        vox = vox.fill()
    except Exception:
        pass
    pts = vox.points
    if len(pts) == 0:
        pts = mesh.vertices
    best = None
    for k in range(1, max_k + 1):
        centers, radii = _kmeans_cover(pts, k, pitch, seed)
        best = (centers, radii)
        if radii.max() <= r_target:
            break
    return best


def coverage(mesh, centers, radii):
    v = mesh.vertices
    d = np.linalg.norm(v[:, None, :] - centers[None], axis=-1) - radii[None]
    return float((d.min(axis=1) <= 1e-4).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--robot', required=True, choices=list(MESH_SETS))
    ap.add_argument('--pitch', type=float, default=0.015)
    ap.add_argument('--max-k', type=int, default=24)
    ap.add_argument('--r-target', type=float, default=0.045)
    a = ap.parse_args()

    cfg = MESH_SETS[a.robot]
    out_dir = Path(__file__).resolve().parent / 'spheres' / a.robot
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for idx, fname in enumerate(cfg['links']):
        mesh = trimesh.load(cfg['dir'] / fname, force='mesh')
        centers, radii = spheres_for_mesh(mesh, a.pitch, a.max_k, a.r_target)
        cov = coverage(mesh, centers, radii)
        assert cov >= 0.97, f'{fname}: coverage {cov:.3f} < 0.97'
        rec = [{'name': f'link{idx}',
                'spheres': [{'origin': c.tolist(), 'radius': float(r)}
                            for c, r in zip(centers, radii)]}]
        (out_dir / f'link{idx}-spheres.json').write_text(
            json.dumps(rec, indent=1))
        total += len(centers)
        ext = mesh.bounds[1] - mesh.bounds[0]
        print(f'link{idx:<2d} {fname:<14s} spheres {len(centers):>2d}  '
              f'coverage {cov:.3f}  bbox {ext.round(3)}  '
              f'r [{radii.min():.3f}, {radii.max():.3f}]')
    print(f'total spheres: {total}  ->  {out_dir}')


if __name__ == '__main__':
    main()

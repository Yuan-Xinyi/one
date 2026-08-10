"""Render link meshes with their generated collision spheres overlaid.

One figure per robot: columns are poses (home and one bent configuration),
meshes in grey, spheres as translucent red surfaces. This is the eyeball
check that the sphere sets neither swallow the arm (over-approximation kills
the task pool) nor leave link surface uncovered (under-approximation lets
strokes pass through contact).

Usage:
    python -m Yuan.IJRR.kinematics.visualize_spheres --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib                      # before torch: CXXABI in this env
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import trimesh
import torch

from Yuan.IJRR.kinematics.batched_chain_kin import BatchedChainKinematics
from Yuan.IJRR.kinematics.gen_link_spheres import MESH_SETS

HERE = Path(__file__).resolve().parent

# A visibly bent posture per robot (within limits), so sphere fit is judged
# away from the home pose too.
BENT = {
    'xarm7':   [0.4, 0.9, -0.5, 1.6, 0.3, 1.2, 0.5],
    'cobotta': [0.5, 0.9, 1.6, 0.8, 1.0, 0.6],
}


def load_spheres(robot, n_links):
    d = HERE / 'spheres' / robot
    out = []
    for i in range(n_links):
        rec = json.loads((d / f'link{i}-spheres.json').read_text())
        sph = rec[0]['spheres'] if isinstance(rec, list) else rec['spheres']
        out.append([(np.array(s['origin']), s['radius']) for s in sph])
    return out


def draw(ax, robot):
    kin = BatchedChainKinematics(robot, device='cpu')
    cfg = MESH_SETS[robot]
    spheres = load_spheres(robot, kin.n_joints + 1)
    for qs, alpha in ((kin.q_mid.numpy(), 1.0),):
        pass
    return kin, cfg, spheres


def plot_pose(ax, kin, cfg, spheres, q, title):
    tfs = kin.link_transforms(
        torch.as_tensor(q, dtype=torch.float32).unsqueeze(0))[0].numpy()
    u, v = np.mgrid[0:2 * np.pi:14j, 0:np.pi:8j]
    for i, fname in enumerate(cfg['links']):
        T = tfs[i]
        mesh = trimesh.load(cfg['dir'] / fname, force='mesh')
        vts = (T[:3, :3] @ mesh.vertices.T).T + T[:3, 3]
        ax.plot_trisurf(vts[:, 0], vts[:, 1], vts[:, 2],
                        triangles=mesh.faces, color='0.55', alpha=0.9,
                        linewidth=0, shade=True)
        for c, r in spheres[i]:
            cw = T[:3, :3] @ c + T[:3, 3]
            xs = cw[0] + r * np.cos(u) * np.sin(v)
            ys = cw[1] + r * np.sin(u) * np.sin(v)
            zs = cw[2] + r * np.cos(v)
            ax.plot_surface(xs, ys, zs, color='crimson', alpha=0.16,
                            linewidth=0)
    all_pts = tfs[:, :3, 3]
    c = all_pts.mean(axis=0)
    span = max(0.45, 0.7 * np.abs(all_pts - c).max())
    ax.set_xlim(c[0] - span, c[0] + span)
    ax.set_ylim(c[1] - span, c[1] + span)
    ax.set_zlim(max(-0.05, c[2] - span), c[2] + span)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(title, fontsize=10)
    ax.view_init(elev=18, azim=-60)
    ax.set_axis_off()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=str(HERE / 'spheres'))
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for robot in ('xarm7', 'cobotta'):
        kin = BatchedChainKinematics(robot, device='cpu')
        cfg = MESH_SETS[robot]
        spheres = load_spheres(robot, kin.n_joints + 1)
        fig = plt.figure(figsize=(11, 6))
        for k, (q, name) in enumerate((
                (kin.q_mid.numpy(), 'home (q_mid)'),
                (np.array(BENT[robot], dtype=np.float32), 'bent'))):
            ax = fig.add_subplot(1, 2, k + 1, projection='3d')
            plot_pose(ax, kin, cfg, spheres, q, f'{robot} — {name}')
        n_sph = sum(len(s) for s in spheres)
        fig.suptitle(f'{robot}: {n_sph} collision spheres over link meshes',
                     fontsize=12)
        fig.tight_layout()
        p = out / f'spheres_{robot}.png'
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f'wrote {p}')


if __name__ == '__main__':
    main()

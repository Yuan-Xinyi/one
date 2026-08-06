"""Classical reachability analysis for the FR3 + pen used in the ISRR paper.

This is the textbook inverse-kinematics-sampling reachability map (Zacharias'
capability map / Reuleaux's reachability map), instantiated for the simulated
robot of the paper: a 7-DoF Franka Research 3 whose TCP sits at the pen tip
(flange + 0.1034 m hand + 0.10 m pen).

The Cartesian workspace is cut into voxels. Each voxel centre is paired with a
set of tool-axis directions sampled uniformly on the sphere. A pose is called
*reachable* when a joint configuration exists that

  1. puts the pen tip within ``pos_tol`` of the voxel centre,
  2. aligns the tool z-axis within ``ang_tol`` of the sampled direction,
  3. respects the FR3 joint limits, and
  4. is free of self-collision.

The reachability index of a voxel is then

    D(voxel) = (# reachable directions) / (# sampled directions)  in [0, 1].

Only the tool axis is sampled, not the full orientation: the pen is a body of
revolution, so rotation about its own axis does not change the task. This is
the usual reduction for axis-symmetric tools and keeps the direction grid
affordable.

Outputs (npz):
    voxel_xyz     (V, 3) f32  voxel centres, world frame
    D             (V,)   f32  reachability index
    n_solved      (V,)   i32  reachable direction count
    manip_max     (V,)   f32  best Yoshikawa manipulability over directions
    dirs          (M, 3) f32  the sampled tool-axis directions
    solved        (V, M) bool per-(voxel, direction) feasibility
    meta          json string with every parameter

Usage:
    python -m Yuan.reachability.reach_map --res 0.05 --n-dirs 50 --out runs/reach_5cm.npz
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from one.robots.manipulators.franka.fr3_pen.batched_fr3_kin import (
    BatchedFR3Kinematics,
)
from one.robots.manipulators.franka.fr3.sphere_collision import FR3SphereCollision

# Shoulder height of the FR3: the reachable set is roughly a ball centred here.
BASE_CENTER = (0.0, 0.0, 0.333)
# link7 -> flange 0.107 is already inside BatchedFR3Kinematics; add hand + pen.
DEFAULT_TCP_OFFSET = 0.1034 + 0.10


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
def fibonacci_directions(n: int) -> np.ndarray:
    """``n`` nearly-uniform unit vectors on the sphere (spherical Fibonacci)."""
    i = np.arange(n, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    phi = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def mean_direction_spacing(dirs: np.ndarray) -> float:
    """Mean nearest-neighbour angle of a direction set [rad]."""
    c = np.clip(dirs @ dirs.T, -1.0, 1.0)
    np.fill_diagonal(c, -1.0)
    return float(np.arccos(c.max(axis=1)).mean())


def voxel_grid(res: float, reach: float, pad: float = 1.0) -> np.ndarray:
    """Voxel centres of the axis-aligned box around the reachable ball.

    Voxels farther than ``reach`` from the shoulder are dropped up front: no
    amount of IK will place the tip there, and they would otherwise dominate
    the run time.
    """
    cx, cy, cz = BASE_CENTER
    r = reach + pad * res
    axes = [np.arange(c - r, c + r + 1e-9, res) for c in (cx, cy, cz)]
    gx, gy, gz = np.meshgrid(*axes, indexing='ij')
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    d = np.linalg.norm(pts - np.array(BASE_CENTER), axis=1)
    return pts[d <= r].astype(np.float32)


# --------------------------------------------------------------------------- #
# IK
# --------------------------------------------------------------------------- #
def _z_axis_rotvec(R_cur: torch.Tensor, z_tgt: torch.Tensor) -> torch.Tensor:
    """Rotation vector taking the current tool z-axis onto ``z_tgt``."""
    z_cur = R_cur[:, :, 2]
    axis = torch.cross(z_cur, z_tgt, dim=-1)
    s = axis.norm(dim=-1)
    c = (z_cur * z_tgt).sum(dim=-1).clamp(-1.0, 1.0)
    angle = torch.atan2(s, c)
    return axis / s.clamp_min(1e-9).unsqueeze(-1) * angle.unsqueeze(-1)


def _dls_pinv(J: torch.Tensor, damping: float) -> torch.Tensor:
    JT = J.transpose(-1, -2)
    A = J @ JT
    A = A + (damping ** 2) * torch.eye(6, device=J.device, dtype=J.dtype)
    return JT @ torch.linalg.inv(A)


def ik_z_axis(kin: BatchedFR3Kinematics,
              p_tgt: torch.Tensor,
              z_tgt: torch.Tensor,
              q_seed: torch.Tensor,
              pos_tol: float,
              ang_tol: float,
              max_iters: int = 80,
              damping: float = 1e-3):
    """Damped-least-squares IK on (position, tool-axis direction).

    Returns ``(q, ok)``. ``ok`` also requires the solution to sit inside the
    joint limits; the iterate is clamped every step so this is about whether
    the clamp had to fight the update, not a separate check.
    """
    q = q_seed.clamp(kin.lmt_lo, kin.lmt_up)
    ok = torch.zeros(q.shape[0], device=q.device, dtype=torch.bool)
    active = torch.ones_like(ok)

    for _ in range(max_iters):
        p, R, J, _ = kin.tcp_fk_jac(q)
        dp = p_tgt - p
        dth = _z_axis_rotvec(R, z_tgt)
        pos_err = dp.norm(dim=-1)
        ang_err = dth.norm(dim=-1)
        in_lmt = ((q >= kin.lmt_lo - 1e-5) & (q <= kin.lmt_up + 1e-5)).all(dim=-1)
        conv = (pos_err <= pos_tol) & (ang_err <= ang_tol) & in_lmt
        ok |= active & conv
        active &= ~conv
        if not active.any():
            break

        # Trust region, same bounds as the pipeline's projection step.
        dp = dp * torch.where(pos_err > 0.1, 0.1 / pos_err.clamp_min(1e-12),
                              torch.ones_like(pos_err)).unsqueeze(-1)
        dth = dth * torch.where(ang_err > 0.3, 0.3 / ang_err.clamp_min(1e-12),
                                torch.ones_like(ang_err)).unsqueeze(-1)
        dx = torch.cat([dp, dth], dim=-1)

        Jpinv = _dls_pinv(J, damping)
        dq = (Jpinv @ dx.unsqueeze(-1)).squeeze(-1)
        # Null-space pull toward the joint centres: keeps the iterate off the
        # limits so a solution is not rejected just for grazing one.
        N = torch.eye(7, device=q.device, dtype=q.dtype).expand(q.shape[0], 7, 7)
        N = N - Jpinv @ J
        dq = dq + (N @ (0.2 * (kin.q_mid - q)).unsqueeze(-1)).squeeze(-1)
        q_next = (q + dq).clamp(kin.lmt_lo, kin.lmt_up)
        q = torch.where(active.unsqueeze(-1), q_next, q)

    return q, ok


def manipulability(kin: BatchedFR3Kinematics, q: torch.Tensor) -> torch.Tensor:
    """Yoshikawa index on the translational Jacobian, ``sqrt(det(Jp Jp^T))``."""
    _, _, J, _ = kin.tcp_fk_jac(q)
    Jp = J[:, :3, :]
    return torch.sqrt(torch.linalg.det(Jp @ Jp.transpose(-1, -2)).clamp_min(0.0))


# --------------------------------------------------------------------------- #
# main sweep
# --------------------------------------------------------------------------- #
def build_reach_map(res: float = 0.05,
                    n_dirs: int = 50,
                    n_restarts: int = 8,
                    pos_tol: float = 5e-3,
                    ang_tol: float | None = None,
                    reach: float = 1.06,
                    chunk: int = 200_000,
                    max_iters: int = 80,
                    seed: int = 0,
                    self_collision: bool = True,
                    device: str = 'cuda',
                    verbose: bool = True) -> dict:
    dev = torch.device(device if torch.cuda.is_available() or device == 'cpu'
                       else 'cpu')
    kin = BatchedFR3Kinematics(device=dev, tcp_offset=DEFAULT_TCP_OFFSET)
    coll = FR3SphereCollision(device=dev) if self_collision else None

    dirs_np = fibonacci_directions(n_dirs)
    if ang_tol is None:
        # Half the direction grid's own resolution: finer would only measure
        # the discretisation, coarser would blur neighbouring directions.
        ang_tol = 0.5 * mean_direction_spacing(dirs_np)
    vox_np = voxel_grid(res, reach)
    V, M = len(vox_np), n_dirs
    if verbose:
        print(f'[reach] {V} voxels x {M} directions = {V * M} poses, '
              f'res={res} m, ang_tol={math.degrees(ang_tol):.2f} deg, dev={dev}')

    vox = torch.as_tensor(vox_np, device=dev)
    dirs = torch.as_tensor(dirs_np, device=dev, dtype=vox.dtype)

    # (V, M) flattened pose list.
    p_all = vox.repeat_interleave(M, dim=0)
    z_all = dirs.repeat(V, 1)

    solved = torch.zeros(V * M, device=dev, dtype=torch.bool)
    manip = torch.zeros(V * M, device=dev, dtype=vox.dtype)

    gen = torch.Generator(device=dev).manual_seed(seed)
    t0 = time.time()

    # Restart rounds: everything starts from the joint centres, then only the
    # poses still unsolved get another random seed. Reachable poses mostly
    # converge in the first round, so later rounds are cheap.
    for rnd in range(n_restarts):
        todo = (~solved).nonzero(as_tuple=True)[0]
        if todo.numel() == 0:
            break
        if verbose:
            print(f'[reach]   restart {rnd}: {todo.numel()} poses open '
                  f'({time.time() - t0:.0f}s)')
        for s in range(0, todo.numel(), chunk):
            idx = todo[s:s + chunk]
            if rnd == 0:
                q_seed = kin.q_mid.unsqueeze(0).expand(idx.numel(), 7).clone()
            else:
                q_seed = kin.rand_conf_batch(idx.numel(), generator=gen)
            q, ok = ik_z_axis(kin, p_all[idx], z_all[idx], q_seed,
                              pos_tol, ang_tol, max_iters=max_iters)
            if coll is not None and ok.any():
                hit = coll.is_collided(kin.link_transforms(q[ok]))
                ok_idx = ok.nonzero(as_tuple=True)[0]
                ok[ok_idx[hit]] = False
            if ok.any():
                good = idx[ok]
                solved[good] = True
                manip[good] = manipulability(kin, q[ok])

    solved = solved.view(V, M)
    manip = manip.view(V, M)
    n_solved = solved.sum(dim=1)
    out = {
        'voxel_xyz': vox_np,
        'D': (n_solved.float() / M).cpu().numpy().astype(np.float32),
        'n_solved': n_solved.cpu().numpy().astype(np.int32),
        'manip_max': manip.amax(dim=1).cpu().numpy().astype(np.float32),
        'dirs': dirs_np.astype(np.float32),
        'solved': solved.cpu().numpy(),
        'meta': json.dumps({
            'res': res, 'n_dirs': n_dirs, 'n_restarts': n_restarts,
            'pos_tol': pos_tol, 'ang_tol_rad': ang_tol,
            'ang_tol_deg': math.degrees(ang_tol), 'reach': reach,
            'max_iters': max_iters, 'seed': seed,
            'self_collision': self_collision,
            'tcp_offset': DEFAULT_TCP_OFFSET,
            'joint_limits_lo': kin.lmt_lo.cpu().tolist(),
            'joint_limits_up': kin.lmt_up.cpu().tolist(),
            'elapsed_s': time.time() - t0,
        }),
    }
    if verbose:
        d = out['D']
        print(f'[reach] done in {time.time() - t0:.0f}s | '
              f'reachable voxels (D>0): {(d > 0).sum()} / {V} | '
              f'mean D over those: {d[d > 0].mean():.3f} | '
              f'voxel volume {res ** 3 * (d > 0).sum() * 1e3:.1f} L')
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--res', type=float, default=0.05, help='voxel size [m]')
    p.add_argument('--n-dirs', type=int, default=50)
    p.add_argument('--n-restarts', type=int, default=8)
    p.add_argument('--pos-tol', type=float, default=5e-3)
    p.add_argument('--ang-tol-deg', type=float, default=None,
                   help='default: half the direction grid spacing')
    p.add_argument('--reach', type=float, default=1.06,
                   help='pre-filter radius from the shoulder [m]')
    p.add_argument('--chunk', type=int, default=200_000)
    p.add_argument('--max-iters', type=int, default=80)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no-self-collision', action='store_true')
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', default='Yuan/reachability/runs/reach_map.npz')
    return p.parse_args()


def main():
    a = parse_args()
    out = build_reach_map(
        res=a.res, n_dirs=a.n_dirs, n_restarts=a.n_restarts,
        pos_tol=a.pos_tol,
        ang_tol=None if a.ang_tol_deg is None else math.radians(a.ang_tol_deg),
        reach=a.reach, chunk=a.chunk, max_iters=a.max_iters, seed=a.seed,
        self_collision=not a.no_self_collision, device=a.device)
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    print(f'[reach] wrote {path}')


if __name__ == '__main__':
    main()

"""Nozzle-down feasibility field for a fixed-base arm.

A 3-D printer's constraint is exactly the task class of the paper: the tip
tracks a prescribed path and the tool axis must stay within a cone of a
reference direction, with the spin about that axis free. For planar-layer
printing the reference direction is straight down everywhere, so pointwise
feasibility is a property of POSITION ALONE once the tilt tolerance is fixed.
That collapses "largest printable object" into a query on one scalar field:

    F_theta(p) = 1  iff  there is a collision-free, in-limit q with the tip at
                        p and the tool axis within theta of -z_world

and the largest printable box at a given bed placement is the largest box
contained in {F = 1}. Computed once per tolerance, queried in closed form.

Three tolerances are swept so the nesting F_5 < F_30 < F_90 prices the nozzle
tilt allowance directly. For every feasible voxel the witness configuration's
lowest link-sphere bottom is recorded, so a bed at any height can be screened
afterwards without recomputing the field.
"""
import sys, math, time
from pathlib import Path

REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
OUT = Path('/home/lqin/one/Yuan/IJRR/runs/paper_fill/print_analysis')
sys.path.insert(0, str(REPO))
import numpy as np
import torch
from scipy.spatial import cKDTree

from Yuan.IJRR.eval import line_bound as lb
from Yuan.IJRR.stage1_seed.cone_ik import _build_R_with_z
from Yuan.IJRR.kinematics.batched_rollout import _batched_ik_project
from Yuan.IJRR.stage1_seed.iksel_clean_pilot import _minimal_rotvec, POS_SCALE

DEV = torch.device('cuda')
STEP = 0.02
TUBE = 0.002          # print position tolerance [m]; IK projector floor is 5 mm
NREF = np.array([0.0, 0.0, -1.0], np.float32)
TOLS = ((5.0, 8), (30.0, 12), (90.0, 24))


@torch.no_grad()
def feasible_field(env, tree, T, pts, cone_deg, M, tube=TUBE,
                   k_nn=100, n_try=8, chunk=8192, seed=0):
    """Per position: is there an admissible q with the tool axis within
    ``cone_deg`` of straight down?  Returns (ok, q, z_low) where z_low is the
    lowest point of any link sphere of the witness -- the minimum bed height
    that configuration clears."""
    dt_, NJ = env.kin.dtype, int(env.kin.lmt_lo.shape[0])
    P = pts.shape[0]
    cos_lim = math.cos(math.radians(cone_deg))

    rng = np.random.default_rng(seed)
    ang = rng.uniform(0.0, math.radians(cone_deg * 0.9), M)
    az = rng.uniform(0.0, 2 * math.pi, M)
    dirs = np.stack([np.sin(ang) * np.cos(az),
                     np.sin(ang) * np.sin(az),
                     -np.cos(ang)], 1).astype(np.float32)
    dirs[0] = NREF                                  # the axis itself, always

    ok = np.zeros(P, bool)
    q_out = np.full((P, NJ), np.nan, np.float32)
    hint = torch.tensor([1.0, 0.0, 0.0], dtype=dt_, device=DEV)
    Tpos, Tzax, Tq, Tji = T['pos'], T['zax'], T['q'], T['jinv6']
    nref_t = torch.as_tensor(NREF, device=DEV, dtype=dt_)

    # Directions are tried outer-most: a cheap direction that works for most
    # of the workspace resolves the bulk of the voxels in the first pass and
    # the pending set shrinks before the expensive warm starts are reached.
    for m in range(M):
        pend = np.nonzero(~ok)[0]
        if not len(pend):
            break
        z_m = np.broadcast_to(dirs[m], (len(pend), 3)).copy()
        cand = np.empty((len(pend), n_try, NJ), np.float32)
        for lo in range(0, len(pend), chunk):
            hi = min(lo + chunk, len(pend))
            rows = pend[lo:hi]
            feat = np.concatenate([pts[rows] * POS_SCALE, z_m[lo:hi]], 1)
            _, ids = tree.query(feat.astype(np.float32), k=k_nn, workers=-1)
            C = hi - lo
            dp = pts[rows][:, None, :] - Tpos[ids]
            rv = _minimal_rotvec(Tzax[ids].reshape(-1, 3),
                                 np.repeat(z_m[lo:hi], k_nn, 0)).reshape(C, k_nn, 3)
            d6 = np.concatenate([dp, rv], -1).astype(np.float32)
            dq = np.einsum('ckje,cke->ckj', Tji[ids], d6)
            order = (dq * dq).sum(-1).argsort(1)[:, :n_try]
            cand[lo:hi] = Tq[np.take_along_axis(ids, order, 1)]

        sub = np.arange(len(pend))
        for t in range(n_try):
            if not len(sub):
                break
            for c_lo in range(0, len(sub), chunk):
                s = sub[c_lo:c_lo + chunk]
                rows = pend[s]
                q0 = torch.as_tensor(cand[s, t], device=DEV, dtype=dt_)
                p_t = torch.as_tensor(pts[rows], device=DEV, dtype=dt_)
                R_t = _build_R_with_z(
                    torch.as_tensor(z_m[s], device=DEV, dtype=dt_), hint)
                q_o, _, _ = _batched_ik_project(env.kin, q0, p_t, R_t,
                                                branch_action=None)
                coll = env.collision.is_collided(env.kin.link_transforms(q_o))
                p_fk, R_fk, _, _ = env.kin.tcp_fk_jac(q_o)
                in_lmt = ((q_o >= env.kin.lmt_lo - 1e-5)
                          & (q_o <= env.kin.lmt_up + 1e-5)).all(dim=-1)
                fine = ((~coll) & in_lmt
                        & ((p_fk - p_t).norm(dim=-1) <= tube)
                        & ((R_fk[:, :, 2] * nref_t).sum(-1) >= cos_lim))
                f = fine.cpu().numpy()
                ok[rows[f]] = True
                q_out[rows[f]] = q_o[fine].cpu().numpy()
            sub = sub[~ok[pend[sub]]]

    # Lowest link-sphere bottom of each witness: the highest bed the arm clears.
    # link0 is excluded -- it is the mount, it extends 39 mm below the mounting
    # plane by construction, and demanding that the fixed base clear the table
    # it is bolted to would rule out every configuration at bed height zero.
    z_low = np.full(P, np.nan, np.float32)
    arm = (env.collision.link_indices >= 1)
    rad = env.collision.radii[arm]
    idx = np.nonzero(ok)[0]
    for lo in range(0, len(idx), chunk):
        rows = idx[lo:lo + chunk]
        q_t = torch.as_tensor(q_out[rows], device=DEV, dtype=dt_)
        cen = env.collision.sphere_positions(env.kin.link_transforms(q_t))
        z_low[rows] = (cen[:, arm, 2] - rad[None, :]).amin(dim=-1).cpu().numpy()
    return ok, q_out, z_low


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = lb.build_env(DEV, 'stock', 512)
    print(f'[field] tcp_offset {float(env.kin.tcp_offset):.4f} m, '
          f'{int(env.kin.lmt_lo.shape[0])} joints, '
          f'{len(env.collision.radii)} collision spheres', flush=True)
    T = np.load(REPO / lb.TABLE)
    tree = cKDTree(np.concatenate([T['pos'] * POS_SCALE, T['zax']], 1)
                   .astype(np.float32))
    ptree = cKDTree(T['pos'])

    gx = np.arange(-1.00, 1.0001, STEP, dtype=np.float32)
    # down to -0.50: the arm reaches half a metre below its own mount, so a bed
    # on the floor beside a pedestal is inside the workspace
    gz = np.arange(-0.50, 1.1001, STEP, dtype=np.float32)
    NX, NZ = len(gx), len(gz)
    X, Y, Z = np.meshgrid(gx, gx, gz, indexing='ij')
    grid = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1).astype(np.float32)

    # prune: outside the arm's reach, or far from every warm-start sample
    keep = np.linalg.norm(grid, axis=1) <= 1.12
    dnn, _ = ptree.query(grid[keep], k=1, workers=-1)
    keep[np.nonzero(keep)[0][dnn > 0.15]] = False
    pts = grid[keep]
    print(f'[field] grid {NX}x{NX}x{NZ} = {len(grid)} voxels at {STEP*100:.0f} mm, '
          f'{len(pts)} survive the reach prune', flush=True)

    res = {}
    for cone, M in TOLS:
        t0 = time.time()
        ok, q, zl = feasible_field(env, tree, T, pts, cone, M)
        full = np.zeros(len(grid), bool); full[keep] = ok
        zlf = np.full(len(grid), np.nan, np.float32); zlf[keep] = zl
        res[f'F{cone:g}'] = full.reshape(NX, NX, NZ)
        res[f'ZL{cone:g}'] = zlf.reshape(NX, NX, NZ)
        print(f'[field] cone {cone:4.0f} deg  feasible {ok.mean():.3f} of probed, '
              f'{full.sum()} voxels, {time.time()-t0:6.1f}s', flush=True)

    np.savez_compressed(OUT / 'down_field.npz', gx=gx, gz=gz, step=np.float32(STEP),
                        tube=np.float32(TUBE), **res)
    print(f'[field] wrote {OUT / "down_field.npz"}')


if __name__ == '__main__':
    main()

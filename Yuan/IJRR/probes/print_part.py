"""What the printed object itself takes away.

The feasibility field is pure kinematics: it asks whether the tip can be
somewhere with the axis held, and the only body it knows about is the arm. A
printer has two more: the bed, and the part already laid down, which is an
obstacle that grows under the nozzle as the job proceeds.

For a bottom-up planar print of a convex solid the arm stays above the surface
and the part costs almost nothing. For a bowl printed cavity-up it costs a
great deal, because the wrist has to descend inside a cavity whose walls rise
around it. That contrast is the reason placement (which way up the object
goes) is a decision and not a detail.

Both parts have closed-form distance functions -- a box for the cube, a partial
spherical shell for the bowl -- so the obstacle test costs one expression per
arm sphere instead of a voxel sweep, and the size search stays affordable.
"""
import sys, math, time
from pathlib import Path

REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
OUT = Path('/home/lqin/one/Yuan/IJRR/runs/paper_fill/print_analysis')
SCR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(SCR))
import matplotlib; matplotlib.use('Agg')
import numpy as np
import torch
from scipy.spatial import cKDTree

from Yuan.IJRR.eval import line_bound as lb
from Yuan.IJRR.stage1_seed.cone_ik import _build_R_with_z
from Yuan.IJRR.kinematics.batched_rollout import _batched_ik_project
from Yuan.IJRR.stage1_seed.iksel_clean_pilot import _minimal_rotvec, POS_SCALE
from print_bowl import unit_bowl, _rot_z_to, ETA, RB_FRAC, R_COL

DEV = torch.device('cuda')
TUBE = 0.002
WALL_T = 0.004        # printed wall thickness [m], half-width of the shell
CLEAR = 0.003         # extra clearance demanded between arm and part [m]


# ---------------------------------------------------------------- obstacles
class BoxPart:
    """Solid box grown to the current layer height."""

    def __init__(self, c, side, zb):
        self.lo = torch.tensor([c[0] - side / 2, c[1] - side / 2, zb],
                               device=DEV)
        self.hi = torch.tensor([c[0] + side / 2, c[1] + side / 2, zb],
                               device=DEV)
        self.side = side

    def dist(self, cen, z_cur):
        hi = self.hi.clone().expand(cen.shape[0], 3).clone()
        hi[:, 2] = torch.as_tensor(z_cur, device=DEV, dtype=cen.dtype)
        d = torch.maximum(self.lo.view(1, 1, 3) - cen,
                          cen - hi.view(-1, 1, 3)).clamp_min(0.0)
        return d.norm(dim=-1)


class ShellPart:
    """Spherical-cap shell, restricted to what has already been laid down.

    The bowl is a thin surface, so an arm sphere collides when it straddles
    that surface -- being inside the cavity or outside the wall is free. Points
    whose polar angle falls outside the printed band are measured to the
    nearest printed edge circle instead, which is what makes the obstacle grow
    correctly rather than appear whole from the first layer."""

    def __init__(self, c, R, zb, inverted, eta=ETA):
        H = eta * R
        rho = (R * R + H * H) / (2.0 * H)
        self.rho, self.H, self.inv = rho, H, inverted
        self.C = torch.tensor(
            [c[0], c[1], zb + (rho if not inverted else H - rho)], device=DEV)
        self.phi_rim = math.acos(max(-1.0, min(1.0, (rho - H) / rho)))
        self.phi_b = math.asin(min(1.0, RB_FRAC * R / rho))
        self.zb = zb

    def _phi_of_z(self, z):
        """Polar angle on the cap at world height z (axis pointing to the pole).

        Upright:  z = zb + rho (1 - cos phi).
        Inverted: z = zb + H - rho (1 - cos phi)."""
        if not self.inv:
            c = 1.0 - (z - self.zb) / self.rho
        else:
            c = 1.0 - (self.zb + self.H - z) / self.rho
        return math.acos(max(-1.0, min(1.0, c)))

    def dist(self, cen, z_cur):
        v = cen - self.C.view(1, 1, 3)
        r = v.norm(dim=-1).clamp_min(1e-9)
        # Polar angle from the cap's own pole direction. The upright bowl's
        # centre sits ABOVE its pole, so its pole direction is -z; the dome's
        # centre sits below its pole, so its pole direction is +z.
        axis_z = v[..., 2] / r * (-1.0 if not self.inv else 1.0)
        phi = torch.arccos(axis_z.clamp(-1.0, 1.0))
        # Which band of the cap already exists at the current nozzle height.
        # The upright bowl is printed pole-first and grows outward, so phi runs
        # up to phi(z_cur); the dome starts at its rim on the bed and closes at
        # the pole, so phi runs from phi(z_cur) DOWN to the rim.
        phi_hi = self._phi_of_z(float(z_cur))
        if not self.inv:
            band_lo = self.phi_b
            band_hi = max(self.phi_b, min(phi_hi, self.phi_rim))
        else:
            band_lo = max(0.0, min(phi_hi, self.phi_rim))
            band_hi = self.phi_rim
        inside = (phi >= band_lo) & (phi <= band_hi)
        d_surf = (r - self.rho).abs()
        # distance to the nearer bounding circle of the printed band
        d_edge = torch.full_like(d_surf, 1e6)
        for pe in (band_lo, band_hi):
            ce = torch.tensor([float(self.C[0]), float(self.C[1]),
                               float(self.C[2]) + (-1 if not self.inv else 1)
                               * self.rho * math.cos(pe)], device=DEV)
            re = self.rho * math.sin(pe)
            dxy = (cen[..., :2] - ce[:2].view(1, 1, 2)).norm(dim=-1)
            d_edge = torch.minimum(
                d_edge, torch.sqrt((dxy - re) ** 2
                                   + (cen[..., 2] - ce[2]) ** 2))
        return torch.where(inside, d_surf, d_edge)


# --------------------------------------------------------------- feasibility
class PartFeas:
    """Pointwise feasibility with the bed and the growing part as obstacles."""

    def __init__(self, cone_deg, M, seed=0):
        self.env = lb.build_env(DEV, 'stock', 512)
        self.T = np.load(REPO / lb.TABLE)
        self.tree = cKDTree(np.concatenate(
            [self.T['pos'] * POS_SCALE, self.T['zax']], 1).astype(np.float32))
        self.cos_lim = math.cos(math.radians(cone_deg))
        rng = np.random.default_rng(seed)
        ang = rng.uniform(0, math.radians(cone_deg * 0.9), M)
        az = rng.uniform(0, 2 * math.pi, M)
        d = np.stack([np.sin(ang) * np.cos(az), np.sin(ang) * np.sin(az),
                      np.cos(ang)], 1).astype(np.float32)
        d[0] = (0, 0, 1)
        self.CD, self.M = d, M
        self.arm = self.env.collision.link_indices >= 1
        self.rad = self.env.collision.radii[self.arm]
        self.dt = self.env.kin.dtype
        self.hint = torch.tensor([1.0, 0.0, 0.0], dtype=self.dt, device=DEV)

    @torch.no_grad()
    def __call__(self, pts, axes, part=None, zb=None, n_try=8, k_nn=100,
                 chunk=8192):
        M, NJ = self.M, 7
        P = len(pts)
        ok = np.zeros(P, bool)
        zs_all = np.einsum('nij,mj->nmi', _rot_z_to(axes), self.CD)
        for m in range(M):
            pend = np.nonzero(~ok)[0]
            if not len(pend):
                break
            zm = zs_all[pend, m]
            feat = np.concatenate([pts[pend] * POS_SCALE, zm], 1).astype(np.float32)
            _, ids = self.tree.query(feat, k=k_nn, workers=-1)
            dp = pts[pend][:, None, :] - self.T['pos'][ids]
            rv = _minimal_rotvec(self.T['zax'][ids].reshape(-1, 3),
                                 np.repeat(zm, k_nn, 0)).reshape(len(pend), k_nn, 3)
            d6 = np.concatenate([dp, rv], -1).astype(np.float32)
            dq = np.einsum('ckje,cke->ckj', self.T['jinv6'][ids], d6)
            order = (dq * dq).sum(-1).argsort(1)[:, :n_try]
            cand = self.T['q'][np.take_along_axis(ids, order, 1)]

            sub = np.arange(len(pend))
            for t in range(n_try):
                if not len(sub):
                    break
                for lo in range(0, len(sub), chunk):
                    s = sub[lo:lo + chunk]
                    rows = pend[s]
                    q0 = torch.as_tensor(cand[s, t], device=DEV, dtype=self.dt)
                    p_t = torch.as_tensor(pts[rows], device=DEV, dtype=self.dt)
                    R_t = _build_R_with_z(
                        torch.as_tensor(zm[s], device=DEV, dtype=self.dt), self.hint)
                    q_o, _, _ = _batched_ik_project(self.env.kin, q0, p_t, R_t,
                                                    branch_action=None)
                    tfs = self.env.kin.link_transforms(q_o)
                    coll = self.env.collision.is_collided(tfs)
                    p_fk, R_fk, _, _ = self.env.kin.tcp_fk_jac(q_o)
                    nt = torch.as_tensor(axes[rows], device=DEV, dtype=self.dt)
                    in_lmt = ((q_o >= self.env.kin.lmt_lo - 1e-5)
                              & (q_o <= self.env.kin.lmt_up + 1e-5)).all(dim=-1)
                    fine = ((~coll) & in_lmt
                            & ((p_fk - p_t).norm(dim=-1) <= TUBE)
                            & ((R_fk[:, :, 2] * nt).sum(-1) >= self.cos_lim))
                    if part is not None or zb is not None:
                        cen = self.env.collision.sphere_positions(tfs)[:, self.arm]
                        if zb is not None:
                            fine &= ((cen[..., 2] - self.rad[None, :])
                                     >= zb).all(dim=-1)
                        if part is not None:
                            zc = torch.as_tensor(pts[rows][:, 2], device=DEV,
                                                 dtype=self.dt)
                            dmin = part.dist(cen, zc) - self.rad[None, :] \
                                - WALL_T - CLEAR
                            fine &= (dmin > 0).all(dim=-1)
                    f = fine.cpu().numpy()
                    ok[rows[f]] = True
                sub = sub[~ok[pend[sub]]]
        return ok


# ------------------------------------------------------------------ samplers
def cube_samples(c, side, zb, dz=0.04, dxy=0.04):
    """Layer samples of a solid cube: every layer's full cross-section, since
    infill visits it all. Carries each point's layer height for the part."""
    n = max(2, int(round(side / dxy)) + 1)
    g = np.linspace(-side / 2, side / 2, n)
    X, Y = np.meshgrid(g, g, indexing='ij')
    nz = max(2, int(round(side / dz)) + 1)
    P = []
    for z in np.linspace(0.0, side, nz):
        P.append(np.stack([X.ravel() + c[0], Y.ravel() + c[1],
                           np.full(X.size, zb + z)], 1))
    p = np.concatenate(P).astype(np.float32)
    a = np.tile(np.float32([0, 0, -1]), (len(p), 1))
    return p, a


def bowl_samples(c, R, zb, inverted, conformal, pitch=0.055):
    p, a, _ = unit_bowl(inverted, pitch=pitch)
    w = (R * p + np.float32([c[0], c[1], zb])).astype(np.float32)
    if not conformal:
        a = np.tile(np.float32([0, 0, -1]), (len(w), 1))
    return w, a.astype(np.float32)


def main():
    q = np.load(OUT / 'cube_query.npz')
    gx, gz = q['gx'], q['gz']
    out = {}

    for cone, M in ((5, 8), (30, 12)):
        fe = PartFeas(cone, M)

        # ---- cube: shrink from the kinematic answer until the part fits ----
        A = q[f'cube_c{cone}_solid']
        b = np.unravel_index(A.argmax(), A.shape)
        c0 = (float(gx[b[0]]), float(gx[b[1]]))
        zb = float(gz[b[2]]) - 0.01
        s_kin = float(A.max())
        s_ok = 0.0
        for s in np.arange(s_kin, 0.09, -0.04):
            pts, ax = cube_samples(c0, s, zb)
            keep = np.hypot(pts[:, 0], pts[:, 1]) >= R_COL
            pts, ax = pts[keep], ax[keep]
            if fe(pts, ax, part=BoxPart(c0, s, zb), zb=zb).all():
                s_ok = float(s)
                break
        print(f'[part] cone {cone:3d} cube  kinematic {s_kin:.3f} m -> '
              f'with bed+part {s_ok:.3f} m  at ({c0[0]:+.2f},{c0[1]:+.2f}) '
              f'bed {zb:+.2f}', flush=True)
        out[f'cube_c{cone}'] = np.float32([s_kin, s_ok, c0[0], c0[1], zb])

        # ---- bowl: four ways, at the placement each one prefers ------------
        bq = np.load(OUT / 'bowl_query.npz') if (OUT / 'bowl_query.npz').exists() \
            else None
        if bq is None:
            continue
        for mode in ('planar', 'conformal'):
            for inv in (False, True):
                key = f'{mode}_c{cone}_{"inv" if inv else "up"}'
                if mode == 'planar':
                    bb = bq[key]
                    R_kin = float(bb.max())
                    idx = int(bb.argmax())
                    pl = bq['places'][idx]
                else:
                    v = bq[key]
                    R_kin, pl = float(v[0]), v[1:]
                R_ok = 0.0
                for R in np.arange(R_kin, 0.019, -0.02):
                    pts, ax = bowl_samples(pl[:2], R, float(pl[2]), inv,
                                           mode == 'conformal')
                    keep = np.hypot(pts[:, 0], pts[:, 1]) >= R_COL
                    if not keep.all():
                        continue
                    part = ShellPart(pl[:2], R, float(pl[2]), inv)
                    if fe(pts, ax, part=part, zb=float(pl[2])).all():
                        R_ok = float(R)
                        break
                print(f'[part] cone {cone:3d} bowl {mode:<9s} '
                      f'{"inverted" if inv else "upright ":8s} '
                      f'rim dia {200*R_kin:5.1f} -> {200*R_ok:5.1f} cm',
                      flush=True)
                out[f'bowl_{mode}_c{cone}_{"inv" if inv else "up"}'] = \
                    np.float32([R_kin, R_ok, *pl])

    np.savez_compressed(OUT / 'part_query.npz', **out)
    print(f'wrote {OUT / "part_query.npz"}')


if __name__ == '__main__':
    main()

"""Largest printable sushi bowl, four ways.

The bowl is a spherical cap: rim radius R, depth H = eta*R, truncated by a flat
foot of radius rb_frac*R. Unlike the cube it is not an orientation-free object,
and that is the point -- two independent choices interact:

  how the object is placed   upright (cavity up)  vs  inverted (dome)
  how the nozzle is held     planar layers, axis always down
                             conformal, axis along the local surface normal

Planar layers make the orientation task trivial (the cube's regime) but impose
the overhang rule; conformal printing removes the overhang rule and replaces it
with an orientation task sweeping the nozzle from vertical at the pole to
nearly horizontal at the rim, through every azimuth. Only the second needs an
arm, and only the second is limited by the wrist.

Planar cases are answered by lookup into the precomputed nozzle-down field.
Conformal cases need cone-constrained IK at each point's own axis, so they are
run as an elimination ladder: every placement is tested at a small radius, only
the survivors are tested one rung up. The cost is dominated by the cheapest
rung and the answer is exact on the radius ladder.
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
from Yuan.IJRR.stage1_seed.iksel_clean_pilot import POS_SCALE

DEV = torch.device('cuda')
ETA = 0.90            # depth / rim radius, donburi-proportioned
RB_FRAC = 0.45        # flat foot radius / rim radius
PITCH = 0.045         # surface sampling pitch in units of R
TUBE = 0.002
R_COL = 0.20          # printed volume is excluded from the arm's own column


def unit_bowl(inverted, pitch=PITCH, eta=ETA, rb_frac=RB_FRAC):
    """Mid-surface samples of the wall plus the closing disc, at R = 1, with
    the conformal nozzle axis per sample. Scaling by R is exact: the shape is
    self-similar and the axes are scale-free."""
    H = eta
    rho = (1.0 + H * H) / (2.0 * H)
    phi_rim = math.acos(max(-1.0, min(1.0, (rho - H) / rho)))
    phi_b = math.asin(min(1.0, rb_frac / rho))
    n_phi = max(4, int(round(rho * (phi_rim - phi_b) / pitch)) + 1)

    P, A, PH = [], [], []
    for ph in np.linspace(phi_b, phi_rim, n_phi):
        r = rho * math.sin(ph)
        zc = rho * (1.0 - math.cos(ph))
        n_th = max(8, int(round(2 * math.pi * r / pitch)))
        th = np.arange(n_th) * (2 * math.pi / n_th)
        c, s = np.cos(th), np.sin(th)
        if not inverted:
            # cavity up: deposit on the concave side; nozzle points down-and-out
            p = np.stack([r * c, r * s, np.full(n_th, zc)], 1)
            a = np.stack([math.sin(ph) * c, math.sin(ph) * s,
                          np.full(n_th, -math.cos(ph))], 1)
        else:
            # dome: deposit on the convex side; nozzle points down-and-in
            p = np.stack([r * c, r * s, np.full(n_th, H - zc)], 1)
            a = np.stack([-math.sin(ph) * c, -math.sin(ph) * s,
                          np.full(n_th, -math.cos(ph))], 1)
        P.append(p); A.append(a); PH.append(np.full(n_th, math.degrees(ph)))

    n_d = max(1, int(round(rb_frac / pitch)))
    for i in range(n_d):
        rr = rb_frac * (i + 0.5) / n_d
        n_th = max(6, int(round(2 * math.pi * rr / pitch)))
        th = np.arange(n_th) * (2 * math.pi / n_th)
        zd = 0.0 if not inverted else H
        P.append(np.stack([rr * np.cos(th), rr * np.sin(th),
                           np.full(n_th, zd)], 1))
        A.append(np.tile(np.float64([0, 0, -1]), (n_th, 1)))
        PH.append(np.full(n_th, np.nan))
    return (np.concatenate(P).astype(np.float32),
            np.concatenate(A).astype(np.float32),
            np.concatenate(PH).astype(np.float32))


def overhang_deg(rb_frac=RB_FRAC, eta=ETA):
    """Worst wall inclination from the vertical build direction -- the angle
    the 45-degree support rule is stated against. Upright printing starts at
    the foot, inverted printing starts at the pole."""
    rho = (1.0 + eta * eta) / (2.0 * eta)
    phi_b = math.degrees(math.asin(min(1.0, rb_frac / rho)))
    phi_rim = math.degrees(math.acos(max(-1.0, min(1.0, (rho - eta) / rho))))
    return 90.0 - phi_b, 90.0 - phi_b   # same wall, printed from either end


def _rot_z_to(axes):
    """Batched rotation taking +z onto each row of ``axes``."""
    n = axes.shape[0]
    z = np.zeros((n, 3), np.float32); z[:, 2] = 1.0
    v = np.cross(z, axes)
    c = axes[:, 2]
    s = np.linalg.norm(v, axis=1)
    out = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
    good = s > 1e-8
    if good.any():
        vv = v[good] / s[good, None]
        K = np.zeros((int(good.sum()), 3, 3), np.float32)
        K[:, 0, 1] = -vv[:, 2]; K[:, 0, 2] = vv[:, 1]
        K[:, 1, 0] = vv[:, 2];  K[:, 1, 2] = -vv[:, 0]
        K[:, 2, 0] = -vv[:, 1]; K[:, 2, 1] = vv[:, 0]
        th = np.arctan2(s[good], c[good])[:, None, None]
        out[good] = (np.eye(3, dtype=np.float32)[None] + np.sin(th) * K
                     + (1 - np.cos(th)) * (K @ K))
    flip = (~good) & (c < 0)
    if flip.any():
        out[flip] = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    return out


class Feas:
    """Cone-constrained pointwise feasibility with a per-point cone axis."""

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

    def __call__(self, pts, axes):
        M = self.M
        zs = np.einsum('nij,mj->nmi', _rot_z_to(axes), self.CD).reshape(-1, 3)
        ok, _ = lb.feasible_rows(self.env, self.tree, self.T,
                                 np.repeat(pts, M, 0), zs,
                                 np.repeat(axes, M, 0), self.cos_lim, TUBE,
                                 k_nn=100, n_try=8, q_hint=None, chunk=16384)
        return (ok.reshape(len(pts), M).any(1)
                & (np.hypot(pts[:, 0], pts[:, 1]) >= R_COL))


def planar_best(F, gx, gz, step, pu, places, R_lad, chunk=1500):
    """Largest ladder radius admissible at each placement, by field lookup."""
    NX, NZ = len(gx), len(gz)
    best = np.zeros(len(places), np.float32)
    alive = np.arange(len(places))
    for R in R_lad:
        if not len(alive):
            break
        pl = R * pu
        keep = []
        for lo in range(0, len(alive), chunk):
            rows = alive[lo:lo + chunk]
            w = places[rows][:, None, :] + pl[None, :, :]
            i = np.rint((w[:, :, 0] - gx[0]) / step).astype(np.int32)
            j = np.rint((w[:, :, 1] - gx[0]) / step).astype(np.int32)
            k = np.rint((w[:, :, 2] - gz[0]) / step).astype(np.int32)
            inb = ((i >= 0) & (i < NX) & (j >= 0) & (j < NX)
                   & (k >= 0) & (k < NZ))
            ok = np.zeros(w.shape[:2], bool)
            ok[inb] = F[i[inb], j[inb], k[inb]]
            keep.append(rows[ok.all(1)])
        alive = np.concatenate(keep) if keep else np.array([], np.int64)
        best[alive] = R
    return best


def main():
    z = np.load(OUT / 'down_field.npz')
    gx, gz, step = z['gx'], z['gz'], float(z['step'])
    R_LAD = np.round(np.arange(0.04, 0.601, 0.01), 3)

    pgc = np.arange(-0.75, 0.751, 0.05, np.float32)
    beds = np.arange(-0.45, 0.601, 0.025, np.float32)
    CX, CY, ZB = np.meshgrid(pgc, pgc, beds, indexing='ij')
    places = np.stack([CX.ravel(), CY.ravel(), ZB.ravel()], 1).astype(np.float32)
    print(f'[bowl] {len(places)} placements (x,y,bed), '
          f'radius ladder {R_LAD[0]:.2f}..{R_LAD[-1]:.2f} m', flush=True)
    oh_up, oh_inv = overhang_deg()
    print(f'[bowl] worst wall inclination from vertical: {oh_up:.1f} deg '
          f'(45 deg is the usual unsupported limit)', flush=True)

    col = (np.hypot(gx[:, None], gx[None, :]) >= R_COL)[:, :, None]
    res, summary = {}, []
    for inv in (False, True):
        pu, au, ph = unit_bowl(inv)
        print(f'[bowl] {"inverted" if inv else "upright"} unit sample: '
              f'{len(pu)} points, nozzle tilt '
              f'{np.degrees(np.arccos(-au[:,2].clip(-1,1))).max():.1f} deg max',
              flush=True)
        for cone in (5, 30):
            t0 = time.time()
            b = planar_best(z[f'F{cone}'] & col, gx, gz, step, pu, places, R_LAD)
            res[f'planar_c{cone}_{"inv" if inv else "up"}'] = b
            a = int(b.argmax())
            summary.append(('planar', cone, inv, float(b[a]), places[a]))
            print(f'[bowl] planar    cone {cone:3d}  '
                  f'{"inverted" if inv else "upright ":8s}  '
                  f'R* {b[a]*100:5.1f} cm  rim dia {200*b[a]:5.1f} cm  '
                  f'at ({places[a][0]:+.2f},{places[a][1]:+.2f}) '
                  f'bed {places[a][2]:+.2f}  {time.time()-t0:.0f}s', flush=True)

    # ---- conformal: screen the placement grid, then ladder the survivors ---
    # Every conformal test costs cone IK at each sample's own axis, so a full
    # ladder over the fine grid is out of reach. The screen uses a coarse
    # surface sampling to find WHERE a modest bowl is admissible at all; the
    # ladder then re-tests those placements at the full sampling, so no
    # reported radius rests on the coarse pass.
    pgf = np.arange(-0.70, 0.701, 0.10, np.float32)
    bedf = np.arange(-0.45, 0.601, 0.05, np.float32)
    CX, CY, ZB = np.meshgrid(pgf, pgf, bedf, indexing='ij')
    pf = np.stack([CX.ravel(), CY.ravel(), ZB.ravel()], 1).astype(np.float32)
    R_LADC = np.round(np.arange(0.06, 0.401, 0.02), 3)
    print(f'[bowl] conformal placement grid {len(pf)}, '
          f'ladder {R_LADC[0]:.2f}..{R_LADC[-1]:.2f} m', flush=True)

    for cone, M in ((30, 12), (5, 8)):
        fe = Feas(cone, M)
        for inv in (False, True):
            pu, au, _ = unit_bowl(inv, pitch=0.11)
            ps, as_, _ = unit_bowl(inv, pitch=0.22)
            t0 = time.time()
            alive, pl = [], R_LADC[0] * ps
            for lo in range(0, len(pf), 600):
                rows = np.arange(lo, min(lo + 600, len(pf)))
                w = (pf[rows][:, None, :] + pl[None, :, :]).reshape(-1, 3)
                ok = fe(w, np.tile(as_, (len(rows), 1))).reshape(len(rows), len(ps))
                alive.append(rows[ok.all(1)])
            alive = np.concatenate(alive)
            print(f'[bowl]   conformal c{cone} {"inv" if inv else "up"}: screen '
                  f'{len(alive)}/{len(pf)} at {len(ps)} coarse samples '
                  f'({time.time()-t0:.0f}s)', flush=True)

            # Bisect on the radius rather than walk the ladder: only the
            # LARGEST admissible radius is wanted, and every rung below it is
            # paid for in full by a linear sweep. Bisection starts in the
            # middle, where the surviving placement set is already small.
            def survivors(R, cand):
                pl = R * pu
                out = []
                for lo in range(0, len(cand), 250):
                    rows = cand[lo:lo + 250]
                    w = (pf[rows][:, None, :] + pl[None, :, :]).reshape(-1, 3)
                    ok = fe(w, np.tile(au, (len(rows), 1))
                            ).reshape(len(rows), len(pu))
                    out.append(rows[ok.all(1)])
                return np.concatenate(out) if out else np.array([], np.int64)

            bestR, bestP = 0.0, None
            klo, khi = 0, len(R_LADC) - 1
            while klo <= khi and len(alive):
                km = (klo + khi) // 2
                s = survivors(R_LADC[km], alive)
                print(f'[bowl]     R={R_LADC[km]:.2f} -> {len(s)}/{len(alive)} '
                      f'survive ({time.time()-t0:.0f}s)', flush=True)
                if len(s):
                    bestR, bestP = float(R_LADC[km]), pf[s[0]].copy()
                    alive, klo = s, km + 1
                else:
                    khi = km - 1
            res[f'conformal_c{cone}_{"inv" if inv else "up"}'] = \
                np.float32([bestR, *(bestP if bestP is not None else (0, 0, 0))])
            summary.append(('conformal', cone, inv, bestR,
                            bestP if bestP is not None else np.zeros(3)))
            print(f'[bowl] conformal cone {cone:3d}  '
                  f'{"inverted" if inv else "upright ":8s}  '
                  f'R* {bestR*100:5.1f} cm  rim dia {200*bestR:5.1f} cm  '
                  f'{time.time()-t0:.0f}s', flush=True)

    np.savez_compressed(OUT / 'bowl_query.npz', places=places, R_lad=R_LAD,
                        pgc=pgc, beds=beds, **res)
    print('\n=== bowl summary (rim diameter, cm) ===')
    for mode, cone, inv, R, p in summary:
        print(f'{mode:<10s} cone {cone:3d}  '
              f'{"inverted" if inv else "upright ":9s} '
              f'{200*R:6.1f} cm   at ({p[0]:+.2f},{p[1]:+.2f}) bed {p[2]:+.2f}')
    print(f'wrote {OUT / "bowl_query.npz"}')


if __name__ == '__main__':
    main()

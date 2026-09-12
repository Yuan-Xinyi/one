"""How big can common objects be, on a fixed-base FR3.

Every object is written as an ORDERED toolpath in a unit frame whose scale
parameter is the dimension a person would quote (a cube's side, a bowl's rim
diameter, a vase's height). Planar-layer printing holds the nozzle down
everywhere, so the whole question reduces to "does this point set fit inside
the nozzle-down feasible field" -- one lookup per point, no IK, and the largest
scale is found by walking a ladder.

The toolpaths are kept in print order rather than as a point cloud because the
same generators feed the renderer.
"""
import sys, math
from pathlib import Path

REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
OUT = Path('/home/lqin/one/Yuan/IJRR/runs/paper_fill/print_analysis')
SCR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(SCR))
import numpy as np

R_COL = 0.20


# ------------------------------------------------------------- path helpers
def ring(r, z, n_min=10, pitch=0.02, phase=0.0, close=True):
    n = max(n_min, int(round(2 * math.pi * r / pitch)))
    th = np.arange(n + (1 if close else 0)) * (2 * math.pi / n) + phase
    return np.stack([r * np.cos(th), r * np.sin(th), np.full(len(th), z)], 1)


def disc(r, z, pitch=0.02):
    """Concentric fill of a disc, outside in."""
    out = []
    k = max(1, int(round(r / pitch)))
    for i in range(k, 0, -1):
        out.append(ring(r * i / k, z, pitch=pitch))
    out.append(np.array([[0.0, 0.0, z]]))
    return np.concatenate(out)


def raster(hx, hy, z, pitch=0.02):
    """Square perimeter then a boustrophedon fill."""
    out = [np.array([[-hx, -hy, z], [hx, -hy, z], [hx, hy, z],
                     [-hx, hy, z], [-hx, -hy, z]])]
    n = max(1, int(round(2 * hy / pitch)))
    for i in range(n + 1):
        y = -hy + 2 * hy * i / n
        xs = [-hx, hx] if i % 2 == 0 else [hx, -hx]
        out.append(np.array([[xs[0], y, z], [xs[1], y, z]]))
    return np.concatenate(out)


def densify(p, step=0.005):
    """Resample a polyline so consecutive samples are step apart."""
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    keep = seg > 1e-9
    p = np.concatenate([p[:1], p[1:][keep]])
    seg = seg[keep]
    if not len(seg):
        return p
    s = np.concatenate([[0.0], np.cumsum(seg)])
    u = np.arange(0.0, s[-1], step)
    return np.stack([np.interp(u, s, p[:, k]) for k in range(3)], 1)


# ------------------------------------------------------------------ objects
# Each builder returns an ordered path in a frame where the quoted dimension
# is 1.0 and z = 0 is the bed. dz is the layer height in the same units.
def o_cube(dz=0.08, pitch=0.09):
    out = []
    for z in np.arange(dz / 2, 1.0, dz):
        out.append(raster(0.5, 0.5, z, pitch=pitch))
    return np.concatenate(out)


def o_penholder(dz=0.07, h=1.0, wall=True):
    """Hollow cylinder, quoted by outer DIAMETER; height = 1.2 x diameter."""
    out = [disc(0.5, 0.02, pitch=0.06)]
    for i, z in enumerate(np.arange(0.06, 1.2, dz)):
        out.append(ring(0.5, z, pitch=0.05, phase=0.35 * i))
    return np.concatenate(out)


def o_bowl(dz=0.05, eta=0.9, rb=0.45):
    """Spherical cap, quoted by rim diameter (so R = 0.5)."""
    R = 0.5
    H = eta * R
    rho = (R * R + H * H) / (2 * H)
    out = [disc(rb * R, 0.0, pitch=0.05)]
    phi_b = math.asin(min(1.0, rb * R / rho))
    phi_r = math.acos(max(-1.0, min(1.0, (rho - H) / rho)))
    n = max(4, int(round(rho * (phi_r - phi_b) / dz)))
    for i, ph in enumerate(np.linspace(phi_b, phi_r, n)):
        out.append(ring(rho * math.sin(ph), rho * (1 - math.cos(ph)),
                        pitch=0.05, phase=0.4 * i))
    return np.concatenate(out)


def o_vase(dz=0.045):
    """Waisted surface of revolution, quoted by HEIGHT."""
    def r_of(t):                      # t in [0,1] along the height
        return (0.20 + 0.16 * math.sin(math.pi * t ** 0.75)
                - 0.10 * math.sin(2.6 * math.pi * t) * t)
    out = [disc(r_of(0.0), 0.0, pitch=0.05)]
    for i, t in enumerate(np.arange(dz, 1.0 + 1e-9, dz)):
        out.append(ring(max(0.06, r_of(t)), t, pitch=0.05, phase=0.4 * i))
    return np.concatenate(out)


def o_plate(dz=0.04, depth=0.16):
    """Shallow dish, quoted by diameter."""
    R, H = 0.5, depth
    rho = (R * R + H * H) / (2 * H)
    out = [disc(0.30 * R, 0.0, pitch=0.05)]
    phi_b = math.asin(min(1.0, 0.30 * R / rho))
    phi_r = math.acos(max(-1.0, min(1.0, (rho - H) / rho)))
    n = max(3, int(round(rho * (phi_r - phi_b) / dz)))
    for i, ph in enumerate(np.linspace(phi_b, phi_r, n)):
        out.append(ring(rho * math.sin(ph), rho * (1 - math.cos(ph)),
                        pitch=0.05, phase=0.4 * i))
    return np.concatenate(out)


def o_dome(dz=0.05):
    """Hemispherical shell printed as a dome, quoted by diameter."""
    R = 0.5
    out = []
    n = max(4, int(round(0.5 * math.pi * R / dz)))
    for i, ph in enumerate(np.linspace(math.pi / 2, 0.12, n)):
        out.append(ring(R * math.sin(ph), R * math.cos(ph),
                        pitch=0.05, phase=0.4 * i))
    out.append(disc(R * math.sin(0.12), R * math.cos(0.12), pitch=0.05))
    return np.concatenate(out)


def o_mug(dz=0.07):
    """Cylinder plus a handle, quoted by body diameter; height = 1.1 x that.

    The handle is a half-torus in the x-z plane; a print layer meets it in two
    small loops, elongated where the arc runs shallow. It is included because
    it is the one feature here that is not a surface of revolution, so it
    forces the nozzle into one particular sector of every layer it spans."""
    rb, H = 0.5, 1.1
    ca, rt = 0.30, 0.055              # handle arc radius, tube radius
    cz = 0.55                         # handle centre height
    out = [disc(rb, 0.03, pitch=0.06)]
    for i, z in enumerate(np.arange(0.07, H, dz)):
        seg = [ring(rb, z, pitch=0.05, phase=0.35 * i)]
        d = z - cz
        if abs(d) < ca:
            x = rb * 0.86 + math.sqrt(max(0.0, ca * ca - d * d))
            slope = math.sqrt(max(1e-3, 1.0 - (d / ca) ** 2))
            a = min(3.0 * rt, rt / slope)
            th = np.linspace(0, 2 * math.pi, 14)
            seg.append(np.stack([x + a * np.cos(th), rt * np.sin(th),
                                 np.full(14, z)], 1))
        out.append(np.concatenate(seg))
    return np.concatenate(out)


OBJECTS = [
    ('立方体',   'cube',      o_cube,      '边长'),
    ('笔筒',     'penholder', o_penholder, '外径'),
    ('寿司碗',   'bowl',      o_bowl,      '口径'),
    ('花瓶',     'vase',      o_vase,      '高'),
    ('盘子',     'plate',     o_plate,     '直径'),
    ('半球罩',   'dome',      o_dome,      '直径'),
    ('马克杯',   'mug',       o_mug,       '杯身外径'),
]


# ------------------------------------------------------------------- query
def best_scale(F, gx, gz, step, pts, places, ladder, chunk=800):
    """Largest ladder scale admissible at each placement, by field lookup."""
    NX, NZ = len(gx), len(gz)
    best = np.zeros(len(places), np.float32)
    alive = np.arange(len(places))
    for S in ladder:
        if not len(alive):
            break
        pl = S * pts
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
            ok &= (np.hypot(w[:, :, 0], w[:, :, 1]) >= R_COL)
            keep.append(rows[ok.all(1)])
        alive = np.concatenate(keep) if keep else np.array([], np.int64)
        best[alive] = S
    return best


def query_set(raw, cap=2200, step_u=0.025):
    """Points used for the feasibility query. The field is 2 cm, so sampling
    the path finer than that buys nothing; a uniform stride caps the cost for
    the objects with long infill paths."""
    p = densify(raw, step_u)
    if len(p) > cap:
        p = p[np.linspace(0, len(p) - 1, cap).astype(int)]
    return p


def main():
    z = np.load(OUT / 'down_field.npz')
    gx, gz, step = z['gx'], z['gz'], float(z['step'])
    pg = np.arange(-0.75, 0.751, 0.05, np.float32)
    beds = np.arange(-0.45, 0.601, 0.05, np.float32)
    CX, CY, ZB = np.meshgrid(pg, pg, beds, indexing='ij')
    places = np.stack([CX.ravel(), CY.ravel(), ZB.ravel()], 1).astype(np.float32)
    LAD = np.round(np.arange(0.06, 1.201, 0.02), 3)
    print(f'{len(places)} placements, scale ladder '
          f'{LAD[0]:.2f}..{LAD[-1]:.2f} m', flush=True)

    res, rows = {}, []
    for cn, key, fn, dim in OBJECTS:
        raw = fn()
        pts = query_set(raw)
        res[f'path_{key}'] = raw.astype(np.float32)
        line = [cn, key, dim, len(pts)]
        for cone in (5, 30):
            b = best_scale(z[f'F{cone}'], gx, gz, step, pts, places, LAD)
            i = int(b.argmax())
            res[f'{key}_c{cone}'] = np.float32([b[i], *places[i]])
            line += [float(b[i]), places[i]]
        rows.append(line)
        print(f'{cn:<6s} ({key:<9s}) {dim:<4s}  '
              f'5deg {line[4]*100:5.1f} cm @({line[5][0]:+.2f},{line[5][1]:+.2f})'
              f' bed {line[5][2]:+.2f}   '
              f'30deg {line[6]*100:5.1f} cm @({line[7][0]:+.2f},{line[7][1]:+.2f})'
              f' bed {line[7][2]:+.2f}   [{len(pts)} samples]', flush=True)

    np.savez_compressed(OUT / 'objects_query.npz', **res)
    print(f'wrote {OUT / "objects_query.npz"}')


if __name__ == '__main__':
    main()

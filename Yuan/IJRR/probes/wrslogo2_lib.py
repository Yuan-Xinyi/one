"""Faithful WRS mark: thick interlocking frames, ONE Euler stroke.

Each frame is a rectangular annulus -- outer and inner contour. The frames
overlap, so contours cross at even-degree vertices and the whole arrangement
carries an Euler circuit; an inner contour that nothing crosses is stitched
in with a doubled (retraced) bridge, the same device as the letter R's
counter. Frame coordinates are transcribed from the original artwork
(pixel units of the 1811 x 1330 file, y down, converted below).
"""
import numpy as np
import sys
sys.path.insert(0, '.')
from onestroke_lib import euler_path

# (x1, y1, x2, y2, border) in artwork pixels, y down
FRAMES = [
    (540,   28, 1352, 615, 55),   # top wide (bottom bar crosses mid-height)
    (383,  130,  862, 632, 55),   # left big
    (1068,  88, 1550, 618, 55),   # top right tall
    (300,  352,  782, 838, 55),   # left mid
    (237,  468,  523, 753, 52),   # far left small
    (348,  400,  662, 702, 50),   # left inner small
    (988,  352, 1462, 838, 55),   # right big
    (1158, 438, 1678, 942, 55),   # right wide
    (1160, 645, 1512, 1042, 55),  # right bottom
    (1218, 442, 1470, 700, 48),   # right inner small
]
H_PX = 1042.0 - 28.0              # mark height in artwork pixels


def rects():
    """[(outer, inner)] rectangles in unit-height coordinates, y up,
    centred on the origin."""
    out = []
    for x1, y1, x2, y2, t in FRAMES:
        o = np.array([x1, y1, x2, y2], float)
        i = np.array([x1 + t, y1 + t, x2 - t, y2 - t], float)
        out.append((o, i))
    allr = np.array([o for o, _ in out])
    cx = 0.5 * (allr[:, 0].min() + allr[:, 2].max())
    cy = 0.5 * (allr[:, 1].min() + allr[:, 3].max())
    res = []
    for o, i in out:
        def cv(r):
            x1 = (r[0] - cx) / H_PX
            x2 = (r[2] - cx) / H_PX
            y1 = -(r[3] - cy) / H_PX     # flip y
            y2 = -(r[1] - cy) / H_PX
            return np.array([x1, y1, x2, y2])
        res.append((cv(o), cv(i)))
    return res


def mid_rects():
    """One centreline rectangle per frame (user: no hollow -- a single
    outline per frame looks cleaner than outer+inner contours)."""
    out = []
    for (x1, y1, x2, y2, t), _ in zip(FRAMES, range(len(FRAMES))):
        out.append((x1 + t / 2, y1 + t / 2, x2 - t / 2, y2 - t / 2))
    allr = np.array([[x1, y1, x2, y2] for x1, y1, x2, y2, _ in FRAMES], float)
    cx = 0.5 * (allr[:, 0].min() + allr[:, 2].max())
    cy = 0.5 * (allr[:, 1].min() + allr[:, 3].max())
    res = []
    for x1, y1, x2, y2 in out:
        res.append(np.array([(x1 - cx) / H_PX, -(y2 - cy) / H_PX,
                             (x2 - cx) / H_PX, -(y1 - cy) / H_PX]))
    return res


def _rect_segs(r):
    x1, y1, x2, y2 = r
    c = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
    return [(np.array(c[k]), np.array(c[k + 1])) for k in range(4)]


def one_stroke(pitch=0.008, hollow=False):
    segs = []
    if hollow:
        for o, i in rects():
            segs += _rect_segs(o) + _rect_segs(i)
    else:
        for r in mid_rects():
            segs += _rect_segs(r)
    # split every segment at crossings (axis-aligned: H x V only)
    pts_on = [[] for _ in segs]
    for a, (p, q) in enumerate(segs):
        horiz = abs(p[1] - q[1]) < 1e-12
        for b, (r, s) in enumerate(segs):
            if a == b:
                continue
            if horiz and abs(r[0] - s[0]) < 1e-12:
                y, x = p[1], r[0]
                if (min(p[0], q[0]) + 1e-9 < x < max(p[0], q[0]) - 1e-9 and
                        min(r[1], s[1]) + 1e-9 < y < max(r[1], s[1]) - 1e-9):
                    pts_on[a].append((x, y))
            elif (not horiz) and abs(r[1] - s[1]) < 1e-12:
                x, y = p[0], r[1]
                if (min(p[1], q[1]) + 1e-9 < y < max(p[1], q[1]) - 1e-9 and
                        min(r[0], s[0]) + 1e-9 < x < max(r[0], s[0]) - 1e-9):
                    pts_on[a].append((x, y))
    key = lambda pt: (round(pt[0], 7), round(pt[1], 7))
    vid, verts, edges = {}, [], []

    def V(pt):
        k = key(pt)
        if k not in vid:
            vid[k] = len(verts)
            verts.append(k)
        return vid[k]

    for a, (p, q) in enumerate(segs):
        chain = [tuple(p)] + sorted(
            pts_on[a], key=lambda pt: np.hypot(pt[0] - p[0], pt[1] - p[1])) \
            + [tuple(q)]
        for u, w in zip(chain[:-1], chain[1:]):
            if np.hypot(w[0] - u[0], w[1] - u[1]) > 1e-9:
                edges.append((V(u), V(w)))
    # connect components with doubled bridges (retraced pen lines)
    n = len(verts)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)
    comps = {}
    for v in range(n):
        comps.setdefault(find(v), []).append(v)
    comp_list = sorted(comps.values(), key=len, reverse=True)
    n_bridge = 0
    P = np.array(verts)
    while len(comp_list) > 1:
        base = comp_list[0]
        best = (1e18, None, None, None)
        for ci in range(1, len(comp_list)):
            du = ((P[base][:, None, :] -
                   P[comp_list[ci]][None, :, :]) ** 2).sum(-1)
            i, j = np.unravel_index(np.argmin(du), du.shape)
            if du[i, j] < best[0]:
                best = (du[i, j], ci, base[i], comp_list[ci][j])
        _, ci, u, w = best
        edges += [(u, w), (u, w)]
        n_bridge += 1
        base += comp_list.pop(ci)
    deg = np.zeros(n, int)
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    odd = int((deg % 2 == 1).sum())
    assert odd == 0, f'{odd} odd vertices'
    order = euler_path(n, edges)
    Q = np.array([verts[k] for k in order], float)
    out = [Q[:1]]
    for a, b in zip(Q[:-1], Q[1:]):
        m = max(2, int(round(np.hypot(*(b - a)) / pitch)))
        out.append(a + (b - a) * np.linspace(0, 1, m)[1:, None])
    return np.concatenate(out), len(edges), n_bridge


if __name__ == '__main__':
    Q, ne, nb = one_stroke()
    L = float(np.linalg.norm(np.diff(Q, axis=0), axis=1).sum())
    print(f'ONE stroke: {ne} edges, {nb} bridges, length {L:.2f} u, '
          f'closed gap {np.linalg.norm(Q[0] - Q[-1]):.4f}')

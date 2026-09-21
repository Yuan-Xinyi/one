"""Hollow one-stroke WRS letters, and the WRS mark as ONE Euler stroke.

Letters: the thick letter's outline. W and S are simply connected, so the
outline is one closed curve, built by offsetting the centreline. R's bowl
encloses a counter -- a hole -- and the only one-stroke solution is a bridge:
the pen runs a thin line into the inner contour, traces it, and returns along
the same line. The stroke never lifts.

Mark: the ten squares overlap, every outline crossing has degree four and
every corner degree two -- all even -- so the arrangement graph of the whole
mark has an Euler circuit: ONE closed stroke covers every visible line of all
ten squares with no connectors and no pen lift.
"""
import numpy as np
import sys
sys.path.insert(0, '.')
from onestroke_lib import euler_path
import wrslogo_lib
import wrs_lib


# ---------------------------------------------------------------- outline
def _offset_open(poly, t):
    """Closed outline of an open polyline thickened by t (bevel joins,
    square caps)."""
    p = np.asarray(poly, float)
    seg = np.diff(p, axis=0)
    L = np.linalg.norm(seg, axis=1)
    d = seg / L[:, None]
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    left, right = [], []
    left.append(p[0] + n[0] * t / 2)
    right.append(p[0] - n[0] * t / 2)
    for i in range(1, len(p) - 1):
        nm = n[i - 1] + n[i]
        nn = np.linalg.norm(nm)
        if nn < 1e-9:
            nm, nn = n[i - 1], 1.0
        nm = nm / nn
        scale = min(1.0 / max(nm @ n[i - 1], 0.35), 2.6)
        left.append(p[i] + nm * t / 2 * scale)
        right.append(p[i] - nm * t / 2 * scale)
    left.append(p[-1] + n[-1] * t / 2)
    right.append(p[-1] - n[-1] * t / 2)
    return np.concatenate([np.asarray(left), np.asarray(right)[::-1],
                           np.asarray(left[:1])])


def hollow_W(t=0.15):
    c = np.array([[0.0, 1.0], [0.20, 0.0], [0.40, 0.72],
                  [0.60, 0.0], [0.80, 1.0]])
    return _offset_open(c, t)


def hollow_S(t=0.15):
    return _offset_open(wrs_lib.letter_S(), t)


def hollow_R(t=0.15):
    """Outer contour + counter, joined by a retraced bridge (one stroke)."""
    def _arc(c, r, a0, a1, n=40):
        th = np.linspace(np.radians(a0), np.radians(a1), n)
        return np.stack([c[0] + r*np.cos(th), c[1] + r*np.sin(th)], 1)
    # outer boundary, counterclockwise from bottom-left
    outer = [np.array([[0.0, 0.0], [0.55, 0.0]])]           # base under leg? no: stem base
    outer = []
    outer.append(np.array([[0.0, 0.0], [t, 0.0], [t, 0.40]]))
    outer.append(np.array([[0.30, 0.0]]))                    # leg inner bottom
    outer.append(np.array([[0.47, 0.0]]))                    # leg outer bottom
    outer.append(np.array([[0.26, 0.47]]))                   # leg meets bowl
    outer.append(_arc((0.20, 0.725), 0.325, -55.0, 90.0))    # bowl outer
    outer.append(np.array([[0.0, 1.05 - 0.0], [0.0, 0.0]]))  # top-left, close
    # assemble carefully instead: explicit walk
    A = [(0.0, 0.0), (t, 0.0), (t, 0.40), (0.30, 0.0), (0.47, 0.0),
         (0.28, 0.44)]
    bowl_out = _arc((0.16, 0.72), 0.335, -57.0, 90.0)
    B = [(0.0, 1.055), (0.0, 0.0)]
    out = np.concatenate([np.asarray(A, float), bowl_out, np.asarray(B, float)])
    # counter (inner contour), clockwise
    ctr = _arc((0.16, 0.72), 0.335 - t, 82.0, -50.0)
    ctr = np.concatenate([np.array([[t, 0.905]]), ctr,
                          np.array([[t, 0.55], [t, 0.905]])])
    # bridge from outer stem-left mid-bowl into the counter and back
    br_a = np.array([0.0, 0.73])
    br_b = np.array([t, 0.73])
    path = [out]
    path.append(np.array([br_a, br_b]))
    path.append(ctr)
    path.append(np.array([br_b, br_a]))
    return np.concatenate(path)


def hollow_word(gap=0.16):
    letters = [('W', hollow_W()), ('R', hollow_R()), ('S', hollow_S())]
    out, x = [], 0.0
    for name, p in letters:
        p = p.copy()
        p[:, 0] += x - p[:, 0].min()
        x = p[:, 0].max() + gap
        out.append((name, p))
    w = max(p[:, 0].max() for _, p in out)
    for _, p in out:
        p[:, 0] -= w / 2
        p[:, 1] -= 0.5
    return out, w


# ------------------------------------------------------- mark, one stroke
def mark_one_stroke(pitch=0.01):
    """Euler circuit over the arrangement of the ten squares."""
    segs = []
    for _, sq in wrslogo_lib.mark():
        for i in range(4):
            segs.append((sq[i], sq[i + 1]))
    # split every segment at its crossings with the others
    pts_on = [[] for _ in segs]
    for i, (p, q) in enumerate(segs):
        for j, (r, s) in enumerate(segs):
            if i == j:
                continue
            hp = abs(p[1] - q[1]) < 1e-12
            hv = abs(r[0] - s[0]) < 1e-12
            if hp and hv:
                y, x = p[1], r[0]
                if (min(p[0], q[0]) + 1e-9 < x < max(p[0], q[0]) - 1e-9 and
                        min(r[1], s[1]) + 1e-9 < y < max(r[1], s[1]) - 1e-9):
                    pts_on[i].append((x, y))
            elif (not hp) and (not hv):
                continue
            elif (not hp) and hv:
                continue
            elif (not hp):
                continue
        # vertical segment against horizontal others
    for i, (p, q) in enumerate(segs):
        vp = abs(p[0] - q[0]) < 1e-12
        if not vp:
            continue
        for j, (r, s) in enumerate(segs):
            if i == j or abs(r[1] - s[1]) > 1e-12:
                continue
            x, y = p[0], r[1]
            if (min(p[1], q[1]) + 1e-9 < y < max(p[1], q[1]) - 1e-9 and
                    min(r[0], s[0]) + 1e-9 < x < max(r[0], s[0]) - 1e-9):
                pts_on[i].append((x, y))
    key = lambda pt: (round(pt[0], 7), round(pt[1], 7))
    vid, verts, edges = {}, [], []
    def _v(pt):
        k = key(pt)
        if k not in vid:
            vid[k] = len(verts); verts.append(k)
        return vid[k]
    for i, (p, q) in enumerate(segs):
        chain = [tuple(p)] + sorted(
            pts_on[i], key=lambda pt: np.hypot(pt[0]-p[0], pt[1]-p[1])) \
            + [tuple(q)]
        for a, b in zip(chain[:-1], chain[1:]):
            if np.hypot(b[0]-a[0], b[1]-a[1]) > 1e-9:
                edges.append((_v(a), _v(b)))
    deg = np.zeros(len(verts), int)
    for a, b in edges:
        deg[a] += 1; deg[b] += 1
    assert (deg % 2 == 0).all(), f'odd vertices: {(deg % 2 == 1).sum()}'
    order = euler_path(len(verts), edges)
    P = np.array([verts[k] for k in order], float)
    # densify
    out = [P[:1]]
    for a, b in zip(P[:-1], P[1:]):
        n = max(2, int(round(np.hypot(*(b - a)) / pitch)))
        out.append(a + (b - a) * np.linspace(0, 1, n)[1:, None])
    Q = np.concatenate(out)
    return Q, len(edges), len(verts)


if __name__ == '__main__':
    ws, w = hollow_word()
    for n, p in ws:
        print(f'{n}: {len(p)} pts, closed gap '
              f'{np.linalg.norm(p[0]-p[-1]):.3f}')
    Q, ne, nv = mark_one_stroke()
    L = float(np.linalg.norm(np.diff(Q, axis=0), axis=1).sum())
    print(f'mark: {ne} edges, {nv} vertices, ONE stroke length {L:.2f} u, '
          f'closed gap {np.linalg.norm(Q[0]-Q[-1]):.4f}')


def word_one_stroke(pitch=0.01):
    """The whole hollow word as ONE stroke.

    Letters are joined by doubled (retraced) connector lines -- the same
    device as R's internal bridge -- which makes every vertex of the word
    multigraph even, so an Euler circuit covers all three outlines and both
    connectors in one closed stroke, pen never lifting.
    """
    letters, _ = hollow_word()
    polys = [np.asarray(pl, float) for _, pl in letters]
    key = lambda pt: (round(pt[0], 6), round(pt[1], 6))
    vid, verts, edges = {}, [], []

    def V(pt):
        k = key(pt)
        if k not in vid:
            vid[k] = len(verts); verts.append(k)
        return vid[k]

    for pl in polys:
        for a, b in zip(pl[:-1], pl[1:]):
            if np.hypot(*(b - a)) > 1e-9:
                edges.append((V(a), V(b)))
    # connectors between nearest vertex pairs of consecutive letters,
    # doubled so their endpoints stay even
    for A, B in ((polys[0], polys[1]), (polys[1], polys[2])):
        d = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        i, j = np.unravel_index(np.argmin(d), d.shape)
        for _ in range(2):
            edges.append((V(A[i]), V(B[j])))
    deg = np.zeros(len(verts), int)
    for a, b in edges:
        deg[a] += 1; deg[b] += 1
    odd = int((deg % 2 == 1).sum())
    # R's outline walk is open (its two ends sit on the stem's
    # left edge), so 2 odd vertices are legitimate: the Euler
    # path starts and ends there -- still one unbroken stroke
    assert odd in (0, 2), f'{odd} odd vertices'
    order = euler_path(len(verts), edges)
    P = np.array([verts[k] for k in order], float)
    out = [P[:1]]
    for a, b in zip(P[:-1], P[1:]):
        n = max(2, int(round(np.hypot(*(b - a)) / pitch)))
        out.append(a + (b - a) * np.linspace(0, 1, n)[1:, None])
    return np.concatenate(out), len(edges)

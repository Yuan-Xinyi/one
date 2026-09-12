"""Spanning Tree Coverage as a one-stroke figure.

Gabriely & Rimon's construction: tile the square with M x M megacells, build a
spanning tree over them, and circumnavigate the tree through the subcell
centres. The result is a single CLOSED cycle that visits every subcell of the
2M x 2M grid exactly once -- an area-coverage path that is a one-stroke figure
by construction, which is precisely the property the whole study is about.
"""
import numpy as np


def stc_path(M=4, seed=3, pitch=0.01):
    rng = np.random.default_rng(seed)
    # spanning tree over the megacell grid, randomized depth-first search
    tree = set()
    seen = {(0, 0)}
    stack = [(0, 0)]
    while stack:
        c = stack[-1]
        nb = [(c[0] + d[0], c[1] + d[1])
              for d in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        nb = [n for n in nb if 0 <= n[0] < M and 0 <= n[1] < M
              and n not in seen]
        if not nb:
            stack.pop()
            continue
        n = nb[rng.integers(len(nb))]
        tree.add((c, n)); tree.add((n, c))
        seen.add(n)
        stack.append(n)

    def edge(c, d):
        return (c, (c[0] + d[0], c[1] + d[1])) in tree

    # circumnavigate: subcells SW,SE,NE,NW; the standard counterclockwise
    # transition rules -- through the boundary where a tree edge crosses it,
    # around the corner of the own megacell otherwise
    s = 1.0 / M
    def sub(c, k):
        dx = (-1, 1, 1, -1)[k] * s / 4
        dy = (-1, -1, 1, 1)[k] * s / 4
        return (-0.5 + (c[0] + 0.5) * s + dx, -0.5 + (c[1] + 0.5) * s + dy)

    def nxt(c, k):
        if k == 1:   # SE
            return ((c[0] + 1, c[1]), 0) if edge(c, (1, 0)) else (c, 2)
        if k == 2:   # NE
            return ((c[0], c[1] + 1), 1) if edge(c, (0, 1)) else (c, 3)
        if k == 3:   # NW
            return ((c[0] - 1, c[1]), 2) if edge(c, (-1, 0)) else (c, 0)
        # k == 0, SW
        return ((c[0], c[1] - 1), 3) if edge(c, (0, -1)) else (c, 1)

    start = ((0, 0), 0)
    pts = [sub(*start)]
    cur = nxt(*start)
    guard = 0
    while cur != start:
        pts.append(sub(*cur))
        cur = nxt(*cur)
        guard += 1
        assert guard < 16 * M * M + 8, 'walk did not close'
    pts.append(sub(*start))                     # close the cycle
    P = np.asarray(pts, np.float64)
    assert len(P) - 1 == 4 * M * M, 'cycle must visit every subcell once'
    # densify straight segments to the requested pitch
    out = [P[:1]]
    for a, b in zip(P[:-1], P[1:]):
        n = max(2, int(round(np.linalg.norm(b - a) / pitch)))
        out.append((a + (b - a) * np.linspace(0, 1, n)[1:, None]))
    return np.concatenate(out).astype(np.float32)


if __name__ == '__main__':
    for M in (3, 4, 5):
        p = stc_path(M)
        L = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
        print(f'M={M}: {4*M*M} subcells, unit length {L:.2f}, '
              f'closed gap {np.linalg.norm(p[0]-p[-1]):.1e}')


def stc_progressive(Ms=(2, 4, 8, 16), seed=3, pitch=0.01):
    """One planar stroke that covers the square at increasing density.

    Each stage is a closed STC cycle over the same square at grid density M;
    subcell lattices of different M never share a lane (odd numerators over
    distinct powers of two), so stages meet only at perpendicular crossings
    and the union reads as a multi-scale weave. Consecutive stages are joined
    by a short in-plane connector near the shared corner -- the pen never
    lifts, so the whole thing remains a single stroke.

    Returns (path, stage_arcs): the polyline and the arc length at which each
    stage begins, for per-stage colouring.
    """
    parts, marks, arc = [], [], 0.0
    prev_end = None
    for M in Ms:
        cyc = stc_path(M, seed=seed, pitch=pitch)[:, :2]
        if prev_end is not None:
            a, b = prev_end, cyc[0]
            n = max(2, int(round(np.linalg.norm(b - a) / pitch)))
            conn = a + (b - a) * np.linspace(0, 1, n)[1:, None]
            parts.append(conn.astype(np.float32))
            arc += float(np.linalg.norm(b - a))
        marks.append(arc)
        seg = cyc if prev_end is None else cyc[1:]
        parts.append(seg.astype(np.float32))
        arc += float(np.linalg.norm(np.diff(cyc, axis=0), axis=1).sum())
        prev_end = cyc[-1]
    return np.concatenate(parts), np.asarray(marks)

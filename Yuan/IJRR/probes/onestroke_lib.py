"""Classic one-stroke figures, as figures rather than as point clouds.

A primitive has to be drawn without lifting the pen: a seam in the middle of a
circle is not a circle. So a figure here is either a smooth closed curve or a
GRAPH, and for the graph figures the drawing order is not hand-written -- it is
an Euler path computed by Hierholzer's algorithm, which visits every edge
exactly once. If no Euler path exists the figure is not a one-stroke figure and
the builder refuses it, so the demo cannot quietly cheat by lifting the pen.
"""
import math

import numpy as np


# ------------------------------------------------------------ Euler path
def euler_path(n_v, edges):
    """Hierholzer: an edge ordering that uses every edge exactly once.

    Returns the vertex sequence, or raises if the figure is not one-stroke
    drawable (more than two odd-degree vertices, or disconnected)."""
    adj = [[] for _ in range(n_v)]
    for ei, (a, b) in enumerate(edges):
        adj[a].append((b, ei))
        adj[b].append((a, ei))
    deg = [len(a) for a in adj]
    odd = [v for v in range(n_v) if deg[v] % 2]
    if len(odd) not in (0, 2):
        raise ValueError(f'{len(odd)} odd-degree vertices: not one-stroke')
    seen = set(),
    # connectivity over the edge-carrying vertices
    start = odd[0] if odd else next(v for v in range(n_v) if deg[v])
    stack, comp = [start], {start}
    while stack:
        v = stack.pop()
        for w, _ in adj[v]:
            if w not in comp:
                comp.add(w); stack.append(w)
    if any(deg[v] and v not in comp for v in range(n_v)):
        raise ValueError('figure is disconnected: not one-stroke')

    used = [False] * len(edges)
    it = [0] * n_v
    stack, path = [start], []
    while stack:
        v = stack[-1]
        while it[v] < len(adj[v]) and used[adj[v][it[v]][1]]:
            it[v] += 1
        if it[v] == len(adj[v]):
            path.append(stack.pop())
        else:
            w, ei = adj[v][it[v]]
            used[ei] = True
            stack.append(w)
    assert all(used), 'Hierholzer left edges unused'
    return path[::-1]


def from_graph(V, E, pitch=0.01):
    """Polyline of the Euler path through a straight-edge figure."""
    order = euler_path(len(V), E)
    pts = [np.asarray(V[order[0]], np.float64)]
    for a, b in zip(order[:-1], order[1:]):
        p, q = np.asarray(V[a], np.float64), np.asarray(V[b], np.float64)
        n = max(2, int(round(np.linalg.norm(q - p) / pitch)))
        pts.extend(p + (q - p) * t for t in np.linspace(0, 1, n)[1:])
    return np.asarray(pts, np.float32), order


# ------------------------------------------------------------ the figures
def f_circle(pitch=0.01):
    n = max(64, int(round(math.pi / pitch)))
    t = np.linspace(0, 2 * math.pi, n + 1)
    return np.stack([0.5 * np.cos(t), 0.5 * np.sin(t)], 1).astype(np.float32)


def f_square(pitch=0.01):
    V = [(-.5, -.5), (.5, -.5), (.5, .5), (-.5, .5)]
    E = [(0, 1), (1, 2), (2, 3), (3, 0)]
    return from_graph(V, E, pitch)[0]


def f_triangle(pitch=0.01):
    V = [(math.cos(math.radians(90 + 120 * k)) * 0.5,
          math.sin(math.radians(90 + 120 * k)) * 0.5) for k in range(3)]
    return from_graph(V, [(0, 1), (1, 2), (2, 0)], pitch)[0]


def f_star(pitch=0.01):
    """Pentagram: five chords of a circle,每个顶点度 2, 闭合欧拉回路."""
    V = [(0.5 * math.cos(math.radians(90 + 72 * k)),
          0.5 * math.sin(math.radians(90 + 72 * k))) for k in range(5)]
    E = [(k, (k + 2) % 5) for k in range(5)]
    return from_graph(V, E, pitch)[0]


def f_envelope(pitch=0.01):
    """The classic envelope puzzle: a rectangle, both diagonals, and a roof.

    Exactly two odd-degree vertices (the two bottom corners), so it is
    one-stroke drawable and the stroke has to start at one of them."""
    V = [(-.5, -.4), (.5, -.4), (.5, .2), (-.5, .2), (0.0, .5)]
    E = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3), (3, 4), (4, 2)]
    return from_graph(V, E, pitch)[0]


def f_eight(pitch=0.01):
    """Lemniscate of Gerono: a smooth closed figure-eight, one stroke."""
    n = max(96, int(round(2 * math.pi / pitch)))
    t = np.linspace(0, 2 * math.pi, n + 1)
    return np.stack([0.5 * np.cos(t), 0.5 * np.sin(t) * np.cos(t)],
                    1).astype(np.float32)


def f_spiral(pitch=0.01, turns=3.0):
    n = max(160, int(round(2 * math.pi * turns * 0.35 / pitch)))
    t = np.linspace(0, 2 * math.pi * turns, n)
    r = 0.06 + (0.5 - 0.06) * t / t[-1]
    return np.stack([r * np.cos(t), r * np.sin(t)], 1).astype(np.float32)


FIGURES = [
    ('圆',     'circle',   f_circle,   True),
    ('正方形', 'square',   f_square,   True),
    ('三角形', 'triangle', f_triangle, True),
    ('五角星', 'star',     f_star,     True),
    ('信封',   'envelope', f_envelope, False),
    ('双环',   'eight',    f_eight,    True),
    ('螺线',   'spiral',   f_spiral,   False),
]


def build(key, pitch=0.01):
    for cn, k, fn, closed in FIGURES:
        if k == key:
            p = fn(pitch)
            L = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
            return cn, p, L, closed
    raise KeyError(key)


def resample(p2, step):
    """Arc-length resample of a planar polyline at a fixed step."""
    seg = np.linalg.norm(np.diff(p2, axis=0), axis=1)
    keep = seg > 1e-12
    p2 = np.concatenate([p2[:1], p2[1:][keep]])
    s = np.concatenate([[0.0], np.cumsum(seg[keep])])
    u = np.arange(0.0, s[-1] + 1e-12, step)
    return np.stack([np.interp(u, s, p2[:, 0]),
                     np.interp(u, s, p2[:, 1])], 1).astype(np.float32)


if __name__ == '__main__':
    for cn, k, fn, closed in FIGURES:
        _, p, L, c = build(k)
        d = np.linalg.norm(p[0] - p[-1])
        print(f'{cn:<4s} {k:<9s} {len(p):5d} pts  length {L:.3f} (unit)  '
              f'{"closed" if c else "open"}  endpoint gap {d:.4f}')

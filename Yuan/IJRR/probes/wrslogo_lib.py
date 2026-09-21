"""The WRS mark: a cluster of interlocking square outlines.

Eyeballed from the reference: squares of several sizes overlapping along a
loose diagonal, each drawn as one closed stroke, pen lifted between squares.
Coordinates in a unit frame (overall height 1)."""
import numpy as np

# (centre x, centre y, side)
SQUARES = [
    (0.38, 0.62, 0.58),
    (0.60, 0.50, 0.46),
    (0.33, 0.38, 0.36),
    (0.16, 0.52, 0.24),
    (0.12, 0.32, 0.16),
    (0.52, 0.82, 0.26),
    (0.88, 0.38, 0.40),
    (0.98, 0.20, 0.24),
    (1.06, 0.46, 0.16),
    (0.48, 0.16, 0.20),
]


def square(cx, cy, s):
    h = s / 2
    return np.array([[cx-h, cy-h], [cx+h, cy-h], [cx+h, cy+h],
                     [cx-h, cy+h], [cx-h, cy-h]])


def mark():
    """[(name, closed path)], centred on the origin."""
    paths = [(f'sq{i}', square(*p)) for i, p in enumerate(SQUARES)]
    allp = np.concatenate([p for _, p in paths])
    c = 0.5 * (allp.min(0) + allp.max(0))
    for _, p in paths:
        p -= c
    return paths


if __name__ == '__main__':
    ps = mark()
    allp = np.concatenate([p for _, p in ps])
    w, h = allp.max(0) - allp.min(0)
    print(f'{len(ps)} squares, bbox {w:.2f} x {h:.2f}, '
          f'total stroke {sum(4*(SQUARES[i][2]) for i in range(len(ps))):.2f} u')

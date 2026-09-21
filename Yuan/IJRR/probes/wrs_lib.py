"""The letters W, R, S as one-stroke paths, unit height."""
import numpy as np


def _arc(c, r, a0, a1, n=48):
    t = np.linspace(np.radians(a0), np.radians(a1), n)
    return np.stack([c[0] + r*np.cos(t), c[1] + r*np.sin(t)], 1)


def letter_W():
    return np.array([[0.0, 1.0], [0.20, 0.0], [0.40, 0.75],
                     [0.60, 0.0], [0.80, 1.0]])


def letter_R():
    stem = np.array([[0.0, 0.0], [0.0, 1.0]])
    bump = _arc((0.10, 0.745), 0.255, 90.0, -90.0)   # (0.10,1.0) -> (0.10,0.49)
    bump[0] = (0.0, 1.0)
    bump[-1] = (0.0, 0.49)
    leg = np.array([[0.0, 0.49], [0.42, 0.0]])
    return np.concatenate([stem, bump[1:], leg[1:]])


def letter_S():
    up = _arc((0.24, 0.74), 0.24, 20.0, 268.0)       # over the top, CCW
    lo = _arc((0.24, 0.26), 0.24, 88.0, -160.0)      # under the bottom, CW
    return np.concatenate([up, lo[1:]])


def word(gap=0.18):
    """[(name, path)], laid out left to right; unit height, y in [0,1]."""
    letters = [('W', letter_W()), ('R', letter_R()), ('S', letter_S())]
    out, x = [], 0.0
    for name, p in letters:
        p = p.copy()
        p[:, 0] += x - p[:, 0].min()
        x = p[:, 0].max() + gap
        out.append((name, p))
    w = max(p[:, 0].max() for _, p in out)
    for _, p in out:                  # centre the word on the origin
        p[:, 0] -= w / 2
        p[:, 1] -= 0.5
    return out, w


def densify(p, pitch):
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    keep = seg > 1e-12
    p = np.concatenate([p[:1], p[1:][keep]]); seg = seg[keep]
    a = np.concatenate([[0.0], np.cumsum(seg)])
    g = np.arange(0.0, a[-1] + 1e-12, pitch)
    return np.stack([np.interp(g, a, p[:, k]) for k in range(2)], 1)


if __name__ == '__main__':
    ws, w = word()
    for name, p in ws:
        L = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
        print(f'{name}: {len(p)} pts, stroke {L:.2f} (unit), '
              f'x [{p[:,0].min():+.2f},{p[:,0].max():+.2f}]')
    print(f'word width {w:.2f} x height 1.00')

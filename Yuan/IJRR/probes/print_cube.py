"""Largest printable cube from the nozzle-down feasibility field.

A planar-layer print visits, layer by layer, every point of the object's
horizontal cross-section, so a SOLID cube is printable pointwise iff the whole
box lies inside the feasible set. A HOLLOW cube (shelled walls, solid caps)
only needs the boundary, which is the larger and the more honest answer for a
"how big can you go" question.

The query is exact on the voxel grid: a 3-D prefix sum of the infeasible
indicator answers "is this box entirely feasible" in O(1), so every
(footprint centre, bed height) pair can be binary-searched for its largest
side. The result is a placement map, not a single number -- with a fixed base
the placement is the only spatial freedom there is.
"""
import sys
from pathlib import Path

import numpy as np

OUT = Path('/home/lqin/one/Yuan/IJRR/runs/paper_fill/print_analysis')
STEP = 0.02
WALL_VOX = 1          # shell thickness in voxels for the hollow variant
CAP_VOX = 2           # solid top/bottom layers
# The object has to share the room with the thing printing it. link0 reaches
# 0.175 m from the axis, and anything mounting the arm above a lower bed
# occupies the same column, so the printed volume is excluded from a vertical
# cylinder of this radius -- a cube that swallows its own robot is not a cube
# the robot can print, however reachable its points are.
R_COL = 0.20


def prefix3(bad):
    """3-D inclusive prefix sum, padded by one zero plane on each low side."""
    S = np.zeros((bad.shape[0] + 1, bad.shape[1] + 1, bad.shape[2] + 1), np.int32)
    S[1:, 1:, 1:] = bad
    np.cumsum(S, axis=0, out=S)
    np.cumsum(S, axis=1, out=S)
    np.cumsum(S, axis=2, out=S)
    return S


def boxsum(S, i0, i1, j0, j1, k0, k1):
    """Inclusive index box sum; all arguments broadcastable integer arrays."""
    return (S[i1 + 1, j1 + 1, k1 + 1] - S[i0, j1 + 1, k1 + 1]
            - S[i1 + 1, j0, k1 + 1] - S[i1 + 1, j1 + 1, k0]
            + S[i0, j0, k1 + 1] + S[i0, j1 + 1, k0] + S[i1 + 1, j0, k0]
            - S[i0, j0, k0])


def _admissible(S, I, J, K, r, NX, NZ, hollow):
    ok = ((I - r >= 0) & (I + r < NX) & (J - r >= 0) & (J + r < NX)
          & (K >= 0) & (K + 2 * r < NZ))
    i0 = np.clip(I - r, 0, NX - 1); i1 = np.clip(I + r, 0, NX - 1)
    j0 = np.clip(J - r, 0, NX - 1); j1 = np.clip(J + r, 0, NX - 1)
    k0 = np.clip(K, 0, NZ - 1);     k1 = np.clip(K + 2 * r, 0, NZ - 1)
    cnt = boxsum(S, i0, i1, j0, j1, k0, k1)
    if hollow:
        # interior that an unfilled print never visits: strip the walls and the
        # solid caps, then subtract. A degenerate (negative-extent) interior
        # contributes nothing, which is the right answer for a thin shell.
        a0, a1 = i0 + WALL_VOX, i1 - WALL_VOX
        b0, b1 = j0 + WALL_VOX, j1 - WALL_VOX
        c0, c1 = k0 + CAP_VOX, k1 - CAP_VOX
        solid = (a0 <= a1) & (b0 <= b1) & (c0 <= c1)
        cl = lambda u, n: np.clip(u, 0, n - 1)
        inner = np.where(solid, boxsum(S, cl(a0, NX), cl(a1, NX),
                                       cl(b0, NX), cl(b1, NX),
                                       cl(c0, NZ), cl(c1, NZ)), 0)
        cnt = cnt - inner
    return ok & (cnt == 0)


def max_side(S, NX, NZ, hollow=False, k_only=None):
    """For every (i, j, k) = (footprint centre, bed layer) the largest r with
    the cube [i +/- r] x [j +/- r] x [k .. k + 2r] admissible. Side = (2r+1)h.

    Solid boxes nest, so the search bisects. Shells do NOT nest -- a larger
    shell can clear a region a smaller one cuts through -- so the hollow case
    is scanned, which is also what makes its answer the true maximum rather
    than the first monotone crossing."""
    ks = np.arange(NZ) if k_only is None else np.array([k_only])
    I, J, K = np.meshgrid(np.arange(NX), np.arange(NX), ks, indexing='ij')
    r_max = min(NX // 2, NZ // 2)
    if hollow:
        best = np.zeros(I.shape, np.int32)
        for r in range(r_max + 1):
            best = np.where(_admissible(S, I, J, K, r, NX, NZ, True), r, best)
        return best
    lo = np.zeros(I.shape, np.int32)
    hi = np.full(I.shape, r_max, np.int32)
    for _ in range(int(np.ceil(np.log2(r_max + 1))) + 1):
        mid = (lo + hi + 1) // 2
        good = _admissible(S, I, J, K, mid, NX, NZ, False)
        lo = np.where(good, mid, lo)
        hi = np.where(good, hi, mid - 1)
    return lo


def main():
    z = np.load(OUT / 'down_field.npz')
    gx, gz = z['gx'], z['gz']
    NX, NZ = len(gx), len(gz)
    res = {}
    print(f'grid {NX}x{NX}x{NZ}, step {STEP*100:.0f} cm, '
          f'z {gz[0]:+.2f}..{gz[-1]:+.2f}, base column r={R_COL:.2f} m')
    col = (np.hypot(gx[:, None], gx[None, :]) >= R_COL)[:, :, None]

    for cone in (5, 30, 90):
        F = z[f'F{cone}'] & col
        ZL = z[f'ZL{cone}']
        for bedcheck in (False, True):
            if bedcheck:
                # the bed is a floor for the ARM, not only for the tip: every
                # link sphere of the witness must sit above the bed plane. The
                # screen is applied per candidate bed height, which is why the
                # field stores the witness clearance rather than a verdict.
                sides = np.zeros((NX, NX, NZ), np.float32)
                for k in range(NZ):
                    zb = gz[k] - STEP / 2
                    Fk = F & (np.nan_to_num(ZL, nan=-9.0) >= zb)
                    Sk = prefix3((~Fk).astype(np.int32))
                    r = max_side(Sk, NX, NZ, k_only=k)[:, :, 0]
                    sides[:, :, k] = (2 * r + 1) * STEP
                res[f'cube_c{cone}_bed'] = sides
            else:
                S = prefix3((~F).astype(np.int32))
                for hollow in (False, True):
                    r = max_side(S, NX, NZ, hollow=hollow)
                    tag = 'hollow' if hollow else 'solid'
                    res[f'cube_c{cone}_{tag}'] = (2 * r + 1) * STEP

        k0 = int(np.argmin(np.abs(gz - STEP / 2)))     # bed level with the mount
        for tag in ('solid', 'hollow', 'bed'):
            key = f'cube_c{cone}_{tag}'
            if key not in res:
                continue
            A = res[key]
            b = np.unravel_index(A.argmax(), A.shape)
            T0 = A[:, :, k0]
            t0 = np.unravel_index(T0.argmax(), T0.shape)
            print(f'cone {cone:3d} deg  {tag:<7s}  max side {A.max():.3f} m  '
                  f'at centre ({gx[b[0]]:+.2f}, {gx[b[1]]:+.2f}) '
                  f'bed z={gz[b[2]]-STEP/2:+.3f}  '
                  f'reach r={np.hypot(gx[b[0]], gx[b[1]]):.3f}   |   '
                  f'bed level with mount: {T0.max():.3f} m at '
                  f'({gx[t0[0]]:+.2f}, {gx[t0[1]]:+.2f})')

    np.savez_compressed(OUT / 'cube_query.npz', gx=gx, gz=gz,
                        step=np.float32(STEP), **res)
    print(f'wrote {OUT / "cube_query.npz"}')


if __name__ == '__main__':
    main()

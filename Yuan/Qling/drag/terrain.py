"""Generic 2-D terrain resting-pose solver for the dragged-bottle line.

A terrain is a piecewise-linear height profile h(y) (extruded along x).
For a lying box whose bottom trace along the travel direction is a
segment of half-span w_eff, the RESTING pose at center-y is the lowest
non-penetrating placement:

    (z*, phi*) = argmin_z  s.t.  the segment [y_c - w cos phi,
                 y_c + w cos phi] at height/tilt (z, phi) clears h(y)

solved numerically over a phi grid (the argmin recovers the two-contact
closed form on a concave fold and produces the edge-pivot family on a
convex fold; a physical snap-through at a convex edge shows up as a
kink, which the field table's finite resolution smooths over ~1 cm --
documented approximation, the object is HELD so no dynamic fall
occurs).

Outputs the same (dz, phi, d/dy, d/dpsi) tables BottleSlopeEnv uses.
"""
from __future__ import annotations

import math

import numpy as np


class Terrain:
    """Piecewise-linear profile through (y_i, z_i) knots; flat
    extrapolation before the first and after the last knot."""

    def __init__(self, knots):
        self.ky = np.array([k[0] for k in knots], dtype=np.float64)
        self.kz = np.array([k[1] for k in knots], dtype=np.float64)
        assert np.all(np.diff(self.ky) > 0)

    def h(self, y):
        return np.interp(y, self.ky, self.kz)

    def h_t(self, y):
        """Torch-friendly height for probes/render (numpy in, out)."""
        return np.interp(y, self.ky, self.kz)

    def max_slope(self):
        s = np.diff(self.kz) / np.diff(self.ky)
        return float(np.abs(s).max())


def resting_fields(terrain: Terrain, w_eff: float, body_r: float,
                   yg: np.ndarray, phi_grid: int = 181,
                   phi_max: float | None = None):
    """(dz, phi) arrays over center-y grid `yg` for one w_eff.

    For each y_c and each candidate tilt phi, the lowest z of the
    bottom-face center such that the face segment clears the terrain is
    max over sample points of [h(y) - offset]; the resting pose takes
    the phi minimizing the BODY-CENTER height z_face + r cos phi ...
    with the face segment sampled densely (terrain kinks fall between
    samples at <0.5 mm resolution)."""
    # candidate tilts are restricted per-y to the envelope of terrain
    # slopes under the footprint: an unrestricted argmin-z finds
    # non-physical "corner-hugging" poses (tilting against a ramp to
    # wedge an end-corner lower at fixed y)
    seg_ang = np.arctan(np.diff(terrain.kz) / np.diff(terrain.ky))
    span = w_eff + body_r + 0.01
    lo_ang = np.zeros_like(yg)
    hi_ang = np.zeros_like(yg)
    for i, y in enumerate(yg):
        touch = [0.0]        # flat extrapolation always a candidate
        for k in range(len(seg_ang)):
            if terrain.ky[k] <= y + span and terrain.ky[k + 1] >= y - span:
                touch.append(float(seg_ang[k]))
            elif terrain.ky[k] > y + span:
                break
        lo_ang[i], hi_ang[i] = min(touch), max(touch)
    if phi_max is None:
        phi_max = math.atan(terrain.max_slope()) + 0.05
    phis = np.linspace(-phi_max, phi_max, phi_grid)
    s = np.linspace(-1.0, 1.0, 81)                    # face param
    z_best = np.full(yg.shape, np.inf)
    ph_best = np.zeros_like(yg)
    for phi in phis:
        # face point at param s for BODY center y: world y =
        # y + r sin(phi) + s w cos(phi); its height above the face
        # center is s w sin(phi); body center is r cos(phi) above the
        # face center along the face normal
        ys = (yg[:, None] + body_r * math.sin(phi)
              + s[None, :] * w_eff * math.cos(phi))
        hs = terrain.h(ys)
        z_center = ((hs - s[None, :] * w_eff * math.sin(phi)).max(axis=1)
                    + body_r * math.cos(phi))
        ok = (phi >= lo_ang - 1e-9) & (phi <= hi_ang + 1e-9)
        take = ok & (z_center < z_best)
        z_best[take] = z_center[take]
        ph_best[take] = phi
    return z_best, ph_best


def build_tables(terrain: Terrain, body_r: float, half_len: float,
                 y_lo: float, y_hi: float, ny: int = 601,
                 npsi: int = 61):
    """Full (psi, y) tables: dz (center height minus flat), phi, and
    the y/psi derivatives, matching BottleSlopeEnv's table contract."""
    yg = np.linspace(y_lo, y_hi, ny)
    pg = np.linspace(-math.pi, math.pi, npsi)
    dz = np.zeros((npsi, ny))
    ph = np.zeros((npsi, ny))
    for j, psi in enumerate(pg):
        w_eff = (body_r * abs(math.cos(psi))
                 + half_len * abs(math.sin(psi)))
        z, p = resting_fields(terrain, w_eff, body_r, yg)
        dz[j] = z - body_r          # relative to flat resting height
        ph[j] = p
    c = np.gradient(dz, yg, axis=1)
    k = np.gradient(ph, yg, axis=1)
    zp = np.gradient(dz, pg, axis=0)
    pp = np.gradient(ph, pg, axis=0)
    return dict(y=yg, psi=pg, dz=dz, phi=ph, c=c, k=k, zp=zp, pp=pp)

"""Bottle over-the-hill task: table -> 45-deg ramp (8 cm rise) ->
upper plateau. Two folds; the upper one is CONVEX (a genuinely new
contact regime: the bottle pivots over the edge and lays down flat on
the plateau).

Everything reuses BottleSlopeEnv's machinery; the differences are the
terrain-driven field tables (generic resting solver in terrain.py,
slope-envelope-restricted argmin) and the terrain surface clearance.
Goals are resting poses on the TABLE or the PLATEAU (the 8-cm ramp is
a transit feature, too short to place on).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .bottle_slope_env import BottleSlopeEnv
from .terrain import Terrain, build_tables

RAMP_RISE = 0.08
Y_TOP = BottleSlopeEnv.Y_FOLD + RAMP_RISE          # upper fold y (45deg)


class BottleHillEnv(BottleSlopeEnv):

    def _terrain(self):
        yf = self.Y_FOLD
        run = RAMP_RISE / math.tan(self.THETA)
        return Terrain([(-1.0, 0.0), (yf, 0.0),
                        (yf + run, RAMP_RISE), (2.0, RAMP_RISE)])

    def _build_field_table(self):
        self.terrain = self._terrain()
        tab = build_tables(self.terrain, self.BOTTLE_R, self.BOTTLE_L / 2,
                           self.Y_FOLD - 0.35, self.Y_FOLD + 0.45,
                           ny=801, npsi=61)
        self.tab_y = torch.tensor(tab['y'], dtype=torch.float32)
        self.tab_psi = torch.tensor(tab['psi'], dtype=torch.float32)
        self.tab_dz = torch.tensor(tab['dz'], dtype=torch.float32)
        self.tab_phi = torch.tensor(tab['phi'], dtype=torch.float32)
        self.tab_c = torch.tensor(tab['c'], dtype=torch.float32)
        self.tab_k = torch.tensor(tab['k'], dtype=torch.float32)
        self.tab_zp = torch.tensor(tab['zp'], dtype=torch.float32)
        self.tab_pp = torch.tensor(tab['pp'], dtype=torch.float32)
        self.k_max = float(np.abs(tab['k']).max())

    def _anchor_yc(self):
        return Y_TOP + 0.02        # reach scan: 4 grasps feasible here

    def _goal_y_range(self):
        return 0.33, Y_TOP + 0.08  # plateau reachable band ends ~0.64

    def _goal_phi_ok(self, phi):
        # resting = table or plateau (flat); the ramp is transit only
        return phi < 0.02 * self.THETA

    def _surface_clearance(self, pts, rr):
        ky = torch.tensor(self.terrain.ky, dtype=pts.dtype,
                          device=pts.device)
        kz = torch.tensor(self.terrain.kz, dtype=pts.dtype,
                          device=pts.device)
        y = pts[..., 1].clamp(float(ky[0]), float(ky[-1]))
        i = torch.bucketize(y, ky).clamp(1, len(ky) - 1)
        w = (y - ky[i - 1]) / (ky[i] - ky[i - 1])
        hy = kz[i - 1] * (1 - w) + kz[i] * w
        # vertical clearance scaled by cos(max slope): a conservative
        # normal-distance bound on every terrain segment
        return (pts[..., 2] - hy) * math.cos(self.THETA) - rr

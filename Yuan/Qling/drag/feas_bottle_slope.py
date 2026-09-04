"""Kinematic feasibility pre-check for the bottle plane-to-slope task.

Scene: table plane z=0; a 15-deg ramp rises toward +y from the fold
line y = Y_FOLD. The lying bottle (axis along +x, the compare_exp roll
preserved) is pushed from y=0.40 up onto the ramp.

Crossing the fold, the rigid bottle BRIDGES: rear bottom edge on the
table, front bottom edge on the ramp; the tilt angle grows 0 -> theta
in closed form as it advances (2-D geometry in the y-z travel plane,
valid for the axis-parallel crossing).

For each of the 20 compare_exp grasp candidates this script
Newton-continues the arm along the whole path and reports how far it
survives (convergence + joint limits). Collision is NOT checked here
-- this is the reach/limits go/no-go; the env adds collision.

Usage:
    cd /home/lqin/one/Yuan/Qling
    /home/lqin/miniconda3/envs/one/bin/python -m drag.feas_bottle_slope
"""
import matplotlib  # noqa: F401  must precede torch on this box
import math
import os

import numpy as np
import torch

from .ijrr_root import add_ijrr_path
add_ijrr_path()
from Yuan.IJRR.kinematics.batched_chain_kin import (  # noqa: E402
    BatchedChainKinematics)
from .nova2_spec import NOVA2, GRIPPER_TCP_OFFSET     # noqa: E402

EXP_DIR = os.path.join(os.path.dirname(__file__), '..', 'compare_exp')
DATA = os.path.join(os.path.dirname(__file__), 'data')

import sys as _sys
THETA = math.radians(float(_sys.argv[1])
                     if len(_sys.argv) > 1 else 15.0)   # ramp angle
Y_FOLD = 0.48                   # fold line y (parallel to x axis)
BOTTLE_R = 0.0375               # lateral/vertical half extent (lying)
Y_START = 0.40                  # bottle CENTER y at start (flat)
SLOPE_ADV = 0.07                # goal: center this far up-slope from fold
N_WP = 80


def rotx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def bridge_phi(y_r):
    """Tilt angle for rear-bottom-edge y position y_r (2-D closed
    geometry: rear edge on table, front edge on/at the ramp)."""
    w2 = 2 * BOTTLE_R
    if y_r + w2 <= Y_FOLD:
        return 0.0
    if y_r >= Y_FOLD:
        return THETA
    lo, hi = 0.0, THETA
    for _ in range(60):                      # bisection, monotone
        mid = 0.5 * (lo + hi)
        zf = w2 * math.sin(mid)
        yf = y_r + w2 * math.cos(mid)
        if zf < math.tan(THETA) * (yf - Y_FOLD):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def bottle_pose(y_r, R_flat):
    """Body center + rotation for rear-bottom-edge position y_r."""
    phi = bridge_phi(y_r)
    if y_r >= Y_FOLD:                        # fully on ramp
        z_r = math.tan(THETA) * (y_r - Y_FOLD)
    else:
        z_r = 0.0
    # bottom-face center, then body center R above it along the face
    # normal (tilt axis = x)
    yb = y_r + BOTTLE_R * math.cos(phi)
    zb = z_r + BOTTLE_R * math.sin(phi)
    yc = yb - BOTTLE_R * math.sin(phi)
    zc = zb + BOTTLE_R * math.cos(phi)
    R_obj = rotx(phi) @ R_flat
    return np.array([0.0, yc, zc]), R_obj, phi


def read_init_R():
    tok = open(os.path.join(EXP_DIR, 'repair0_init_goal.txt')).read().split()
    i = tok.index('init_rotmat')
    return np.array([float(x) for x in tok[i + 1:i + 10]]).reshape(3, 3)


def main():
    g = np.load(os.path.join(EXP_DIR, 'grasps_G20.npz'))
    g_R = torch.tensor(g['ac_rotmat'], dtype=torch.float32)
    g_p = torch.tensor(g['ac_pos'], dtype=torch.float32)
    G = g_R.shape[0]

    init_R = read_init_R()
    head0 = math.atan2(init_R[1, 2], init_R[0, 2])
    R_flat = rotz(-head0) @ init_R          # axis -> +x, roll preserved

    kin = BatchedChainKinematics(NOVA2, dtype=torch.float32,
                                 tcp_offset=GRIPPER_TCP_OFFSET)
    lmt_lo = kin.q_lo + 1e-4 if hasattr(kin, 'q_lo') else None
    pool = np.load(os.path.join(DATA, 'q0_pool_s0.npz'))
    key = [k for k in pool.files][0]
    q_pool = torch.tensor(pool[key], dtype=torch.float32)
    if lmt_lo is None:
        lo = torch.tensor(NOVA2['q_lo'], dtype=torch.float32) \
            if isinstance(NOVA2, dict) and 'q_lo' in NOVA2 else None
        hi = torch.tensor(NOVA2['q_hi'], dtype=torch.float32) \
            if lo is not None else None
    else:
        lo, hi = kin.q_lo, kin.q_hi
    if lo is None:
        from .drag_env import DragEnvConfig, Nova2DragEnv  # fallback
        env = Nova2DragEnv(DragEnvConfig(n_envs=1, device='cpu'))
        lo, hi = env.lmt_lo, env.lmt_up

    # rear-edge param: center y = y_r + R cos(phi) - R sin(phi) ... use
    # rear edge directly; start/end chosen so center runs Y_START ->
    # fold + SLOPE_ADV (up-slope)
    y_r0 = Y_START - BOTTLE_R
    y_r1 = Y_FOLD + SLOPE_ADV * math.cos(THETA) - BOTTLE_R * 0.0
    path = np.linspace(y_r0, y_r1, N_WP)

    def solve(p_des, R_des, q0, iters=30):
        q = q0.clone()
        for _ in range(iters):
            p, R, J, _ = kin.tcp_fk_jac(q)
            e_p = p_des - p
            e_r = 0.5 * (torch.cross(R[:, :, 0], R_des[:, :, 0], dim=1)
                         + torch.cross(R[:, :, 1], R_des[:, :, 1], dim=1)
                         + torch.cross(R[:, :, 2], R_des[:, :, 2], dim=1))
            e = torch.cat([e_p, e_r], dim=1)
            JJt = J @ J.transpose(-1, -2) + 1e-5 * torch.eye(6)
            q = q + (J.transpose(-1, -2)
                     @ torch.linalg.solve(JJt, e.unsqueeze(-1))
                     ).squeeze(-1).clamp(-0.25, 0.25)
        p, R, J, _ = kin.tcp_fk_jac(q)
        ok = (p_des - p).norm(dim=1) < 2e-3
        e_r = 0.5 * (torch.cross(R[:, :, 0], R_des[:, :, 0], dim=1)
                     + torch.cross(R[:, :, 1], R_des[:, :, 1], dim=1)
                     + torch.cross(R[:, :, 2], R_des[:, :, 2], dim=1))
        ok &= e_r.norm(dim=1) < 0.02
        ok &= ((q > lo) & (q < hi)).all(dim=1)
        return q, ok

    alive = torch.ones(G, dtype=torch.bool)
    died_at = torch.full((G,), -1, dtype=torch.long)
    q_cur = None
    pool_p, _, _, _ = kin.tcp_fk_jac(q_pool)
    phis = []
    for wi, y_r in enumerate(path):
        p_obj, R_obj, phi = bottle_pose(float(y_r), R_flat)
        phis.append(math.degrees(phi))
        Ro = torch.tensor(R_obj, dtype=torch.float32)
        po = torch.tensor(p_obj, dtype=torch.float32)
        R_des = Ro.unsqueeze(0) @ g_R
        p_des = po.unsqueeze(0) + (Ro.unsqueeze(0)
                                   @ g_p.unsqueeze(-1)).squeeze(-1)
        if q_cur is None:
            near = torch.cdist(p_des, pool_p).argmin(dim=1)
            q_cur = q_pool[near]
        q_new, ok = solve(p_des, R_des, q_cur)
        newly_dead = alive & ~ok
        died_at[newly_dead] = wi
        alive &= ok
        q_cur = torch.where(alive.unsqueeze(1), q_new, q_cur)
    print(f'tilt profile: phi(deg) at wp 0/20/40/60/79 = '
          f'{[round(phis[i], 1) for i in (0, 20, 40, 60, 79)]}')
    print(f'path: center y {Y_START:.3f} -> fold {Y_FOLD} -> '
          f'+{SLOPE_ADV} up-slope, {N_WP} wps')
    surv = alive.nonzero().squeeze(1).tolist()
    print(f'full-path survivors (reach+limits): {len(surv)}/{G} -> {surv}')
    for gi in range(G):
        if not alive[gi]:
            print(f'  grasp {gi:2d}: died at wp {int(died_at[gi]):3d} '
                  f'(phi {phis[int(died_at[gi])]:.1f} deg)')


if __name__ == '__main__':
    main()

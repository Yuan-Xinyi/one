"""Symbolic FR3 kinematics and self-collision for CasADi/IPOPT.

A symbolic model that silently disagrees with the one the rest of the study
uses would make every number downstream meaningless, so the only thing this
module promises is that it reproduces ``BatchedFR3Kinematics`` exactly -- the
check at the bottom compares link transforms, tool position and tool axis
against the torch implementation on random configurations and fails loudly.
"""
import math
import sys
from pathlib import Path

REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))
import casadi as ca
import numpy as np

# constants copied from one/robots/.../fr3_pen/batched_fr3_kin.py
HAND_TCP_OFFSET, PEN_LENGTH = 0.1034, 0.10
TCP_OFFSET = HAND_TCP_OFFSET + PEN_LENGTH
LMT_LO = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
LMT_UP = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
ZERO_SPECS = [                       # (rot-x angle, translation)
    (0.0,            [0.0, 0.0, 0.333]),
    (-math.pi / 2,   [0.0, 0.0, 0.0]),
    (math.pi / 2,    [0.0, -0.316, 0.0]),
    (math.pi / 2,    [0.0825, 0.0, 0.0]),
    (-math.pi / 2,   [-0.0825, 0.384, 0.0]),
    (math.pi / 2,    [0.0, 0.0, 0.0]),
    (math.pi / 2,    [0.088, 0.0, 0.0]),
]


def _rotx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def _zero_tfs():
    out = []
    for a, p in ZERO_SPECS:
        T = np.eye(4)
        T[:3, :3] = _rotx(a)
        T[:3, 3] = p
        out.append(T)
    return out


def fk_sym(q):
    """Symbolic link transforms (list of 8 4x4) for a CasADi 7-vector."""
    Z = _zero_tfs()
    T = ca.MX.eye(4)
    links = [T]
    for i in range(7):
        Tj = T @ ca.MX(Z[i])
        c, s = ca.cos(q[i]), ca.sin(q[i])
        Rz = ca.MX.eye(4)
        Rz[0, 0] = c; Rz[0, 1] = -s
        Rz[1, 0] = s; Rz[1, 1] = c
        T = Tj @ Rz
        links.append(T)
    return links


def tcp_sym(q):
    """Symbolic (tool position, tool z axis)."""
    T = fk_sym(q)[-1]
    R = T[:3, :3]
    p = R @ ca.MX(np.array([0.0, 0.0, 0.107 + TCP_OFFSET])) + T[:3, 3]
    return p, R[:, 2]


def sphere_data():
    """Collision spheres and the pair mask, read from the torch model."""
    import matplotlib; matplotlib.use('Agg')          # noqa: E402
    import torch                                      # noqa: E402
    from one.robots.manipulators.franka.fr3.sphere_collision import (
        FR3SphereCollision)
    c = FR3SphereCollision(device='cpu')
    return (c.centers.numpy().astype(float), c.radii.numpy().astype(float),
            c.link_indices.numpy(), c.mask.numpy())


def collision_pairs_sym(q, centers, radii, link_idx, pairs):
    """Symbolic squared centre distance minus squared radius sum, per pair."""
    links = fk_sym(q)
    pos = []
    for k in range(len(centers)):
        T = links[int(link_idx[k])]
        pos.append(T[:3, :3] @ ca.MX(centers[k]) + T[:3, 3])
    out = []
    for (i, j) in pairs:
        d = pos[i] - pos[j]
        out.append(ca.dot(d, d) - (radii[i] + radii[j]) ** 2)
    return ca.vertcat(*out) if out else ca.MX.zeros(0)


def verify(n=64, seed=0):
    import matplotlib; matplotlib.use('Agg')          # noqa: E402
    import torch                                      # noqa: E402
    from one.robots.manipulators.franka.fr3_pen.batched_fr3_kin import (
        BatchedFR3Kinematics)
    kin = BatchedFR3Kinematics(device='cpu', tcp_offset=TCP_OFFSET)
    rng = np.random.default_rng(seed)
    Q = rng.uniform(LMT_LO, LMT_UP, size=(n, 7))
    qs = ca.MX.sym('q', 7)
    p, z = tcp_sym(qs)
    F = ca.Function('F', [qs], [p, z, fk_sym(qs)[4]])
    qt = torch.tensor(Q, dtype=torch.float32)
    pt, Rt, _, _ = kin.tcp_fk_jac(qt)
    Lt = kin.link_transforms(qt)
    ep = ez = el = 0.0
    for k in range(n):
        pc, zc, l4 = [np.array(x).ravel() for x in F(Q[k])]
        ep = max(ep, float(np.abs(pc - pt[k].numpy()).max()))
        ez = max(ez, float(np.abs(zc - Rt[k, :, 2].numpy()).max()))
        el = max(el, float(np.abs(l4.reshape(4, 4) - Lt[k, 4].numpy()).max()))
    print(f'[verify] {n} random configs — max |dp| {ep:.2e} m, '
          f'max |dz| {ez:.2e}, max |dT_link4| {el:.2e}')
    assert ep < 2e-5 and ez < 2e-5 and el < 2e-5, 'symbolic FK disagrees'
    print('[verify] symbolic model matches the torch kinematics')


if __name__ == '__main__':
    verify()
    C, R, L, M = sphere_data()
    ij = np.argwhere(M)
    print(f'[verify] {len(C)} spheres, {len(ij)} unmasked pairs')

"""Single-configuration FR3 kinematics and self-collision in plain numpy.

The planner calls these from inside its sampling loop, one configuration at a
time, so the torch path's batching is all overhead here. Correctness is not
assumed: ``verify`` compares against the torch model the rest of the study uses.
"""
import math
import sys
from pathlib import Path

REPO = Path('/home/lqin/one/Yuan/IJRR/.claude/worktrees/vigilant-hertz-799b05')
sys.path.insert(0, str(REPO))
import numpy as np

TCP_OFFSET = 0.1034 + 0.10
FLANGE = np.array([0.0, 0.0, 0.107 + TCP_OFFSET])
LMT_LO = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
LMT_UP = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
QD = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
_SPECS = [(0.0, [0.0, 0.0, 0.333]), (-math.pi / 2, [0.0, 0.0, 0.0]),
          (math.pi / 2, [0.0, -0.316, 0.0]), (math.pi / 2, [0.0825, 0.0, 0.0]),
          (-math.pi / 2, [-0.0825, 0.384, 0.0]), (math.pi / 2, [0.0, 0.0, 0.0]),
          (math.pi / 2, [0.088, 0.0, 0.0])]


def _zero_tfs():
    out = np.zeros((7, 4, 4))
    for i, (a, p) in enumerate(_SPECS):
        c, s = math.cos(a), math.sin(a)
        out[i] = np.eye(4)
        out[i][:3, :3] = [[1, 0, 0], [0, c, -s], [0, s, c]]
        out[i][:3, 3] = p
    return out


ZT = _zero_tfs()


def links(q):
    """World transforms of link0..link7, shape (8, 4, 4)."""
    T = np.eye(4)
    out = np.empty((8, 4, 4))
    out[0] = T
    for i in range(7):
        Tj = T @ ZT[i]
        c, s = math.cos(q[i]), math.sin(q[i])
        Rz = np.eye(4)
        Rz[0, 0] = c; Rz[0, 1] = -s; Rz[1, 0] = s; Rz[1, 1] = c
        T = Tj @ Rz
        out[i + 1] = T
    return out


def tcp(q):
    """(tool position, tool z axis, joint transforms) for one configuration."""
    L = links(q)
    T = L[7]
    return T[:3, :3] @ FLANGE + T[:3, 3], T[:3, 2], L


def tcp_jac(q):
    """Position Jacobian (3x7) of the tool point."""
    T = np.eye(4)
    axes = np.empty((7, 3))
    orig = np.empty((7, 3))
    for i in range(7):
        Tj = T @ ZT[i]
        axes[i] = Tj[:3, 2]
        orig[i] = Tj[:3, 3]
        c, s = math.cos(q[i]), math.sin(q[i])
        Rz = np.eye(4)
        Rz[0, 0] = c; Rz[0, 1] = -s; Rz[1, 0] = s; Rz[1, 1] = c
        T = Tj @ Rz
    p = T[:3, :3] @ FLANGE + T[:3, 3]
    return np.cross(axes, p - orig).T, p, T[:3, 2]


class Collision:
    def __init__(self):
        import matplotlib; matplotlib.use('Agg')      # noqa: E402
        import torch                                  # noqa: E402
        from one.robots.manipulators.franka.fr3.sphere_collision import (
            FR3SphereCollision)
        c = FR3SphereCollision(device='cpu')
        self.centers = c.centers.numpy().astype(float)
        self.radii = c.radii.numpy().astype(float)
        self.li = c.link_indices.numpy()
        ij = np.argwhere(c.mask.numpy())
        self.i, self.j = ij[:, 0], ij[:, 1]
        self.rsum = self.radii[self.i] + self.radii[self.j]
        self.ch = np.concatenate([self.centers, np.ones((len(self.centers), 1))], 1)

    def margin(self, L):
        pos = np.einsum('kij,kj->ki', L[self.li], self.ch)[:, :3]
        d = np.linalg.norm(pos[self.i] - pos[self.j], axis=1)
        return float((d - self.rsum).min())


def verify(n=48, seed=1):
    """Compare against the torch model in its own dtype (float32), so the
    tolerance is float32 round-off and not a dtype conversion artefact."""
    import matplotlib; matplotlib.use('Agg')          # noqa: E402
    import torch                                      # noqa: E402
    from one.robots.manipulators.franka.fr3_pen.batched_fr3_kin import (
        BatchedFR3Kinematics)
    from one.robots.manipulators.franka.fr3.sphere_collision import (
        FR3SphereCollision)
    kin = BatchedFR3Kinematics(device='cpu', tcp_offset=TCP_OFFSET)
    cc = FR3SphereCollision(device='cpu')
    col = Collision()
    rng = np.random.default_rng(seed)
    Q = rng.uniform(LMT_LO, LMT_UP, size=(n, 7))
    qt = torch.tensor(Q, dtype=torch.float32)
    pt, Rt, Jt, _ = kin.tcp_fk_jac(qt)
    mt = cc.min_margin(kin.link_transforms(qt))
    ep = ez = ej = em = 0.0
    for k in range(n):
        J, p, z = tcp_jac(Q[k])
        ep = max(ep, abs(p - pt[k].numpy()).max())
        ez = max(ez, abs(z - Rt[k, :, 2].numpy()).max())
        ej = max(ej, abs(J - Jt[k, :3].numpy()).max())
        em = max(em, abs(col.margin(links(Q[k])) - float(mt[k])))
    print(f'[verify] numpy vs torch on {n} configs — dp {ep:.2e} m, dz {ez:.2e}, '
          f'dJ {ej:.2e}, d(collision margin) {em:.2e} m')
    assert max(ep, ez, ej, em) < 5e-6, 'numpy model disagrees with torch'
    print('[verify] numpy model matches the torch model to float32 round-off')


if __name__ == '__main__':
    verify()

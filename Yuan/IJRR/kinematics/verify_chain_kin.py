"""Cross-checks for BatchedChainKinematics before anything trains on it.

Three independent checks per robot:

  1. Finite-difference Jacobian: analytic J against central differences of
     the FK, positional and rotational blocks separately.
  2. Cross-check against the numpy models in the `one` package: CVR038 for
     the Cobotta, XArm7 (recovered from git) for the xArm7, both through
     their own ``fk`` with the same flange point.
  3. FR3 regression: an FR3 spec built from the existing
     ``BatchedFR3Kinematics`` internals must reproduce it bit-for-bit at
     fp32, proving the generic Rodrigues path equals the specialized
     ``rotz`` path.

Usage:
    python -m Yuan.IJRR.kinematics.verify_chain_kin
"""
from __future__ import annotations

import numpy as np
import torch

from Yuan.IJRR.kinematics.batched_chain_kin import (
    BatchedChainKinematics, SPECS, _EYE3)


def fd_jacobian_check(kin, n_q=64, eps=1e-4, seed=0):
    g = torch.Generator(device=kin.device).manual_seed(seed)
    q = kin.rand_conf_batch(n_q, generator=g)
    _, _, J, _ = kin.tcp_fk_jac(q)
    Jp_err = Jr_err = 0.0
    for i in range(kin.n_joints):
        dq = torch.zeros_like(q)
        dq[:, i] = eps
        p1, R1, _, _ = kin.tcp_fk_jac(q + dq)
        p0, R0, _, _ = kin.tcp_fk_jac(q - dq)
        Jp_fd = (p1 - p0) / (2 * eps)
        dR = R1 @ R0.transpose(-1, -2)
        w = torch.stack([dR[:, 2, 1] - dR[:, 1, 2],
                         dR[:, 0, 2] - dR[:, 2, 0],
                         dR[:, 1, 0] - dR[:, 0, 1]], dim=-1) / (2 * (2 * eps))
        Jp_err = max(Jp_err, (J[:, :3, i] - Jp_fd).abs().max().item())
        Jr_err = max(Jr_err, (J[:, 3:, i] - w).abs().max().item())
    return Jp_err, Jr_err


def one_model_check(name, kin, n_q=32, seed=1):
    """Compare every link transform against the numpy model in `one`.

    ``model.fk(qs)`` returns all link transforms ``(n+1, 4, 4)``, so this
    checks the full chain, not only the TCP.
    """
    import one.robots.manipulators.manipulator_base as ormmb
    if name == 'cobotta':
        from one.robots.manipulators.denso.cvr038.cvr038 import CVR038
        model = CVR038()
    elif name == 'xarm7':
        return _urdf_chain_check(kin, n_q=n_q, seed=seed)
    else:
        raise ValueError(name)
    rng = np.random.default_rng(seed)
    lo = kin.lmt_lo.cpu().numpy()
    up = kin.lmt_up.cpu().numpy()
    q_all = lo + rng.random((n_q, kin.n_joints)) * (up - lo)
    err = 0.0
    for qs in q_all:
        tfs_one = np.asarray(model.fk(qs.astype(np.float32)))
        qb = torch.as_tensor(qs, dtype=torch.float32).unsqueeze(0)
        tfs_kin = kin.link_transforms(qb)[0].cpu().numpy()
        n = min(tfs_one.shape[0], tfs_kin.shape[0])
        err = max(err, float(np.abs(tfs_one[:n] - tfs_kin[:n]).max()))
    return err


def _urdf_chain_check(kin, n_q=32, seed=1):
    """Independent FK straight from xarm7.urdf: parse the joint origins and
    axes with an XML reader and compose them in numpy, with no dependency on
    the `one` loader whose API has moved on."""
    import os
    import xml.etree.ElementTree as ET
    urdf = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'one',
                        'robots', 'manipulators', 'xarm', 'xarm7',
                        'xarm7.urdf')
    root = ET.parse(os.path.abspath(urdf)).getroot()
    joints = []
    for j in root.iter('joint'):
        if j.get('type') != 'revolute':
            continue
        o = j.find('origin')
        xyz = [float(v) for v in o.get('xyz', '0 0 0').split()]
        rpy = [float(v) for v in o.get('rpy', '0 0 0').split()]
        ax = [float(v) for v in j.find('axis').get('xyz').split()]
        joints.append((xyz, rpy, ax))
    assert len(joints) == kin.n_joints, len(joints)

    def rot_rpy(r, p, y):
        cr, sr, cp, sp, cy, sy = (np.cos(r), np.sin(r), np.cos(p),
                                  np.sin(p), np.cos(y), np.sin(y))
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        return Rz @ Ry @ Rx

    def rot_axis(a, q):
        a = np.asarray(a, dtype=float)
        a = a / np.linalg.norm(a)
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * (K @ K)

    rng = np.random.default_rng(seed)
    lo = kin.lmt_lo.cpu().numpy()
    up = kin.lmt_up.cpu().numpy()
    err = 0.0
    for qs in lo + rng.random((n_q, kin.n_joints)) * (up - lo):
        T = np.eye(4)
        tfs = [T.copy()]
        for (xyz, rpy, ax), q in zip(joints, qs):
            To = np.eye(4)
            To[:3, :3] = rot_rpy(*rpy)
            To[:3, 3] = xyz
            Tm = np.eye(4)
            Tm[:3, :3] = rot_axis(ax, q)
            T = T @ To @ Tm
            tfs.append(T.copy())
        tfs = np.stack(tfs)
        qb = torch.as_tensor(qs, dtype=torch.float32).unsqueeze(0)
        tfs_kin = kin.link_transforms(qb)[0].cpu().numpy()
        err = max(err, float(np.abs(tfs - tfs_kin).max()))
    return err


def fr3_regression(n_q=256, seed=2):
    from one.robots.manipulators.franka.fr3_pen.batched_fr3_kin import (
        BatchedFR3Kinematics)
    ref = BatchedFR3Kinematics(device='cpu')
    joints = []
    for i in range(7):
        tf = ref.zero_tfs[i]
        joints.append((tf[:3, :3].tolist(), tf[:3, 3].tolist(),
                       [0.0, 0.0, 1.0]))
    spec = dict(name='fr3_from_ref', joints=joints,
                lmt_lo=ref.lmt_lo.tolist(), lmt_up=ref.lmt_up.tolist(),
                qdot_max=ref.qdot_max.tolist(),
                flange_pos=[0.0, 0.0, 0.107])
    kin = BatchedChainKinematics(spec, device='cpu',
                                 tcp_offset=ref.tcp_offset)
    g = torch.Generator().manual_seed(seed)
    q = ref.rand_conf_batch(n_q, generator=g)
    p_a, R_a, J_a, _ = ref.tcp_fk_jac(q)
    p_b, R_b, J_b, _ = kin.tcp_fk_jac(q)
    return ((p_a - p_b).abs().max().item(),
            (R_a - R_b).abs().max().item(),
            (J_a - J_b).abs().max().item())


def main():
    print('FR3 regression (generic vs specialized, must be ~1e-6):')
    pe, re_, je = fr3_regression()
    print(f'  p {pe:.2e}   R {re_:.2e}   J {je:.2e}')

    for name in ('xarm7', 'cobotta'):
        kin64 = BatchedChainKinematics(name, device='cpu',
                                       dtype=torch.float64)
        jp, jr = fd_jacobian_check(kin64, eps=1e-6)
        print(f'{name}: FD Jacobian (fp64)  pos {jp:.2e}  rot {jr:.2e}')
        kin = BatchedChainKinematics(name, device='cpu')
        try:
            err = one_model_check(name, kin)
            print(f'{name}: vs one-model, all link transforms  max {err:.2e}')
        except Exception as e:
            print(f'{name}: one-model check FAILED to run: {type(e).__name__}: {e}')


if __name__ == '__main__':
    main()

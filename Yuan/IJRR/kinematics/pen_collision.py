"""Self-collision including the hand and the pen.

The shared ``FR3SphereCollision`` model covers link0..link7 only — 62 spheres
on the bare arm. The controlled TCP, however, sits 0.2034 m past the flange
(Franka Hand acting centre + a 10 cm pen), and none of that is represented.
Measured consequence: with the pen tip at (0, 0, 0.30) and the tool pointing
down, the pen shaft passes 4.6 cm inside the link1/link2 spheres and the stock
checker reports no collision.

This module attaches spheres to link7 covering the hand and the pen, so the
same check the rest of the pipeline already calls (``is_collided``) also sees
the tool. Geometry along link7's local +z:

    link7 origin --0.107 m--> flange --tcp_offset--> TCP

so the hand+pen occupies local z in [0.107, 0.107 + tcp_offset].

The spheres carry link index 7, which makes the existing pair mask check them
against link0..link4 and ignore link5/link6/link7 (adjacent or explicitly
ignored) — the pairs that matter here, since the measured interference is with
link1 and link2.

The radii are an approximation of the Franka Hand body and a thin pen shaft,
not a CAD-derived fit; they are deliberately on the generous side for the hand
so the check does not silently miss interference.
"""
from __future__ import annotations

import torch

from one.robots.manipulators.franka.fr3.sphere_collision import (
    FR3SphereCollision, fr3_self_collision_mask,
)

FLANGE_FROM_LINK7 = 0.107
HAND_RADIUS = 0.040
PEN_RADIUS = 0.015
# Fraction of the tool offset taken up by the hand body; the rest is the pen.
HAND_FRACTION = 0.1034 / 0.2034


class PenSphereCollision(FR3SphereCollision):
    """FR3 self-collision with the hand and pen included."""

    def __init__(self, tcp_offset: float, *args,
                 hand_radius: float = HAND_RADIUS,
                 pen_radius: float = PEN_RADIUS,
                 n_hand: int = 3, n_pen: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        dev, dt = self.device, self.dtype
        z0 = FLANGE_FROM_LINK7
        z_hand_end = z0 + HAND_FRACTION * tcp_offset
        z_tip = z0 + tcp_offset

        zs, rs = [], []
        for i in range(n_hand):
            t = (i + 0.5) / n_hand
            zs.append(z0 + t * (z_hand_end - z0))
            rs.append(hand_radius)
        for i in range(n_pen):
            t = (i + 0.5) / n_pen
            zs.append(z_hand_end + t * (z_tip - z_hand_end))
            rs.append(pen_radius)

        extra_c = torch.tensor([[0.0, 0.0, z] for z in zs], device=dev, dtype=dt)
        extra_r = torch.tensor(rs, device=dev, dtype=dt)
        extra_l = torch.full((len(zs),), 7, device=dev, dtype=torch.long)

        self.n_tool_spheres = len(zs)
        self.centers = torch.cat([self.centers, extra_c], 0)
        self.radii = torch.cat([self.radii, extra_r], 0)
        self.link_indices = torch.cat([self.link_indices, extra_l], 0)
        self.mask = fr3_self_collision_mask(self.link_indices)


def _selftest() -> None:  # pragma: no cover
    import numpy as np
    from scipy.spatial import cKDTree
    from one.robots.manipulators.franka.fr3_pen.batched_fr3_kin import (
        BatchedFR3Kinematics, DEFAULT_TCP_OFFSET)
    from Yuan.IJRR.stage1_seed.cone_ik import _build_R_with_z, _sample_in_cone
    from Yuan.IJRR.kinematics.batched_rollout import _batched_ik_project

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kin = BatchedFR3Kinematics(device=dev, tcp_offset=DEFAULT_TCP_OFFSET)
    bare = FR3SphereCollision(device=dev)
    withpen = PenSphereCollision(DEFAULT_TCP_OFFSET, device=dev)
    print(f'bare spheres {len(bare.radii)}, with tool {len(withpen.radii)} '
          f'(+{withpen.n_tool_spheres})')

    # The tool tip really is where the kinematics says it is.
    q = kin.rand_conf_batch(64, generator=torch.Generator(device=dev).manual_seed(0))
    p, R, _, _ = kin.tcp_fk_jac(q)
    tf = kin.link_transforms(q)
    cen = withpen.sphere_positions(tf)
    tip = cen[:, -1]                      # last tool sphere sits nearest the tip
    err = (tip - p).norm(dim=-1)
    assert err.max() < 0.03, err.max()
    print(f'last tool sphere to TCP: max {err.max()*1e3:.1f} mm (expected < 30)')

    T = np.load('Yuan/IJRR/runs/iksel_clean_v1/cvt_table_201600.npz')
    tree = cKDTree(np.concatenate([T['pos'] * 20.0, T['zax']], 1).astype(np.float32))
    nv = np.array([0, 0, -1], np.float32)
    dirs = _sample_in_cone(torch.as_tensor(nv), 29.5, 32,
                           np.random.default_rng(0)).numpy()
    dirs[0] = nv
    hint = torch.tensor([1., 0, 0], device=dev, dtype=kin.dtype)
    print(f"\n{'x':>7} {'solutions':>10} {'bare says coll':>15} "
          f"{'with tool says coll':>20}")
    for x in (-0.30, -0.12, 0.0, 0.12, 0.30):
        tgt = np.array([[x, 0.0, 0.30]], np.float32)
        found = []
        for k in range(len(dirs)):
            z = dirs[k:k + 1]
            _, ids = tree.query(np.concatenate([tgt * 20.0, z], 1), k=12)
            for t in range(12):
                q0 = torch.as_tensor(T['q'][ids[:, t]], device=dev, dtype=kin.dtype)
                Rt = _build_R_with_z(torch.as_tensor(z, device=dev, dtype=kin.dtype),
                                     hint)
                qo, cv, _ = _batched_ik_project(
                    kin, q0, torch.as_tensor(tgt, device=dev, dtype=kin.dtype),
                    Rt, branch_action=None)
                if bool(cv[0]):
                    pp, _, _, _ = kin.tcp_fk_jac(qo)
                    if float((pp[0] - torch.as_tensor(
                            tgt[0], device=dev, dtype=kin.dtype)).norm()) < 5e-3:
                        found.append(qo[0])
                    break
        if not found:
            print(f'{x:>7.2f} {"0":>10}')
            continue
        Q = torch.stack(found)
        tf = kin.link_transforms(Q)
        print(f'{x:>7.2f} {len(found):>10} {int(bare.is_collided(tf).sum()):>15} '
              f'{int(withpen.is_collided(tf).sum()):>20}')
    print('\nExpect: no collision at |x| = 0.30, and the tool-aware model '
          'flagging most solutions near the base axis.')


if __name__ == '__main__':
    _selftest()

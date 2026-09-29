"""Torch batched FK/Jacobian for a serial chain with arbitrary joint axes.

Generalizes ``BatchedFR3Kinematics`` so the same environment runs on the
xArm7 and the Cobotta CVR038. The FR3 class hardcodes all joints about the
local z axis; the Cobotta model in ``one`` rotates three of its joints about
local y, so this class takes the axis per joint and rotates by Rodrigues'
formula. Each link frame reproduces the frame of the source model (the
recovered ``xarm7.urdf`` for the xArm7; ``one/robots/.../cvr038.py`` for the
Cobotta), which is what makes the cross-checks in
``verify_chain_kin.py`` possible.

The interface mirrors ``BatchedFR3Kinematics`` exactly, plus ``n_joints``:
``lmt_lo/lmt_up/q_mid/jnt_ranges/qdot_max``, ``link_transforms``,
``fk_jac``, ``tcp_fk_jac``, ``fk_batch``, ``rand_conf_batch``.

The pen is mounted directly on the flange for both arms (no hand), so the
TCP sits ``tcp_offset`` along the flange +z; the experiment keeps the FR3
pen length of 0.10 m.
"""
from __future__ import annotations

import math

import torch

PEN_LENGTH = 0.10


def _rotx(a):
    c, s = math.cos(a), math.sin(a)
    return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]


_EYE3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

# ---------------------------------------------------------------------------
# Robot specs. Each joint is (loc_rotmat, loc_pos, axis) in the parent link
# frame; the joint rotates about ``axis`` expressed in its own frame.
# ---------------------------------------------------------------------------

# Recovered from one/robots/manipulators/xarm/xarm7/xarm7.urdf (git 4e02032):
# every joint origin carries an rpy about x and every joint rotates about
# local z, the same pattern as the FR3.
XARM7 = dict(
    name='xarm7',
    joints=[
        (_EYE3,            [0.0, 0.0, 0.267],       [0.0, 0.0, 1.0]),
        (_rotx(-math.pi/2), [0.0, 0.0, 0.0],        [0.0, 0.0, 1.0]),
        (_rotx(math.pi/2),  [0.0, -0.293, 0.0],     [0.0, 0.0, 1.0]),
        (_rotx(math.pi/2),  [0.0525, 0.0, 0.0],     [0.0, 0.0, 1.0]),
        (_rotx(math.pi/2),  [0.0775, -0.3425, 0.0], [0.0, 0.0, 1.0]),
        (_rotx(math.pi/2),  [0.0, 0.0, 0.0],        [0.0, 0.0, 1.0]),
        (_rotx(-math.pi/2), [0.076, 0.097, 0.0],    [0.0, 0.0, 1.0]),
    ],
    lmt_lo=[-math.pi, -2.18, -math.pi, -0.11, -math.pi, -1.75, -math.pi],
    lmt_up=[math.pi, 2.18, math.pi, math.pi, math.pi, math.pi, math.pi],
    qdot_max=[3.14] * 7,
    # link7 frame is the flange plate; only the pen extends past it.
    flange_pos=[0.0, 0.0, 0.0],
)

# From one/robots/manipulators/denso/cvr038/cvr038.py (official Denso xacro):
# axes Z, Y, Y, Z, Y, Z with identity local rotations.
COBOTTA = dict(
    name='cobotta',
    joints=[
        (_EYE3, [0.0, 0.0, 0.0],           [0.0, 0.0, 1.0]),
        (_EYE3, [0.0, 0.0, 0.18],          [0.0, 1.0, 0.0]),
        (_EYE3, [0.0, 0.0, 0.165],         [0.0, 1.0, 0.0]),
        (_EYE3, [-0.012, 0.02, -0.345],    [0.0, 0.0, 1.0]),
        (_EYE3, [0.0, -0.02, 0.5225],      [0.0, 1.0, 0.0]),
        (_EYE3, [0.0, -0.0445, 0.042],     [0.0, 0.0, 1.0]),
    ],
    lmt_lo=[-2.61799388, -1.04719755, 0.31415927,
            -2.96705973, -1.65806279, -2.96705973],
    lmt_up=[2.61799388, 1.74532925, 2.44346095,
            2.96705973, 2.35619449, 2.96705973],
    qdot_max=[3.14] * 6,
    flange_pos=[0.0, 0.0, 0.0],
)

SPECS = {'xarm7': XARM7, 'cobotta': COBOTTA}


def tool_rotmat(tilt_deg: float, azimuth_deg: float = 0.0):
    """Rotation (numpy 3x3) taking the flange z to the tool axis
    a0 = (sin b cos p, sin b sin p, cos b): the shortest rotation, about
    z x a0. tilt 0 returns the exact identity."""
    import numpy as _np
    b = math.radians(float(tilt_deg)); p = math.radians(float(azimuth_deg))
    if b == 0.0:
        return _np.eye(3)
    a0 = _np.array([math.sin(b) * math.cos(p), math.sin(b) * math.sin(p), math.cos(b)])
    k = _np.cross([0.0, 0.0, 1.0], a0); k /= _np.linalg.norm(k)
    K = _np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return _np.eye(3) + math.sin(b) * K + (1.0 - math.cos(b)) * (K @ K)


def tool_rotmats(tilt: torch.Tensor, azimuth: torch.Tensor) -> torch.Tensor:
    """Batched tool_rotmat: (B,) tilt / azimuth [rad] -> (B,3,3), exact
    identity where tilt == 0."""
    B = tilt.shape[0]
    a0 = torch.stack([torch.sin(tilt) * torch.cos(azimuth),
                      torch.sin(tilt) * torch.sin(azimuth), torch.cos(tilt)], -1)
    z = torch.zeros_like(a0); z[:, 2] = 1.0
    k = torch.linalg.cross(z, a0, dim=-1)
    k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    K = torch.zeros(B, 3, 3, device=tilt.device, dtype=tilt.dtype)
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    s, c = torch.sin(tilt).view(B, 1, 1), torch.cos(tilt).view(B, 1, 1)
    R = torch.eye(3, device=tilt.device, dtype=tilt.dtype).expand(B, 3, 3) + s * K + (1 - c) * (K @ K)
    return torch.where((tilt.abs() > 1e-12).view(B, 1, 1), R,
                       torch.eye(3, device=tilt.device, dtype=tilt.dtype).expand(B, 3, 3))


def _skew(v: torch.Tensor) -> torch.Tensor:
    K = torch.zeros((3, 3), device=v.device, dtype=v.dtype)
    K[0, 1], K[0, 2] = -v[2], v[1]
    K[1, 0], K[1, 2] = v[2], -v[0]
    K[2, 0], K[2, 1] = -v[1], v[0]
    return K


class BatchedChainKinematics:
    """Batched FK/Jacobian for one of the specs above.

    Same call signatures as ``BatchedFR3Kinematics`` so
    ``NSRLBatchedEnv`` can hold either without caring which.
    """

    def __init__(self, spec: dict | str, device=None, dtype=torch.float32,
                 tcp_offset: float = PEN_LENGTH, tool_xyz=None,
                 tool_tilt_deg: float = 0.0, tool_azimuth_deg: float = 0.0):
        if isinstance(spec, str):
            spec = SPECS[spec]
        self.name = spec['name']
        self.device = torch.device('cpu' if device is None else device)
        self.dtype = dtype
        self.n_joints = len(spec['joints'])

        tfs, axes = [], []
        for rot, pos, axis in spec['joints']:
            tf = torch.eye(4, device=self.device, dtype=dtype)
            tf[:3, :3] = torch.as_tensor(rot, device=self.device, dtype=dtype)
            tf[:3, 3] = torch.as_tensor(pos, device=self.device, dtype=dtype)
            tfs.append(tf)
            a = torch.as_tensor(axis, device=self.device, dtype=dtype)
            axes.append(a / a.norm())
        self.zero_tfs = torch.stack(tfs, dim=0)
        self.axes = torch.stack(axes, dim=0)             # (n, 3)
        self._K = torch.stack([_skew(a) for a in self.axes], dim=0)
        self._K2 = self._K @ self._K

        self.lmt_lo = torch.as_tensor(spec['lmt_lo'], device=self.device,
                                      dtype=dtype)
        self.lmt_up = torch.as_tensor(spec['lmt_up'], device=self.device,
                                      dtype=dtype)
        self.q_mid = 0.5 * (self.lmt_lo + self.lmt_up)
        self.qdot_max = torch.as_tensor(spec['qdot_max'], device=self.device,
                                        dtype=dtype)
        base = torch.as_tensor(spec['flange_pos'], device=self.device,
                               dtype=dtype)
        # tool_xyz: general tool point in the flange frame (e.g. the XHand
        # index fingertip, which sits 2.7 cm off the flange axis); the tool
        # axis stays the flange z. None keeps the on-axis pen at tcp_offset.
        # Tilted tool axis (a bent tool): the tool frame is the flange frame
        # rotated by tool_rotmat, the tip sits tcp_offset along the tilted
        # axis. tilt 0 is the exact identity, so every existing config is
        # untouched.
        R_tool = torch.as_tensor(tool_rotmat(tool_tilt_deg, tool_azimuth_deg),
                                 device=self.device, dtype=dtype)
        if tool_xyz is not None:
            tool = torch.as_tensor(list(tool_xyz), device=self.device, dtype=dtype)
            tcp_offset = float(tool[2])
        elif float(tool_tilt_deg) != 0.0:
            tool = R_tool @ torch.as_tensor([0.0, 0.0, tcp_offset],
                                            device=self.device, dtype=dtype)
        else:
            tool = torch.as_tensor([0.0, 0.0, tcp_offset], device=self.device,
                                   dtype=dtype)
        self.flange_p = base + tool
        self.flange_R = R_tool
        self.tcp_offset = float(tcp_offset)
        self.tool_xyz = tuple(float(v) for v in tool)
        self.tool_tilt_deg = float(tool_tilt_deg)
        self.tool_azimuth_deg = float(tool_azimuth_deg)

    @property
    def jnt_ranges(self) -> torch.Tensor:
        return torch.stack([self.lmt_lo, self.lmt_up], dim=1)

    def _joint_rot(self, i: int, q: torch.Tensor) -> torch.Tensor:
        """Rodrigues rotation about joint i's axis, batched over q (B,)."""
        c = torch.cos(q).view(-1, 1, 1)
        s = torch.sin(q).view(-1, 1, 1)
        eye = torch.eye(3, device=self.device, dtype=self.dtype)
        return eye + s * self._K[i] + (1.0 - c) * self._K2[i]

    def _motion(self, i: int, q: torch.Tensor) -> torch.Tensor:
        b = q.shape[0]
        tf = torch.eye(4, device=self.device, dtype=self.dtype).expand(
            b, 4, 4).clone()
        tf[:, :3, :3] = self._joint_rot(i, q)
        return tf

    def link_transforms(self, q: torch.Tensor) -> torch.Tensor:
        """World transforms for link0..link<n>, shaped (B, n+1, 4, 4)."""
        q = q.to(device=self.device, dtype=self.dtype)
        b = q.shape[0]
        T = torch.eye(4, device=self.device, dtype=self.dtype).expand(
            b, 4, 4).clone()
        links = [T.clone()]
        for i in range(self.n_joints):
            T_j = T @ self.zero_tfs[i].expand(b, 4, 4)
            T = T_j @ self._motion(i, q[:, i])
            links.append(T.clone())
        return torch.stack(links, dim=1)

    def fk_jac(self, q: torch.Tensor, local_point: torch.Tensor | None = None):
        """Return (p_tcp, R_last, J, T_last) for q shaped (B, n)."""
        q = q.to(device=self.device, dtype=self.dtype)
        b = q.shape[0]
        T = torch.eye(4, device=self.device, dtype=self.dtype).expand(
            b, 4, 4).clone()
        jnt_Ts = []
        for i in range(self.n_joints):
            T_j = T @ self.zero_tfs[i].expand(b, 4, 4)
            jnt_Ts.append(T_j)
            T = T_j @ self._motion(i, q[:, i])
        T_last = T
        if local_point is None:
            local_point = self.flange_p
        local_point = local_point.to(device=self.device, dtype=self.dtype)
        # local_point may be one point (3,) or one per batch row (B, 3)
        p_tcp = (T_last[:, :3, :3] @ local_point.reshape(-1, 3, 1)).squeeze(-1)
        p_tcp = p_tcp + T_last[:, :3, 3]

        J = torch.zeros((b, 6, self.n_joints), device=self.device,
                        dtype=self.dtype)
        for i, T_j in enumerate(jnt_Ts):
            axis = (T_j[:, :3, :3] @ self.axes[i].view(1, 3, 1)).squeeze(-1)
            p_j = T_j[:, :3, 3]
            J[:, :3, i] = torch.cross(axis, p_tcp - p_j, dim=-1)
            J[:, 3:, i] = axis
        return p_tcp, T_last[:, :3, :3], J, T_last

    def tcp_fk_jac(self, q: torch.Tensor):
        p_tcp, R_last, J, T_last = self.fk_jac(q, self.flange_p)
        return p_tcp, R_last @ self.flange_R, J, T_last

    def fk_batch(self, q: torch.Tensor):
        p_tcp, R_tcp, _, _ = self.tcp_fk_jac(q)
        return p_tcp, R_tcp

    def rand_conf_batch(self, batch_size: int,
                        generator: torch.Generator | None = None) -> torch.Tensor:
        u = torch.rand((batch_size, self.n_joints), device=self.device,
                       dtype=self.dtype, generator=generator)
        return self.lmt_lo.unsqueeze(0) + u * (self.lmt_up
                                               - self.lmt_lo).unsqueeze(0)

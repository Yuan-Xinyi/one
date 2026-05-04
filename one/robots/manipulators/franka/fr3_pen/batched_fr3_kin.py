"""Torch batched FR3 FK and TCP Jacobian.

Tensor-friendly mirror of the FR3 chain in ``one.robots.manipulators.franka.fr3``.
Skips scene/mesh state so rollout batches stay as plain tensor math.

The kinematic constants (per-joint ``loc_pos`` / ``loc_rotmat``) match those in
``one/robots/manipulators/franka/fr3/fr3.py``. The flange is at ``+0.107`` along
link7's z by FR3 convention; ``tcp_offset`` adds anything mounted past the
flange (e.g. Franka Hand acting center + pen tip).
"""
from __future__ import annotations

import math

import torch


# Default end-effector geometry: bare flange + Franka Hand grasptarget + 10 cm pen.
# Override at construction with ``BatchedFR3Kinematics(tcp_offset=...)`` for
# different end-effectors or to use the bare flange (``tcp_offset=0.0``).
HAND_TCP_OFFSET = 0.1034
PEN_LENGTH = 0.10
DEFAULT_TCP_OFFSET = HAND_TCP_OFFSET + PEN_LENGTH

# FR3 datasheet velocity limits (rad/s). Joints 1-4: 150 deg/s, joints 5-7: 180 deg/s.
QDOT_MAX_DEFAULT = (2.62, 2.62, 2.62, 2.62, 3.14, 3.14, 3.14)


def _as_tensor(data, device, dtype):
    return torch.as_tensor(data, device=device, dtype=dtype)


def _tf_from_rot_pos(rot: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    tf = torch.eye(4, device=rot.device, dtype=rot.dtype)
    tf[:3, :3] = rot
    tf[:3, 3] = pos
    return tf


def _rotx(angle: float, device, dtype) -> torch.Tensor:
    c = math.cos(angle)
    s = math.sin(angle)
    return _as_tensor([[1.0, 0.0, 0.0],
                       [0.0, c, -s],
                       [0.0, s, c]], device, dtype)


def _rotz_batch(q: torch.Tensor) -> torch.Tensor:
    b = q.shape[0]
    c = torch.cos(q)
    s = torch.sin(q)
    rot = torch.zeros((b, 3, 3), device=q.device, dtype=q.dtype)
    rot[:, 0, 0] = c
    rot[:, 0, 1] = -s
    rot[:, 1, 0] = s
    rot[:, 1, 1] = c
    rot[:, 2, 2] = 1.0
    return rot


def _motion_z_batch(q: torch.Tensor) -> torch.Tensor:
    b = q.shape[0]
    tf = torch.eye(4, device=q.device, dtype=q.dtype).expand(b, 4, 4).clone()
    tf[:, :3, :3] = _rotz_batch(q)
    return tf


class BatchedFR3Kinematics:
    """FR3 FK/Jacobian for active joint batches.

    All joints rotate around their local z axes. The returned Jacobian matches
    ``NumIKSolver._forward(..., local_point=flange_tcp_p)``.

    ``tcp_offset`` is the distance along link7's +z from the flange (which is
    already 0.107 m past link7's origin) to the controlled TCP. Default value
    corresponds to the Franka Hand acting center + a 10 cm pen tip.
    """

    def __init__(self, device=None, dtype=torch.float32,
                 tcp_offset: float = DEFAULT_TCP_OFFSET,
                 lmt_lo=None, lmt_up=None,
                 qdot_max=None):
        self.device = torch.device('cpu' if device is None else device)
        self.dtype = dtype
        self.axis_z = _as_tensor([0.0, 0.0, 1.0], self.device, self.dtype)
        self.zero_tfs = self._make_zero_tfs()
        # FR3 (Franka Research 3) joint limits — match one's fr3.py.
        if lmt_lo is None:
            lmt_lo = [-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159]
        if lmt_up is None:
            lmt_up = [2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159]
        self.lmt_lo = _as_tensor(lmt_lo, self.device, self.dtype)
        self.lmt_up = _as_tensor(lmt_up, self.device, self.dtype)
        self.q_mid = 0.5 * (self.lmt_lo + self.lmt_up)
        if qdot_max is None:
            qdot_max = QDOT_MAX_DEFAULT
        self.qdot_max = _as_tensor(qdot_max, self.device, self.dtype)
        self.flange_p = _as_tensor([0.0, 0.0, 0.107 + tcp_offset],
                                   self.device, self.dtype)
        self.flange_R = torch.eye(3, device=self.device, dtype=self.dtype)
        self.tcp_offset = float(tcp_offset)

    @property
    def jnt_ranges(self) -> torch.Tensor:
        """``(7, 2)`` tensor of ``[lower, upper]`` per active joint."""
        return torch.stack([self.lmt_lo, self.lmt_up], dim=1)

    def _make_zero_tfs(self) -> torch.Tensor:
        dev, dt = self.device, self.dtype
        specs = [
            (torch.eye(3, device=dev, dtype=dt), [0.0, 0.0, 0.333]),
            (_rotx(-math.pi / 2, dev, dt), [0.0, 0.0, 0.0]),
            (_rotx(math.pi / 2, dev, dt), [0.0, -0.316, 0.0]),
            (_rotx(math.pi / 2, dev, dt), [0.0825, 0.0, 0.0]),
            (_rotx(-math.pi / 2, dev, dt), [-0.0825, 0.384, 0.0]),
            (_rotx(math.pi / 2, dev, dt), [0.0, 0.0, 0.0]),
            (_rotx(math.pi / 2, dev, dt), [0.088, 0.0, 0.0]),
        ]
        tfs = [_tf_from_rot_pos(rot, _as_tensor(pos, dev, dt))
               for rot, pos in specs]
        return torch.stack(tfs, dim=0)

    def link_transforms(self, q: torch.Tensor) -> torch.Tensor:
        """Return world transforms for link0..link7, shaped ``(B, 8, 4, 4)``."""
        q = q.to(device=self.device, dtype=self.dtype)
        b = q.shape[0]
        T = torch.eye(4, device=self.device, dtype=self.dtype).expand(
            b, 4, 4).clone()
        links = [T.clone()]
        for i in range(7):
            T_j = T @ self.zero_tfs[i].expand(b, 4, 4)
            T = T_j @ _motion_z_batch(q[:, i])
            links.append(T.clone())
        return torch.stack(links, dim=1)

    def fk_jac(self, q: torch.Tensor, local_point: torch.Tensor | None = None):
        """Return ``(p_tcp, R_last, J, T_last)`` for ``q`` shaped ``(B, 7)``."""
        q = q.to(device=self.device, dtype=self.dtype)
        b = q.shape[0]
        T = torch.eye(4, device=self.device, dtype=self.dtype).expand(
            b, 4, 4).clone()
        jnt_Ts = []
        for i in range(7):
            T_j = T @ self.zero_tfs[i].expand(b, 4, 4)
            jnt_Ts.append(T_j)
            T = T_j @ _motion_z_batch(q[:, i])
        T_last = T
        if local_point is None:
            local_point = self.flange_p
        local_point = local_point.to(device=self.device, dtype=self.dtype)
        p_tcp = (T_last[:, :3, :3] @ local_point.view(1, 3, 1)).squeeze(-1)
        p_tcp = p_tcp + T_last[:, :3, 3]

        J = torch.zeros((b, 6, 7), device=self.device, dtype=self.dtype)
        for i, T_j in enumerate(jnt_Ts):
            axis = T_j[:, :3, :3] @ self.axis_z.view(1, 3, 1)
            axis = axis.squeeze(-1)
            p_j = T_j[:, :3, 3]
            J[:, :3, i] = torch.cross(axis, p_tcp - p_j, dim=-1)
            J[:, 3:, i] = axis
        return p_tcp, T_last[:, :3, :3], J, T_last

    def tcp_fk_jac(self, q: torch.Tensor):
        p_tcp, R_last, J, T_last = self.fk_jac(q, self.flange_p)
        R_tcp = R_last @ self.flange_R
        return p_tcp, R_tcp, J, T_last

    def fk_batch(self, q: torch.Tensor):
        """Convenience batched FK returning ``(p_tcp, R_tcp)`` only.

        Differentiable w.r.t. ``q``; matches the legacy
        ``PenFrankaResearch3GPU.robot.fk_batch`` interface.
        """
        q = q.to(device=self.device, dtype=self.dtype)
        b = q.shape[0]
        T = torch.eye(4, device=self.device, dtype=self.dtype).expand(
            b, 4, 4).clone()
        for i in range(7):
            T_j = T @ self.zero_tfs[i].expand(b, 4, 4)
            T = T_j @ _motion_z_batch(q[:, i])
        local_point = self.flange_p
        p_tcp = (T[:, :3, :3] @ local_point.view(1, 3, 1)).squeeze(-1)
        p_tcp = p_tcp + T[:, :3, 3]
        R_tcp = T[:, :3, :3] @ self.flange_R
        return p_tcp, R_tcp

    def rand_conf_batch(self, batch_size: int,
                        generator: torch.Generator | None = None) -> torch.Tensor:
        """Uniform random joint configurations within ``[lmt_lo, lmt_up]``."""
        u = torch.rand((batch_size, 7), device=self.device, dtype=self.dtype,
                       generator=generator)
        return self.lmt_lo.unsqueeze(0) + u * (self.lmt_up - self.lmt_lo).unsqueeze(0)

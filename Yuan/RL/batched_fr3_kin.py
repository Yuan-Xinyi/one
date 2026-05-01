"""Torch batched FR3 FK and TCP Jacobian.

This is a narrow, tensor-friendly version of the FR3 kinematic chain used by
``one``. It intentionally skips scene objects and mesh state so rollout batches
can run as plain tensor math.
"""
from __future__ import annotations

import math

import torch

import Yuan.RL.config as cfg


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
    """

    def __init__(self, device=None, dtype=torch.float32,
                 tcp_offset: float | None = None):
        if tcp_offset is None:
            tcp_offset = float(getattr(cfg, "TCP_OFFSET", 0.0))
        self.device = torch.device('cpu' if device is None else device)
        self.dtype = dtype
        self.axis_z = _as_tensor([0.0, 0.0, 1.0], self.device, self.dtype)
        self.zero_tfs = self._make_zero_tfs()
        self.lmt_lo = _as_tensor(
            [-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159],
            self.device, self.dtype)
        self.lmt_up = _as_tensor(
            [2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159],
            self.device, self.dtype)
        self.q_mid = 0.5 * (self.lmt_lo + self.lmt_up)
        self.qdot_max = _as_tensor(
            [2.62, 2.62, 2.62, 2.62, 3.14, 3.14, 3.14],
            self.device, self.dtype)
        self.flange_p = _as_tensor([0.0, 0.0, 0.107 + tcp_offset],
                                   self.device, self.dtype)
        self.flange_R = torch.eye(3, device=self.device, dtype=self.dtype)

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
        """Return world transforms for link0..link7, shaped ``(B,8,4,4)``."""
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
        """Return ``(p_tcp, R_last, J, T_last)`` for ``q`` shaped ``(B,7)``."""
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

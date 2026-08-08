"""textbook redundant-manipulator nullspace controller, adapted to our env.

Mirrors classical nullspace control (Yoshikawa manipulability + Liegeois JL
avoidance + cone attractor), projected onto our 3-DOF position-only nullspace
and packaged as an action_fn compatible with ``rollout_first_episode``.

Parameters mirror flow_connectivity/config.py defaults; only the angle gate
threshold is retargeted to our 30° cone (vs their 5° tolerance for plane-writing).
"""
from __future__ import annotations

import math

import torch

from Yuan.IJRR.env.env import NSRLBatchedEnv, build_task_aligned_basis


class ClassicalNullspaceController:
    """Null-space controller. Default gains (manip 0.8, jl 0.4, angle 0.2) were
    retuned on the 10k eval set (2026-05-30 grid sweep over q0_seed rollouts).
    angle_boundary_gain is inert in practice (the cone term g(theta) almost
    never activates on this task distribution), so its value is immaterial."""

    def __init__(self, kin,
                 manip_gain: float = 0.8,
                 jl_gain: float = 0.4,
                 angle_boundary_gain: float = 0.2,
                 angle_margin_deg: float = 8.0,
                 theta_max_deg: float = 30.0,
                 manip_damping: float = 1e-3):
        self.kin = kin
        self.manip_gain = manip_gain
        self.jl_gain = jl_gain
        self.angle_boundary_gain = angle_boundary_gain
        self.angle_margin = math.radians(angle_margin_deg)
        self.theta_max = math.radians(theta_max_deg)
        self.manip_damping = manip_damping
        # cached vectors
        self._q_mid = kin.q_mid
        self._span = (kin.lmt_up - kin.lmt_lo).clamp_min(1e-6)

    def _directional_manipulability(self, J_pos: torch.Tensor,
                                    direction: torch.Tensor) -> torch.Tensor:
        eye3 = torch.eye(3, device=J_pos.device, dtype=J_pos.dtype).expand(
            J_pos.shape[0], 3, 3)
        JJt = J_pos @ J_pos.transpose(-1, -2) + (self.manip_damping ** 2) * eye3
        d_col = direction.unsqueeze(-1)
        inv_quad = (d_col.transpose(-1, -2) @ torch.linalg.inv(JJt) @ d_col
                    ).squeeze(-1).squeeze(-1)
        return inv_quad.clamp_min(1e-12).pow(-0.5)

    def q_dot_null(self, q: torch.Tensor, u_hat: torch.Tensor,
                   n_target: torch.Tensor) -> torch.Tensor:
        """Compute raw nullspace velocity request in rad/s, shape (B, 7).
        Forced into grad mode in case the caller wraps in torch.no_grad()."""
        with torch.enable_grad():
            return self._compute_q_dot(q, u_hat, n_target)

    def _compute_q_dot(self, q, u_hat, n_target):
        q_eval = q.detach().clone().requires_grad_(True)
        _, R_tcp, J, _ = self.kin.tcp_fk_jac(q_eval)
        z_cur = R_tcp[:, :, 2]
        cos_theta = (z_cur * n_target).sum(dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        J_pos = J[:, :3, :]

        q_dot = torch.zeros_like(q)

        # 1. Manipulability gradient (push away from singularity in u_hat direction)
        if self.manip_gain != 0.0:
            mu = self._directional_manipulability(J_pos, u_hat)
            grad_mu = torch.autograd.grad(
                mu.sum(), q_eval, retain_graph=True,
                create_graph=False, allow_unused=True)[0]
            if grad_mu is not None:
                q_dot = q_dot + self.manip_gain * grad_mu.detach()

        # 2. JL center pull (linear restoring force toward mid)
        if self.jl_gain != 0.0:
            q_dot = q_dot - self.jl_gain * (q - self._q_mid) / self._span

        # 3. Angle gradient (cone-aware soft boundary)
        if self.angle_boundary_gain != 0.0:
            grad_cos = torch.autograd.grad(
                cos_theta.sum(), q_eval, retain_graph=False,
                create_graph=False, allow_unused=True)[0]
            if grad_cos is not None:
                theta = torch.acos(cos_theta)
                boundary_gate = ((theta - self.theta_max) / self.angle_margin
                                 ).clamp(0.0, 1.0).unsqueeze(-1)
                q_dot = q_dot + self.angle_boundary_gain * boundary_gate * grad_cos.detach()

        return q_dot


def cn_action_fn(controller: ClassicalNullspaceController):
    """Action closure for ``rollout_first_episode``."""

    def _fn(env: NSRLBatchedEnv) -> torch.Tensor:
        with torch.no_grad():
            B_basis, _ = build_task_aligned_basis(
                env.kin, env.q, env.line_dir, env.n_target,
                env.kin.q_mid, env.q_half, env.cfg.manip_damping,
            )

        # Need grad for q_dot_null computation
        q_dot_raw = controller.q_dot_null(env.q, env.line_dir, env.n_target)
        # Project onto nullspace basis: a = B^T q_dot
        with torch.no_grad():
            a_raw = (B_basis.transpose(-1, -2) @ q_dot_raw.unsqueeze(-1)).squeeze(-1)
            return (a_raw / env.a_max).clamp(-1.0, 1.0)

    return _fn

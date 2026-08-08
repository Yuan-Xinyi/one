"""Torch batched IK projection and segmented rollout for v18.

Public surface used by the v18 pipeline:
    _branch_seed_bank          deterministic IK start postures
    _batched_ik_project        DLS IK with branch-aware null-space objective
    batched_rollout_segment    segmented Cartesian DLS rollout from q_init
    _device_from_cfg, _load_fr3_sphere_collision_cls   helpers
"""
from __future__ import annotations

import numpy as np
import torch

import Yuan.IJRR.kinematics.config as cfg
from Yuan.IJRR.kinematics.batched_fr3_kin import BatchedFR3Kinematics


def _normalize(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _normalize_angle_pair(pair: torch.Tensor) -> torch.Tensor:
    norm = pair.norm(dim=-1, keepdim=True)
    default = torch.zeros_like(pair)
    default[:, 0] = 1.0
    return torch.where(norm > 1e-6, pair / norm.clamp_min(1e-12), default)


def _z_axis_error_from_rotmats(R_cur: torch.Tensor,
                               R_tgt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z_cur = R_cur[:, :, 2]
    z_tgt = R_tgt[:, :, 2]
    cross = torch.cross(z_cur, z_tgt, dim=-1)
    cos_th = (z_cur * z_tgt).sum(dim=-1).clamp(-1.0, 1.0)
    theta = torch.acos(cos_th)
    sin_th = cross.norm(dim=-1)
    ref_x = torch.tensor([1.0, 0.0, 0.0], device=R_cur.device, dtype=R_cur.dtype)
    ref_y = torch.tensor([0.0, 1.0, 0.0], device=R_cur.device, dtype=R_cur.dtype)
    fallback_ref = torch.where((z_cur * ref_x.view(1, 3)).abs().sum(dim=-1, keepdim=True) < 0.9,
                               ref_x.view(1, 3), ref_y.view(1, 3))
    fallback_axis = _normalize(torch.cross(z_cur, fallback_ref.expand_as(z_cur), dim=-1))
    axis = torch.where(sin_th.unsqueeze(-1) > 1e-6,
                       cross / sin_th.clamp_min(1e-12).unsqueeze(-1),
                       fallback_axis)
    return axis * theta.unsqueeze(-1), theta


def _target_swivel_dir(p0: torch.Tensor, branch_action: torch.Tensor) -> torch.Tensor:
    """Target elbow direction around the shoulder-target axis."""
    shoulder = torch.tensor([0.0, 0.0, 0.333], device=p0.device, dtype=p0.dtype)
    axis = _normalize(p0 - shoulder.view(1, 3))
    world_z = torch.tensor([0.0, 0.0, 1.0], device=p0.device, dtype=p0.dtype)
    ref = world_z.view(1, 3).expand_as(axis)
    ref = ref - axis * (ref * axis).sum(dim=-1, keepdim=True)
    bad = ref.norm(dim=-1) < 1e-4
    if bad.any():
        world_x = torch.tensor([1.0, 0.0, 0.0], device=p0.device, dtype=p0.dtype)
        ref_alt = world_x.view(1, 3).expand_as(axis)
        ref_alt = ref_alt - axis * (ref_alt * axis).sum(dim=-1, keepdim=True)
        ref = torch.where(bad.unsqueeze(-1), ref_alt, ref)
    ref = _normalize(ref)
    side = torch.cross(axis, ref, dim=-1)
    phi_vec = _normalize_angle_pair(branch_action[:, :2])
    cphi = phi_vec[:, 0:1]
    sphi = phi_vec[:, 1:2]
    return _normalize(cphi * ref + sphi * side)


def _swivel_cost_from_q(kin: BatchedFR3Kinematics,
                        q: torch.Tensor,
                        p0: torch.Tensor,
                        branch_action: torch.Tensor) -> torch.Tensor:
    link_tfs = kin.link_transforms(q)
    shoulder = torch.tensor([0.0, 0.0, 0.333], device=q.device, dtype=q.dtype).view(1, 3)
    elbow = link_tfs[:, 4, :3, 3]
    axis = _normalize(p0 - shoulder)
    cur = elbow - shoulder
    cur = cur - axis * (cur * axis).sum(dim=-1, keepdim=True)
    cur = _normalize(cur)
    tgt = _target_swivel_dir(p0, branch_action)
    return 1.0 - (cur * tgt).sum(dim=-1).clamp(-1.0, 1.0)


def _swivel_grad_fd(kin: BatchedFR3Kinematics,
                    q: torch.Tensor,
                    p0: torch.Tensor,
                    branch_action: torch.Tensor) -> torch.Tensor:
    eps = float(cfg.BRANCH_FD_EPS)
    grads = []
    for j in range(7):
        dq = torch.zeros_like(q)
        dq[:, j] = eps
        cp = _swivel_cost_from_q(kin, (q + dq).clamp(kin.lmt_lo, kin.lmt_up),
                                 p0, branch_action)
        cm = _swivel_cost_from_q(kin, (q - dq).clamp(kin.lmt_lo, kin.lmt_up),
                                 p0, branch_action)
        grads.append((cp - cm) / (2.0 * eps))
    return torch.stack(grads, dim=-1)


def _rotvec_between(R_cur: torch.Tensor, R_tgt: torch.Tensor) -> torch.Tensor:
    R_err = R_tgt @ R_cur.transpose(-1, -2)
    trace = R_err.diagonal(dim1=-2, dim2=-1).sum(-1)
    quat = torch.zeros((R_err.shape[0], 4), device=R_err.device, dtype=R_err.dtype)
    good = trace > 0.0
    if good.any():
        s = torch.sqrt(trace[good] + 1.0).clamp_min(1e-12) * 2.0
        quat[good, 0] = 0.25 * s
        quat[good, 1] = (R_err[good, 2, 1] - R_err[good, 1, 2]) / s
        quat[good, 2] = (R_err[good, 0, 2] - R_err[good, 2, 0]) / s
        quat[good, 3] = (R_err[good, 1, 0] - R_err[good, 0, 1]) / s

    bad = ~good
    if bad.any():
        Rb = R_err[bad]
        idx = torch.argmax(torch.stack([Rb[:, 0, 0], Rb[:, 1, 1], Rb[:, 2, 2]], dim=-1),
                           dim=-1)
        qb = torch.zeros((Rb.shape[0], 4), device=R_err.device, dtype=R_err.dtype)
        m0 = idx == 0
        if m0.any():
            s = torch.sqrt(1.0 + Rb[m0, 0, 0] - Rb[m0, 1, 1] - Rb[m0, 2, 2]).clamp_min(1e-12) * 2.0
            qb[m0, 0] = (Rb[m0, 2, 1] - Rb[m0, 1, 2]) / s
            qb[m0, 1] = 0.25 * s
            qb[m0, 2] = (Rb[m0, 0, 1] + Rb[m0, 1, 0]) / s
            qb[m0, 3] = (Rb[m0, 0, 2] + Rb[m0, 2, 0]) / s
        m1 = idx == 1
        if m1.any():
            s = torch.sqrt(1.0 + Rb[m1, 1, 1] - Rb[m1, 0, 0] - Rb[m1, 2, 2]).clamp_min(1e-12) * 2.0
            qb[m1, 0] = (Rb[m1, 0, 2] - Rb[m1, 2, 0]) / s
            qb[m1, 1] = (Rb[m1, 0, 1] + Rb[m1, 1, 0]) / s
            qb[m1, 2] = 0.25 * s
            qb[m1, 3] = (Rb[m1, 1, 2] + Rb[m1, 2, 1]) / s
        m2 = idx == 2
        if m2.any():
            s = torch.sqrt(1.0 + Rb[m2, 2, 2] - Rb[m2, 0, 0] - Rb[m2, 1, 1]).clamp_min(1e-12) * 2.0
            qb[m2, 0] = (Rb[m2, 1, 0] - Rb[m2, 0, 1]) / s
            qb[m2, 1] = (Rb[m2, 0, 2] + Rb[m2, 2, 0]) / s
            qb[m2, 2] = (Rb[m2, 1, 2] + Rb[m2, 2, 1]) / s
            qb[m2, 3] = 0.25 * s
        quat[bad] = qb

    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    quat = torch.where(quat[:, :1] < 0.0, -quat, quat)
    v = quat[:, 1:]
    v_norm = v.norm(dim=-1)
    theta = 2.0 * torch.atan2(v_norm, quat[:, 0].clamp_min(1e-12))
    axis = torch.where(v_norm.unsqueeze(-1) > 1e-8,
                       v / v_norm.clamp_min(1e-12).unsqueeze(-1),
                       torch.zeros_like(v))
    return axis * theta.unsqueeze(-1)


def _z_axis_rotvec(R_cur: torch.Tensor, R_tgt: torch.Tensor) -> torch.Tensor:
    return _z_axis_error_from_rotmats(R_cur, R_tgt)[0]


def _dls_pinv(J: torch.Tensor, damping: float) -> torch.Tensor:
    b = J.shape[0]
    eye6 = torch.eye(6, device=J.device, dtype=J.dtype).expand(b, 6, 6)
    A = J @ J.transpose(-1, -2) + (float(damping) ** 2) * eye6
    return J.transpose(-1, -2) @ torch.linalg.inv(A)


def _directional_manipulability(J_pos: torch.Tensor,
                                direction: torch.Tensor,
                                damping: float) -> torch.Tensor:
    b = J_pos.shape[0]
    eye3 = torch.eye(3, device=J_pos.device, dtype=J_pos.dtype).expand(b, 3, 3)
    metric = J_pos @ J_pos.transpose(-1, -2) + (float(damping) ** 2) * eye3
    d_col = _normalize(direction).unsqueeze(-1)
    inv_quad = (d_col.transpose(-1, -2) @ torch.linalg.inv(metric) @ d_col).squeeze(-1).squeeze(-1)
    return inv_quad.clamp_min(1e-12).pow(-0.5)




def _batched_ik_project(kin: BatchedFR3Kinematics,
                        q_seed: torch.Tensor,
                        p0: torch.Tensor,
                        R_tgt: torch.Tensor,
                        branch_action: torch.Tensor | None = None,
                        preserve_seed: bool = False):
    q = q_seed.clamp(kin.lmt_lo, kin.lmt_up)
    active = torch.ones((q.shape[0],), device=q.device, dtype=torch.bool)
    converged = torch.zeros_like(active)
    fail_reason = np.array(['init_ik_fail'] * q.shape[0], dtype=object)

    for _ in range(cfg.BATCHED_IK_MAX_ITERS):
        p_tcp, R_tcp, J, _ = kin.tcp_fk_jac(q)
        delta_p = p0 - p_tcp
        if branch_action is None and cfg.INIT_IK_ORIENT_MODE == "z_axis":
            delta_theta = _z_axis_rotvec(R_tcp, R_tgt)
        else:
            delta_theta = _rotvec_between(R_tcp, R_tgt)
        pos_err = delta_p.norm(dim=-1)
        rot_err = delta_theta.norm(dim=-1)
        in_limits = ((q >= kin.lmt_lo - 1e-5)
                     & (q <= kin.lmt_up + 1e-5)).all(dim=-1)
        now_conv = ((pos_err <= float(cfg.EPS_POS_INIT))
                    & (rot_err <= float(cfg.THETA_MAX))
                    & in_limits)
        newly = active & now_conv
        if newly.any():
            converged |= newly
            active &= ~newly
        if not active.any():
            break

        # Trust-region-style step bounding (matches the serial solver).
        pos_scale = torch.where(pos_err > 0.1, 0.1 / pos_err.clamp_min(1e-12),
                                torch.ones_like(pos_err))
        rot_scale = torch.where(rot_err > 0.3, 0.3 / rot_err.clamp_min(1e-12),
                                torch.ones_like(rot_err))
        delta_p = delta_p * pos_scale.unsqueeze(-1)
        delta_theta = delta_theta * rot_scale.unsqueeze(-1)
        delta_x = torch.cat([delta_p, delta_theta], dim=-1)

        Jpinv = _dls_pinv(J, cfg.BATCHED_IK_DAMPING)
        delta_q = (Jpinv @ delta_x.unsqueeze(-1)).squeeze(-1)
        if not preserve_seed:
            # Nullspace pull toward q_mid (or swivel target). Disabled when the
            # caller wants the seed's null-direction offsets preserved, e.g.
            # for 2D landscape slices in the task nullspace.
            N = torch.eye(7, device=q.device, dtype=q.dtype).expand(q.shape[0], 7, 7)
            N = N - Jpinv @ J
            if branch_action is None:
                delta_q_secondary = 0.2 * (kin.q_mid - q)
            else:
                grad_cost = _swivel_grad_fd(kin, q, p0, branch_action)
                branch_gain = torch.where(
                    (pos_err < 0.05) & (rot_err < 0.30),
                    torch.full_like(pos_err, float(cfg.BRANCH_SWIVEL_GAIN)),
                    torch.zeros_like(pos_err),
                )
                delta_q_secondary = -branch_gain.unsqueeze(-1) * grad_cost
            delta_q = delta_q + (N @ delta_q_secondary.unsqueeze(-1)).squeeze(-1)
        q_next = (q + delta_q).clamp(kin.lmt_lo, kin.lmt_up)
        q = torch.where(active.unsqueeze(-1), q_next, q)

    if active.any():
        p_tcp, R_tcp, _, _ = kin.tcp_fk_jac(q)
        pos_err = (p0 - p_tcp).norm(dim=-1)
        if branch_action is None and cfg.INIT_IK_ORIENT_MODE == "z_axis":
            rot_err = _z_axis_rotvec(R_tcp, R_tgt).norm(dim=-1)
        else:
            rot_err = _rotvec_between(R_tcp, R_tgt).norm(dim=-1)
        in_limits = ((q >= kin.lmt_lo - 1e-5)
                     & (q <= kin.lmt_up + 1e-5)).all(dim=-1)
        now_conv = ((pos_err <= float(cfg.EPS_POS_INIT))
                    & (rot_err <= float(cfg.THETA_MAX))
                    & in_limits)
        converged |= active & now_conv

    fail_reason[converged.detach().cpu().numpy()] = 'ik_ok'
    return q.clamp(kin.lmt_lo, kin.lmt_up), converged, fail_reason


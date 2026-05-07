"""Torch batched rollout for the FR3 farsighted-seed task.

This module mirrors ``rollout.py`` but evaluates a full batch together. It is
intended for training data collection; visualization still uses the serial
rollout so it can keep full trajectories and WRS scene state.
"""
from __future__ import annotations

import numpy as np
import torch
import importlib.util
from pathlib import Path

import Yuan.RL.config as cfg
from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics


def _load_fr3_sphere_collision_cls():
    """Load the FR3 sphere checker without importing the top-level one package."""
    root = Path(__file__).resolve().parents[2]
    path = root / 'one/robots/manipulators/franka/fr3/sphere_collision.py'
    spec = importlib.util.spec_from_file_location('_fr3_sphere_collision', path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.FR3SphereCollision


def _device_from_cfg() -> torch.device:
    if cfg.BATCHED_ROLLOUT_DEVICE == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(cfg.BATCHED_ROLLOUT_DEVICE)


def _normalize(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _normalize_angle_pair(pair: torch.Tensor) -> torch.Tensor:
    norm = pair.norm(dim=-1, keepdim=True)
    default = torch.zeros_like(pair)
    default[:, 0] = 1.0
    return torch.where(norm > 1e-6, pair / norm.clamp_min(1e-12), default)


def build_target_rotmat_batch(d: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """TCP_z = -n (tool faces INTO the surface, opposite to surface
    outward normal). TCP_x = d projected onto z-perp. TCP_y = z x x."""
    z = _normalize(-n)                                # ← flip: TCP_z = -n
    x = d - z * (d * z).sum(dim=-1, keepdim=True)
    x = _normalize(x)
    y = torch.cross(z, x, dim=-1)
    return torch.stack([x, y, z], dim=-1)


def build_branch_rotmat_batch(d: torch.Tensor, n: torch.Tensor,
                              branch_action: torch.Tensor) -> torch.Tensor:
    """Build full TCP rotation from path frame plus tool-roll descriptor psi."""
    R0 = build_target_rotmat_batch(d, n)
    psi_vec = _normalize_angle_pair(branch_action[:, 2:4])
    cpsi = psi_vec[:, 0:1]
    spsi = psi_vec[:, 1:2]
    x0 = R0[:, :, 0]
    y0 = R0[:, :, 1]
    z = R0[:, :, 2]
    x = _normalize(cpsi * x0 + spsi * y0)
    y = torch.cross(z, x, dim=-1)
    return torch.stack([x, y, z], dim=-1)


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


def _branch_seed_bank(kin: BatchedFR3Kinematics) -> torch.Tensor:
    """Deterministic set of FR3 postures covering common IK branches.
    Expanded to 16 for full-sphere n distributions (some normals only have
    IK solutions reachable from "wrist-flipped" or "back-pose" seeds)."""
    seeds = torch.tensor([
        # canonical (FR3 home, elbow up/down ±)
        [0.0, -0.785398163, 0.0, -2.35619449, 0.0, 1.57079632679, 0.785398163397],
        [0.0, -0.4, 0.0, -2.2, 0.0, 1.8, 0.0],
        [0.0,  0.4, 0.0, -2.2, 0.0, 1.8, 0.0],
        # shoulder rotated ±, with various elbow
        [ 1.0, 0.8,  1.0, -2.1,  1.2, 1.0,  0.5],
        [-1.0, 0.8, -1.0, -2.1, -1.2, 1.0, -0.5],
        [ 1.0, 1.2, -1.0, -2.2,  1.5, 1.0,  0.5],
        [-1.0, 1.2,  1.0, -2.2, -1.5, 1.0, -0.5],
        [ 0.0, 1.2,  1.2, -2.0,  1.2, 1.2,  0.0],
        [ 0.0, 1.2, -1.2, -2.0, -1.2, 1.2,  0.0],
        # wrist-flipped (J6 negative -> tool pointing UP, useful for n in lower hemisphere)
        [ 0.0, -0.4,  0.0, -2.2,  0.0, -1.5,  0.0],
        [ 0.0,  0.4,  0.0, -2.2,  0.0, -1.5,  0.0],
        [ 1.5, -0.4,  0.0, -2.2,  0.0, -1.5,  0.0],
        [-1.5, -0.4,  0.0, -2.2,  0.0, -1.5,  0.0],
        # extreme back / side reaches
        [ 2.5,  0.5, -2.0, -1.0,  0.5, 1.5,  0.0],
        [-2.5,  0.5,  2.0, -1.0, -0.5, 1.5,  0.0],
        [ 0.0, -1.0,  0.0, -1.5,  0.0,  2.5,  0.0],
    ], device=kin.device, dtype=torch.float32)
    num = max(1, min(int(cfg.BRANCH_IK_NUM_STARTS), seeds.shape[0]))
    return seeds[:num].clamp(kin.lmt_lo, kin.lmt_up)


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


def _batched_nullspace_objective_grad(kin: BatchedFR3Kinematics,
                                      q: torch.Tensor,
                                      direction: torch.Tensor,
                                      R_tgt: torch.Tensor,
                                      q_ref: torch.Tensor,
                                      gains: dict | None = None) -> torch.Tensor:
    """Nullspace gradient. `gains` is an optional dict that overrides cfg's
    NULL_* scalar gains with PER-ROW (B,) tensors. Used by sequential
    rollouts that switch nullspace preset mid-trajectory. Keys recognized:
        'manip', 'jlm', 'angle_boundary', 'angle_attract', 'k_null'
    Each value can be a python float (broadcast) or a (B,) tensor.
    """
    g = gains or {}
    def _to_t(name, default):
        v = g.get(name, default)
        if isinstance(v, torch.Tensor):
            return v.to(q.device, dtype=q.dtype)
        return torch.full((q.shape[0],), float(v), device=q.device, dtype=q.dtype)
    g_manip   = _to_t('manip',          float(cfg.NULL_MANIP_GAIN))
    g_jlm     = _to_t('jlm',            float(cfg.NULL_JOINT_LIMIT_GAIN))
    g_a_bnd   = _to_t('angle_boundary', float(cfg.NULL_ANGLE_GAIN))
    g_a_att   = _to_t('angle_attract',  float(cfg.NULL_ANGLE_ATTRACT_GAIN))
    g_knull   = _to_t('k_null',         float(cfg.K_NULL))

    q_eval = q.detach().clone().requires_grad_(True)
    _, R_tcp, J, _ = kin.tcp_fk_jac(q_eval)
    z_cur = R_tcp[:, :, 2]
    z_tgt = R_tgt[:, :, 2]
    cos_theta = (z_cur * z_tgt).sum(dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    J_pos = J[:, :3, :]

    q_dot_null = torch.zeros_like(q)
    if cfg.NULL_USE_MANIPULABILITY and float(g_manip.abs().max()) > 0.0:
        mu = _directional_manipulability(J_pos, direction, cfg.NULL_MANIP_DAMPING)
        grad_mu = torch.autograd.grad(mu.sum(), q_eval, retain_graph=True,
                                      create_graph=False, allow_unused=True)[0]
        if grad_mu is not None:
            q_dot_null = q_dot_null + g_manip.unsqueeze(-1) * grad_mu.detach()

    lower = kin.lmt_lo.view(1, 7)
    upper = kin.lmt_up.view(1, 7)
    center = 0.5 * (lower + upper)
    span = (upper - lower).clamp_min(1e-6)
    q_dot_joint = -g_jlm.unsqueeze(-1) * (q - center) / span
    q_dot_null = q_dot_null + q_dot_joint

    grad_cos = torch.autograd.grad(cos_theta.sum(), q_eval, retain_graph=False,
                                   create_graph=False, allow_unused=True)[0]
    if grad_cos is not None:
        theta = torch.acos(cos_theta)
        theta_max = torch.as_tensor(float(cfg.THETA_MAX), device=q.device, dtype=q.dtype)
        margin_den = max(float(cfg.NULL_ANGLE_MARGIN), 1e-6)
        boundary_gate = ((theta - theta_max) / margin_den).clamp(0.0, 1.0).unsqueeze(-1)
        interior_gate = theta.unsqueeze(-1)
        angle_gain = (g_a_bnd.unsqueeze(-1) * boundary_gate
                      + g_a_att.unsqueeze(-1) * interior_gate)
        q_dot_null = q_dot_null + angle_gain * grad_cos.detach()

    q_dot_null = q_dot_null + g_knull.unsqueeze(-1) * (q_ref - q)
    return q_dot_null


def _batched_ik_project(kin: BatchedFR3Kinematics,
                        q_seed: torch.Tensor,
                        p0: torch.Tensor,
                        R_tgt: torch.Tensor,
                        branch_action: torch.Tensor | None = None):
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

        # Match the serial solver's trust-region style.
        pos_scale = torch.where(pos_err > 0.1, 0.1 / pos_err.clamp_min(1e-12),
                                torch.ones_like(pos_err))
        rot_scale = torch.where(rot_err > 0.3, 0.3 / rot_err.clamp_min(1e-12),
                                torch.ones_like(rot_err))
        delta_p = delta_p * pos_scale.unsqueeze(-1)
        delta_theta = delta_theta * rot_scale.unsqueeze(-1)
        delta_x = torch.cat([delta_p, delta_theta], dim=-1)

        Jpinv = _dls_pinv(J, cfg.BATCHED_IK_DAMPING)
        delta_q = (Jpinv @ delta_x.unsqueeze(-1)).squeeze(-1)
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


def _joint_margin_norm(kin: BatchedFR3Kinematics,
                       q: torch.Tensor) -> torch.Tensor:
    """Per-row min normalized distance to nearest joint limit, in [0, 0.5].
    Higher = farther from any limit (safer for the next control step)."""
    span = (kin.lmt_up - kin.lmt_lo).clamp_min(1e-6)
    lo_d = (q - kin.lmt_lo) / span
    up_d = (kin.lmt_up - q) / span
    return torch.minimum(lo_d, up_d).min(dim=-1).values


def branch_project_multistart(kin: BatchedFR3Kinematics,
                              p0: torch.Tensor,
                              R_tgt: torch.Tensor,
                              branch_action: torch.Tensor):
    """Project branch descriptors with deterministic multi-start IK.

    Score among the num_starts converged candidates per task:
        pos_err + rot_err + IK_SWIVEL_W * swivel_cost - IK_MARGIN_W * margin
    The margin term steers selection away from near-joint-limit q so that
    step-0 joint_limit terminations during rollout are reduced.
    """
    batch_size = branch_action.shape[0]
    seed_bank = _branch_seed_bank(kin)
    num_starts = seed_bank.shape[0]
    q_seed = seed_bank.view(1, num_starts, 7).expand(batch_size, num_starts, 7)
    q_seed = q_seed.reshape(batch_size * num_starts, 7).clone()
    p_rep = p0[:, None, :].expand(batch_size, num_starts, 3).reshape(-1, 3)
    R_rep = R_tgt[:, None, :, :].expand(batch_size, num_starts, 3, 3).reshape(-1, 3, 3)
    a_rep = branch_action[:, None, :].expand(batch_size, num_starts, 4).reshape(-1, 4)

    q_all, ok_all, _ = _batched_ik_project(kin, q_seed, p_rep, R_rep,
                                           branch_action=a_rep)
    p_tcp, R_tcp, _, _ = kin.tcp_fk_jac(q_all)
    pos_err = (p_rep - p_tcp).norm(dim=-1)
    rot_err = _rotvec_between(R_tcp, R_rep).norm(dim=-1)
    swivel_cost = _swivel_cost_from_q(kin, q_all, p_rep, a_rep)
    margin = _joint_margin_norm(kin, q_all)
    score = (pos_err + rot_err
             + float(cfg.IK_SWIVEL_W) * swivel_cost
             - float(cfg.IK_MARGIN_W) * margin)
    score = torch.where(ok_all, score, torch.full_like(score, float('inf')))
    score = score.view(batch_size, num_starts)
    best_idx = score.argmin(dim=-1)
    ik_ok = torch.isfinite(score.gather(1, best_idx.view(-1, 1)).squeeze(1))
    q_all = q_all.view(batch_size, num_starts, 7)
    q_best = q_all[torch.arange(batch_size, device=kin.device), best_idx]
    reasons = np.array(['init_ik_fail'] * batch_size, dtype=object)
    reasons[ik_ok.detach().cpu().numpy()] = 'ik_ok'
    return q_best, ik_ok, reasons


def batched_rollout(q_seed_np: np.ndarray,
                    c_np: np.ndarray,
                    v_path_np: np.ndarray,
                    eps_p_np: np.ndarray,
                    T_np: np.ndarray,
                    action_mode: str | None = None) -> dict:
    """Evaluate a batch of seed actions.

    Args:
        q_seed_np: ``(B, 7)`` active joint seeds.
        c_np: ``(B, 9)`` contexts ``[p0, d, n]``.
        v_path_np: ``(B,)`` path speeds.
        eps_p_np: ``(B,)`` position tolerances.
        T_np: ``(B,)`` rollout horizons.
    """
    device = _device_from_cfg()
    kin = BatchedFR3Kinematics(device=device)
    if cfg.USE_COLLISION_CHECK and cfg.BATCHED_COLLISION_CHECK:
        sphere_cls = _load_fr3_sphere_collision_cls()
        sphere_cc = sphere_cls(device=device,
                               margin=cfg.BATCHED_COLLISION_MARGIN)
    else:
        sphere_cc = None
    action = torch.as_tensor(q_seed_np, device=device, dtype=torch.float32)
    c = torch.as_tensor(c_np, device=device, dtype=torch.float32)
    v_path = torch.as_tensor(v_path_np, device=device, dtype=torch.float32)
    eps_p = torch.as_tensor(eps_p_np, device=device, dtype=torch.float32)
    T = torch.as_tensor(T_np, device=device, dtype=torch.long)

    p0 = c[:, :3]
    d = c[:, 3:6]
    n = c[:, 6:9]
    if action_mode is None:
        action_mode = cfg.ACTION_MODE

    if action_mode == "branch_descriptor":
        branch_action = action
        R_tgt = build_branch_rotmat_batch(d, n, branch_action)
        q_seed = _branch_seed_bank(kin)[0].view(1, 7).expand(action.shape[0], 7).clone()
    else:
        branch_action = None
        R_tgt = build_target_rotmat_batch(d, n)
        q_seed = action.clamp(kin.lmt_lo, kin.lmt_up)
    seed_p, seed_R, _, _ = kin.tcp_fk_jac(q_seed)
    seed_pos_err = (p0 - seed_p).norm(dim=-1)
    seed_orient_err = _z_axis_rotvec(seed_R, R_tgt).norm(dim=-1)

    if action_mode == "branch_descriptor":
        q, ik_ok, reasons = branch_project_multistart(kin, p0, R_tgt, branch_action)
    else:
        q, ik_ok, reasons = _batched_ik_project(kin, q_seed, p0, R_tgt,
                                                branch_action=branch_action)
    lengths = torch.zeros((q.shape[0],), device=device, dtype=torch.long)
    alive = ik_ok.clone()
    q_ref = q.clone()
    max_T = int(T.max().item()) if T.numel() else 0
    last_pos_err = torch.zeros_like(v_path)
    last_orient_err = torch.zeros_like(v_path)

    eye7 = torch.eye(7, device=device, dtype=torch.float32).expand(q.shape[0], 7, 7)
    theta_max = torch.as_tensor(float(cfg.THETA_MAX), device=device,
                                dtype=torch.float32)
    dt = float(cfg.DT)

    for step in range(1, max_T + 1):
        in_horizon = step <= T
        step_alive = alive & in_horizon
        if not step_alive.any():
            break

        p_ref = p0 + (step * dt) * v_path.unsqueeze(-1) * d
        p_dot_ff = v_path.unsqueeze(-1) * d
        p_tcp, R_tcp, J, _ = kin.tcp_fk_jac(q)
        z_tgt = R_tgt[:, :, 2]
        omega_err, theta = _z_axis_error_from_rotmats(R_tcp, R_tgt)
        x_dot = torch.cat([
            p_dot_ff + float(cfg.KP_LIN) * (p_ref - p_tcp),
            float(cfg.KOMEGA) * omega_err,
        ], dim=-1)

        Jpinv = _dls_pinv(J, float(cfg.DLS_LAMBDA))
        q_dot = (Jpinv @ x_dot.unsqueeze(-1)).squeeze(-1)
        N = eye7 - Jpinv @ J
        q_dot_secondary = _batched_nullspace_objective_grad(
            kin, q, d, R_tgt, q_ref)
        q_dot = q_dot + (N @ q_dot_secondary.unsqueeze(-1)).squeeze(-1)
        q_dot = q_dot.clamp(-kin.qdot_max, kin.qdot_max)
        q_new_raw = q + q_dot * dt
        joint_limit_hit = ((q_new_raw < kin.lmt_lo - 1e-6)
                           | (q_new_raw > kin.lmt_up + 1e-6)).any(dim=-1)
        q_new = q_new_raw.clamp(kin.lmt_lo, kin.lmt_up)

        p_new, R_new, _, _ = kin.tcp_fk_jac(q_new)
        pos_err = (p_ref - p_new).norm(dim=-1)
        orient_err = torch.acos(
            (R_new[:, :, 2] * z_tgt).sum(dim=-1).clamp(-1.0, 1.0))
        if sphere_cc is None:
            self_collision = torch.zeros_like(step_alive)
        else:
            self_collision = sphere_cc.is_collided(kin.link_transforms(q_new))
        last_pos_err = torch.where(step_alive, pos_err, last_pos_err)
        last_orient_err = torch.where(step_alive, orient_err, last_orient_err)

        fail_pos = step_alive & (pos_err > eps_p)
        fail_ori = step_alive & (orient_err > theta_max)
        fail_lmt = step_alive & joint_limit_hit
        fail_col = step_alive & self_collision
        ok = step_alive & ~(fail_pos | fail_ori | fail_lmt | fail_col)

        lengths = torch.where(ok, torch.full_like(lengths, step), lengths)
        q = torch.where(ok.unsqueeze(-1), q_new, q)

        fail_pos_np = fail_pos.detach().cpu().numpy()
        fail_ori_np = fail_ori.detach().cpu().numpy()
        fail_lmt_np = fail_lmt.detach().cpu().numpy()
        reasons[fail_pos_np] = 'pos_err_exceeded'
        reasons[fail_ori_np] = 'orient_err_exceeded'
        reasons[fail_lmt_np] = 'joint_limit'
        reasons[fail_col.detach().cpu().numpy()] = 'self_collision'
        alive = alive & in_horizon & ~(fail_pos | fail_ori | fail_lmt | fail_col)

    complete = (ik_ok & (lengths >= T)).detach().cpu().numpy()
    reasons[complete] = 'max_steps'
    return {
        'lengths': lengths.detach().cpu().numpy().astype(np.int32),
        'reasons': reasons.tolist(),
        'pos_err': last_pos_err.detach().cpu().numpy(),
        'orient_err': last_orient_err.detach().cpu().numpy(),
        'seed_pos_err': seed_pos_err.detach().cpu().numpy(),
        'seed_orient_err': seed_orient_err.detach().cpu().numpy(),
    }


def batched_rollout_segment(q_init: torch.Tensor,
                            R_tgt: torch.Tensor,
                            branch_action: torch.Tensor,
                            p0: torch.Tensor,
                            d_dir: torch.Tensor,
                            v_path: torch.Tensor,
                            eps_p: torch.Tensor,
                            T_total: torch.Tensor,
                            start_step: int,
                            end_step: int,
                            preset_gains: dict | None = None,
                            alive_mask: torch.Tensor | None = None,
                            sphere_cc=None,
                            kin: BatchedFR3Kinematics | None = None,
                            is_phantom: bool = False,
                            q_ref: torch.Tensor | None = None) -> dict:
    """Run controller from step `start_step` (exclusive) to `end_step` (inclusive)
    on a (B, 7) joint state. Same dynamics as batched_rollout but:
      - takes pre-built R_tgt, p0, d_dir (caller computes them once)
      - takes per-batch nullspace gain overrides via `preset_gains` dict
      - is_phantom=True skips the nullspace term entirely
      - alive_mask lets caller carry over which rows are still active

    Returns dict with:
      'q_final'    (B, 7)  — joint state at end_step (or last alive step)
      'lengths'    (B,)    — last step the row was still alive (relative to 0,
                            absolute index up to end_step)
      'alive_out'  (B,)    — bool, still alive after end_step
      'last_pos_err','last_orient_err' (B,)
    """
    device = q_init.device
    B = q_init.shape[0]
    if kin is None:
        kin = BatchedFR3Kinematics(device=device)
    eye7 = torch.eye(7, device=device, dtype=torch.float32).expand(B, 7, 7)
    dt = float(cfg.DT)
    theta_max = float(cfg.THETA_MAX)
    z_tgt = R_tgt[:, :, 2]

    if alive_mask is None:
        alive = torch.ones((B,), device=device, dtype=torch.bool)
    else:
        alive = alive_mask.clone()
    q = q_init.clone()
    if q_ref is None:
        q_ref = q.clone()
    else:
        q_ref = q_ref.clone()
    lengths = torch.full((B,), float(start_step),
                         device=device, dtype=torch.float32).long()
    last_pos_err = torch.zeros_like(v_path)
    last_orient_err = torch.zeros_like(v_path)
    in_horizon_global = torch.ones_like(alive)

    for step in range(start_step + 1, end_step + 1):
        in_horizon = step <= T_total
        step_alive = alive & in_horizon
        if not step_alive.any():
            break

        p_ref = p0 + (step * dt) * v_path.unsqueeze(-1) * d_dir
        p_dot_ff = v_path.unsqueeze(-1) * d_dir
        p_tcp, R_tcp, J, _ = kin.tcp_fk_jac(q)
        omega_err, _ = _z_axis_error_from_rotmats(R_tcp, R_tgt)
        x_dot = torch.cat([
            p_dot_ff + float(cfg.KP_LIN) * (p_ref - p_tcp),
            float(cfg.KOMEGA) * omega_err,
        ], dim=-1)
        Jpinv = _dls_pinv(J, float(cfg.DLS_LAMBDA))
        q_dot = (Jpinv @ x_dot.unsqueeze(-1)).squeeze(-1)
        if not is_phantom:
            N = eye7 - Jpinv @ J
            q_dot_secondary = _batched_nullspace_objective_grad(
                kin, q, d_dir, R_tgt, q_ref, gains=preset_gains)
            q_dot = q_dot + (N @ q_dot_secondary.unsqueeze(-1)).squeeze(-1)
        q_dot = q_dot.clamp(-kin.qdot_max, kin.qdot_max)
        q_new_raw = q + q_dot * dt
        joint_limit_hit = ((q_new_raw < kin.lmt_lo - 1e-6)
                           | (q_new_raw > kin.lmt_up + 1e-6)).any(dim=-1)
        q_new = q_new_raw.clamp(kin.lmt_lo, kin.lmt_up)
        p_new, R_new, _, _ = kin.tcp_fk_jac(q_new)
        pos_err = (p_ref - p_new).norm(dim=-1)
        orient_err = torch.acos(
            (R_new[:, :, 2] * z_tgt).sum(dim=-1).clamp(-1.0, 1.0))
        if sphere_cc is None:
            self_collision = torch.zeros_like(step_alive)
        else:
            self_collision = sphere_cc.is_collided(kin.link_transforms(q_new))
        last_pos_err = torch.where(step_alive, pos_err, last_pos_err)
        last_orient_err = torch.where(step_alive, orient_err, last_orient_err)
        fail_pos = step_alive & (pos_err > eps_p)
        fail_ori = step_alive & (orient_err > theta_max)
        fail_lmt = step_alive & joint_limit_hit
        fail_col = step_alive & self_collision
        ok = step_alive & ~(fail_pos | fail_ori | fail_lmt | fail_col)
        lengths = torch.where(ok, torch.full_like(lengths, step), lengths)
        q = torch.where(ok.unsqueeze(-1), q_new, q)
        alive = alive & in_horizon & ~(fail_pos | fail_ori | fail_lmt | fail_col)
        in_horizon_global = in_horizon

    return {
        'q_final': q,
        'lengths': lengths,
        'alive_out': alive & in_horizon_global,
        'last_pos_err': last_pos_err,
        'last_orient_err': last_orient_err,
    }


def batched_rollout_contact(q_seed_np: np.ndarray,
                            c_np: np.ndarray,
                            v_path_np: np.ndarray,
                            eps_p_np: np.ndarray,
                            T_np: np.ndarray,
                            action_mode: str | None = None,
                            k_n: float | None = None,
                            pen_target: float | None = None,
                            pen_min: float | None = None,
                            pen_max: float | None = None,
                            grace_steps: int | None = None,
                            use_dynamics: bool | None = None,
                            tip_mass: float | None = None,
                            grip_k: float | None = None,
                            grip_c: float | None = None,
                            n_substeps: int | None = None) -> dict:
    """Contact-mode rollout: same dynamics as batched_rollout, plus a linear
    spring contact model along the surface normal.

    Surface (per task): plane through p0 with outward normal n_out = c[6:9].
    Pen tip target is OFFSET into the surface by `pen_target` so equilibrium
    contact force F = k_n * pen_target.

    NEW failure modes:
        force_high : penetration > pen_max  (any step after grace)
        force_low  : penetration < pen_min  (any step after grace, lost contact)

    Existing failure modes (joint_limit, pos_err, orient_err, self_collision)
    still apply, but pos_err is checked against p_ref shifted into the surface
    (so loose normal tracking + spring force handle the contact direction).
    """
    if k_n        is None: k_n        = float(cfg.CONTACT_K_N)
    if pen_target is None: pen_target = float(cfg.CONTACT_PENETRATION_TARGET)
    if pen_min    is None: pen_min    = float(cfg.CONTACT_PEN_MIN)
    if pen_max    is None: pen_max    = float(cfg.CONTACT_PEN_MAX)
    if grace_steps is None: grace_steps = int(cfg.CONTACT_GRACE_STEPS)
    if use_dynamics is None:
        use_dynamics = bool(getattr(cfg, "CONTACT_USE_DYNAMICS", False))
    if tip_mass   is None: tip_mass   = float(getattr(cfg, "CONTACT_TIP_MASS", 0.5))
    if grip_k     is None: grip_k     = float(getattr(cfg, "CONTACT_GRIP_K", 20000.0))
    if grip_c     is None: grip_c     = float(getattr(cfg, "CONTACT_GRIP_C", 20.0))
    if n_substeps is None: n_substeps = int(getattr(cfg, "CONTACT_N_SUBSTEPS", 20))

    device = _device_from_cfg()
    kin = BatchedFR3Kinematics(device=device)
    if cfg.USE_COLLISION_CHECK and cfg.BATCHED_COLLISION_CHECK:
        sphere_cls = _load_fr3_sphere_collision_cls()
        sphere_cc = sphere_cls(device=device,
                               margin=cfg.BATCHED_COLLISION_MARGIN)
    else:
        sphere_cc = None
    action = torch.as_tensor(q_seed_np, device=device, dtype=torch.float32)
    c = torch.as_tensor(c_np, device=device, dtype=torch.float32)
    v_path = torch.as_tensor(v_path_np, device=device, dtype=torch.float32)
    eps_p = torch.as_tensor(eps_p_np, device=device, dtype=torch.float32)
    T = torch.as_tensor(T_np, device=device, dtype=torch.long)

    p0 = c[:, :3]
    d_dir = c[:, 3:6]
    n_out = c[:, 6:9]                                      # outward normal
    if action_mode is None:
        action_mode = cfg.ACTION_MODE
    if action_mode == "branch_descriptor":
        branch_action = action
        R_tgt = build_branch_rotmat_batch(d_dir, n_out, branch_action)
    else:
        branch_action = None
        R_tgt = build_target_rotmat_batch(d_dir, n_out)

    if action_mode == "branch_descriptor":
        q, ik_ok, reasons = branch_project_multistart(kin, p0, R_tgt, branch_action)
    else:
        q_seed = action.clamp(kin.lmt_lo, kin.lmt_up)
        q, ik_ok, reasons = _batched_ik_project(kin, q_seed, p0, R_tgt,
                                                branch_action=branch_action)

    lengths = torch.zeros((q.shape[0],), device=device, dtype=torch.long)
    alive = ik_ok.clone()
    q_ref = q.clone()
    max_T = int(T.max().item()) if T.numel() else 0
    last_force = torch.zeros_like(v_path)

    eye7 = torch.eye(7, device=device, dtype=torch.float32).expand(q.shape[0], 7, 7)
    dt = float(cfg.DT)
    theta_max = float(cfg.THETA_MAX)
    z_tgt = R_tgt[:, :, 2]

    # path target shifted into surface so equilibrium contact = F_target
    # In v2 (use_dynamics), grip-spring/contact split shifts equilibrium:
    #   z_eq = K_grip / (K_grip + K_n) · z_kin
    # so commanded z_kin must be (K_grip+K_n)/K_grip · pen_target to land
    # at F = K_n · pen_target in steady state.
    if use_dynamics:
        z_kin_target = pen_target * (grip_k + k_n) / max(grip_k, 1e-6)
    else:
        z_kin_target = pen_target
    n_offset = z_kin_target * n_out                        # outward-normal × pen
    # p_ref(step) = p0 + (step·dt·v)·d  -  z_kin_target · n_out

    # ---- v2 dynamics state per task ----
    if use_dynamics:
        # tip starts at gripper position projected on surface normal
        p0_init, _, _, _ = kin.tcp_fk_jac(q)
        z_kin_init = (-((p0_init - p0) * n_out).sum(dim=-1))   # can be < 0
        z_dyn = z_kin_init.clamp(min=0.0).clone()              # start above surface = 0
        z_dot = torch.zeros_like(z_dyn)
        dt_sub = dt / max(n_substeps, 1)
        m_inv = 1.0 / max(tip_mass, 1e-6)
    else:
        z_dyn = torch.zeros(q.shape[0], device=device, dtype=torch.float32)
        z_dot = torch.zeros_like(z_dyn)

    for step in range(1, max_T + 1):
        in_horizon = step <= T
        step_alive = alive & in_horizon
        if not step_alive.any():
            break

        p_ref = p0 + (step * dt) * v_path.unsqueeze(-1) * d_dir - n_offset
        p_dot_ff = v_path.unsqueeze(-1) * d_dir
        p_tcp, R_tcp, J, _ = kin.tcp_fk_jac(q)
        omega_err, _ = _z_axis_error_from_rotmats(R_tcp, R_tgt)
        x_dot = torch.cat([
            p_dot_ff + float(cfg.KP_LIN) * (p_ref - p_tcp),
            float(cfg.KOMEGA) * omega_err,
        ], dim=-1)
        Jpinv = _dls_pinv(J, float(cfg.DLS_LAMBDA))
        q_dot = (Jpinv @ x_dot.unsqueeze(-1)).squeeze(-1)
        N = eye7 - Jpinv @ J
        q_dot_secondary = _batched_nullspace_objective_grad(
            kin, q, d_dir, R_tgt, q_ref)
        q_dot = q_dot + (N @ q_dot_secondary.unsqueeze(-1)).squeeze(-1)
        q_dot = q_dot.clamp(-kin.qdot_max, kin.qdot_max)
        q_new_raw = q + q_dot * dt
        joint_limit_hit = ((q_new_raw < kin.lmt_lo - 1e-6)
                           | (q_new_raw > kin.lmt_up + 1e-6)).any(dim=-1)
        q_new = q_new_raw.clamp(kin.lmt_lo, kin.lmt_up)

        p_new, R_new, _, _ = kin.tcp_fk_jac(q_new)
        # gripper-side penetration along outward normal (kinematic; can be < 0)
        z_kin = -((p_new - p0) * n_out).sum(dim=-1)
        if use_dynamics:
            # 1-DOF mass-spring: tip lags z_kin via stiff spring + light damping;
            # surface pushes back on tip when z_dyn > 0. Substep explicit Euler.
            for _ in range(n_substeps):
                f_grip = grip_k * (z_kin - z_dyn) - grip_c * z_dot
                f_surf = -k_n * z_dyn.clamp(min=0.0)
                z_ddot = (f_grip + f_surf) * m_inv
                z_dot = z_dot + z_ddot * dt_sub
                z_dyn = z_dyn + z_dot * dt_sub
            pen = z_dyn.clamp(min=0.0)
        else:
            pen = z_kin.clamp(min=0.0)
        force = k_n * pen
        # tangential pos_err only (drop n-direction since spring handles it)
        delta = (p_ref - p_new)
        delta_n = (delta * n_out).sum(dim=-1, keepdim=True) * n_out
        pos_err_tan = (delta - delta_n).norm(dim=-1)
        orient_err = torch.acos(
            (R_new[:, :, 2] * z_tgt).sum(dim=-1).clamp(-1.0, 1.0))
        if sphere_cc is None:
            self_collision = torch.zeros_like(step_alive)
        else:
            self_collision = sphere_cc.is_collided(kin.link_transforms(q_new))

        last_force = torch.where(step_alive, force, last_force)
        fail_pos = step_alive & (pos_err_tan > eps_p)
        fail_ori = step_alive & (orient_err > theta_max)
        fail_lmt = step_alive & joint_limit_hit
        fail_col = step_alive & self_collision
        if step > grace_steps:
            fail_fhi = step_alive & (pen > pen_max)
            fail_flo = step_alive & (pen < pen_min)
        else:
            fail_fhi = torch.zeros_like(step_alive)
            fail_flo = torch.zeros_like(step_alive)
        ok = step_alive & ~(fail_pos | fail_ori | fail_lmt | fail_col
                            | fail_fhi | fail_flo)
        lengths = torch.where(ok, torch.full_like(lengths, step), lengths)
        q = torch.where(ok.unsqueeze(-1), q_new, q)

        reasons[fail_pos.detach().cpu().numpy()] = 'pos_err_exceeded'
        reasons[fail_ori.detach().cpu().numpy()] = 'orient_err_exceeded'
        reasons[fail_lmt.detach().cpu().numpy()] = 'joint_limit'
        reasons[fail_col.detach().cpu().numpy()] = 'self_collision'
        reasons[fail_fhi.detach().cpu().numpy()] = 'force_high'
        reasons[fail_flo.detach().cpu().numpy()] = 'force_low'
        alive = alive & in_horizon & ~(fail_pos | fail_ori | fail_lmt | fail_col
                                       | fail_fhi | fail_flo)

    complete = (ik_ok & (lengths >= T)).detach().cpu().numpy()
    reasons[complete] = 'max_steps'
    return {
        'lengths':     lengths.detach().cpu().numpy().astype(np.int32),
        'reasons':     reasons.tolist(),
        'last_force':  last_force.detach().cpu().numpy(),
    }


def phantom_rollout(action_np: np.ndarray,
                    c_np: np.ndarray,
                    v_path_np: np.ndarray,
                    eps_p_np: np.ndarray,
                    T_np: np.ndarray,
                    use_collision: bool | None = None) -> dict:
    """Kinematic phantom rollout: same path-tracking dynamics as the real
    rollout but **without** the nullspace controller (manipulability gradient,
    joint-limit attraction, angle attraction, K_NULL pull). Reports the same
    `lengths` array.

    Purpose: cheap-but-pathwise predictor of geometric failure (joint limit,
    IK divergence, collision). Skipping the nullspace removes the autograd
    backward pass that dominates per-step cost in batched_rollout.

    Assumes ACTION_MODE == 'branch_descriptor' (which is our deployed mode).
    """
    device = _device_from_cfg()
    kin = BatchedFR3Kinematics(device=device)
    if use_collision is None:
        use_collision = bool(cfg.USE_COLLISION_CHECK and cfg.BATCHED_COLLISION_CHECK)
    if use_collision:
        sphere_cls = _load_fr3_sphere_collision_cls()
        sphere_cc = sphere_cls(device=device,
                               margin=cfg.BATCHED_COLLISION_MARGIN)
    else:
        sphere_cc = None

    action = torch.as_tensor(action_np, device=device, dtype=torch.float32)
    c = torch.as_tensor(c_np, device=device, dtype=torch.float32)
    v_path = torch.as_tensor(v_path_np, device=device, dtype=torch.float32)
    eps_p = torch.as_tensor(eps_p_np, device=device, dtype=torch.float32)
    T = torch.as_tensor(T_np, device=device, dtype=torch.long)

    p0 = c[:, :3]
    d_dir = c[:, 3:6]
    n_dir = c[:, 6:9]
    branch_action = action
    R_tgt = build_branch_rotmat_batch(d_dir, n_dir, branch_action)
    q, ik_ok, _ = branch_project_multistart(kin, p0, R_tgt, branch_action)

    lengths = torch.zeros((q.shape[0],), device=device, dtype=torch.long)
    alive = ik_ok.clone()
    max_T = int(T.max().item()) if T.numel() else 0
    dt = float(cfg.DT)
    theta_max = float(cfg.THETA_MAX)
    z_tgt = R_tgt[:, :, 2]

    for step in range(1, max_T + 1):
        in_horizon = step <= T
        step_alive = alive & in_horizon
        if not step_alive.any():
            break

        p_ref = p0 + (step * dt) * v_path.unsqueeze(-1) * d_dir
        p_dot_ff = v_path.unsqueeze(-1) * d_dir
        p_tcp, R_tcp, J, _ = kin.tcp_fk_jac(q)
        omega_err, _ = _z_axis_error_from_rotmats(R_tcp, R_tgt)
        x_dot = torch.cat([
            p_dot_ff + float(cfg.KP_LIN) * (p_ref - p_tcp),
            float(cfg.KOMEGA) * omega_err,
        ], dim=-1)
        Jpinv = _dls_pinv(J, float(cfg.DLS_LAMBDA))
        q_dot = (Jpinv @ x_dot.unsqueeze(-1)).squeeze(-1)
        # No nullspace term — that's the speedup vs batched_rollout.
        q_dot = q_dot.clamp(-kin.qdot_max, kin.qdot_max)
        q_new_raw = q + q_dot * dt
        joint_limit_hit = ((q_new_raw < kin.lmt_lo - 1e-6)
                           | (q_new_raw > kin.lmt_up + 1e-6)).any(dim=-1)
        q_new = q_new_raw.clamp(kin.lmt_lo, kin.lmt_up)
        p_new, R_new, _, _ = kin.tcp_fk_jac(q_new)
        pos_err = (p_ref - p_new).norm(dim=-1)
        orient_err = torch.acos(
            (R_new[:, :, 2] * z_tgt).sum(dim=-1).clamp(-1.0, 1.0))
        if sphere_cc is None:
            self_collision = torch.zeros_like(step_alive)
        else:
            self_collision = sphere_cc.is_collided(kin.link_transforms(q_new))
        fail_pos = step_alive & (pos_err > eps_p)
        fail_ori = step_alive & (orient_err > theta_max)
        fail_lmt = step_alive & joint_limit_hit
        fail_col = step_alive & self_collision
        ok = step_alive & ~(fail_pos | fail_ori | fail_lmt | fail_col)
        lengths = torch.where(ok, torch.full_like(lengths, step), lengths)
        q = torch.where(ok.unsqueeze(-1), q_new, q)
        alive = alive & in_horizon & ~(fail_pos | fail_ori | fail_lmt | fail_col)

    return {
        'lengths': lengths.detach().cpu().numpy().astype(np.int32),
    }

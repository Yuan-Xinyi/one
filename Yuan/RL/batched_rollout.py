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


def build_target_rotmat_batch(d: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    z = _normalize(n)
    x = d - z * (d * z).sum(dim=-1, keepdim=True)
    x = _normalize(x)
    y = torch.cross(z, x, dim=-1)
    return torch.stack([x, y, z], dim=-1)


def _rotvec_between(R_cur: torch.Tensor, R_tgt: torch.Tensor) -> torch.Tensor:
    R_err = R_tgt @ R_cur.transpose(-1, -2)
    trace = R_err.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_th = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos_th)
    vee = torch.stack([
        R_err[:, 2, 1] - R_err[:, 1, 2],
        R_err[:, 0, 2] - R_err[:, 2, 0],
        R_err[:, 1, 0] - R_err[:, 0, 1],
    ], dim=-1)
    sin_th = torch.sin(theta)
    scale = torch.where(sin_th.abs() > 1e-6,
                        theta / (2.0 * sin_th.clamp_min(1e-12)),
                        torch.full_like(theta, 0.5))
    return vee * scale.unsqueeze(-1)


def _dls_pinv(J: torch.Tensor, damping: float) -> torch.Tensor:
    b = J.shape[0]
    eye6 = torch.eye(6, device=J.device, dtype=J.dtype).expand(b, 6, 6)
    A = J @ J.transpose(-1, -2) + (float(damping) ** 2) * eye6
    return J.transpose(-1, -2) @ torch.linalg.inv(A)


def _batched_ik_project(kin: BatchedFR3Kinematics,
                        q_seed: torch.Tensor,
                        p0: torch.Tensor,
                        R_tgt: torch.Tensor):
    q = q_seed.clamp(kin.lmt_lo, kin.lmt_up)
    prev_err = torch.full((q.shape[0],), float('inf'),
                          device=q.device, dtype=q.dtype)
    active = torch.ones((q.shape[0],), device=q.device, dtype=torch.bool)
    converged = torch.zeros_like(active)
    fail_reason = np.array(['init_ik_fail'] * q.shape[0], dtype=object)

    for _ in range(cfg.BATCHED_IK_MAX_ITERS):
        p_tcp, R_tcp, J, _ = kin.tcp_fk_jac(q)
        delta_p = p0 - p_tcp
        delta_theta = _rotvec_between(R_tcp, R_tgt)
        pos_err = delta_p.norm(dim=-1)
        rot_err = delta_theta.norm(dim=-1)
        in_limits = ((q >= kin.lmt_lo - 1e-5)
                     & (q <= kin.lmt_up + 1e-5)).all(dim=-1)
        now_conv = ((pos_err <= cfg.BATCHED_IK_TOL_POS)
                    & (rot_err <= cfg.BATCHED_IK_TOL_ROT)
                    & in_limits)
        newly = active & now_conv
        if newly.any():
            converged |= newly
            active &= ~newly
        if not active.any():
            break

        err = torch.cat([delta_p, delta_theta], dim=-1)
        err_norm = err.norm(dim=-1)
        increased = active & (err_norm > prev_err)
        if increased.any():
            active &= ~increased
        if not active.any():
            break
        prev_err = torch.where(active, err_norm, prev_err)

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
        delta_q_secondary = 0.2 * (kin.q_mid - q)
        delta_q = delta_q + (N @ delta_q_secondary.unsqueeze(-1)).squeeze(-1)
        q = torch.where(active.unsqueeze(-1), q + delta_q, q)

    if active.any():
        p_tcp, R_tcp, _, _ = kin.tcp_fk_jac(q)
        pos_err = (p0 - p_tcp).norm(dim=-1)
        rot_err = _rotvec_between(R_tcp, R_tgt).norm(dim=-1)
        in_limits = ((q >= kin.lmt_lo - 1e-5)
                     & (q <= kin.lmt_up + 1e-5)).all(dim=-1)
        now_conv = ((pos_err <= cfg.BATCHED_IK_TOL_POS)
                    & (rot_err <= cfg.BATCHED_IK_TOL_ROT)
                    & in_limits)
        converged |= active & now_conv

    fail_reason[converged.detach().cpu().numpy()] = 'ik_ok'
    return q.clamp(kin.lmt_lo, kin.lmt_up), converged, fail_reason


def batched_rollout(q_seed_np: np.ndarray,
                    c_np: np.ndarray,
                    v_path_np: np.ndarray,
                    eps_p_np: np.ndarray,
                    T_np: np.ndarray) -> dict:
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
    q_seed = torch.as_tensor(q_seed_np, device=device, dtype=torch.float32)
    c = torch.as_tensor(c_np, device=device, dtype=torch.float32)
    v_path = torch.as_tensor(v_path_np, device=device, dtype=torch.float32)
    eps_p = torch.as_tensor(eps_p_np, device=device, dtype=torch.float32)
    T = torch.as_tensor(T_np, device=device, dtype=torch.long)

    p0 = c[:, :3]
    d = c[:, 3:6]
    n = c[:, 6:9]
    R_tgt = build_target_rotmat_batch(d, n)

    q, ik_ok, reasons = _batched_ik_project(kin, q_seed, p0, R_tgt)
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
        z_cur = R_tcp[:, :, 2]
        z_tgt = R_tgt[:, :, 2]
        cross = torch.cross(z_cur, z_tgt, dim=-1)
        cos_th = (z_cur * z_tgt).sum(dim=-1).clamp(-1.0, 1.0)
        theta = torch.acos(cos_th)
        sin_th = cross.norm(dim=-1)
        axis = torch.where(sin_th.unsqueeze(-1) > 1e-6,
                           cross / sin_th.clamp_min(1e-12).unsqueeze(-1),
                           torch.zeros_like(cross))
        omega_err = axis * theta.unsqueeze(-1)
        x_dot = torch.cat([
            p_dot_ff + float(cfg.KP_LIN) * (p_ref - p_tcp),
            float(cfg.KOMEGA) * omega_err,
        ], dim=-1)

        Jpinv = _dls_pinv(J, float(cfg.DLS_LAMBDA))
        q_dot = (Jpinv @ x_dot.unsqueeze(-1)).squeeze(-1)
        N = eye7 - Jpinv @ J
        q_dot_secondary = float(cfg.K_NULL) * (q_ref - q)
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
    }

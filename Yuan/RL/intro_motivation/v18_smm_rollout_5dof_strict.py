"""5-DOF strict rollout: tight position + z-axis, tool spin free.

Compared to v18_smm_rollout_6dof's full 6-DOF strict tracking, this
releases rotation about the tool's z-axis (pen spin) as a second
redundant DOF. The result:

  * Task constraint: pos (3D) + z-axis direction (2 effective DOF, the
    spin component is zero). Total 5-DOF.
  * Null space: 2D (vs 1D in 6-DOF strict).
  * JL avoidance: gets a 2D direction to push in, so more JL configs
    are recoverable than in the 1D-null 6-DOF mode.
  * Failure modes: pos_err > 5mm, z-axis err > 3°, JL hit.

Same tolerances as 6-DOF strict, only difference is `omega_err` uses
the z-axis-only formula (perpendicular-to-z_cur axis-angle), and null
space dimensionality is 2 instead of 1.
"""
from __future__ import annotations

import numpy as np
import torch

import Yuan.RL.config as cfg
from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.RL.batched_rollout import _dls_pinv, _z_axis_error_from_rotmats
from Yuan.RL.v18_data_prep import _build_R_from_normal_direction


V_PATH = 0.10
CHUNK_SIZE = 1024
EPS_POS_5DOF_STRICT = 0.005           # 5 mm
EPS_ORI_5DOF_STRICT = np.deg2rad(3.0)  # 3° z-axis tolerance
JLIMIT_MARGIN = 0.20
JLIMIT_GAIN = 4.0


def _as_tensor(x, device):
    return torch.as_tensor(x, device=device, dtype=torch.float32)


def _build_perp_basis(z_cur: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """For each row of z_cur (B, 3), return (e1, e2) orthonormal basis
    perpendicular to z_cur. Robust against z_cur || ex by falling back
    to ey."""
    B = z_cur.shape[0]
    device = z_cur.device
    dtype = z_cur.dtype
    ex = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype).view(1, 3).expand(B, 3)
    ey = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype).view(1, 3).expand(B, 3)
    e1_x = ex - (ex * z_cur).sum(-1, keepdim=True) * z_cur
    e1_x_norm = e1_x.norm(dim=-1, keepdim=True)
    e1_y = ey - (ey * z_cur).sum(-1, keepdim=True) * z_cur
    e1_y_norm = e1_y.norm(dim=-1, keepdim=True)
    use_x = (e1_x_norm.squeeze(-1) > 0.3)
    e1 = torch.where(use_x.unsqueeze(-1),
                     e1_x / e1_x_norm.clamp_min(1e-9),
                     e1_y / e1_y_norm.clamp_min(1e-9))
    e2 = torch.cross(z_cur, e1, dim=-1)
    e2 = e2 / e2.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return e1, e2


def _dls_pinv_5x7(J5: torch.Tensor, damping: float) -> torch.Tensor:
    """DLS pinv for a 5×7 task Jacobian. Returns (B, 7, 5)."""
    B = J5.shape[0]
    eye5 = torch.eye(5, device=J5.device, dtype=J5.dtype).expand(B, 5, 5)
    A = J5 @ J5.transpose(-1, -2) + (float(damping) ** 2) * eye5
    return J5.transpose(-1, -2) @ torch.linalg.inv(A)


def _batched_segment_5dof_strict(q_init: torch.Tensor,
                                  R_tgt: torch.Tensor,
                                  p0: torch.Tensor,
                                  d_dir: torch.Tensor,
                                  v_path: torch.Tensor,
                                  T_total: torch.Tensor,
                                  n_steps: int,
                                  kin: BatchedFR3Kinematics,
                                  eps_pos: float,
                                  eps_ori: float,
                                  alive_mask: torch.Tensor | None,
                                  enforce_init_pose: bool):
    device = q_init.device
    B = q_init.shape[0]
    eye7 = torch.eye(7, device=device, dtype=torch.float32).expand(B, 7, 7)
    dt = float(cfg.DT)

    alive = (torch.ones((B,), device=device, dtype=torch.bool)
             if alive_mask is None else alive_mask.clone())
    q = q_init.clone()
    lengths = torch.zeros((B,), device=device, dtype=torch.long)

    if enforce_init_pose:
        p_init, R_init, _, _ = kin.tcp_fk_jac(q)
        init_pos_err = (p0 - p_init).norm(dim=-1)
        init_omega, _ = _z_axis_error_from_rotmats(R_init, R_tgt)
        init_ori_err = init_omega.norm(dim=-1)
        init_fail = (init_pos_err > eps_pos) | (init_ori_err > eps_ori)
        alive = alive & ~init_fail

    for step in range(1, n_steps + 1):
        in_horizon = step <= T_total
        step_alive = alive & in_horizon
        if not step_alive.any():
            break

        p_ref = p0 + (step * dt) * v_path.unsqueeze(-1) * d_dir
        p_dot_ff = v_path.unsqueeze(-1) * d_dir
        p_tcp, R_tcp, J, _ = kin.tcp_fk_jac(q)

        # Build 5×7 task Jacobian: rows = (3 pos, 2 z-axis ori in basis ⊥ z_cur).
        # The spin direction (around z_cur) is NOT in the task → 2D null space.
        z_cur = R_tcp[:, :, 2]                                  # (B, 3)
        e1, e2 = _build_perp_basis(z_cur)                       # (B, 3) each
        J_pos = J[:, :3, :]                                     # (B, 3, 7)
        J_ang = J[:, 3:, :]                                     # (B, 3, 7)
        # row e_k of J_ang gives angular velocity along e_k = e_k^T @ J_ang
        J_ang_e1 = (e1.unsqueeze(-1) * J_ang).sum(dim=1, keepdim=True)  # (B,1,7)
        J_ang_e2 = (e2.unsqueeze(-1) * J_ang).sum(dim=1, keepdim=True)
        J_5dof = torch.cat([J_pos, J_ang_e1, J_ang_e2], dim=1)  # (B, 5, 7)

        # 5-DOF task error: pos (3D) + z-axis error projected on (e1,e2) (2D).
        omega_err_3, _ = _z_axis_error_from_rotmats(R_tcp, R_tgt)  # (B, 3)
        omega_err_e1 = (omega_err_3 * e1).sum(-1, keepdim=True)
        omega_err_e2 = (omega_err_3 * e2).sum(-1, keepdim=True)
        x_dot_pos = p_dot_ff + float(cfg.KP_LIN) * (p_ref - p_tcp)
        x_dot_ori = float(cfg.KOMEGA) * torch.cat(
            [omega_err_e1, omega_err_e2], dim=-1)
        x_dot = torch.cat([x_dot_pos, x_dot_ori], dim=-1)        # (B, 5)

        Jpinv5 = _dls_pinv_5x7(J_5dof, float(cfg.DLS_LAMBDA))    # (B, 7, 5)
        q_dot_primary = (Jpinv5 @ x_dot.unsqueeze(-1)).squeeze(-1)
        N = eye7 - Jpinv5 @ J_5dof                               # rank 2 → 2D null

        dist_lo = q - kin.lmt_lo
        dist_hi = kin.lmt_up - q
        danger_lo = (JLIMIT_MARGIN - dist_lo).clamp(min=0.0)
        danger_hi = (JLIMIT_MARGIN - dist_hi).clamp(min=0.0)
        q_dot_jl = JLIMIT_GAIN * (danger_lo - danger_hi)
        q_dot_jl_proj = (N @ q_dot_jl.unsqueeze(-1)).squeeze(-1)

        q_dot = (q_dot_primary + q_dot_jl_proj).clamp(-kin.qdot_max, kin.qdot_max)
        q_new_raw = q + q_dot * dt
        joint_limit_hit = ((q_new_raw < kin.lmt_lo - 1e-6)
                           | (q_new_raw > kin.lmt_up + 1e-6)).any(dim=-1)
        q_new = q_new_raw.clamp(kin.lmt_lo, kin.lmt_up)

        p_new, R_new, _, _ = kin.tcp_fk_jac(q_new)
        pos_err = (p_ref - p_new).norm(dim=-1)
        omega_new, _ = _z_axis_error_from_rotmats(R_new, R_tgt)
        orient_err = omega_new.norm(dim=-1)

        fail_pos = step_alive & (pos_err > eps_pos)
        fail_ori = step_alive & (orient_err > eps_ori)
        fail_lmt = step_alive & joint_limit_hit
        ok = step_alive & ~(fail_pos | fail_ori | fail_lmt)
        lengths = torch.where(ok, torch.full_like(lengths, step), lengths)
        q = torch.where(ok.unsqueeze(-1), q_new, q)
        alive = alive & in_horizon & ~(fail_pos | fail_ori | fail_lmt)

    return lengths, q, alive


def rollout_chunk_5dof_strict(kin, q_init, track_pts, plane_normal,
                                eps_pos=EPS_POS_5DOF_STRICT,
                                eps_ori=EPS_ORI_5DOF_STRICT,
                                enforce_init_pose=True):
    device = kin.device
    B = q_init.shape[0]
    q = q_init.clone()
    alive = torch.ones(B, device=device, dtype=torch.bool)
    lengths_m = torch.zeros(B, device=device, dtype=torch.float32)

    for idx in range(track_pts.shape[0] - 1):
        if not bool(alive.any().item()):
            break
        p0 = track_pts[idx]
        seg_vec = track_pts[idx + 1] - p0
        seg_len = float(seg_vec.norm().item())
        if seg_len < 1e-8:
            continue
        direction = seg_vec / seg_vec.norm().clamp_min(1e-12)
        rot_np = _build_R_from_normal_direction(
            plane_normal.detach().cpu().numpy(),
            direction.detach().cpu().numpy(),
        )
        n_steps = max(1, int(round(seg_len / (V_PATH * float(cfg.DT)))))
        v_path = torch.full((B,), V_PATH, device=device, dtype=torch.float32)
        T_total = torch.full((B,), n_steps, device=device, dtype=torch.long)
        alive_entering = alive.clone()

        lengths_step, q, alive = _batched_segment_5dof_strict(
            q_init=q,
            R_tgt=_as_tensor(rot_np, device).unsqueeze(0).expand(B, 3, 3),
            p0=p0.unsqueeze(0).expand(B, 3),
            d_dir=direction.unsqueeze(0).expand(B, 3),
            v_path=v_path,
            T_total=T_total,
            n_steps=n_steps,
            kin=kin,
            eps_pos=eps_pos,
            eps_ori=eps_ori,
            alive_mask=alive,
            enforce_init_pose=(enforce_init_pose and idx == 0),
        )
        completed = lengths_step.float() / float(n_steps) * seg_len
        lengths_m = torch.where(alive_entering, lengths_m + completed, lengths_m)

    return lengths_m.detach().cpu().numpy()


def rollout_lengths_5dof_strict(kin, q_batch, track_pts, plane_normal,
                                  eps_pos=EPS_POS_5DOF_STRICT,
                                  eps_ori=EPS_ORI_5DOF_STRICT,
                                  enforce_init_pose=True):
    lengths = np.zeros(q_batch.shape[0], dtype=np.float32)
    for start in range(0, q_batch.shape[0], CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, q_batch.shape[0])
        lengths[start:end] = rollout_chunk_5dof_strict(
            kin, q_batch[start:end], track_pts, plane_normal,
            eps_pos=eps_pos, eps_ori=eps_ori,
            enforce_init_pose=enforce_init_pose)
    return lengths

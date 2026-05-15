"""5-DOF strict rollout, Žlajpah 2017 formulation.

Task constraint = position (3D) + pen z-axis direction (2 effective DOF).
Tool spin about its z-axis is free → 2D null space, used for H_JL + H_dir.

Task Jacobian (rank-5, presented as 5×7 in (e1, e2) basis ⊥ z_t):
    J_t = [J_v;  (I - z_t z_t^T) J_ω]
        = [J_v;  e1^T J_ω;  e2^T J_ω]      (B, 5, 7)

Task velocity (5D):
    x_dot = [v·û (3); k_p (z_t × n_target) projected to (e1, e2)]

Deadzone:
    angle(z_t, n_target) < θ_tol_soft  →  rotation command = 0

Secondary tasks (in null space N = I - J_t^+ J_t, rank 2):
    q_dot_secondary = w_jl · q_dot_jl_repulsion + w_dir · (−∇H_dir)
    H_dir(q) = −1 / √(û^T (J_t J_t^T)^{−1} û)      (directional manipulability)
    weights w_jl : w_dir = 1 : 2

Failure:
    pos_err > 5 mm
    angle > θ_tol_hard
    σ_min(J_t) < 1e-3
    any q outside [lmt_lo + 0.05, lmt_up - 0.05]
"""
from __future__ import annotations

import numpy as np
import torch

import Yuan.flow_connectivity.config as cfg
from Yuan.flow_connectivity.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.flow_connectivity.v18_data_prep import _build_R_from_normal_direction


# --- task tolerances ---
V_PATH = 0.10
CHUNK_SIZE = 1024
EPS_POS_5DOF = 0.005                    # 5 mm
THETA_TOL_SOFT = np.deg2rad(5.0)        # deadzone for rotation command
THETA_TOL_HARD = np.deg2rad(15.0)       # failure threshold for angle
SIGMA_MIN_FAIL = 1e-3                   # primary-task singular failure
JL_FAIL_MARGIN = 0.0                    # fail if q within this of any limit
                                         # (0 = only when crossing, matches 6-DOF)

# --- controller params ---
DLS_LAMBDA = 0.05
JL_REPULSION_MARGIN = 0.20              # JL avoidance activates within this
JL_GAIN = 4.0
W_JL = 1.0                              # secondary weight on H_JL
W_DIR = 2.0                             # secondary weight on H_dir


def _as_tensor(x, device):
    return torch.as_tensor(x, device=device, dtype=torch.float32)


def _build_perp_basis(z_cur: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Orthonormal basis ⊥ z_cur (B, 3). Robust against z_cur || ex."""
    B = z_cur.shape[0]
    device, dtype = z_cur.device, z_cur.dtype
    ex = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype).view(1, 3).expand(B, 3)
    ey = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype).view(1, 3).expand(B, 3)
    e1_x = ex - (ex * z_cur).sum(-1, keepdim=True) * z_cur
    e1_y = ey - (ey * z_cur).sum(-1, keepdim=True) * z_cur
    use_x = (e1_x.norm(dim=-1) > 0.3).unsqueeze(-1)
    e1 = torch.where(use_x, e1_x, e1_y)
    e1 = e1 / e1.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    e2 = torch.cross(z_cur, e1, dim=-1)
    e2 = e2 / e2.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return e1, e2


def _build_J_5dof(J: torch.Tensor, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
    """Stack 5×7 task Jacobian from full 6×7 J and (e1, e2) ⊥ z_t."""
    J_v = J[:, :3, :]                                      # (B, 3, 7)
    J_w = J[:, 3:, :]                                      # (B, 3, 7)
    J_w_e1 = (e1.unsqueeze(-1) * J_w).sum(dim=1, keepdim=True)
    J_w_e2 = (e2.unsqueeze(-1) * J_w).sum(dim=1, keepdim=True)
    return torch.cat([J_v, J_w_e1, J_w_e2], dim=1)         # (B, 5, 7)


def _dls_pinv_5x7(J5: torch.Tensor, damping: float) -> torch.Tensor:
    B = J5.shape[0]
    eye5 = torch.eye(5, device=J5.device, dtype=J5.dtype).expand(B, 5, 5)
    A = J5 @ J5.transpose(-1, -2) + (float(damping) ** 2) * eye5
    return J5.transpose(-1, -2) @ torch.linalg.inv(A)


def _H_dir_grad(kin, q, u_hat):
    """Autograd ∇H_dir where H_dir = -1/√(û^T (J_t J_t^T)^{-1} û).
    u_hat is 3D Cartesian path direction (B, 3); embedded as (u, 0, 0) in 5D."""
    device, dtype = q.device, q.dtype
    B = q.shape[0]
    eye5 = torch.eye(5, device=device, dtype=dtype).expand(B, 5, 5)
    q_eval = q.detach().clone().requires_grad_(True)
    _, R_tcp, J, _ = kin.tcp_fk_jac(q_eval)
    z_cur = R_tcp[:, :, 2]
    e1, e2 = _build_perp_basis(z_cur)
    J_t = _build_J_5dof(J, e1, e2)
    u_5d = torch.zeros(B, 5, device=device, dtype=dtype)
    u_5d[:, :3] = u_hat
    JJt = J_t @ J_t.transpose(-1, -2) + 1e-6 * eye5
    JJt_inv_u = torch.linalg.solve(JJt, u_5d.unsqueeze(-1)).squeeze(-1)
    quad = (u_5d * JJt_inv_u).sum(dim=-1).clamp_min(1e-12)
    H = -1.0 / torch.sqrt(quad)
    grad = torch.autograd.grad(H.sum(), q_eval, allow_unused=True)[0]
    return grad if grad is not None else torch.zeros_like(q)


def _batched_segment_5dof(q_init: torch.Tensor, p0: torch.Tensor,
                           d_dir: torch.Tensor, n_target: torch.Tensor,
                           v_path: torch.Tensor, T_total: torch.Tensor,
                           n_steps: int, kin: BatchedFR3Kinematics,
                           alive_mask: torch.Tensor | None,
                           enforce_init_pose: bool,
                           record_traj: bool = False):
    """One segment of 5-DOF Žlajpah rollout. Returns:
       lengths_step (B,), q_final (B, 7), alive (B,) [, q_record (n_steps+1, B, 7), fail_reason (B, str)]"""
    device = q_init.device
    B = q_init.shape[0]
    eye7 = torch.eye(7, device=device, dtype=torch.float32).expand(B, 7, 7)
    dt = float(cfg.DT)
    lo = kin.lmt_lo; hi = kin.lmt_up

    alive = (torch.ones((B,), device=device, dtype=torch.bool)
             if alive_mask is None else alive_mask.clone())
    q = q_init.clone()
    lengths = torch.zeros((B,), device=device, dtype=torch.long)
    q_record = [q.clone()] if record_traj else None
    fail_reason = ['' for _ in range(B)]
    last_pos_err = torch.zeros((B,), device=device, dtype=torch.float32)
    last_ang_err = torch.zeros((B,), device=device, dtype=torch.float32)
    last_sigma = torch.zeros((B,), device=device, dtype=torch.float32)

    if enforce_init_pose:
        p_init, R_init, _, _ = kin.tcp_fk_jac(q)
        init_pos_err = (p0 - p_init).norm(dim=-1)
        z_init = R_init[:, :, 2]
        cos_th = (z_init * n_target).sum(-1).clamp(-1.0, 1.0)
        init_ang = torch.acos(cos_th)
        init_fail = (init_pos_err > EPS_POS_5DOF) | (init_ang > THETA_TOL_HARD)
        alive = alive & ~init_fail
        for i in torch.where(init_fail)[0]:
            fail_reason[int(i)] = 'init_pose_fail'

    for step in range(1, n_steps + 1):
        in_horizon = step <= T_total
        step_alive = alive & in_horizon
        if not step_alive.any():
            if record_traj:
                q_record.append(q.clone())
            continue

        p_ref = p0 + (step * dt) * v_path.unsqueeze(-1) * d_dir
        p_dot_ff = v_path.unsqueeze(-1) * d_dir
        p_tcp, R_tcp, J, _ = kin.tcp_fk_jac(q)
        z_cur = R_tcp[:, :, 2]
        e1, e2 = _build_perp_basis(z_cur)
        J_t = _build_J_5dof(J, e1, e2)

        # --- primary task ---
        x_dot_pos = p_dot_ff + float(cfg.KP_LIN) * (p_ref - p_tcp)

        # Rotation: ω = KOMEGA * (z_cur × n_target) drives z_cur → n_target.
        cross_zn = torch.cross(z_cur, n_target, dim=-1)
        cos_th = (z_cur * n_target).sum(-1).clamp(-1.0, 1.0)
        angle_err = torch.acos(cos_th)
        in_deadzone = angle_err < THETA_TOL_SOFT
        omega_cmd = float(cfg.KOMEGA) * cross_zn
        omega_cmd = torch.where(in_deadzone.unsqueeze(-1),
                                  torch.zeros_like(omega_cmd), omega_cmd)
        rot_e1 = (omega_cmd * e1).sum(-1, keepdim=True)
        rot_e2 = (omega_cmd * e2).sum(-1, keepdim=True)
        x_dot = torch.cat([x_dot_pos, rot_e1, rot_e2], dim=-1)        # (B, 5)

        Jpinv5 = _dls_pinv_5x7(J_t, DLS_LAMBDA)
        q_dot_primary = (Jpinv5 @ x_dot.unsqueeze(-1)).squeeze(-1)

        # --- secondary tasks in null space (2D) ---
        N = eye7 - Jpinv5 @ J_t

        # H_JL: repulsion from joint limits (descent direction implicit).
        dist_lo = q - lo
        dist_hi = hi - q
        danger_lo = (JL_REPULSION_MARGIN - dist_lo).clamp(min=0.0)
        danger_hi = (JL_REPULSION_MARGIN - dist_hi).clamp(min=0.0)
        q_dot_jl = JL_GAIN * (danger_lo - danger_hi)

        # H_dir: descend along -∇H_dir to maximize directional manipulability.
        grad_H_dir = _H_dir_grad(kin, q, d_dir)
        q_dot_dir = -grad_H_dir          # raw descent direction

        q_dot_secondary = W_JL * q_dot_jl + W_DIR * q_dot_dir
        q_dot_null = (N @ q_dot_secondary.unsqueeze(-1)).squeeze(-1)

        q_dot = (q_dot_primary + q_dot_null).clamp(-kin.qdot_max, kin.qdot_max)
        q_new_raw = q + q_dot * dt

        # --- failure detection ---
        margin_new = torch.minimum(q_new_raw - lo, hi - q_new_raw).min(dim=-1).values
        jl_fail = margin_new < JL_FAIL_MARGIN
        # σ_min(J_t) at new q (compute via current J_t since q hasn't been clamped yet)
        sigma_min = torch.linalg.svdvals(J_t)[:, -1]
        sing_fail = sigma_min < SIGMA_MIN_FAIL

        q_new = q_new_raw.clamp(lo, hi)
        p_new, R_new, _, _ = kin.tcp_fk_jac(q_new)
        z_new = R_new[:, :, 2]
        pos_err_new = (p_ref - p_new).norm(dim=-1)
        cos_new = (z_new * n_target).sum(-1).clamp(-1.0, 1.0)
        ang_new = torch.acos(cos_new)
        pos_fail = pos_err_new > EPS_POS_5DOF
        ang_fail = ang_new > THETA_TOL_HARD

        fail_pos = step_alive & pos_fail
        fail_ang = step_alive & ang_fail
        fail_lmt = step_alive & jl_fail
        fail_sng = step_alive & sing_fail
        died = fail_pos | fail_ang | fail_lmt | fail_sng

        last_pos_err = torch.where(step_alive, pos_err_new, last_pos_err)
        last_ang_err = torch.where(step_alive, ang_new, last_ang_err)
        last_sigma = torch.where(step_alive, sigma_min, last_sigma)

        for i in torch.where(died)[0]:
            ii = int(i)
            if fail_reason[ii]:
                continue
            if bool(fail_lmt[ii]):
                fr = 'joint_limit'
            elif bool(fail_sng[ii]):
                fr = 'singular'
            elif bool(fail_pos[ii]):
                fr = 'pos_err'
            else:
                fr = 'angle_err'
            fail_reason[ii] = fr

        ok = step_alive & ~died
        lengths = torch.where(ok, torch.full_like(lengths, step), lengths)
        q = torch.where(ok.unsqueeze(-1), q_new, q)
        alive = alive & in_horizon & ~died
        if record_traj:
            q_record.append(q.clone())

    if record_traj:
        return lengths, q, alive, torch.stack(q_record, dim=0), fail_reason, \
                last_pos_err, last_ang_err, last_sigma
    return lengths, q, alive


def rollout_chunk_5dof(kin, q_init, track_pts, plane_normal,
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
            direction.detach().cpu().numpy())
        n_target_t = _as_tensor(rot_np[:, 2], device).unsqueeze(0).expand(B, 3)
        n_steps = max(1, int(round(seg_len / (V_PATH * float(cfg.DT)))))
        v_path = torch.full((B,), V_PATH, device=device, dtype=torch.float32)
        T_total = torch.full((B,), n_steps, device=device, dtype=torch.long)
        alive_entering = alive.clone()

        lengths_step, q, alive = _batched_segment_5dof(
            q_init=q, p0=p0.unsqueeze(0).expand(B, 3),
            d_dir=direction.unsqueeze(0).expand(B, 3),
            n_target=n_target_t,
            v_path=v_path, T_total=T_total, n_steps=n_steps,
            kin=kin, alive_mask=alive,
            enforce_init_pose=(enforce_init_pose and idx == 0),
        )
        completed = lengths_step.float() * (V_PATH * float(cfg.DT))
        lengths_m = torch.where(alive_entering, lengths_m + completed, lengths_m)
    return lengths_m.detach().cpu().numpy()


def rollout_lengths_5dof(kin, q_batch, track_pts, plane_normal,
                          enforce_init_pose=True):
    lengths = np.zeros(q_batch.shape[0], dtype=np.float32)
    for start in range(0, q_batch.shape[0], CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, q_batch.shape[0])
        lengths[start:end] = rollout_chunk_5dof(
            kin, q_batch[start:end], track_pts, plane_normal,
            enforce_init_pose=enforce_init_pose)
    return lengths


def record_rollout_5dof(kin, q_init, track_pts, plane_normal_np):
    """Step-by-step rollout with q_traj recording. Returns:
       (q_traj (T+1, B, 7), fail_infos list[dict])."""
    device = kin.device
    B = q_init.shape[0]
    q = q_init.clone()
    alive = torch.ones(B, device=device, dtype=torch.bool)
    q_record = [q.clone()]
    fail_info: list[dict | None] = [None] * B
    step_global = 0
    lo = kin.lmt_lo.detach().cpu().numpy()
    hi = kin.lmt_up.detach().cpu().numpy()

    track_pts_np = track_pts.detach().cpu().numpy() if hasattr(track_pts, 'detach') else track_pts

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
            plane_normal_np, direction.detach().cpu().numpy())
        n_target_t = _as_tensor(rot_np[:, 2], device).unsqueeze(0).expand(B, 3)
        n_steps = max(1, int(round(seg_len / (V_PATH * float(cfg.DT)))))
        v_path = torch.full((B,), V_PATH, device=device, dtype=torch.float32)
        T_total = torch.full((B,), n_steps, device=device, dtype=torch.long)
        prev_alive_np = alive.detach().cpu().numpy().copy()
        step_global_before = step_global

        out = _batched_segment_5dof(
            q_init=q, p0=p0.unsqueeze(0).expand(B, 3),
            d_dir=direction.unsqueeze(0).expand(B, 3),
            n_target=n_target_t,
            v_path=v_path, T_total=T_total, n_steps=n_steps,
            kin=kin, alive_mask=alive,
            enforce_init_pose=(idx == 0),
            record_traj=True,
        )
        lengths_step, q, alive, seg_traj, seg_reasons, lpe, lae, lsm = out
        for k in range(1, seg_traj.shape[0]):
            q_record.append(seg_traj[k])
        step_global += n_steps

        new_alive_np = alive.detach().cpu().numpy()
        died = prev_alive_np & ~new_alive_np
        if died.any():
            seg_lengths = lengths_step.detach().cpu().numpy()
            lpe_np = lpe.detach().cpu().numpy()
            lae_np = lae.detach().cpu().numpy()
            lsm_np = lsm.detach().cpu().numpy()
            q_final_np = q.detach().cpu().numpy()
            for i in np.where(died)[0]:
                ii = int(i)
                if fail_info[ii] is not None:
                    continue
                margin = np.minimum(q_final_np[ii] - lo, hi - q_final_np[ii])
                fj = int(margin.argmin())
                reason = seg_reasons[ii] or 'unknown'
                if reason == 'joint_limit':
                    reason = f'joint_limit (j{fj})'
                fail_info[ii] = {
                    'reason': reason,
                    'fail_step': step_global_before + int(seg_lengths[ii]),
                    'fail_joint': fj,
                    'pos_err': float(lpe_np[ii]),
                    'ori_err': float(lae_np[ii]),
                    'sigma_min': float(lsm_np[ii]),
                }

    for i in range(B):
        if fail_info[i] is None:
            fail_info[i] = {
                'reason': 'completed_path', 'fail_step': step_global,
                'fail_joint': -1, 'pos_err': 0.0, 'ori_err': 0.0,
                'sigma_min': 0.0,
            }

    return torch.stack(q_record, dim=0), fail_info

"""6-DOF strict rollout for path-following on the SMM.

Aligns with v18_smm_enumerate's SMM definition (FK(q) = (p_tgt, R_tgt)
within Newton tol 1e-6). Unlike the v18 default pos_priority rollout
(5-DOF: position + z-axis, with 30° dead zone), this controller:

  * Tracks the full 6-DOF pose (p_ref, R_tgt) tightly: pos_err < 5mm
    AND |rotvec(R_cur, R_tgt)| < 3° at every step.
  * Has exactly 1D null space (the SMM tangent at the current q),
    used only for JL avoidance.
  * No orientation dead zone — any drift beyond 3° fails the row.

Path: along the segment, p_ref moves at V_PATH, R_tgt stays constant
(straight-line task → constant tangent → constant R built from
plane_normal + tangent). The robot's only redundant DOF is the SMM
tangent, so it sweeps the 1D SMM as p_ref translates.

Result: q0 that lie on a SMM branch with insufficient JL clearance
along the moving-pose trajectory fail early. This is the clean test
of branch-level path-following capability.
"""
from __future__ import annotations

import numpy as np
import torch

import Yuan.flow_connectivity.config as cfg
from Yuan.flow_connectivity.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.flow_connectivity.batched_rollout import _dls_pinv, _rotvec_between
from Yuan.flow_connectivity.v18_data_prep import _build_R_from_normal_direction


V_PATH = 0.10
CHUNK_SIZE = 1024
EPS_POS_6DOF = 0.005             # 5 mm
EPS_ORI_6DOF = np.deg2rad(3.0)   # 3 deg
JLIMIT_MARGIN = 0.20             # rad
JLIMIT_GAIN = 4.0                # rad/s


def _as_tensor(x, device):
    return torch.as_tensor(x, device=device, dtype=torch.float32)


def _batched_segment_6dof(q_init: torch.Tensor,
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
    """One segment of strict 6-DOF tracking.
    Returns (lengths (B,) max step within segment, q_final, alive_out)."""
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
        init_ori_err = _rotvec_between(R_init, R_tgt).norm(dim=-1)
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

        # Full 6-DOF error: position + full SO(3) rotation vector.
        omega_err = _rotvec_between(R_tcp, R_tgt)
        x_dot_pos = p_dot_ff + float(cfg.KP_LIN) * (p_ref - p_tcp)
        x_dot_ori = float(cfg.KOMEGA) * omega_err
        x_dot = torch.cat([x_dot_pos, x_dot_ori], dim=-1)

        # Primary: full 6-DOF DLS. Null space is exactly 1D (the SMM tangent).
        Jpinv = _dls_pinv(J, float(cfg.DLS_LAMBDA))
        q_dot_primary = (Jpinv @ x_dot.unsqueeze(-1)).squeeze(-1)
        N = eye7 - Jpinv @ J

        # JL avoidance projected onto the 1D null space. With only 1 null
        # direction available, JL is either aligned with it (helps) or
        # orthogonal (helpless) — purely a function of branch geometry.
        dist_lo = q - kin.lmt_lo
        dist_hi = kin.lmt_up - q
        danger_lo = (JLIMIT_MARGIN - dist_lo).clamp(min=0.0)
        danger_hi = (JLIMIT_MARGIN - dist_hi).clamp(min=0.0)
        q_dot_jl = JLIMIT_GAIN * (danger_lo - danger_hi)
        q_dot_jl_proj = (N @ q_dot_jl.unsqueeze(-1)).squeeze(-1)

        q_dot = q_dot_primary + q_dot_jl_proj
        q_dot = q_dot.clamp(-kin.qdot_max, kin.qdot_max)
        q_new_raw = q + q_dot * dt
        joint_limit_hit = ((q_new_raw < kin.lmt_lo - 1e-6)
                           | (q_new_raw > kin.lmt_up + 1e-6)).any(dim=-1)
        q_new = q_new_raw.clamp(kin.lmt_lo, kin.lmt_up)

        p_new, R_new, _, _ = kin.tcp_fk_jac(q_new)
        pos_err = (p_ref - p_new).norm(dim=-1)
        orient_err = _rotvec_between(R_new, R_tgt).norm(dim=-1)

        fail_pos = step_alive & (pos_err > eps_pos)
        fail_ori = step_alive & (orient_err > eps_ori)
        fail_lmt = step_alive & joint_limit_hit
        ok = step_alive & ~(fail_pos | fail_ori | fail_lmt)
        lengths = torch.where(ok, torch.full_like(lengths, step), lengths)
        q = torch.where(ok.unsqueeze(-1), q_new, q)
        alive = alive & in_horizon & ~(fail_pos | fail_ori | fail_lmt)

    return lengths, q, alive


def rollout_chunk_6dof(kin: BatchedFR3Kinematics,
                        q_init: torch.Tensor,
                        track_pts: torch.Tensor,
                        plane_normal: torch.Tensor,
                        eps_pos: float = EPS_POS_6DOF,
                        eps_ori: float = EPS_ORI_6DOF,
                        enforce_init_pose: bool = True) -> np.ndarray:
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

        lengths_step, q, alive = _batched_segment_6dof(
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
        # Step-based: each successful step advances p_ref by V_PATH*DT.
        completed = lengths_step.float() * (V_PATH * float(cfg.DT))
        lengths_m = torch.where(alive_entering, lengths_m + completed, lengths_m)

    return lengths_m.detach().cpu().numpy()


def rollout_lengths_6dof(kin: BatchedFR3Kinematics,
                          q_batch: torch.Tensor,
                          track_pts: torch.Tensor,
                          plane_normal: torch.Tensor,
                          eps_pos: float = EPS_POS_6DOF,
                          eps_ori: float = EPS_ORI_6DOF,
                          enforce_init_pose: bool = True) -> np.ndarray:
    lengths = np.zeros(q_batch.shape[0], dtype=np.float32)
    for start in range(0, q_batch.shape[0], CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, q_batch.shape[0])
        lengths[start:end] = rollout_chunk_6dof(
            kin, q_batch[start:end], track_pts, plane_normal,
            eps_pos=eps_pos, eps_ori=eps_ori,
            enforce_init_pose=enforce_init_pose)
    return lengths


def record_rollout_6dof(kin: BatchedFR3Kinematics,
                         q_init: torch.Tensor,
                         track_pts: torch.Tensor,
                         plane_normal_np: np.ndarray,
                         eps_pos: float = EPS_POS_6DOF,
                         eps_ori: float = EPS_ORI_6DOF):
    """Step-by-step 6-DOF rollout with q_traj recording. Returns:
       q_traj: (T+1, B, 7) torch tensor
       fail_infos: list of dicts per row {reason, fail_step, fail_joint, pos_err, ori_err}
    """
    device = kin.device
    B = q_init.shape[0]
    eye7 = torch.eye(7, device=device, dtype=torch.float32).expand(B, 7, 7)
    dt = float(cfg.DT)
    lo = kin.lmt_lo
    hi = kin.lmt_up

    q = q_init.clone()
    alive = torch.ones(B, device=device, dtype=torch.bool)
    q_record = [q.clone()]
    fail_info: list[dict | None] = [None] * B
    step_global = 0

    seg_dir0 = track_pts[1] - track_pts[0]
    seg_dir0 = seg_dir0 / seg_dir0.norm().clamp_min(1e-12)
    rot0_np = _build_R_from_normal_direction(
        plane_normal_np, seg_dir0.detach().cpu().numpy())
    R_tgt0 = torch.as_tensor(rot0_np, device=device, dtype=torch.float32).unsqueeze(0).expand(B, 3, 3)
    p_init, R_init, _, _ = kin.tcp_fk_jac(q)
    init_pos_err = (track_pts[0] - p_init).norm(dim=-1)
    init_ori_err = _rotvec_between(R_init, R_tgt0).norm(dim=-1)
    init_fail = (init_pos_err > eps_pos) | (init_ori_err > eps_ori)
    alive = alive & ~init_fail
    for i in torch.where(init_fail)[0]:
        ii = int(i)
        fail_info[ii] = {
            'reason': 'init_pose_fail', 'fail_step': 0, 'fail_joint': -1,
            'pos_err': float(init_pos_err[ii]),
            'ori_err': float(init_ori_err[ii]),
        }

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
        R_tgt = torch.as_tensor(rot_np, device=device, dtype=torch.float32).unsqueeze(0).expand(B, 3, 3)
        d_dir = direction.unsqueeze(0).expand(B, 3)
        p0_b = p0.unsqueeze(0).expand(B, 3)
        v_path_v = torch.full((B,), V_PATH, device=device, dtype=torch.float32)
        n_steps = max(1, int(round(seg_len / (V_PATH * dt))))

        for step in range(1, n_steps + 1):
            step_global += 1
            step_alive = alive.clone()
            if not bool(step_alive.any().item()):
                q_record.append(q.clone())
                continue

            p_ref = p0_b + (step * dt) * v_path_v.unsqueeze(-1) * d_dir
            p_dot_ff = v_path_v.unsqueeze(-1) * d_dir
            p_tcp, R_tcp, J, _ = kin.tcp_fk_jac(q)
            omega_err = _rotvec_between(R_tcp, R_tgt)
            x_dot_pos = p_dot_ff + float(cfg.KP_LIN) * (p_ref - p_tcp)
            x_dot_ori = float(cfg.KOMEGA) * omega_err
            x_dot = torch.cat([x_dot_pos, x_dot_ori], dim=-1)

            Jpinv = _dls_pinv(J, float(cfg.DLS_LAMBDA))
            q_dot_primary = (Jpinv @ x_dot.unsqueeze(-1)).squeeze(-1)
            N = eye7 - Jpinv @ J

            dist_lo = q - lo
            dist_hi = hi - q
            danger_lo = (JLIMIT_MARGIN - dist_lo).clamp(min=0.0)
            danger_hi = (JLIMIT_MARGIN - dist_hi).clamp(min=0.0)
            q_dot_jl = JLIMIT_GAIN * (danger_lo - danger_hi)
            q_dot_jl_proj = (N @ q_dot_jl.unsqueeze(-1)).squeeze(-1)

            q_dot = (q_dot_primary + q_dot_jl_proj).clamp(-kin.qdot_max, kin.qdot_max)
            q_new_raw = q + q_dot * dt
            jl_out_lo = (q_new_raw < lo - 1e-6)
            jl_out_hi = (q_new_raw > hi + 1e-6)
            joint_limit_hit = (jl_out_lo | jl_out_hi).any(dim=-1)
            q_new = q_new_raw.clamp(lo, hi)

            p_new, R_new, _, _ = kin.tcp_fk_jac(q_new)
            pos_err = (p_ref - p_new).norm(dim=-1)
            orient_err = _rotvec_between(R_new, R_tgt).norm(dim=-1)

            fail_pos = step_alive & (pos_err > eps_pos)
            fail_ori = step_alive & (orient_err > eps_ori)
            fail_lmt = step_alive & joint_limit_hit
            died = fail_pos | fail_ori | fail_lmt

            for i in torch.where(died)[0]:
                ii = int(i)
                if fail_info[ii] is not None:
                    continue
                if bool(fail_lmt[ii]):
                    joints_out = torch.where(jl_out_lo[ii] | jl_out_hi[ii])[0]
                    fj = int(joints_out[0]) if len(joints_out) > 0 else -1
                    reason = f'joint_limit (j{fj})'
                elif bool(fail_pos[ii]):
                    fj, reason = -1, 'pos_err'
                else:
                    fj, reason = -1, 'ori_err'
                fail_info[ii] = {
                    'reason': reason, 'fail_step': step_global, 'fail_joint': fj,
                    'pos_err': float(pos_err[ii]),
                    'ori_err': float(orient_err[ii]),
                }

            ok = step_alive & ~died
            q = torch.where(ok.unsqueeze(-1), q_new, q)
            alive = alive & ~died
            q_record.append(q.clone())

    for i in range(B):
        if fail_info[i] is None:
            fail_info[i] = {
                'reason': 'completed_path', 'fail_step': step_global,
                'fail_joint': -1, 'pos_err': 0.0, 'ori_err': 0.0,
            }

    return torch.stack(q_record, dim=0), fail_info

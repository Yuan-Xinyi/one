"""Controller-based rollout for the farsighted-seed problem.

Two phases:
  Phase A (t = 0): IK projection. q_seed lives anywhere in joint space; we
                   run one warm-started Newton IK to obtain q_0 satisfying
                   FK(q_0) = (p_0, R*). This is the only place we use IK,
                   matching the problem statement (the *initial* config is
                   what we choose, not the per-step setpoints).
                   If the projection fails, length = 0.

  Phase B (t = 1..T): differential-IK Cartesian controller (DLSController)
                   tracks p_t = p_0 + t*dt*v_path*d  with z(q_t) -> n.
                   Per step we check:
                     - position-error <= EPS_POS
                     - orientation-error <= THETA_MAX
                     - joint limits (margin 1e-6)
                     - joint-velocity saturation
                     - self-collision (MJCollider, if provided)
                   First violation -> rollout terminates. length = t-1.

Returns dict: {length, success, reason, q_traj, qs0, traj_info}.
"""
from __future__ import annotations
import numpy as np
import torch

import Yuan.RL.config as cfg
from Yuan.RL.controller import DLSController
from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.RL.batched_rollout import (
    build_branch_rotmat_batch,
    build_target_rotmat_batch,
    branch_project_multistart,
)


# ---------------- target rotmat from (d, n) ----------------
def build_target_rotmat(d: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Build TCP target rotation. **TCP_z = -n** (tool points INTO the
    surface, opposite to the outward surface normal n). TCP_x is the
    motion direction d re-orthogonalised against TCP_z. TCP_y = z x x.
    Assumes d _|_ n.
    """
    z = -n / (np.linalg.norm(n) + 1e-12)              # ← flip: TCP_z = -n
    x = d - z * (d @ z)
    x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    R = np.empty((3, 3), dtype=np.float32)
    R[:, 0] = x; R[:, 1] = y; R[:, 2] = z
    return R


# ---------------- Phase A: IK projection q_seed -> q_0 ----------------
def _ik_project(arm, R_tgt: np.ndarray, p_tgt: np.ndarray,
                qs_init: np.ndarray, max_iter: int = 50):
    """Newton IK with qs_init as initial guess. Returns full-qs or None."""
    if cfg.INIT_IK_ORIENT_MODE == "z_axis":
        return _ik_project_pos_z(arm, R_tgt, p_tgt, qs_init, max_iter)

    solver = arm._solver                                  # NumIKSolver
    chain = arm._chain
    qs_init = np.asarray(qs_init, dtype=np.float32)
    if qs_init.shape[0] == arm.qs.shape[0]:
        qs_init = qs_init[chain.active_mask]

    flange_local = arm._loc_flange_tf @ arm._loc_tcp_tf
    R_lastlnk = R_tgt @ flange_local[:3, :3].T
    p_lastlnk = (p_tgt
                 - R_lastlnk @ flange_local[:3, 3]).astype(np.float32)

    qs_active, info = solver._backward(
        arm.rotmat, arm.pos,
        R_lastlnk.astype(np.float32), p_lastlnk,
        qs_active_init=qs_init,
        max_iter=max_iter)
    if not info["converged"]:
        return None, info
    return chain.embed_active_qs(qs_active, arm.qs), info


def _branch_project(arm, branch_action: np.ndarray,
                    p0: np.ndarray, d: np.ndarray, n: np.ndarray):
    # Match the device that batched_rollout uses; otherwise borderline-IK
    # actions selected by GPU eval can fail to converge on CPU due to
    # accumulator-order floating-point differences.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kin = BatchedFR3Kinematics(device=device)
    action = torch.as_tensor(branch_action[None], dtype=torch.float32, device=device)
    p0_t = torch.as_tensor(p0[None], dtype=torch.float32, device=device)
    d_t = torch.as_tensor(d[None], dtype=torch.float32, device=device)
    n_t = torch.as_tensor(n[None], dtype=torch.float32, device=device)
    R_tgt_t = build_branch_rotmat_batch(d_t, n_t, action)
    q, ok, _ = branch_project_multistart(kin, p0_t, R_tgt_t, action)
    R_tgt = R_tgt_t.squeeze(0).cpu().numpy().astype(np.float32)
    if not bool(ok[0].item()):
        return None, R_tgt, {"converged": False, "mode": "branch_descriptor"}
    return q.squeeze(0).cpu().numpy().astype(np.float32), R_tgt, {
        "converged": True,
        "mode": "branch_descriptor",
    }


def _z_axis_error(R_cur: np.ndarray, R_tgt: np.ndarray) -> tuple[np.ndarray, float]:
    z_cur = R_cur[:, 2]
    z_tgt = R_tgt[:, 2]
    cross = np.cross(z_cur, z_tgt)
    cos_th = float(np.clip(z_cur @ z_tgt, -1.0, 1.0))
    theta = float(np.arccos(cos_th))
    sin_th = float(np.linalg.norm(cross))
    if sin_th > 1e-6:
        return (cross / sin_th * theta).astype(np.float32), theta
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(z_cur @ ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    axis = np.cross(z_cur, ref)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm > 1e-6:
        return (axis / axis_norm * theta).astype(np.float32), theta
    return np.zeros(3, dtype=np.float32), theta


def _ik_project_pos_z(arm, R_tgt: np.ndarray, p_tgt: np.ndarray,
                      qs_init: np.ndarray, max_iter: int = 50):
    """Project seed to the task manifold: TCP position + TCP z-axis.

    This matches the rollout termination condition and leaves yaw around TCP z
    free, avoiding an artificial full-orientation branch choice.
    """
    chain = arm._chain
    qs_init = np.asarray(qs_init, dtype=np.float32)
    if qs_init.shape[0] == arm.qs.shape[0]:
        q = qs_init[chain.active_mask].copy()
    else:
        q = qs_init.copy()
    ctrl = DLSController(arm)
    q_mid = (chain.lmt_lo + chain.lmt_up) * 0.5
    err = np.zeros(6, dtype=np.float32)

    for it in range(int(max_iter)):
        p_tcp, R_tcp, J = ctrl.fk_with_jac(q)
        delta_p = (p_tgt - p_tcp).astype(np.float32)
        delta_theta, theta = _z_axis_error(R_tcp, R_tgt)
        pos_err = float(np.linalg.norm(delta_p))
        if pos_err <= cfg.EPS_POS_INIT and theta <= cfg.THETA_MAX:
            if (np.any(q < chain.lmt_lo - 1e-5)
                    or np.any(q > chain.lmt_up + 1e-5)):
                return None, {"converged": False, "iters": it,
                              "reason": "joint_limits_exceeded"}
            return chain.embed_active_qs(q, arm.qs), {
                "converged": True,
                "iters": it,
                "pos_err": pos_err,
                "orient_err": theta,
                "mode": "pos_z",
            }
        if pos_err > 0.1:
            delta_p = delta_p / (pos_err + 1e-12) * 0.1
        if theta > 0.3:
            delta_theta = delta_theta / (theta + 1e-12) * 0.3
        delta_x = np.concatenate([delta_p, delta_theta]).astype(np.float32)
        err = delta_x
        JJt = J @ J.T
        A = JJt + (cfg.DLS_LAMBDA ** 2) * np.eye(6, dtype=np.float32)
        Jpinv = J.T @ np.linalg.inv(A)
        delta_q = Jpinv @ delta_x
        N = np.eye(chain.n_active_jnts, dtype=np.float32) - Jpinv @ J
        delta_q = delta_q + N @ (0.2 * (q_mid - q))
        q = np.clip(q + delta_q, chain.lmt_lo, chain.lmt_up)

    return None, {"converged": False, "iters": max_iter,
                  "err": err, "reason": "max_iters_reached",
                  "mode": "pos_z"}


# ---------------- main rollout ----------------
def rollout(arm,
            q_seed: np.ndarray,
            p0: np.ndarray,
            d: np.ndarray,
            n: np.ndarray,
            mjc=None,
            max_steps: int = cfg.MAX_STEPS,
            dt: float = cfg.DT,
            v_path: float = cfg.V_PATH,
            eps_p: float = cfg.EPS_POS,
            theta_max: float = cfg.THETA_MAX,
            action_mode: str | None = None) -> dict:
    chain = arm._chain
    p0 = p0.astype(np.float32)
    d = d.astype(np.float32)
    if action_mode is None:
        action_mode = cfg.ACTION_MODE

    # ----- Phase A: project the action onto the initial task manifold -----
    if action_mode == "branch_descriptor":
        q0_full, R_tgt, info0 = _branch_project(arm, q_seed, p0, d, n)
    else:
        R_tgt = build_target_rotmat(d, n)
        q0_full, info0 = _ik_project(arm, R_tgt, p0, q_seed)
    if q0_full is None:
        return {"length": 0, "success": False, "reason": "init_ik_fail",
                "q_traj": [], "qs0": None, "init_info": info0}

    # ----- Phase B: differential controller -----
    ctrl = DLSController(arm)
    # tweak null-space attractor toward q0 itself (avoid "snap to mid" jumps)
    ctrl.q_ref = q0_full[chain.active_mask].copy()

    q_active = q0_full[chain.active_mask].copy()
    q_traj = [q0_full.copy()]
    p_dot_ff = (v_path * d).astype(np.float32)
    last_info = {"reason": "max_steps"}

    for t in range(1, max_steps + 1):
        p_ref = (p0 + t * dt * v_path * d).astype(np.float32)
        info = ctrl.step(q_active, p_ref, R_tgt, p_dot_ff, dt=dt)

        if info["pos_err"] > eps_p:
            return {"length": t - 1, "success": False,
                    "reason": "pos_err_exceeded",
                    "q_traj": q_traj, "qs0": q0_full, "fail_info": info}
        if info["orient_err"] > theta_max:
            return {"length": t - 1, "success": False,
                    "reason": "orient_err_exceeded",
                    "q_traj": q_traj, "qs0": q0_full, "fail_info": info}
        if info["joint_limit_hit"]:
            return {"length": t - 1, "success": False,
                    "reason": "joint_limit",
                    "q_traj": q_traj, "qs0": q0_full, "fail_info": info}
        # advance state
        q_active = info["q_new"]
        q_full = chain.embed_active_qs(q_active, arm.qs)
        # self-collision (after the move)
        if mjc is not None and cfg.USE_COLLISION_CHECK:
            if mjc.is_collided(q_full):
                return {"length": t - 1, "success": False,
                        "reason": "self_collision",
                        "q_traj": q_traj, "qs0": q0_full,
                        "fail_info": info}
        q_traj.append(q_full.copy())
        last_info = info

    return {"length": max_steps, "success": True, "reason": "max_steps",
            "q_traj": q_traj, "qs0": q0_full, "fail_info": last_info}

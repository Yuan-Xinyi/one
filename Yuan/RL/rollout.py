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

import Yuan.RL.config as cfg
from Yuan.RL.controller import DLSController


# ---------------- target rotmat from (d, n) ----------------
def build_target_rotmat(d: np.ndarray, n: np.ndarray) -> np.ndarray:
    """TCP_z = n, TCP_x = d (re-orthogonalised), TCP_y = z x x. Assumes d _|_ n."""
    z = n / (np.linalg.norm(n) + 1e-12)
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
            theta_max: float = cfg.THETA_MAX) -> dict:
    chain = arm._chain
    R_tgt = build_target_rotmat(d, n)
    p0 = p0.astype(np.float32)
    d = d.astype(np.float32)

    # ----- Phase A: IK project the seed onto (p0, R_tgt) -----
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

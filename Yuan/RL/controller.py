"""DLS-pseudoinverse Cartesian controller for path tracking.

One control step: given current q, a Cartesian reference (p_ref, R_tgt) and
a feedforward linear velocity p_dot_ff, compute joint velocity via

    q_dot = J_lambda^+ ( [p_dot_ff + Kp*(p_ref - p_tcp);
                          K_omega * (z_tcp x z_tgt)] )
            + (I - J_lambda^+ J) * k_null * (q_ref - q)

with damped-least-squares pseudoinverse  J_lambda^+ = J^T (J J^T + lam^2 I)^-1.

Joint velocities are clipped to FR3 vendor limits, then integrated with dt;
joint positions are clipped to limits afterwards.

Termination is decided OUTSIDE the controller (in rollout.py): the controller
just reports the realised pose error so the rollout can decide.
"""
from __future__ import annotations
import numpy as np

import one.utils.math as oum
import Yuan.RL.config as cfg


class DLSController:

    def __init__(self, arm,
                 kp_lin: float = cfg.KP_LIN,
                 k_omega: float = cfg.KOMEGA,
                 dls_lambda: float = cfg.DLS_LAMBDA,
                 k_null: float = cfg.K_NULL,
                 qdot_max: np.ndarray | None = None):
        self.arm = arm
        self.solver = arm._solver           # provides _forward (FK + Jacobian)
        self.chain = arm._chain
        self.kp_lin = float(kp_lin)
        self.k_omega = float(k_omega)
        self.lam2 = float(dls_lambda) ** 2
        self.k_null = float(k_null)
        self.qdot_max = (cfg.QDOT_MAX if qdot_max is None
                         else np.asarray(qdot_max, dtype=np.float32))
        # cache local flange/TCP transform
        flange_tf = (arm._loc_flange_tf @ arm._loc_tcp_tf).astype(np.float32)
        self.flange_R = flange_tf[:3, :3].copy()
        self.flange_p = flange_tf[:3, 3].copy()      # last_lnk -> TCP, in last_lnk frame
        self.lmt_lo = self.chain.lmt_lo.astype(np.float32)
        self.lmt_up = self.chain.lmt_up.astype(np.float32)
        self.ndof = int(self.chain.n_active_jnts)
        # default null-space attractor: middle of joint limits
        self.q_ref = (0.5 * (self.lmt_lo + self.lmt_up)).astype(np.float32)

    # ---------------- forward kinematics with Jacobian at TCP ----------------
    def fk_with_jac(self, q_active: np.ndarray):
        root_tf = oum.tf_from_rotmat_pos(self.arm.rotmat, self.arm.pos)
        # local_point shifts the Jacobian linear part to the TCP position
        wd_p_tcp, jac_at_tcp, lastlnk_tf = self.solver._forward(
            q_active.astype(np.float32), root_tf, local_point=self.flange_p)
        R_tcp = lastlnk_tf[:3, :3] @ self.flange_R
        return wd_p_tcp, R_tcp, jac_at_tcp

    # ---------------- one control step ----------------
    def step(self, q_active: np.ndarray,
             p_ref: np.ndarray, R_tgt: np.ndarray,
             p_dot_ff: np.ndarray, dt: float = cfg.DT) -> dict:
        p_tcp, R_tcp, J = self.fk_with_jac(q_active)

        # --- Cartesian errors ---
        e_p = (p_ref - p_tcp).astype(np.float32)
        z_cur = R_tcp[:, 2]
        z_tgt = R_tgt[:, 2]
        cross = np.cross(z_cur, z_tgt)
        cos_th = float(np.clip(z_cur @ z_tgt, -1.0, 1.0))
        theta = float(np.arccos(cos_th))
        sin_th = float(np.linalg.norm(cross))
        if sin_th > 1e-6:
            omega_err_rotvec = (cross / sin_th) * theta   # axis*angle
        else:
            omega_err_rotvec = np.zeros(3, dtype=np.float32)

        v_cmd = p_dot_ff + self.kp_lin * e_p
        omega_cmd = self.k_omega * omega_err_rotvec
        x_dot = np.concatenate([v_cmd, omega_cmd]).astype(np.float32)  # (6,)

        # --- DLS pseudoinverse ---
        JJt = J @ J.T                                               # (6,6)
        A = JJt + self.lam2 * np.eye(6, dtype=np.float32)
        Jpinv = J.T @ np.linalg.inv(A)                              # (n,6)
        q_dot_task = Jpinv @ x_dot                                  # (n,)

        # --- null-space term: pull toward q_ref ---
        if self.k_null > 0.0:
            q_dot_secondary = self.k_null * (self.q_ref - q_active)
            N = np.eye(self.ndof, dtype=np.float32) - Jpinv @ J
            q_dot = q_dot_task + N @ q_dot_secondary
        else:
            q_dot = q_dot_task

        # --- velocity limits (per-joint clamp, preserves direction less but
        #     keeps things simple; alternative: scale uniformly to fit) ---
        sat = np.any(np.abs(q_dot) > self.qdot_max)
        q_dot = np.clip(q_dot, -self.qdot_max, self.qdot_max)

        q_new = q_active + q_dot * float(dt)
        # joint limit clamp (we report violation in info, rollout decides)
        out_of_limit = ((q_new < self.lmt_lo - 1e-6).any()
                        or (q_new > self.lmt_up + 1e-6).any())
        q_new = np.clip(q_new, self.lmt_lo, self.lmt_up).astype(np.float32)

        # post-step FK so the rollout can check the *realised* tracking error
        p_tcp_new, R_tcp_new, _ = self.fk_with_jac(q_new)
        z_new = R_tcp_new[:, 2]
        cos_post = float(np.clip(z_new @ z_tgt, -1.0, 1.0))
        post_orient = float(np.arccos(cos_post))

        return {
            "q_new": q_new,
            "p_tcp": p_tcp,
            "R_tcp": R_tcp,
            "p_tcp_new": p_tcp_new,
            "R_tcp_new": R_tcp_new,
            "q_dot": q_dot,
            "pre_pos_err":  float(np.linalg.norm(e_p)),    # input error
            "pre_orient_err": theta,
            # post-step (realised) errors used by the rollout for termination
            "pos_err":    float(np.linalg.norm(p_ref - p_tcp_new)),
            "orient_err": post_orient,
            "vel_saturated": bool(sat),
            "joint_limit_hit": bool(out_of_limit),
        }

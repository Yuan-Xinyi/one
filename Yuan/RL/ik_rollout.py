"""Rollout a straight-line Cartesian path through warm-started numerical IK.

Bypasses ``arm.ik_tcp_nearest`` (which is the CVT-seeded SELIKSolver and only
uses ref_qs to *rank* database candidates) so that q_seed is the actual
Newton initial guess. This gives the RL policy a continuous mapping
q_seed -> q_0.

Steps:
  1. Solve IK at p_0 with target frame R*=R(d,n), seeded by q_seed.
  2. March along d by PATH_STEP, warm-starting from the previous q.
  3. Stop at the first step where IK fails (numik flags joint-limit
     violations as non-convergence).

The IK target rotmat is built so that TCP_z = n, TCP_x = d (assumes d _|_ n).
"""
from __future__ import annotations
import numpy as np
import one.utils.math as oum
import Yuan.RL.config as cfg


def build_target_rotmat(d: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Construct a 3x3 rotation with z=n, x=d, y=z x x. Assumes d _|_ n."""
    z = n / (np.linalg.norm(n) + 1e-12)
    x = d - z * (d @ z)               # re-orthogonalize d w.r.t. z
    x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    R = np.empty((3, 3), dtype=np.float32)
    R[:, 0] = x
    R[:, 1] = y
    R[:, 2] = z
    return R


def _solve_with_seed(arm, R_tgt: np.ndarray, p_tgt: np.ndarray,
                     qs_init: np.ndarray, max_iter: int = 50):
    """Newton-style IK via NumIKSolver._backward; returns q (active) or None.

    Bypasses SELIKSolver: qs_init is fed directly as the iteration init.
    Account for the chain root pose so this works even if the arm is mounted.
    """
    solver = arm._solver               # SELIKSolver inherits NumIKSolver
    chain = arm._chain
    # arm.gl_lnk_tfarr[base_lidx] is the chain's root in world coords; use
    # arm.rotmat / arm.pos directly since FR3 lives at the world origin by
    # default (the same shortcut ik_tcp() uses).
    qs_init = np.asarray(qs_init, dtype=np.float32)
    if qs_init.shape[0] == arm.qs.shape[0]:
        qs_init = qs_init[chain.active_mask]
    qs, info = solver._backward(
        arm.rotmat, arm.pos,
        R_tgt.astype(np.float32), p_tgt.astype(np.float32),
        qs_active_init=qs_init,
        max_iter=max_iter)
    if not info["converged"]:
        return None, info
    return chain.embed_active_qs(qs, arm.qs), info


def rollout(arm,
            q_seed: np.ndarray,
            p0: np.ndarray,
            d: np.ndarray,
            n: np.ndarray,
            max_steps: int = cfg.MAX_STEPS,
            step_size: float = cfg.PATH_STEP) -> dict:
    """Run an IK rollout. Returns a dict with:
       length     : int, number of successful path increments (0..max_steps)
       success    : bool, whether the full path was completed
       reason     : str, termination reason
       q_traj     : list[np.ndarray], realized joint trajectory (incl. q_0)
       qs0        : np.ndarray | None, configuration at p_0 (None if init IK failed)
    """
    R_tgt = build_target_rotmat(d, n)
    flange_local = arm._loc_flange_tf @ arm._loc_tcp_tf
    # what numik solves for is the LAST link's pose, not the TCP. Convert.
    R_lastlnk = R_tgt @ flange_local[:3, :3].T
    p_offset_local = flange_local[:3, 3]   # last_lnk -> TCP offset, in last_lnk frame

    def _solve_step(p_tcp, qs_init):
        # last-link target: position is TCP target minus the rotated offset
        p_lastlnk = p_tcp - R_lastlnk @ p_offset_local
        return _solve_with_seed(arm, R_lastlnk, p_lastlnk, qs_init)

    q0, info0 = _solve_step(p0.astype(np.float32), q_seed)
    if q0 is None:
        return {"length": 0, "success": False, "reason": "init_ik_fail",
                "q_traj": [], "qs0": None,
                "init_info": info0}

    q_traj = [q0]
    q = q0
    for t in range(1, max_steps + 1):
        p_t = (p0 + t * step_size * d).astype(np.float32)
        q_new, info_t = _solve_step(p_t, q)
        if q_new is None:
            return {"length": t - 1, "success": False, "reason": "ik_fail",
                    "q_traj": q_traj, "qs0": q0,
                    "fail_info": info_t}
        q_traj.append(q_new)
        q = q_new

    return {"length": max_steps, "success": True, "reason": "max_steps",
            "q_traj": q_traj, "qs0": q0}

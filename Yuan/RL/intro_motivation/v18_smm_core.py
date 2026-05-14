"""SMM 1D branch enumeration core (Guri & Kantor 2025, arXiv:2507.21957).

Pure-numerics module: no GUI, no plotting, no I/O. Imported by the task
script and by experiment scripts that need to compute SMM topology.

  fk_J(kin, q)                      — FK + Jacobian (numpy at single q)
  null_vec(J)                       — unit null vector + sigma_min
  rotvec_R(R_cur, R_tgt)            — rotation vector R_tgt R_cur^T
  newton_project(kin, q, p_tgt,     — refine q onto SMM (6-DOF)
                  R_tgt, lo, hi)
  walk_branch(kin, q0, ..., h)      — RK4 forward+backward ODE walk
  project_and_filter(kin, Q_seed,   — Newton-refine + JL filter + dedup
                      p_tgt, R_tgt, lo, hi)
  enumerate_branches(kin, Q, p_tgt, — group candidates into branches via
                      R_tgt, h)       ODE walks
  get_task_target_pose(seed, kin,   — build (p_tgt, R_tgt) at v18 line
                       rng)            task start (for the SMM)

The enumeration is exact for r=1 SMM = 6-DOF target pose on a 7-DOF arm.
"""
from __future__ import annotations

import numpy as np
import torch

from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.RL.batched_rollout import _batched_ik_project
from Yuan.RL.v18_curve_eval import sample_curve_task
from Yuan.RL.v18_data_prep import _build_R_from_normal_direction


# ---- task knobs (line task built at a v18 seed) ----
N_CHECKPOINTS = 5
LINE_L_RANGE = (0.30, 0.40)
TARGET_PATH_M = 1.5    # extend past reach so every rollout fails partway


# ---- SMM enumeration knobs ----
NEWTON_TOL = 1e-6
NEWTON_MAX_ITERS = 30
JOINT_MARGIN = 0.05        # candidates closer than this to a limit are dropped
SIGMA_FLOOR = 1e-3         # walk stops below this σ_min(J)
DEDUP_RAD = 0.08           # rad; two projected candidates within this are merged
DEFAULT_H = 0.03
CLOSE_TOL_MULT = 4.0       # closure tolerance = CLOSE_TOL_MULT * h
CLOSE_MIN_ARC = 0.5        # rad of walking before closure check enables
MEMBER_EPS_MULT = 2.0      # candidate within MEMBER_EPS_MULT * h of curve = member


# ---- task path helpers ----
def as_tensor(x, device):
    return torch.as_tensor(x, device=device, dtype=torch.float32)


def path_length(path: np.ndarray) -> float:
    return float(np.linalg.norm(path[1:] - path[:-1], axis=1).sum())


def sample_line_task(rng: np.random.Generator,
                     kin: BatchedFR3Kinematics,
                     l_range: tuple[float, float] | None = None) -> dict:
    """Rejection-sample a feasible straight-line task. None → use sample_curve_task default."""
    for _ in range(100):
        if l_range is None:
            task = sample_curve_task(rng, kin, 'line', N_CHECKPOINTS)
        else:
            task = sample_curve_task(rng, kin, 'line', N_CHECKPOINTS, L_range=l_range)
        if task is not None:
            return task
    raise RuntimeError('failed to sample a feasible straight-line task')


def sample_free_line_task(rng: np.random.Generator,
                           kin: BatchedFR3Kinematics,
                           l_range: tuple[float, float] = LINE_L_RANGE,
                           max_tries: int = 200,
                           n_ik_seeds: int = 32) -> dict:
    """Permissive line task: p0 anywhere in FR3-reachable C-space (radial
    box [0.20, 0.85] m, z > 0.02 m). plane_normal sampled uniformly on the
    full sphere — pen can point in ANY direction (down/side/up/back). The
    only feasibility check is that the start pose (p0, R built from
    plane_normal + tangent) admits at least one IK solution."""
    device = kin.device
    lo = kin.lmt_lo.cpu().numpy()[None, :]
    hi = kin.lmt_up.cpu().numpy()[None, :]
    for _ in range(max_tries):
        # plane_normal uniform on S^2
        u = rng.normal(size=3)
        u = (u / max(np.linalg.norm(u), 1e-12)).astype(np.float32)
        plane_normal = u
        # tangent ⊥ normal: take a random hint and orthogonalize
        hint = rng.normal(size=3)
        e_axis = hint - hint.dot(plane_normal) * plane_normal
        e_axis_n = float(np.linalg.norm(e_axis))
        if e_axis_n < 1e-6:
            continue
        e_axis = (e_axis / e_axis_n).astype(np.float32)
        # p0 in a generous box, filter by FR3 reach
        p0 = rng.uniform(np.array([-0.85, -0.85, 0.02], dtype=np.float32),
                          np.array([ 0.85,  0.85, 1.00], dtype=np.float32)).astype(np.float32)
        r = float(np.linalg.norm(p0))
        if not (0.20 < r < 0.85):
            continue
        # Path stays above ground and within reach
        L = float(rng.uniform(*l_range))
        n_pts = 120
        ts = np.linspace(0.0, L, n_pts, dtype=np.float32)
        fine = p0[None, :] + ts[:, None] * e_axis[None, :]
        if (fine[:, 2] < 0.02).any():
            continue
        norms = np.linalg.norm(fine, axis=1)
        if norms.max() > 0.90 or norms.min() < 0.15:
            continue
        # IK feasibility at the start pose
        R0 = _build_R_from_normal_direction(plane_normal, e_axis).astype(np.float32)
        p0_t = torch.as_tensor(p0, device=device, dtype=torch.float32)
        R0_t = torch.as_tensor(R0, device=device, dtype=torch.float32)
        seeds_np = rng.uniform(lo, hi, size=(n_ik_seeds, 7)).astype(np.float32)
        q_seed = torch.as_tensor(seeds_np, device=device, dtype=torch.float32)
        p_rep = p0_t.unsqueeze(0).expand(n_ik_seeds, 3)
        R_rep = R0_t.unsqueeze(0).expand(n_ik_seeds, 3, 3)
        _, ok, _ = _batched_ik_project(kin, q_seed, p_rep, R_rep,
                                        branch_action=None)
        if not bool(ok.any().item()):
            continue
        return {'fine_path_pts': fine, 'plane_normal': plane_normal}
    raise RuntimeError(f'sample_free_line_task: rejected {max_tries} draws — '
                        f'all infeasible at start IK')


def extend_task_path(task: dict, target_L: float) -> dict:
    """Stretch a feasible task's fine_path_pts so the total length equals target_L."""
    fine = task['fine_path_pts'].copy()
    p0 = fine[0]; p1 = fine[-1]
    seg_vec = p1 - p0
    current_L = float(np.linalg.norm(seg_vec))
    if current_L < 1e-6 or target_L <= current_L:
        return task
    direction = seg_vec / current_L
    n_pts = max(fine.shape[0], int(round(120 * target_L / current_L)))
    ts = np.linspace(0.0, target_L, n_pts, dtype=np.float32)
    new_fine = p0[None, :] + ts[:, None] * direction[None, :]
    out = dict(task)
    out['fine_path_pts'] = new_fine.astype(np.float32)
    return out


# ---- single-q numerics ----
def fk_J(kin: BatchedFR3Kinematics, q_np: np.ndarray):
    q_t = torch.as_tensor(q_np, device=kin.device, dtype=torch.float32).unsqueeze(0)
    p, R, J, _ = kin.tcp_fk_jac(q_t)
    return (p[0].detach().cpu().numpy(),
            R[0].detach().cpu().numpy(),
            J[0].detach().cpu().numpy())


def null_vec(J_np: np.ndarray) -> tuple[np.ndarray, float]:
    _, S, Vt = np.linalg.svd(J_np, full_matrices=True)
    return Vt[-1], float(S[-1])


def rotvec_R(R_cur: np.ndarray, R_tgt: np.ndarray) -> np.ndarray:
    R_err = R_tgt @ R_cur.T
    cos_th = np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0)
    th = float(np.arccos(cos_th))
    if abs(th) < 1e-8:
        return np.zeros(3)
    s = 2.0 * np.sin(th) + 1e-12
    axis = np.array([R_err[2, 1] - R_err[1, 2],
                     R_err[0, 2] - R_err[2, 0],
                     R_err[1, 0] - R_err[0, 1]])
    return axis * (th / s)


def newton_project(kin, q, p_tgt, R_tgt, lo, hi,
                   max_iter=NEWTON_MAX_ITERS, tol=NEWTON_TOL):
    q = q.copy()
    for _ in range(max_iter):
        p, R, J = fk_J(kin, q)
        e = np.concatenate([p_tgt - p, rotvec_R(R, R_tgt)])
        err = float(np.linalg.norm(e))
        if err < tol:
            return q, True, err
        dq = np.linalg.pinv(J) @ e
        step_norm = float(np.linalg.norm(dq))
        if step_norm > 0.3:
            dq *= 0.3 / step_norm
        q = q + dq
        q = np.clip(q, lo, hi)
    p, R, _ = fk_J(kin, q)
    e = np.concatenate([p_tgt - p, rotvec_R(R, R_tgt)])
    err = float(np.linalg.norm(e))
    return q, err < 10.0 * tol, err


# ---- ODE walk ----
def _walk_null(kin, q0, p_tgt, R_tgt, lo, hi,
               direction: float, h: float,
               max_steps: int, close_tol: float, close_min_step: int):
    """RK4 + Newton corrector. direction=±1 picks sign of first null vector."""
    _, _, J0 = fk_J(kin, q0)
    n0, s0 = null_vec(J0)
    if s0 < SIGMA_FLOOR:
        return np.array([q0]), False, 'start_singular'
    n_prev = direction * n0

    traj = [q0.copy()]
    for step in range(max_steps):
        q = traj[-1]

        def vel(qx, ref):
            _, _, J = fk_J(kin, qx)
            n, sigma = null_vec(J)
            if sigma < SIGMA_FLOOR:
                return None
            if float(np.dot(n, ref)) < 0:
                n = -n
            return n

        k1 = vel(q, n_prev)
        if k1 is None: return np.array(traj), False, 'singular'
        k2 = vel(q + 0.5 * h * k1, k1)
        if k2 is None: return np.array(traj), False, 'singular'
        k3 = vel(q + 0.5 * h * k2, k2)
        if k3 is None: return np.array(traj), False, 'singular'
        k4 = vel(q + h * k3, k3)
        if k4 is None: return np.array(traj), False, 'singular'
        q_new = q + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if np.any(q_new < lo) or np.any(q_new > hi):
            return np.array(traj), False, 'joint_limit'
        q_new, ok, err = newton_project(kin, q_new, p_tgt, R_tgt, lo, hi)
        if not ok:
            return np.array(traj), False, f'project_fail(err={err:.2e})'
        if np.any(q_new < lo + 1e-4) or np.any(q_new > hi - 1e-4):
            return np.array(traj), False, 'joint_limit_after_project'
        traj.append(q_new)
        n_prev = k4
        if step > close_min_step and np.linalg.norm(q_new - traj[0]) < close_tol:
            return np.array(traj), True, 'closed'
    return np.array(traj), False, 'max_steps'


def walk_branch(kin, q0, p_tgt, R_tgt, lo, hi, h):
    """Bi-directional ODE walk: forward, then backward if not closed.
    Returns (traj, closed, reason)."""
    close_tol = CLOSE_TOL_MULT * h
    close_min_step = max(20, int(CLOSE_MIN_ARC / h))
    max_steps = max(2000, int(20.0 / h))
    fwd, closed, why_fwd = _walk_null(kin, q0, p_tgt, R_tgt, lo, hi,
                                      +1.0, h, max_steps, close_tol, close_min_step)
    if closed:
        return fwd, True, why_fwd
    bwd, _, why_bwd = _walk_null(kin, q0, p_tgt, R_tgt, lo, hi,
                                 -1.0, h, max_steps, close_tol, close_min_step)
    full = (np.concatenate([bwd[::-1], fwd[1:]], axis=0)
            if fwd.shape[0] > 1 else bwd[::-1])
    return full, False, f'fwd:{why_fwd}|bwd:{why_bwd}'


# ---- candidate pool + branch grouping ----
def project_and_filter(kin, Q_seed, p_tgt, R_tgt, lo, hi,
                        joint_margin=JOINT_MARGIN, dedup_rad=DEDUP_RAD,
                        verbose: bool = True):
    """Newton-refine seeds, drop non-converged/near-limit/duplicates."""
    Q_clean = []
    n_no_converge = 0
    n_near_limit = 0
    n_dup = 0
    for i in range(Q_seed.shape[0]):
        q, ok, _err = newton_project(kin, Q_seed[i], p_tgt, R_tgt, lo, hi)
        if not ok:
            n_no_converge += 1
            continue
        margin = float(np.min(np.minimum(q - lo, hi - q)))
        if margin < joint_margin:
            n_near_limit += 1
            continue
        is_dup = False
        for q_e in Q_clean:
            if np.linalg.norm(q - q_e) < dedup_rad:
                is_dup = True
                break
        if is_dup:
            n_dup += 1
            continue
        Q_clean.append(q)
    if verbose:
        print(f'    filter: no_converge={n_no_converge}, '
              f'near_limit(<{joint_margin})={n_near_limit}, '
              f'dup(<{dedup_rad})={n_dup}')
    return np.array(Q_clean) if Q_clean else np.zeros((0, 7))


def enumerate_branches(kin, Q, p_tgt, R_tgt, h=DEFAULT_H):
    """For each unassigned q in Q, walk its branch and assign nearby
    candidates as members. Returns (branches list, assigned int array)."""
    lo = kin.lmt_lo.detach().cpu().numpy()
    hi = kin.lmt_up.detach().cpu().numpy()
    N = Q.shape[0]
    assigned = -np.ones(N, dtype=int)
    branches = []
    eps = MEMBER_EPS_MULT * h
    for i in range(N):
        if assigned[i] >= 0:
            continue
        traj, closed, why = walk_branch(kin, Q[i], p_tgt, R_tgt, lo, hi, h)
        bid = len(branches)
        branches.append({'traj': traj, 'closed': closed, 'reason': why,
                          'seed_idx': i})
        for j in range(N):
            if assigned[j] >= 0:
                continue
            d = np.linalg.norm(traj - Q[j][None, :], axis=1).min()
            if d < eps:
                assigned[j] = bid
        assigned[i] = bid
    return branches, assigned


# ---- task-pose helper ----
def get_task_target_pose(seed: int, kin: BatchedFR3Kinematics,
                          rng: np.random.Generator,
                          free: bool = False):
    """Build the 6-DOF target pose at the start of a line task.
       free=False (default): v18's constrained sampler (pen-down on tilted plane).
       free=True: permissive sampler (any direction, anywhere reachable).
    Returns (p_tgt, R_tgt, task_dict)."""
    if free:
        task = sample_free_line_task(rng, kin, l_range=LINE_L_RANGE)
    else:
        task = sample_line_task(rng, kin, l_range=LINE_L_RANGE)
    task = extend_task_path(task, TARGET_PATH_M)
    path = task['fine_path_pts']
    tangent = path[1] - path[0]
    tangent /= max(np.linalg.norm(tangent), 1e-12)
    R = _build_R_from_normal_direction(task['plane_normal'], tangent)
    return path[0].astype(np.float32), R.astype(np.float32), task

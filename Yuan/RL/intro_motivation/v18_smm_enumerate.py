"""SMM branch enumeration at a fixed 6-DOF target pose (Guri & Kantor 2025).

The paper's SMM-IVP algorithm is exact for 1D self-motion manifolds, i.e.
r = n - m = 1. For a 7-DOF arm this means a 6-DOF (fully-constrained)
target pose. v18's task is 5-DOF (z-axis only, r=2) and would need a
mesh-based extension; we deliberately stick to the 1D case here.

Pipeline:
  1. Pick a target pose (p_tgt, R_tgt). Two modes:
       --mode simple : hand-picked pose in the FR3 workspace interior
       --mode task   : full pose at the start of a v18 line task
  2. Sample IK candidates via mixed seeding (`_dense_ik_at`).
  3. Newton-refine every candidate to tight tolerance (1e-6) and drop
     ones that don't converge or sit within JOINT_MARGIN of a limit.
  4. For each unassigned candidate q, RK4 walk null(J(q)) forward then
     backward until closed loop / joint limit / singularity. Sign of
     n̂(q) chosen by inner product with previous step.
  5. Member assignment: candidates within EPS of any walked curve point
     are marked as members of that branch.

Output:
  * stdout: per-branch summary (closed/open, arc length, members)
  * jsonl: meta + subsampled trajectories
  * PNG: 2D PCA visualization of all branches and member candidates

Usage:
    python -m Yuan.RL.intro_motivation.v18_smm_enumerate
    python -m Yuan.RL.intro_motivation.v18_smm_enumerate --mode task --seed 118
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.RL.intro_motivation.v18_motivation_core import (
    LINE_L_RANGE,
    TARGET_PATH_M,
    extend_task_path,
    sample_line_task,
)
from Yuan.RL.batched_rollout import _branch_seed_bank
from Yuan.RL.v18_data_prep import _build_R_from_normal_direction, _dense_ik_at


# ---- knobs ----
NEWTON_TOL = 1e-6
NEWTON_MAX_ITERS = 30
JOINT_MARGIN = 0.05        # candidates closer than this to a limit are dropped
SIGMA_FLOOR = 1e-3         # walk stops below this σ_min(J)
DEDUP_RAD = 0.08           # rad; two projected candidates within this are merged
DEFAULT_H = 0.03
CLOSE_TOL_MULT = 4.0       # closure tolerance = CLOSE_TOL_MULT * h
CLOSE_MIN_ARC = 0.5        # rad; need this much walking before checking closure
MEMBER_EPS_MULT = 2.0      # candidate within MEMBER_EPS_MULT * h of curve = member


# ---- numerics ----
def fk_J(kin, q_np):
    q_t = torch.as_tensor(q_np, device=kin.device, dtype=torch.float32).unsqueeze(0)
    p, R, J, _ = kin.tcp_fk_jac(q_t)
    return (p[0].detach().cpu().numpy(),
            R[0].detach().cpu().numpy(),
            J[0].detach().cpu().numpy())


def null_vec(J_np):
    _, S, Vt = np.linalg.svd(J_np, full_matrices=True)
    return Vt[-1], float(S[-1])


def rotvec_R(R_cur, R_tgt):
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
        # cap step to keep Newton stable
        step_norm = float(np.linalg.norm(dq))
        if step_norm > 0.3:
            dq *= 0.3 / step_norm
        q = q + dq
        q = np.clip(q, lo, hi)
    p, R, _ = fk_J(kin, q)
    e = np.concatenate([p_tgt - p, rotvec_R(R, R_tgt)])
    err = float(np.linalg.norm(e))
    return q, err < 10.0 * tol, err


def walk_null(kin, q0, p_tgt, R_tgt, lo, hi,
              direction: float, h: float,
              max_steps: int, close_tol: float, close_min_step: int):
    """Forward RK4 with predictor-corrector. direction = ±1 to pick which
    side of the kernel to walk on the first step."""
    # initial direction
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
    close_tol = CLOSE_TOL_MULT * h
    close_min_step = max(20, int(CLOSE_MIN_ARC / h))
    max_steps = max(2000, int(20.0 / h))
    fwd, closed, why_fwd = walk_null(kin, q0, p_tgt, R_tgt, lo, hi,
                                     +1.0, h, max_steps, close_tol, close_min_step)
    if closed:
        return fwd, True, why_fwd
    bwd, _, why_bwd = walk_null(kin, q0, p_tgt, R_tgt, lo, hi,
                                -1.0, h, max_steps, close_tol, close_min_step)
    # bwd starts at q0; concatenate as reversed bwd + fwd[1:]
    full = np.concatenate([bwd[::-1], fwd[1:]], axis=0) if fwd.shape[0] > 1 else bwd[::-1]
    return full, False, f'fwd:{why_fwd}|bwd:{why_bwd}'


def project_and_filter(kin, Q_seed, p_tgt, R_tgt, lo, hi,
                       joint_margin=JOINT_MARGIN, dedup_rad=DEDUP_RAD):
    """Newton-refine every seed, drop non-converged + near-limit + duplicates."""
    Q_clean = []
    n_no_converge = 0
    n_near_limit = 0
    n_dup = 0
    for i in range(Q_seed.shape[0]):
        q, ok, err = newton_project(kin, Q_seed[i], p_tgt, R_tgt, lo, hi)
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
    print(f'    filter breakdown: no_converge={n_no_converge}, '
          f'near_limit(<{joint_margin})={n_near_limit}, '
          f'dup(<{dedup_rad})={n_dup}')
    return np.array(Q_clean) if Q_clean else np.zeros((0, 7))


def enumerate_branches(kin, Q, p_tgt, R_tgt, h):
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


def plot_branches(branches, Q, assigned, out_png):
    all_pts = np.concatenate([b['traj'] for b in branches] + [Q], axis=0)
    mu = all_pts.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(all_pts - mu, full_matrices=False)
    W = Vt[:2].T

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap('tab10')
    for bid, b in enumerate(branches):
        traj_2d = (b['traj'] - mu) @ W
        n_m = int((assigned == bid).sum())
        ax.plot(traj_2d[:, 0], traj_2d[:, 1], '-',
                color=cmap(bid % 10), alpha=0.6, linewidth=1.5,
                label=f"branch {bid} ({'closed' if b['closed'] else 'open'}, "
                      f"T={b['traj'].shape[0]}, {n_m} members)")
        ax.scatter(traj_2d[0, 0], traj_2d[0, 1], s=80, c=[cmap(bid % 10)],
                   edgecolors='black', linewidth=1.0, marker='*', zorder=6)
    Q_2d = (Q - mu) @ W
    for j in range(Q.shape[0]):
        bid = assigned[j]
        c = cmap(bid % 10) if bid >= 0 else 'gray'
        ax.scatter(Q_2d[j, 0], Q_2d[j, 1], s=35, c=[c],
                   edgecolors='black', linewidth=0.5, zorder=5)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(f'SMM branches in 7-DOF joint space (PCA → 2D)\n'
                 f'{len(branches)} branches, {Q.shape[0]} candidates')
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)


def get_simple_target_pose():
    """A clean test pose: TCP in front of robot, pointing down (-z).
    Workspace interior, no obvious singularity."""
    p = np.array([0.45, 0.0, 0.40], dtype=np.float32)
    # R such that z_tool points -z_world (typical pen-down pose)
    R = np.array([[1.0, 0.0,  0.0],
                  [0.0, -1.0, 0.0],
                  [0.0, 0.0, -1.0]], dtype=np.float32)
    return p, R, 'simple test pose: TCP=(0.45,0,0.40), z_tool=-z_world'


def get_task_target_pose(seed, kin, rng):
    """Build target pose at the start of a v18 line task (matches branch_comparison)."""
    task = sample_line_task(rng, kin, l_range=LINE_L_RANGE)
    task = extend_task_path(task, TARGET_PATH_M)
    path = task['fine_path_pts']
    tangent = path[1] - path[0]
    tangent /= max(np.linalg.norm(tangent), 1e-12)
    R = _build_R_from_normal_direction(task['plane_normal'], tangent)
    return path[0].astype(np.float32), R.astype(np.float32), f'task start (seed={seed})'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['simple', 'task'], default='simple')
    parser.add_argument('--seed', type=int, default=118,
                        help='used by --mode task and by IK candidate sampler')
    parser.add_argument('--n-seeds', type=int, default=128)
    parser.add_argument('--h', type=float, default=DEFAULT_H)
    parser.add_argument('--joint-margin', type=float, default=JOINT_MARGIN,
                        help='drop candidates within this rad of any joint limit')
    parser.add_argument('--dedup-rad', type=float, default=DEDUP_RAD,
                        help='merge candidates within this rad after projection')
    parser.add_argument('--out-png', type=str,
                        default='Yuan/RL/intro_motivation/data/smm_branches.png')
    parser.add_argument('--out-jsonl', type=str,
                        default='Yuan/RL/intro_motivation/data/smm_branches.jsonl')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kin = BatchedFR3Kinematics(device=device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    if args.mode == 'simple':
        p, R, label = get_simple_target_pose()
    else:
        p, R, label = get_task_target_pose(args.seed, kin, rng)
    print(f'target pose: {label}')
    print(f'  p_tgt = {p}')
    print(f'  R_tgt[:,2] (z) = {R[:,2]}')

    p_t = torch.as_tensor(p, device=device, dtype=torch.float32)
    R_t = torch.as_tensor(R, device=device, dtype=torch.float32)
    # Augment _dense_ik_at's random seeds with the hand-curated branch
    # seed bank — it includes wrist-flipped and back-reach postures that
    # uniform random sampling rarely hits.
    extra_bank = _branch_seed_bank(kin).detach().cpu().numpy()
    Q_seed_t, _ = _dense_ik_at(kin, p_t, R_t, args.n_seeds, rng,
                               extra_seeds=extra_bank)
    if Q_seed_t.shape[0] == 0:
        raise RuntimeError('no IK candidates found at target pose')
    Q_seed = Q_seed_t.detach().cpu().numpy()
    print(f'  raw IK candidates: {Q_seed.shape[0]} '
          f'(incl. {extra_bank.shape[0]} curated bank seeds)')

    lo = kin.lmt_lo.detach().cpu().numpy()
    hi = kin.lmt_up.detach().cpu().numpy()
    Q = project_and_filter(kin, Q_seed, p, R, lo, hi,
                            joint_margin=args.joint_margin,
                            dedup_rad=args.dedup_rad)
    print(f'  after Newton projection (tol={NEWTON_TOL}), JL margin filter '
          f'(>{args.joint_margin}), dedup (>{args.dedup_rad}): '
          f'{Q.shape[0]} candidates')
    if Q.shape[0] == 0:
        raise RuntimeError('no valid candidates after filtering')

    branches, assigned = enumerate_branches(kin, Q, p, R, args.h)

    print(f'\n=== {len(branches)} SMM branches ===')
    for bid, b in enumerate(branches):
        n_m = int((assigned == bid).sum())
        arc = float(np.sum(np.linalg.norm(np.diff(b['traj'], axis=0), axis=1)))
        sig = tuple(int(np.sign(b['traj'][0][k])) for k in (0, 3, 5))
        status = 'CLOSED' if b['closed'] else f"OPEN ({b['reason']})"
        print(f'  branch {bid}: {status}, T={b["traj"].shape[0]}, '
              f'arc={arc:.2f} rad, members={n_m}/{Q.shape[0]}, '
              f'seed_q signature={sig}')

    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, 'w') as f:
        f.write(json.dumps({
            'type': 'meta', 'mode': args.mode, 'seed': args.seed,
            'p_tgt': p.tolist(), 'R_tgt': R.tolist(),
            'h': args.h, 'n_candidates': int(Q.shape[0]),
            'n_branches': len(branches),
        }) + '\n')
        for bid, b in enumerate(branches):
            arc = float(np.sum(np.linalg.norm(np.diff(b['traj'], axis=0), axis=1)))
            f.write(json.dumps({
                'branch_id': bid, 'closed': bool(b['closed']),
                'reason': b['reason'],
                'n_steps': int(b['traj'].shape[0]),
                'arc_length_rad': arc,
                'n_members': int((assigned == bid).sum()),
                'traj_subsampled': b['traj'][::max(1, b['traj'].shape[0] // 80)].tolist(),
            }) + '\n')

    plot_branches(branches, Q, assigned, Path(args.out_png))
    print(f'\nsaved: {args.out_jsonl}')
    print(f'saved: {args.out_png}')


if __name__ == '__main__':
    main()

"""Shared pre-compute for the four Part-1 figures in this folder.

All `fig_*.py` scripts here draw from the SAME representative line task
(seed 118 by default). To keep each figure script fast and independent
of the others, the heavy SMM enumeration and rollout work is computed
once and cached to disk; subsequent scripts load instantly.

Cache layout (npz at `cache/seed{S}_data.npz`):
  task_path       (Npts, 3) float32 — fine path points
  L_max           ()        float32 — total task length
  p_tgt           (3,)      float32
  R_tgt           (3, 3)    float32
  plane_normal    (3,)      float32
  Q_clean         (N, 7)    float32 — filtered IK candidates
  branch_traj_*   per-branch (T_b, 7) — variable length, stored as object array
  branch_closed   (B,)      bool
  q0_best         (B, 7)    float32 — best q0 per branch (by L_self)
  L_self_best     (B,)      float32 — normalized L for that q0
  q_traj_best     (T+1, B, 7) float32 — recorded rollout
  fail_step       (B,)      int32
  fail_reason     list[str] (json field)
  ee_xyz_best     (T+1, B, 3) float32 — TCP positions along rollout
  sigma_min_t     list of (T_alive,) float32 per branch
  margin_min_t    list of (T_alive,) float32 per branch
  no_jl_traj_*    per-branch no-JL re-walk trajectory (object array)
  no_jl_closed    (B,) bool
  pairwise_min_d  (B, B) float32 — min 7D distance between no-JL walks
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from Yuan.flow_connectivity.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.flow_connectivity.batched_rollout import _branch_seed_bank
from Yuan.flow_connectivity.intro_motivation.v18_smm_core import (
    DEDUP_RAD, DEFAULT_H, JOINT_MARGIN,
    as_tensor, enumerate_branches, path_length,
    project_and_filter, walk_branch,
)
from Yuan.flow_connectivity.intro_motivation.v18_smm_rollout_6dof import (
    record_rollout_6dof, rollout_lengths_6dof,
)
from Yuan.flow_connectivity.intro_motivation.v18_smm_task import (
    pick_representative_q0, sample_branch_q0s,
)
from Yuan.flow_connectivity.v18_data_prep import _dense_ik_at

# Local replacement for v18_smm_core.get_task_target_pose with a geometry
# rejection filter. See task_sampler.py.
import sys as _sys
_HERE_SHARED = Path(__file__).resolve().parent
if str(_HERE_SHARED) not in _sys.path:
    _sys.path.insert(0, str(_HERE_SHARED))
from task_sampler import get_clean_task_target_pose  # noqa: E402


DEFAULT_SEED = 17
DEFAULT_N_IK_SEEDS = 256
DEFAULT_N_PER_BRANCH = 50
CACHE_DIR = Path(__file__).resolve().parent / 'cache'
FIG_DIR = Path(__file__).resolve().parent / 'figs'
NO_JL_LIMIT_MULT = 5.0   # ± this * pi for the no-JL re-walk
DISCONNECT_EPS_MULT = 2.0  # pair_min_d < this * h ⇒ JL-cut artifact


def _ee_xyz_from_q_traj(kin: BatchedFR3Kinematics, q_traj_np: np.ndarray) -> np.ndarray:
    """q_traj_np: (T, B, 7) → ee_xyz: (T, B, 3) via tcp_fk_jac batched per step."""
    T, B, _ = q_traj_np.shape
    out = np.zeros((T, B, 3), dtype=np.float32)
    q_flat = torch.as_tensor(q_traj_np.reshape(T * B, 7),
                              device=kin.device, dtype=torch.float32)
    chunk = 4096
    for s in range(0, q_flat.shape[0], chunk):
        e = min(s + chunk, q_flat.shape[0])
        p_tcp, _, _, _ = kin.tcp_fk_jac(q_flat[s:e])
        out.reshape(T * B, 3)[s:e] = p_tcp.detach().cpu().numpy()
    return out


def _per_branch_metrics(kin: BatchedFR3Kinematics, q_traj_np: np.ndarray,
                         fail_steps: list[int]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Along alive prefix per branch: σ_min(J(t)) and min joint-limit margin(t)."""
    lo = kin.lmt_lo.detach().cpu().numpy()
    hi = kin.lmt_up.detach().cpu().numpy()
    sig_list, mgn_list = [], []
    B = q_traj_np.shape[1]
    for bid in range(B):
        T_alive = max(1, min(fail_steps[bid] + 1, q_traj_np.shape[0]))
        q_alive = q_traj_np[:T_alive, bid, :]
        q_t = torch.as_tensor(q_alive, device=kin.device, dtype=torch.float32)
        _, _, J_t, _ = kin.tcp_fk_jac(q_t)
        J_np = J_t.detach().cpu().numpy()
        sig = np.array([float(np.linalg.svd(J_np[t], compute_uv=False).min())
                        for t in range(T_alive)], dtype=np.float32)
        mgn = np.array([float(np.min(np.minimum(q_alive[t] - lo, hi - q_alive[t])))
                        for t in range(T_alive)], dtype=np.float32)
        sig_list.append(sig)
        mgn_list.append(mgn)
    return sig_list, mgn_list


def _no_jl_rewalk(kin: BatchedFR3Kinematics, branches, p_tgt, R_tgt, h):
    """Re-walk each branch's start with FR3 JL replaced by ±5π."""
    orig_lo = kin.lmt_lo.clone()
    orig_hi = kin.lmt_up.clone()
    big = float(NO_JL_LIMIT_MULT) * np.pi
    fake_lo = np.full(7, -big, dtype=np.float32)
    fake_hi = np.full(7,  big, dtype=np.float32)
    kin.lmt_lo = torch.as_tensor(fake_lo, device=kin.device, dtype=torch.float32)
    kin.lmt_up = torch.as_tensor(fake_hi, device=kin.device, dtype=torch.float32)
    trajs, closed = [], []
    for b in branches:
        q0 = b['traj'][0]
        traj, cl, _ = walk_branch(kin, q0, p_tgt, R_tgt, fake_lo, fake_hi, h)
        trajs.append(traj.astype(np.float32))
        closed.append(bool(cl))
    kin.lmt_lo = orig_lo
    kin.lmt_up = orig_hi
    return trajs, np.array(closed, dtype=bool)


def _pairwise_min_dist(trajs):
    """B x B float32 matrix of min 7D distance between no-JL walk pairs."""
    B = len(trajs)
    D = np.zeros((B, B), dtype=np.float32)
    for i in range(B):
        for j in range(B):
            if i == j:
                continue
            ti = trajs[i]; tj = trajs[j]
            d_min = float('inf')
            for q in ti:
                d = float(np.linalg.norm(tj - q[None, :], axis=1).min())
                if d < d_min:
                    d_min = d
            D[i, j] = d_min
    return D


def build_or_load(seed: int = DEFAULT_SEED,
                   n_ik_seeds: int = DEFAULT_N_IK_SEEDS,
                   n_per_branch: int = DEFAULT_N_PER_BRANCH,
                   h: float = DEFAULT_H,
                   force: bool = False,
                   verbose: bool = True) -> dict:
    """Build the cached bundle of task / branches / rollout / no-JL data.

    Returns a dict with the canonical fields. Each fig script calls this
    once and reads only the fields it needs. Cache file is at
    `25_spring_pre/cache/seed{S}_data.npz` plus `seed{S}_meta.json`.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = CACHE_DIR / f'seed{seed}_data.npz'
    meta_path = CACHE_DIR / f'seed{seed}_meta.json'

    if not force and npz_path.exists() and meta_path.exists():
        if verbose:
            print(f'  [shared] loading cache: {npz_path}')
        with np.load(npz_path, allow_pickle=True) as z:
            d = {k: z[k] for k in z.files}
        with open(meta_path) as f:
            d['meta'] = json.load(f)
        d['branches'] = [
            {'traj': d[f'branch_traj_{bid}'].astype(np.float32),
             'closed': bool(d['branch_closed'][bid])}
            for bid in range(int(d['meta']['n_branches']))
        ]
        d['no_jl_trajs'] = [d[f'no_jl_traj_{bid}'].astype(np.float32)
                             for bid in range(int(d['meta']['n_branches']))]
        d['sigma_min_t'] = [d[f'sigma_t_{bid}'] for bid in range(int(d['meta']['n_branches']))]
        d['margin_min_t'] = [d[f'margin_t_{bid}'] for bid in range(int(d['meta']['n_branches']))]
        return d

    if verbose:
        print(f'  [shared] computing fresh (seed={seed}) — this may take ~1 min')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kin = BatchedFR3Kinematics(device=device)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    p_tgt, R_tgt, task = get_clean_task_target_pose(seed, kin, rng)
    task_path = task['fine_path_pts'].astype(np.float32)
    L_max = float(path_length(task_path))
    plane_normal = task['plane_normal'].astype(np.float32)
    track_pts = as_tensor(task_path, device)
    plane_normal_t = as_tensor(plane_normal, device)

    p_t = torch.as_tensor(p_tgt, device=device, dtype=torch.float32)
    R_t = torch.as_tensor(R_tgt, device=device, dtype=torch.float32)
    extra = _branch_seed_bank(kin).detach().cpu().numpy()
    Q_seed_t, _ = _dense_ik_at(kin, p_t, R_t, n_ik_seeds, rng, extra_seeds=extra)
    lo = kin.lmt_lo.detach().cpu().numpy()
    hi = kin.lmt_up.detach().cpu().numpy()
    Q_clean = project_and_filter(kin, Q_seed_t.detach().cpu().numpy(), p_tgt, R_tgt,
                                  lo, hi, joint_margin=JOINT_MARGIN,
                                  dedup_rad=DEDUP_RAD, verbose=verbose)
    if Q_clean.shape[0] == 0:
        raise RuntimeError(f'no IK candidates after filter (seed={seed})')

    branches, assigned = enumerate_branches(kin, Q_clean, p_tgt, R_tgt, h)
    B = len(branches)
    if verbose:
        print(f'  [shared] {B} SMM branches')

    # Dense per-q0 rollout for the violin plots.
    all_q, all_bid, all_arc = sample_branch_q0s(branches, n_per_branch)
    q_batch = torch.as_tensor(all_q, device=device, dtype=torch.float32)
    L_abs_all = rollout_lengths_6dof(kin, q_batch, track_pts, plane_normal_t)
    L_rel_all = (L_abs_all / L_max).astype(np.float32)

    # Best-q0 per branch for the trajectory rollout.
    rep = pick_representative_q0(branches, kin, track_pts, plane_normal_t,
                                  L_max, mode='best')
    q0_best = np.stack([r['q0'] for r in rep], axis=0).astype(np.float32)
    L_self_best = np.array([r['L_self_norm'] for r in rep], dtype=np.float32)

    q_init = torch.as_tensor(q0_best, device=device, dtype=torch.float32)
    q_traj_t, fail_infos = record_rollout_6dof(
        kin, q_init, track_pts, plane_normal)
    q_traj_np = q_traj_t.detach().cpu().numpy().astype(np.float32)  # (T+1, B, 7)
    fail_steps = [int(fi['fail_step']) for fi in fail_infos]
    fail_reasons = [str(fi['reason']) for fi in fail_infos]
    ee_xyz = _ee_xyz_from_q_traj(kin, q_traj_np)

    sig_list, mgn_list = _per_branch_metrics(kin, q_traj_np, fail_steps)

    # No-JL re-walk for the disconnection test (fig 6).
    no_jl_trajs, no_jl_closed = _no_jl_rewalk(kin, branches, p_tgt, R_tgt, h)
    D = _pairwise_min_dist(no_jl_trajs)

    out = {
        'task_path': task_path,
        'L_max': np.float32(L_max),
        'p_tgt': p_tgt.astype(np.float32),
        'R_tgt': R_tgt.astype(np.float32),
        'plane_normal': plane_normal,
        'Q_clean': Q_clean.astype(np.float32),
        'assigned': assigned.astype(np.int32),
        'branch_closed': no_jl_closed.copy() & False,  # placeholder, overwrite below
        'q0_best': q0_best,
        'L_self_best': L_self_best,
        'q_traj_best': q_traj_np,
        'ee_xyz_best': ee_xyz,
        'fail_step': np.array(fail_steps, dtype=np.int32),
        'all_q0_bid': all_bid.astype(np.int32),
        'all_q0_arc': all_arc.astype(np.float32),
        'all_q0_L_rel': L_rel_all,
        'no_jl_closed': no_jl_closed,
        'pairwise_min_d': D,
        'h': np.float32(h),
        'lmt_lo': lo.astype(np.float32),
        'lmt_up': hi.astype(np.float32),
    }
    # Variable-length per-branch arrays as separate keys (avoid pickled object dtype).
    out['branch_closed'] = np.array([b['closed'] for b in branches], dtype=bool)
    for bid in range(B):
        out[f'branch_traj_{bid}'] = branches[bid]['traj'].astype(np.float32)
        out[f'no_jl_traj_{bid}'] = no_jl_trajs[bid].astype(np.float32)
        out[f'sigma_t_{bid}'] = sig_list[bid]
        out[f'margin_t_{bid}'] = mgn_list[bid]

    np.savez(npz_path, **out)
    meta = {
        'seed': int(seed),
        'n_branches': int(B),
        'n_ik_seeds': int(n_ik_seeds),
        'n_per_branch': int(n_per_branch),
        'h': float(h),
        'L_max': float(L_max),
        'p_tgt': p_tgt.tolist(),
        'R_tgt': R_tgt.tolist(),
        'plane_normal': plane_normal.tolist(),
        'fail_steps': fail_steps,
        'fail_reasons': fail_reasons,
        'L_self_best': L_self_best.tolist(),
        'no_jl_closed': no_jl_closed.tolist(),
        'no_jl_limit_mult': float(NO_JL_LIMIT_MULT),
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    # Re-load for consistent return shape.
    return build_or_load(seed=seed, n_ik_seeds=n_ik_seeds,
                          n_per_branch=n_per_branch, h=h, force=False,
                          verbose=False)

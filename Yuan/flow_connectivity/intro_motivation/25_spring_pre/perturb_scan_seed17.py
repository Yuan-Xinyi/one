"""Perturbation scan around the seed=17 baseline line task.

Three independent scans:
  A. rotate d around n0 by theta in [-30, +30] deg, 31 points.
  B. rotate n around d0 by theta in [-20, +20] deg, 21 points.
     d is re-orthogonalized to lie in the new tangent plane.
  C. translate p0 by delta in [-5, +5] cm along {d0, n0, d0 x n0}, 11 pts each.

For each perturbed task c_i = (p0_i, d_i, n_i) we run the same pipeline as
25_spring_pre/_shared.py: dense IK at the start pose, Newton-project +
filter, ODE-walk to enumerate SMM branches, then the 6-DOF strict
rollout on n_per_branch q0 per branch. We persist:
  * <out>/scan_{A,B,C}_data.npz   raw per-point results (object arrays)
  * <out>/scan_{A,B,C}.csv        flat table (one row per perturbation point)
  * <out>/scan_{A,B,C}.png        4-subplot summary (A/B) or 12-subplot (C)

Run:
    python -m Yuan.flow_connectivity.intro_motivation.25_spring_pre.perturb_scan_seed17

Override defaults:
    python -m Yuan.flow_connectivity.intro_motivation.25_spring_pre.perturb_scan_seed17 \
        --n-ik-seeds 256 --n-per-branch 50 --only A,B
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from tqdm import tqdm

from Yuan.flow_connectivity.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.flow_connectivity.batched_rollout import _branch_seed_bank
from Yuan.flow_connectivity.intro_motivation.v18_smm_core import (
    DEDUP_RAD, DEFAULT_H, JOINT_MARGIN,
    as_tensor, enumerate_branches, project_and_filter,
)
from Yuan.flow_connectivity.intro_motivation.v18_smm_rollout_6dof import (
    rollout_lengths_6dof,
)
from Yuan.flow_connectivity.intro_motivation.v18_smm_task import sample_branch_q0s
from Yuan.flow_connectivity.v18_data_prep import _build_R_from_normal_direction, _dense_ik_at

import sys as _sys
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
from task_sampler import get_clean_task_target_pose  # noqa: E402  (local module)


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / 'perturb_out'
TARGET_PATH_M = 1.5      # match _shared.py / v18_smm_core
MAX_BRANCHES_REPORT = 6  # CSV columns reserve room for up to this many global IDs


# ---------------------------------------------------------------------------
# Task / kinematics helpers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TaskParams:
    p0: np.ndarray            # (3,) float32 start position
    d:  np.ndarray            # (3,) float32 unit direction
    n:  np.ndarray            # (3,) float32 unit surface normal
    n_pts: int                # path discretization (kept fixed across scan
                              # so rollout integration matches the baseline)


def baseline_params(seed: int, kin: BatchedFR3Kinematics) -> TaskParams:
    """Reproduce the seed-17 task (p0, d, n) from the clean sampler.

    n_pts is read from the baseline's actual extended path. The 6-DOF
    rollout in v18_smm_rollout_6dof does (V_PATH * DT) = 0.002 m per step
    and rounds n_steps per segment with banker's rounding — so segment
    length affects how much length is "lost" at each segment boundary.
    Using a different n_pts than the baseline systematically biases L
    even though the geometry is identical. Keep the baseline's value.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    p_tgt, R_tgt, task = get_clean_task_target_pose(seed, kin, rng)
    path = task['fine_path_pts']
    tangent = path[1] - path[0]
    tangent = (tangent / np.linalg.norm(tangent)).astype(np.float32)
    return TaskParams(
        p0=p_tgt.astype(np.float32),
        d=tangent,
        n=task['plane_normal'].astype(np.float32),
        n_pts=int(path.shape[0]),
    )


def task_path_from_params(tp: TaskParams, L: float = TARGET_PATH_M) -> np.ndarray:
    """Straight line of length L from p0 in direction d, with tp.n_pts samples."""
    ts = np.linspace(0.0, L, tp.n_pts, dtype=np.float32)
    return (tp.p0[None, :] + ts[:, None] * tp.d[None, :]).astype(np.float32)


def rotmat_axis_angle(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues."""
    a = axis / (np.linalg.norm(axis) + 1e-12)
    c, s = np.cos(theta), np.sin(theta)
    K = np.array([[0, -a[2], a[1]],
                  [a[2], 0, -a[0]],
                  [-a[1], a[0], 0]], dtype=np.float64)
    R = np.eye(3) + s * K + (1 - c) * (K @ K)
    return R.astype(np.float32)


# ---------------------------------------------------------------------------
# Single-perturbation-point compute
# ---------------------------------------------------------------------------
def compute_point(tp: TaskParams,
                  kin: BatchedFR3Kinematics,
                  rng: np.random.Generator,
                  n_ik_seeds: int,
                  n_per_branch: int,
                  h: float) -> dict:
    """Enumerate SMM branches at c=(p0,d,n) and roll out n_per_branch q0 each.

    Returns a dict with: branches (list of {'traj'}), assigned, all_q0_bid,
    all_q0_arc, L_rel, p_tgt, R_tgt, plane_normal, L_max, n_branches,
    branch_centroids (B,7), branch_L_max (B,), branch_L_std (B,).
    """
    p_tgt = tp.p0.copy()
    R_tgt = _build_R_from_normal_direction(tp.n, tp.d).astype(np.float32)
    task_path = task_path_from_params(tp, L=TARGET_PATH_M)
    L_max = float(np.linalg.norm(task_path[-1] - task_path[0]))
    track_pts = as_tensor(task_path, kin.device)
    plane_normal_t = as_tensor(tp.n, kin.device)

    p_t = torch.as_tensor(p_tgt, device=kin.device, dtype=torch.float32)
    R_t = torch.as_tensor(R_tgt, device=kin.device, dtype=torch.float32)
    extra = _branch_seed_bank(kin).detach().cpu().numpy()
    Q_seed_t, _ = _dense_ik_at(kin, p_t, R_t, n_ik_seeds, rng, extra_seeds=extra)
    if Q_seed_t.shape[0] == 0:
        return _empty_result(p_tgt, R_tgt, tp.n, L_max)
    lo = kin.lmt_lo.detach().cpu().numpy()
    hi = kin.lmt_up.detach().cpu().numpy()
    Q_clean = project_and_filter(
        kin, Q_seed_t.detach().cpu().numpy(), p_tgt, R_tgt, lo, hi,
        joint_margin=JOINT_MARGIN, dedup_rad=DEDUP_RAD, verbose=False)
    if Q_clean.shape[0] == 0:
        return _empty_result(p_tgt, R_tgt, tp.n, L_max)

    branches, assigned = enumerate_branches(kin, Q_clean, p_tgt, R_tgt, h)
    B = len(branches)

    all_q, all_bid, all_arc = sample_branch_q0s(branches, n_per_branch)
    q_batch = torch.as_tensor(all_q, device=kin.device, dtype=torch.float32)
    L_abs = rollout_lengths_6dof(kin, q_batch, track_pts, plane_normal_t)
    L_rel = (L_abs / L_max).astype(np.float32)

    centroids = np.stack([b['traj'].mean(axis=0) for b in branches], axis=0).astype(np.float32)
    L_max_per = np.zeros(B, dtype=np.float32)
    L_std_per = np.zeros(B, dtype=np.float32)
    for bid in range(B):
        m = all_bid == bid
        if m.sum() > 0:
            L_max_per[bid] = float(L_rel[m].max())
            L_std_per[bid] = float(L_rel[m].std())

    return dict(
        branches=[{'traj': b['traj'].astype(np.float32),
                   'closed': bool(b['closed'])} for b in branches],
        assigned=assigned.astype(np.int32),
        all_q0=all_q.astype(np.float32),
        all_q0_bid=all_bid.astype(np.int32),
        all_q0_arc=all_arc.astype(np.float32),
        L_rel=L_rel,
        p_tgt=p_tgt, R_tgt=R_tgt, plane_normal=tp.n.copy(),
        L_max=np.float32(L_max),
        n_branches=int(B),
        branch_centroids=centroids,
        branch_L_max=L_max_per,
        branch_L_std=L_std_per,
    )


def _empty_result(p_tgt, R_tgt, plane_normal, L_max):
    return dict(
        branches=[], assigned=np.zeros(0, dtype=np.int32),
        all_q0=np.zeros((0, 7), dtype=np.float32),
        all_q0_bid=np.zeros(0, dtype=np.int32),
        all_q0_arc=np.zeros(0, dtype=np.float32),
        L_rel=np.zeros(0, dtype=np.float32),
        p_tgt=p_tgt, R_tgt=R_tgt, plane_normal=plane_normal,
        L_max=np.float32(L_max),
        n_branches=0,
        branch_centroids=np.zeros((0, 7), dtype=np.float32),
        branch_L_max=np.zeros(0, dtype=np.float32),
        branch_L_std=np.zeros(0, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Branch matching across consecutive perturbation points
# ---------------------------------------------------------------------------
def hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Hausdorff distance between two q-curves (Na,7) and (Nb,7)."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float('inf')
    d_ab = float(np.max(np.min(np.linalg.norm(
        a[:, None, :] - b[None, :, :], axis=-1), axis=1)))
    d_ba = float(np.max(np.min(np.linalg.norm(
        b[:, None, :] - a[None, :, :], axis=-1), axis=1)))
    return max(d_ab, d_ba)


def match_global_ids(points: list[dict], match_threshold: float = 1.5) -> list[np.ndarray]:
    """Assign a persistent global branch ID to each (point, local_bid).

    Strategy: greedy centroid matching to the PREVIOUS point. Local branch
    j at point t inherits the global id of the previous-point branch whose
    centroid is closest in 7D joint space, provided that distance is below
    `match_threshold` (rad). Otherwise it gets a fresh global id.

    Centroid-based matching is fast (single 7-vector compare per branch
    pair). For verification we also check Hausdorff between matched pairs;
    if Hausdorff far exceeds threshold the centroid match is rejected.

    Returns: list of arrays. global_ids_per_point[t] has length n_branches[t]
             with the persistent global id for each local branch at point t.
    """
    global_ids_per_point: list[np.ndarray] = []
    next_global_id = 0
    prev_global_to_traj: dict[int, np.ndarray] = {}
    prev_global_to_centroid: dict[int, np.ndarray] = {}

    for t, pt in enumerate(points):
        B = pt['n_branches']
        ids = -np.ones(B, dtype=np.int32)
        if B == 0:
            global_ids_per_point.append(ids)
            prev_global_to_traj = {}
            prev_global_to_centroid = {}
            continue

        if not prev_global_to_centroid:
            # Initial seeding: assign fresh global ids in current order.
            for j in range(B):
                ids[j] = next_global_id
                next_global_id += 1
        else:
            cur_centroids = pt['branch_centroids']
            prev_ids = list(prev_global_to_centroid.keys())
            prev_C = np.stack([prev_global_to_centroid[g] for g in prev_ids], axis=0)
            # Cost matrix (cur x prev) in centroid space.
            cost = np.linalg.norm(
                cur_centroids[:, None, :] - prev_C[None, :, :], axis=-1)
            # Greedy assignment: iteratively pick smallest entry.
            used_prev = set()
            unfilled_cur = set(range(B))
            while unfilled_cur and len(used_prev) < len(prev_ids):
                # Mask used entries.
                cost_mask = cost.copy()
                for cj in range(B):
                    if cj not in unfilled_cur:
                        cost_mask[cj, :] = np.inf
                for pj in range(len(prev_ids)):
                    if pj in used_prev:
                        cost_mask[:, pj] = np.inf
                cj, pj = np.unravel_index(int(np.argmin(cost_mask)), cost_mask.shape)
                if not np.isfinite(cost_mask[cj, pj]):
                    break
                if cost_mask[cj, pj] > match_threshold:
                    break
                # Optional Hausdorff verification.
                g_id = prev_ids[pj]
                hd = hausdorff(pt['branches'][cj]['traj'],
                               prev_global_to_traj[g_id])
                if hd > 2.5 * match_threshold:
                    # Centroid lied: skip this pair, force the next-best.
                    cost[cj, pj] = np.inf
                    continue
                ids[cj] = g_id
                used_prev.add(pj)
                unfilled_cur.discard(cj)
            # Unmatched current branches get fresh ids.
            for cj in unfilled_cur:
                ids[cj] = next_global_id
                next_global_id += 1

        global_ids_per_point.append(ids)
        prev_global_to_traj = {int(ids[j]): pt['branches'][j]['traj']
                                for j in range(B)}
        prev_global_to_centroid = {int(ids[j]): pt['branch_centroids'][j]
                                    for j in range(B)}
    return global_ids_per_point


# ---------------------------------------------------------------------------
# Bimodality check
# ---------------------------------------------------------------------------
def is_bimodal(L: np.ndarray, min_n: int = 8) -> tuple[bool, int]:
    """Count peaks of a Gaussian-KDE over the L distribution. Returns
    (bimodal_flag, n_peaks). A peak must rise above 5% of the global max
    and be separated by at least 0.05 in L from neighboring peaks."""
    if L.size < min_n:
        return False, 0
    if float(np.std(L)) < 1e-4:
        return False, 1
    try:
        kde = gaussian_kde(L, bw_method=0.25)
    except Exception:
        return False, 0
    grid = np.linspace(max(0.0, float(L.min()) - 0.02),
                        max(float(L.max()) + 0.02, 0.1), 256)
    dens = kde(grid)
    peaks, _ = find_peaks(dens, height=float(dens.max()) * 0.05,
                          distance=int(256 * 0.05 / (grid[-1] - grid[0])))
    return bool(len(peaks) >= 2), int(len(peaks))


# ---------------------------------------------------------------------------
# Scan-level orchestration
# ---------------------------------------------------------------------------
def run_scan(scan_label: str,
              taskparam_list: list[TaskParams],
              x_values: np.ndarray,
              x_label: str,
              extra_cols: dict | None,
              kin: BatchedFR3Kinematics,
              n_ik_seeds: int,
              n_per_branch: int,
              h: float,
              base_seed: int) -> dict:
    """Compute every perturbation point sequentially. Returns the bundle
    dict ready for save_npz / df / plotting."""
    points = []
    rng = np.random.default_rng(base_seed)  # one rng, shared across points
    pbar = tqdm(taskparam_list, desc=f'scan {scan_label}', ncols=78)
    for tp in pbar:
        t0 = time.time()
        res = compute_point(tp, kin, rng, n_ik_seeds, n_per_branch, h)
        dt = time.time() - t0
        pbar.set_postfix(B=res['n_branches'], t=f'{dt:.1f}s')
        points.append(res)
    global_ids = match_global_ids(points)
    return dict(scan_label=scan_label, x_label=x_label, x_values=x_values,
                points=points, global_ids=global_ids,
                extra_cols=extra_cols or {})


def build_dataframe(bundle: dict) -> pd.DataFrame:
    """One row per perturbation point. Branch columns indexed by GLOBAL id."""
    rows = []
    all_global_ids = sorted({int(g) for arr in bundle['global_ids']
                              for g in arr if g >= 0})
    if len(all_global_ids) == 0:
        return pd.DataFrame()
    # Cap at MAX_BRANCHES_REPORT for CSV width.
    g_kept = all_global_ids[:MAX_BRANCHES_REPORT]
    for i, (pt, ids) in enumerate(zip(bundle['points'], bundle['global_ids'])):
        row = {'i': i, bundle['x_label']: float(bundle['x_values'][i]),
               'K_branches': pt['n_branches']}
        for key, vals in bundle['extra_cols'].items():
            row[key] = vals[i]
        # Compute argmax + gap from the GLOBAL view (drop branches with no q0).
        gid_to_Lmax = {}
        gid_to_Lstd = {}
        for j, g in enumerate(ids):
            if g < 0 or j >= pt['n_branches']:
                continue
            if pt['branch_L_max'].size == 0:
                continue
            gid_to_Lmax[int(g)] = float(pt['branch_L_max'][j])
            gid_to_Lstd[int(g)] = float(pt['branch_L_std'][j])
        # argmax (global id) and top1-top2 gap.
        if gid_to_Lmax:
            sorted_pairs = sorted(gid_to_Lmax.items(),
                                   key=lambda kv: kv[1], reverse=True)
            row['argmax_gid'] = int(sorted_pairs[0][0])
            row['top1'] = float(sorted_pairs[0][1])
            row['top2'] = float(sorted_pairs[1][1]) if len(sorted_pairs) > 1 else float('nan')
            row['top1_top2_gap'] = (row['top1'] - row['top2']) if len(sorted_pairs) > 1 else float('nan')
        else:
            row['argmax_gid'] = -1
            row['top1'] = float('nan')
            row['top2'] = float('nan')
            row['top1_top2_gap'] = float('nan')
        # Per-global-id columns.
        for g in g_kept:
            row[f'Lmax_g{g}'] = gid_to_Lmax.get(g, float('nan'))
            row[f'Lstd_g{g}'] = gid_to_Lstd.get(g, float('nan'))
        # Bimodality of the argmax branch's L distribution.
        if gid_to_Lmax:
            # Find LOCAL bid of argmax_gid.
            g_top = row['argmax_gid']
            local_top = next((j for j, gg in enumerate(ids) if int(gg) == g_top), -1)
            if local_top >= 0:
                m = pt['all_q0_bid'] == local_top
                bm, npk = is_bimodal(pt['L_rel'][m])
                row['top_branch_bimodal'] = int(bm)
                row['top_branch_npeaks'] = npk
            else:
                row['top_branch_bimodal'] = -1
                row['top_branch_npeaks'] = -1
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _branch_cmap(gid: int):
    return plt.get_cmap('tab10')(int(gid) % 10)


def _crossings(x: np.ndarray, ya: np.ndarray, yb: np.ndarray) -> list[float]:
    """X values where (ya - yb) changes sign (linearly interpolated)."""
    mask = np.isfinite(ya) & np.isfinite(yb)
    if mask.sum() < 2:
        return []
    xi = x[mask]; ai = ya[mask]; bi = yb[mask]
    diff = ai - bi
    out = []
    for k in range(len(diff) - 1):
        if diff[k] == 0:
            out.append(float(xi[k]))
        elif diff[k] * diff[k + 1] < 0:
            # Linear interp.
            t = float(diff[k] / (diff[k] - diff[k + 1]))
            out.append(float(xi[k] + t * (xi[k + 1] - xi[k])))
    return out


def _plot_panel(axes_row, bundle, mask, x_sub_label, title_suffix=''):
    """Render (Lmax-per-branch, argmax, gap, strip) into 4 axes."""
    ax_lmax, ax_amax, ax_gap, ax_strip = axes_row
    x = bundle['x_values'][mask]
    pts = [bundle['points'][i] for i in np.where(mask)[0]]
    ids = [bundle['global_ids'][i] for i in np.where(mask)[0]]
    all_gids = sorted({int(g) for arr in ids for g in arr if g >= 0})

    # --- (1) L_max per branch vs x ---
    gid_curves = {g: np.full(len(x), np.nan) for g in all_gids}
    for ti, pt in enumerate(pts):
        for j, g in enumerate(ids[ti]):
            if g < 0 or pt['branch_L_max'].size == 0:
                continue
            gid_curves[int(g)][ti] = float(pt['branch_L_max'][j])
    for g, ys in gid_curves.items():
        ax_lmax.plot(x, ys, '-o', color=_branch_cmap(g), markersize=3,
                      linewidth=1.4, label=f'g{g}')
    # Crossings between every visible pair.
    crossings_all = []
    keys = list(gid_curves.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            for xc in _crossings(x, gid_curves[keys[i]], gid_curves[keys[j]]):
                crossings_all.append(xc)
    for xc in crossings_all:
        ax_lmax.axvline(xc, color='red', linestyle='--', alpha=0.45, linewidth=0.9)
    ax_lmax.set_xlabel(x_sub_label); ax_lmax.set_ylabel(r'$L_{\max}$ / $L_{\rm task}$')
    ax_lmax.set_title(f'L_max per branch  {title_suffix}')
    ax_lmax.legend(fontsize=7, loc='best'); ax_lmax.grid(alpha=0.3)
    finite_vals = [v[np.isfinite(v)] for v in gid_curves.values()]
    finite_vals = [v for v in finite_vals if v.size > 0]
    y_top = float(np.concatenate(finite_vals).max()) if finite_vals else 1.0
    ax_lmax.set_ylim(-0.02, max(0.15, 1.1 * y_top))

    # --- (2) argmax branch id vs x (steps) ---
    argmax_gid = np.full(len(x), -1, dtype=int)
    for ti, pt in enumerate(pts):
        if pt['branch_L_max'].size == 0:
            continue
        local_argmax = int(np.argmax(pt['branch_L_max']))
        argmax_gid[ti] = int(ids[ti][local_argmax])
    valid = argmax_gid >= 0
    if valid.any():
        ax_amax.step(x[valid], argmax_gid[valid], where='post', color='black')
        for ti in np.where(valid)[0]:
            ax_amax.plot(x[ti], argmax_gid[ti], 'o',
                          color=_branch_cmap(argmax_gid[ti]), markersize=5)
    # Highlight switch positions.
    switches = []
    for k in range(1, len(argmax_gid)):
        if argmax_gid[k] != argmax_gid[k - 1] and argmax_gid[k] >= 0 and argmax_gid[k - 1] >= 0:
            switches.append(0.5 * (x[k] + x[k - 1]))
    for xs in switches:
        ax_amax.axvline(xs, color='red', linestyle='--', alpha=0.5, linewidth=0.9)
    ax_amax.set_xlabel(x_sub_label); ax_amax.set_ylabel('argmax global id')
    ax_amax.set_title(f'argmax branch  ({len(switches)} switches)  {title_suffix}')
    ax_amax.grid(alpha=0.3)

    # --- (3) top1 - top2 gap ---
    gap = np.full(len(x), np.nan)
    for ti, pt in enumerate(pts):
        Ls = pt['branch_L_max']
        if Ls.size >= 2:
            s = np.sort(Ls)[::-1]
            gap[ti] = float(s[0] - s[1])
    ax_gap.plot(x, gap, '-o', color='purple', markersize=3, linewidth=1.4)
    ax_gap.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax_gap.set_xlabel(x_sub_label); ax_gap.set_ylabel('top1 - top2 gap')
    ax_gap.set_title(f'cliff strength  {title_suffix}')
    ax_gap.grid(alpha=0.3)

    # --- (4) intra-branch L distribution strip plot ---
    rng_j = np.random.default_rng(0)
    x_span = (x.max() - x.min()) if len(x) > 1 else 1.0
    jitter = 0.015 * x_span
    for ti, pt in enumerate(pts):
        if pt['L_rel'].size == 0:
            continue
        for j, g in enumerate(ids[ti]):
            if g < 0:
                continue
            m = pt['all_q0_bid'] == j
            L = pt['L_rel'][m]
            if L.size == 0:
                continue
            jx = rng_j.uniform(-jitter, jitter, size=L.size) + x[ti]
            ax_strip.scatter(jx, L, c=[_branch_cmap(int(g))], s=8,
                              alpha=0.55, edgecolors='none')
    ax_strip.set_xlabel(x_sub_label); ax_strip.set_ylabel(r'$L$ / $L_{\rm task}$')
    ax_strip.set_title(f'intra-branch L distribution  {title_suffix}')
    ax_strip.grid(alpha=0.3)


def save_plot_AB(bundle: dict, out_png: Path):
    """A or B scan: one 4-subplot row."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.0))
    mask = np.ones(len(bundle['x_values']), dtype=bool)
    _plot_panel(axes, bundle, mask, bundle['x_label'])
    fig.suptitle(f"scan {bundle['scan_label']}  (seed 17 baseline)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close(fig)


def save_plot_C(bundle: dict, out_png: Path):
    """C scan: 3 directions x 4 columns."""
    fig, axes = plt.subplots(3, 4, figsize=(22, 13.0))
    dir_col = np.array(bundle['extra_cols']['direction'])
    for row_idx, dname in enumerate(['d0', 'n0', 'd0xn0']):
        mask = (dir_col == dname)
        _plot_panel(axes[row_idx], bundle, mask, bundle['x_label'],
                    title_suffix=f'(dir={dname})')
    fig.suptitle("scan C  (translate p0, seed 17 baseline)",
                 fontsize=12, y=1.005)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_bundle_npz(bundle: dict, out_path: Path):
    """Stash raw per-point data via pickle inside an npz wrapper (object arr)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path.with_suffix('.pkl'), 'wb') as f:
        pickle.dump(bundle, f, protocol=4)
    # Also write a lightweight npz with x_values + global_ids matrix padded.
    max_B = max((pt['n_branches'] for pt in bundle['points']), default=0)
    gid_mat = np.full((len(bundle['points']), max_B), -1, dtype=np.int32)
    Lmax_mat = np.full((len(bundle['points']), max_B), np.nan, dtype=np.float32)
    for i, (pt, ids) in enumerate(zip(bundle['points'], bundle['global_ids'])):
        for j in range(pt['n_branches']):
            gid_mat[i, j] = int(ids[j])
            Lmax_mat[i, j] = float(pt['branch_L_max'][j])
    np.savez(out_path,
              x_values=bundle['x_values'],
              x_label=str(bundle['x_label']),
              scan_label=str(bundle['scan_label']),
              gid_per_local=gid_mat,
              Lmax_per_local=Lmax_mat)


# ---------------------------------------------------------------------------
# Summary printout
# ---------------------------------------------------------------------------
def summarize(bundle: dict) -> str:
    Ks = np.array([p['n_branches'] for p in bundle['points']])
    lines = []
    lines.append(f"=== scan {bundle['scan_label']} ({len(Ks)} points) ===")
    lines.append(f"  branch count: min={int(Ks.min())}, max={int(Ks.max())}, "
                 f"unique={sorted(set(int(k) for k in Ks))}")
    # argmax switches: respect block boundaries if extra_cols defines a
    # 'direction' (Scan C) — switches that span two different directions
    # are not physical and must be excluded.
    argmax_gid = []
    for pt, ids in zip(bundle['points'], bundle['global_ids']):
        if pt['branch_L_max'].size == 0:
            argmax_gid.append(-1)
        else:
            argmax_gid.append(int(ids[int(np.argmax(pt['branch_L_max']))]))
    argmax_gid = np.array(argmax_gid)
    dir_col = bundle['extra_cols'].get('direction')  # list[str] or None
    switches = []
    for k in range(1, len(argmax_gid)):
        if dir_col is not None and dir_col[k] != dir_col[k - 1]:
            continue  # block boundary; not a real switch
        if argmax_gid[k] != argmax_gid[k - 1] and argmax_gid[k] >= 0 and argmax_gid[k - 1] >= 0:
            tag = f" [{dir_col[k]}]" if dir_col is not None else ""
            switches.append((float(bundle['x_values'][k]),
                              int(argmax_gid[k - 1]), int(argmax_gid[k]), tag))
    lines.append(f"  argmax switches: {len(switches)}")
    for xs, ga, gb, tag in switches:
        lines.append(f"    at {bundle['x_label']}~{xs:+.3f}{tag}: g{ga} -> g{gb}")
    # Top1-top2 gap min.
    gaps = []
    for pt in bundle['points']:
        Ls = pt['branch_L_max']
        if Ls.size >= 2:
            s = np.sort(Ls)[::-1]
            gaps.append(float(s[0] - s[1]))
    if gaps:
        lines.append(f"  min top1-top2 gap: {min(gaps):+.4f}")
    # Branch-0 (g0) bimodality across the scan.
    bm_count = 0; bm_total = 0
    for pt, ids in zip(bundle['points'], bundle['global_ids']):
        local0 = next((j for j, g in enumerate(ids) if int(g) == 0), -1)
        if local0 < 0 or pt['L_rel'].size == 0:
            continue
        m = pt['all_q0_bid'] == local0
        L = pt['L_rel'][m]
        bm_total += 1
        if is_bimodal(L)[0]:
            bm_count += 1
    if bm_total > 0:
        frac = bm_count / bm_total
        lines.append(f"  g0 bimodal in {bm_count}/{bm_total} = {frac*100:.1f}% "
                     f"of points where g0 present")
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Scan builders
# ---------------------------------------------------------------------------
def build_scan_A(tp0: TaskParams) -> tuple[list[TaskParams], np.ndarray, str, dict]:
    """Rotate d around n0."""
    thetas_deg = np.linspace(-30.0, 30.0, 31, dtype=np.float32)
    tps = []
    for th in thetas_deg:
        R = rotmat_axis_angle(tp0.n, float(np.deg2rad(th)))
        d_new = (R @ tp0.d).astype(np.float32)
        d_new = d_new / (np.linalg.norm(d_new) + 1e-12)
        tps.append(TaskParams(p0=tp0.p0.copy(), d=d_new, n=tp0.n.copy(), n_pts=tp0.n_pts))
    return tps, thetas_deg, 'theta_d_deg', {}


def build_scan_B(tp0: TaskParams) -> tuple[list[TaskParams], np.ndarray, str, dict]:
    """Rotate n around d0; re-orthogonalize d_new to lie in new tangent plane."""
    thetas_deg = np.linspace(-20.0, 20.0, 21, dtype=np.float32)
    tps = []
    for th in thetas_deg:
        R = rotmat_axis_angle(tp0.d, float(np.deg2rad(th)))
        n_new = (R @ tp0.n).astype(np.float32)
        n_new = n_new / (np.linalg.norm(n_new) + 1e-12)
        # Re-orthogonalize d so d_new perp n_new (project + renorm).
        d_new = tp0.d - (tp0.d @ n_new) * n_new
        d_new = (d_new / (np.linalg.norm(d_new) + 1e-12)).astype(np.float32)
        tps.append(TaskParams(p0=tp0.p0.copy(), d=d_new, n=n_new, n_pts=tp0.n_pts))
    return tps, thetas_deg, 'theta_n_deg', {}


def build_scan_C(tp0: TaskParams) -> tuple[list[TaskParams], np.ndarray, str, dict]:
    """Translate p0 by delta in {d0, n0, d0 x n0}, 11 pts each."""
    deltas_cm = np.linspace(-5.0, 5.0, 11, dtype=np.float32)
    bn = np.cross(tp0.d, tp0.n).astype(np.float32)
    bn = bn / (np.linalg.norm(bn) + 1e-12)
    dirs = [('d0', tp0.d), ('n0', tp0.n), ('d0xn0', bn)]
    tps = []
    deltas_all = []
    dir_col = []
    for dname, dvec in dirs:
        for dc in deltas_cm:
            p0_new = (tp0.p0 + (dc * 0.01) * dvec).astype(np.float32)
            tps.append(TaskParams(p0=p0_new, d=tp0.d.copy(), n=tp0.n.copy(), n_pts=tp0.n_pts))
            deltas_all.append(float(dc))
            dir_col.append(dname)
    return tps, np.array(deltas_all, dtype=np.float32), 'delta_cm', {'direction': dir_col}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--n-ik-seeds', type=int, default=256)
    ap.add_argument('--n-per-branch', type=int, default=50)
    ap.add_argument('--h', type=float, default=DEFAULT_H)
    ap.add_argument('--only', type=str, default='A,B,C',
                    help='subset of scans to run, comma-separated')
    ap.add_argument('--out', type=str, default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[device] {device}')
    kin = BatchedFR3Kinematics(device=device)
    tp0 = baseline_params(args.seed, kin)
    print(f'[baseline] seed={args.seed}')
    print(f'  p0 = {tp0.p0}')
    print(f'  d  = {tp0.d}')
    print(f'  n  = {tp0.n}')

    builders = {
        'A': (build_scan_A, save_plot_AB),
        'B': (build_scan_B, save_plot_AB),
        'C': (build_scan_C, save_plot_C),
    }
    chosen = [s.strip().upper() for s in args.only.split(',') if s.strip()]
    summaries = []
    for label in chosen:
        if label not in builders:
            print(f'  skipping unknown scan {label!r}')
            continue
        build_fn, plot_fn = builders[label]
        tps, xs, x_label, extras = build_fn(tp0)
        bundle = run_scan(label, tps, xs, x_label, extras, kin,
                           args.n_ik_seeds, args.n_per_branch, args.h,
                           base_seed=args.seed)
        df = build_dataframe(bundle)
        df.to_csv(out_dir / f'scan_{label}.csv', index=False)
        save_bundle_npz(bundle, out_dir / f'scan_{label}.npz')
        plot_fn(bundle, out_dir / f'scan_{label}.png')
        text = summarize(bundle)
        print('\n' + text + '\n')
        summaries.append(text)
        # Stash run config alongside.
        meta = dict(seed=args.seed, scan=label,
                    n_ik_seeds=args.n_ik_seeds,
                    n_per_branch=args.n_per_branch,
                    h=args.h,
                    p0=tp0.p0.tolist(), d=tp0.d.tolist(), n=tp0.n.tolist())
        with open(out_dir / f'scan_{label}_meta.json', 'w') as f:
            json.dump(meta, f, indent=2)

    summary_path = out_dir / 'summary.txt'
    summary_path.write_text('\n\n'.join(summaries) + '\n')
    print(f'\n[done] outputs in {out_dir}')


if __name__ == '__main__':
    main()

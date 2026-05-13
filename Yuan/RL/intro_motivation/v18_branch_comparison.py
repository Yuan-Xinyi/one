"""Part 1 - Branch comparison: 16 motivation slices for diverse IK starts.

Run:
    python -m Yuan.RL.v18_branch_comparison
    python -m Yuan.RL.v18_branch_comparison --seed 3 --grid 121

For a single line task (extended past reachable workspace via
TARGET_PATH_M so rollouts always fail partway):

  * Enumerate IK candidates at the start TCP.
  * Greedy farthest-point sampling in joint space picks 16 representatives
    (anchors A-P) so the picks span both IK branch and intra-branch posture.
  * For each anchor q: build its motivation slice (elbow alpha, wrist
    z-tilt beta), anchor to q's own FK with position-only IK projection,
    run the pos-priority rollout with dead-zone z and init-pose check,
    record per-cell length.
  * Save 16 per-panel JSONLs (used by Parts 2 & 3) plus a 4x4 heatmap
    sharing a global-max colorbar (viridis(L / global_max_L_norm)). A
    red dashed 30 deg contour overlays the actual orient_err vs R_tgt.

Output files (under OUT_DIR/seed{S}/):
    branch_comparison.png|pdf
    branch_comparison_{A..P}.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.RL.v18_curve_eval import branch_signature
from Yuan.RL.v18_data_prep import _build_R_from_normal_direction
from Yuan.RL.intro_motivation.v18_motivation_core import (
    ELBOW_SPAN,
    LINE_L_RANGE,
    ROLLOUT_THETA_MAX,
    ROLLOUT_THETA_MAX_DEG,
    SEED,
    START_POS_TOL,
    START_Z_TOL_DEG,
    TARGET_PATH_M,
    TILT_SPAN,
    as_tensor,
    elbow_wristtilt_basis,
    enumerate_start_iks,
    extend_task_path,
    path_length,
    pos_only_ik_project,
    rollout_lengths,
    sample_line_task,
    seed_dir,
)


N_PICKS = 16
DEFAULT_GRID = 121


def farthest_point_pick(q_np: np.ndarray, n_pick: int, seed_idx: int) -> list[int]:
    n = q_np.shape[0]
    picked = [int(seed_idx)]
    min_dist = np.linalg.norm(q_np - q_np[seed_idx], axis=-1)
    while len(picked) < n_pick and len(picked) < n:
        candidate = int(np.argmax(min_dist))
        picked.append(candidate)
        new_d = np.linalg.norm(q_np - q_np[candidate], axis=-1)
        min_dist = np.minimum(min_dist, new_d)
        for idx in picked:
            min_dist[idx] = -1.0
    return picked


def evaluate_motivation_slice(kin: BatchedFR3Kinematics,
                              q_pick: torch.Tensor,
                              track_pts: torch.Tensor,
                              plane_normal_np: np.ndarray,
                              L_max: float,
                              grid_size: int,
                              path_tangent: np.ndarray,
                              z_tgt: torch.Tensor,
                              plane_normal_t: torch.Tensor) -> dict:
    v_elbow, v_wrist_tilt = elbow_wristtilt_basis(kin, q_pick, path_tangent)

    p_anchor_t, R_anchor_t, _, _ = kin.tcp_fk_jac(q_pick.unsqueeze(0))
    p_anchor = p_anchor_t[0]
    R_anchor = R_anchor_t[0]
    z_anchor = R_anchor[:, 2]

    alphas = np.linspace(-ELBOW_SPAN, ELBOW_SPAN, grid_size, dtype=np.float32)
    betas = np.linspace(-TILT_SPAN, TILT_SPAN, grid_size, dtype=np.float32)
    aa, bb = np.meshgrid(alphas, betas, indexing='xy')
    q0 = q_pick.detach().cpu().numpy()
    q_seed_np = (q0[None, :]
                 + aa.reshape(-1, 1) * v_elbow[None, :]
                 + bb.reshape(-1, 1) * v_wrist_tilt[None, :])
    q_seed = as_tensor(q_seed_np.astype(np.float32), kin.device)

    q_proj, ik_ok = pos_only_ik_project(kin, q_seed, p_anchor)
    p_tcp, R_tcp, _, _ = kin.tcp_fk_jac(q_proj)
    pos_err = (p_tcp - p_anchor.unsqueeze(0)).norm(dim=-1)
    cos_th_anchor = (R_tcp[:, :, 2] * z_anchor.unsqueeze(0)).sum(dim=-1).clamp(-1.0, 1.0)
    z_err_anchor = torch.acos(cos_th_anchor)
    valid = (ik_ok
             & (pos_err <= START_POS_TOL)
             & (z_err_anchor <= np.deg2rad(START_Z_TOL_DEG)))
    valid_np = valid.detach().cpu().numpy().reshape(grid_size, grid_size)

    L_arr = rollout_lengths(kin, q_proj, track_pts, plane_normal_t,
                            theta_max_rad=ROLLOUT_THETA_MAX,
                            enforce_init_pose=True,
                            pos_priority=True) / L_max
    L_norm_grid = L_arr.reshape(grid_size, grid_size).astype(np.float32)
    L_for_plot = L_norm_grid.copy()
    L_for_plot[~valid_np] = np.nan

    cos_th_tgt = (R_tcp[:, :, 2] * z_tgt.unsqueeze(0)).sum(dim=-1).clamp(-1.0, 1.0)
    init_orient_deg = torch.rad2deg(torch.acos(cos_th_tgt)).detach().cpu().numpy()
    orient_grid = init_orient_deg.reshape(grid_size, grid_size)

    L_self = float(rollout_lengths(kin, q_pick.unsqueeze(0), track_pts, plane_normal_t,
                                   theta_max_rad=ROLLOUT_THETA_MAX,
                                   enforce_init_pose=True,
                                   pos_priority=True)[0] / L_max)

    q_proj_grid = q_proj.detach().cpu().numpy().reshape(grid_size, grid_size, 7)
    ik_ok_grid = ik_ok.detach().cpu().numpy().reshape(grid_size, grid_size)
    p_anchor_np = p_anchor.detach().cpu().numpy()
    z_anchor_np = z_anchor.detach().cpu().numpy()
    return {
        'alphas': alphas,
        'betas': betas,
        'L': L_for_plot,
        'L_raw': L_norm_grid,
        'orient': orient_grid,
        'L_self': L_self,
        'q_proj_grid': q_proj_grid,
        'valid_grid': valid_np,
        'ik_ok_grid': ik_ok_grid,
        'p_anchor': p_anchor_np,
        'z_anchor': z_anchor_np,
        'v_elbow': v_elbow.copy(),
        'v_wrist_tilt': v_wrist_tilt.copy(),
    }


def resample_to_signed_orient(alphas: np.ndarray,
                              betas: np.ndarray,
                              L_grid: np.ndarray,
                              orient_grid: np.ndarray,
                              target_y_deg: np.ndarray) -> np.ndarray:
    """For each alpha column, remap L from beta_joint axis to signed orient
    axis. Sign of orient is taken from sign of beta_joint (positive beta =
    positive signed_orient = z tilted toward path tangent)."""
    n_alpha = len(alphas)
    n_y = len(target_y_deg)
    out = np.full((n_y, n_alpha), np.nan, dtype=np.float32)
    sign_per_row = np.sign(betas).astype(np.float32)
    sign_per_row[sign_per_row == 0] = 1.0
    for c in range(n_alpha):
        signed_orient = sign_per_row * orient_grid[:, c]
        L_col = L_grid[:, c]
        valid = np.isfinite(L_col) & np.isfinite(signed_orient)
        if int(valid.sum()) < 2:
            continue
        x = signed_orient[valid]
        y = L_col[valid]
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        _, unique_idx = np.unique(x, return_index=True)
        x = x[unique_idx]
        y = y[unique_idx]
        if len(x) < 2:
            continue
        out[:, c] = np.interp(target_y_deg, x, y, left=np.nan, right=np.nan)
    return out


def render_panel_figure(out_path: Path, panels: list[dict], title: str,
                        threshold_deg: float):
    matplotlib.use('Agg')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(color='0.78')

    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    # Global L_norm max across all 16 panels' all cells. We renormalize every
    # panel by this so a value of 1.0 means "the longest rollout among the 16
    # panels". If any panel has a cell that completes the path, this equals
    # the task L_max normalization (no visual change). For harder tasks
    # where no panel ever completes, it stretches the colormap.
    global_max_L = 0.0
    for p in panels:
        valid = np.isfinite(p['L'])
        if valid.any():
            global_max_L = max(global_max_L, float(np.nanmax(p['L'])))
    if global_max_L <= 0.0:
        global_max_L = 1.0
    print(f'global max L_norm across 16 panels: {global_max_L:.3f}')

    n = len(panels)
    ncol = int(np.ceil(np.sqrt(n)))
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.7 * nrow), squeeze=False)
    im = None
    target_y = np.linspace(-60.0, 60.0, 121, dtype=np.float32)
    for idx, p in enumerate(panels):
        r, c = idx // ncol, idx % ncol
        ax = axes[r][c]
        alphas = p['alphas']
        L_resampled = resample_to_signed_orient(
            alphas, p['betas'], p['L'], p['orient'], target_y)
        L_global_norm = L_resampled / global_max_L
        im = ax.imshow(
            L_global_norm,
            origin='lower',
            extent=[alphas[0], alphas[-1], target_y[0], target_y[-1]],
            aspect='auto',
            vmin=0.0, vmax=1.0,
            cmap=cmap,
        )
        ax.axhline(+threshold_deg, color='red', linestyle='dashed', linewidth=1.0)
        ax.axhline(-threshold_deg, color='red', linestyle='dashed', linewidth=1.0)
        ax.scatter([0.0], [0.0], s=24, c='white', edgecolors='black',
                   linewidths=0.5, zorder=5)
        valid = np.isfinite(L_resampled)
        valid_frac = float(valid.mean()) if valid.size else 0.0
        mean_L = float(L_resampled[valid].mean()) if valid.any() else 0.0
        title_str = (f"q_{p['label']}: branch={p['signature']}\n"
                     f"L_self={p['L_self']:.2f}, valid={valid_frac:.2f}, "
                     f"mean L={mean_L:.2f}")
        ax.set_title(title_str)
        if r == nrow - 1:
            ax.set_xlabel('alpha: elbow [rad]')
        if c == 0:
            ax.set_ylabel('signed z-tilt vs R_tgt [deg]')

    for idx in range(n, nrow * ncol):
        r, c = idx // ncol, idx % ncol
        axes[r][c].axis('off')

    fig.suptitle(title, y=1.02, fontsize=12)
    if im is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cbar_ax,
                     label=f'L(q0) / global_max  (global_max = {global_max_L:.3f})')
    fig.tight_layout(rect=[0, 0, 0.91, 1.0])
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--grid', type=int, default=DEFAULT_GRID,
                        help='motivation slice resolution (per panel)')
    args = parser.parse_args()
    seed = int(args.seed)

    out_dir = seed_dir(seed)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kin = BatchedFR3Kinematics(device=device)

    out_png = out_dir / 'branch_comparison.png'

    task = sample_line_task(rng, kin, l_range=LINE_L_RANGE)
    task = extend_task_path(task, TARGET_PATH_M)
    task_path = task['fine_path_pts']
    L_max = path_length(task_path)
    track_pts = as_tensor(task_path, device)
    plane_normal_np = task['plane_normal']
    plane_normal_t = as_tensor(plane_normal_np, device)
    print(f'seed={seed}, L_max={L_max:.3f}m')

    q_set = enumerate_start_iks(kin, rng, task, track_pts)
    L_start = rollout_lengths(kin, q_set, track_pts, plane_normal_t,
                              theta_max_rad=ROLLOUT_THETA_MAX,
                              enforce_init_pose=True,
                              pos_priority=True)
    print(f'IK candidates: {q_set.shape[0]}')

    # Sample from the FULL IK pool (all IK-feasible q at TCP start), not just
    # successful rollouts, so farthest-point picks truly span joint space and
    # the 16 anchors look visually diverse in the overlay viewer. But first
    # drop any q within JOINT_MARGIN rad of a joint limit: farthest-point on
    # the raw IK pool tends to pull toward corner-of-joint-space configs that
    # die at segment 0 the moment the controller tries to move them.
    JOINT_MARGIN = 0.15
    lo_np = kin.lmt_lo.detach().cpu().numpy()
    hi_np = kin.lmt_up.detach().cpu().numpy()
    q_set_np = q_set.detach().cpu().numpy()
    inbounds = ((q_set_np - lo_np > JOINT_MARGIN)
                & (hi_np - q_set_np > JOINT_MARGIN)).all(axis=1)
    if int(inbounds.sum()) < N_PICKS:
        print(f'WARNING: only {int(inbounds.sum())} inbound candidates; relaxing margin')
        inbounds = np.ones(q_set.shape[0], dtype=bool)
    print(f'in-bound IK pool (margin {JOINT_MARGIN:.2f} rad): '
          f'{int(inbounds.sum())}/{q_set.shape[0]}')
    q_good = q_set[inbounds]
    L_good = L_start[inbounds]
    q_good_np = q_good.detach().cpu().numpy()
    seed_idx = int(np.argmax(L_good))
    picks = farthest_point_pick(q_good_np, min(N_PICKS, q_good.shape[0]), seed_idx)
    q_picks = q_good[picks]
    sigs = [branch_signature(q_good_np[i]) for i in picks]
    print('picked q anchors:')
    for k, (idx_in_good, sig) in enumerate(zip(picks, sigs)):
        label = chr(ord('A') + k)
        print(f'  q_{label}: branch={sig}, L_start={L_good[idx_in_good]:.3f}')

    pairwise = np.linalg.norm(
        q_good_np[picks][:, None, :] - q_good_np[picks][None, :, :], axis=-1)
    print(f'pairwise joint distance: min={pairwise[pairwise > 0].min():.2f} rad, '
          f'max={pairwise.max():.2f} rad')

    seg_dir = (task_path[1] - task_path[0])
    seg_dir = seg_dir / max(np.linalg.norm(seg_dir), 1e-12)
    rot_tgt_np = _build_R_from_normal_direction(plane_normal_np, seg_dir)
    z_tgt = torch.as_tensor(rot_tgt_np[:, 2], device=device, dtype=torch.float32)

    panels = []
    for k in range(len(picks)):
        label = chr(ord('A') + k)
        print(f'\nevaluating panel q_{label}...')
        result = evaluate_motivation_slice(
            kin, q_picks[k], track_pts, plane_normal_np, L_max,
            args.grid, seg_dir, z_tgt, plane_normal_t)
        result['label'] = label
        result['signature'] = sigs[k]
        valid = np.isfinite(result['L'])
        v_frac = float(valid.mean()) if valid.size else 0.0
        s_frac = float(result['L'][valid].mean()) if valid.any() else 0.0
        print(f'  valid={v_frac:.3f}, mean L (valid)={s_frac:.3f}, L_self={result["L_self"]:.3f}')
        print(f'  init orient_err vs R_tgt: max={result["orient"].max():.1f} deg')
        panels.append(result)

    render_panel_figure(
        out_png, panels,
        title=f'Branch comparison (seed={seed}): motivation slice per q anchor',
        threshold_deg=ROLLOUT_THETA_MAX_DEG,
    )

    task_tangent = (task_path[1] - task_path[0])
    task_tangent = task_tangent / max(np.linalg.norm(task_tangent), 1e-12)
    z_tgt_np = z_tgt.detach().cpu().numpy()
    # Compute global max L_norm across all 16 panels' cells so each JSONL's
    # consumers can color-match the heatmap rendering.
    global_max_L_norm = 0.0
    for p in panels:
        L_raw = p['L_raw']
        finite = np.isfinite(L_raw)
        if finite.any():
            global_max_L_norm = max(global_max_L_norm, float(L_raw[finite].max()))
    if global_max_L_norm <= 0.0:
        global_max_L_norm = 1.0
    for k, p in enumerate(panels):
        out_jsonl = out_dir / f'branch_comparison_{p["label"]}.jsonl'
        alphas = p['alphas']
        betas_rad = p['betas']
        L_raw = p['L_raw']
        orient = p['orient']
        valid = p['valid_grid']
        ik_ok = p['ik_ok_grid']
        q_proj_grid = p['q_proj_grid']
        meta = {
            'type': 'meta',
            'seed': seed,
            'anchor_label': p['label'],
            'branch_signature': list(p['signature']),
            'pairwise_pick_rank': k,
            'L_max_m': float(L_max),
            'L_self_normalized': float(p['L_self']),
            'global_max_L_norm': float(global_max_L_norm),
            'grid_size': int(args.grid),
            'elbow_span': float(ELBOW_SPAN),
            'tilt_span_deg': float(np.rad2deg(TILT_SPAN)),
            'rollout_theta_max_deg': float(ROLLOUT_THETA_MAX_DEG),
            'q_anchor': [float(x) for x in q_picks[k].detach().cpu().numpy()],
            'p_anchor': [float(x) for x in p['p_anchor']],
            'z_anchor': [float(x) for x in p['z_anchor']],
            'z_tgt': [float(x) for x in z_tgt_np],
            'v_elbow': [float(x) for x in p['v_elbow']],
            'v_wrist_tilt': [float(x) for x in p['v_wrist_tilt']],
            'path_tangent': [float(x) for x in task_tangent],
            'plane_normal': [float(x) for x in plane_normal_np],
            'task_path': [[float(x) for x in pt] for pt in task_path],
            'alphas': [float(x) for x in alphas],
            'betas_rad': [float(x) for x in betas_rad],
        }
        with open(out_jsonl, 'w') as f:
            f.write(json.dumps(meta) + '\n')
            for bi in range(args.grid):
                beta_v = float(betas_rad[bi])
                for ai in range(args.grid):
                    alpha_v = float(alphas[ai])
                    L_norm = float(L_raw[bi, ai]) if np.isfinite(L_raw[bi, ai]) else None
                    entry = {
                        'alpha_idx': ai,
                        'beta_idx': bi,
                        'alpha_rad': alpha_v,
                        'beta_rad': beta_v,
                        'beta_deg': float(np.rad2deg(beta_v)),
                        'q_init': [float(x) for x in q_proj_grid[bi, ai]],
                        'length_m': (L_norm * float(L_max)) if L_norm is not None else None,
                        'length_normalized': L_norm,
                        'init_orient_err_deg': float(orient[bi, ai]),
                        'valid': bool(valid[bi, ai]),
                        'ik_ok': bool(ik_ok[bi, ai]),
                    }
                    f.write(json.dumps(entry) + '\n')
        print(f'saved: {out_jsonl} (anchor={p["label"]}, branch={p["signature"]}, '
              f'{args.grid * args.grid} grid entries)')

    print(f'\nsaved: {out_png}')


if __name__ == '__main__':
    main()

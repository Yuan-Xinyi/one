"""Motivation slice: elbow posture vs wrist z-tilt.

Run:
    python -m Yuan.RL.v18_motivation_probe
    python -m Yuan.RL.v18_motivation_probe --seed 3

Two physically distinct slice axes through q_A:

  - alpha (elbow): direction in the 5-DOF task nullspace, orthogonalized
    against the pen-spin direction (~joint-7 axis). Same TCP position and
    same pen z-axis as q_A; only the elbow swings.
  - beta (wrist tilt): direction in the position-preserving kernel of
    J_pos that is NOT in the 5-DOF task nullspace. Same TCP position as
    q_A; the pen z-axis tilts toward the path tangent. Beta is calibrated
    so that one unit equals one radian of z-axis tilt.

Position-only DLS IK projection keeps every grid point at p_anchor while
letting z tilt freely. The strict_ok mask requires pos within 2mm and z
within 30 deg of the q_A pose. Expect a success island near (0, 0) and a
failure ring or band where the controller cannot recover the initial z
misalignment before drifting off-path.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.RL.v18_data_prep import _build_R_from_normal_direction
from Yuan.RL.v18_landscape_probe import (
    GRID_SIZE,
    OUT_DIR,
    SEED,
    as_tensor,
    choose_branch_pair,
    enumerate_start_iks,
    path_length,
    print_grid_stats,
    rollout_lengths,
    sample_line_task,
)


ELBOW_SPAN = 0.25
TILT_SPAN_DEG = 60.0
TILT_SPAN = np.deg2rad(TILT_SPAN_DEG)
LINE_L_RANGE = (0.30, 0.40)
TARGET_PATH_M = 1.5  # extend the sampled path to this length (beyond reachable
                     # workspace) so every rollout fails partway; L per cell
                     # then encodes "how far this q got before dying".
ROLLOUT_THETA_MAX_DEG = 30.0
ROLLOUT_THETA_MAX = np.deg2rad(ROLLOUT_THETA_MAX_DEG)
START_POS_TOL = 0.002
START_Z_TOL_DEG = 80.0
POS_IK_MAX_ITERS = 60
POS_IK_DAMPING = 1e-4
POS_IK_TOL = 1e-3


def extend_task_path(task: dict, target_L: float) -> dict:
    """Stretch a feasible task's fine_path_pts along its direction so the
    total length equals target_L. Used when we want rollouts to fail
    partway (path runs outside the reachable workspace) — the resulting
    rollout length per q encodes the maximum distance that q traversed
    before failing, not just '1 = completed task'."""
    fine = task['fine_path_pts'].copy()
    p0 = fine[0]
    p1 = fine[-1]
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


def skew(z: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(z[0])
    return torch.stack([
        torch.stack([zero, -z[2], z[1]]),
        torch.stack([z[2], zero, -z[0]]),
        torch.stack([-z[1], z[0], zero]),
    ])


def elbow_wristtilt_basis(kin: BatchedFR3Kinematics,
                          q_a: torch.Tensor,
                          path_tangent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, R, jac, _ = kin.tcp_fk_jac(q_a.unsqueeze(0))
    j_pos = jac[0, :3, :]
    j_ang = jac[0, 3:, :]
    z = R[0, :, 2]
    j_z = -skew(z) @ j_ang

    j_pos_np = j_pos.detach().cpu().numpy()
    j_z_np = j_z.detach().cpu().numpy()
    z_np = z.detach().cpu().numpy()

    # 4D kernel of J_pos: directions that preserve TCP position.
    _, _, vh_pos = np.linalg.svd(j_pos_np, full_matrices=True)
    pos_null = vh_pos[3:, :]  # 4x7, orthonormal rows

    # Within pos_null, find the 2D subspace that also preserves z-axis
    # (= the 5-DOF task nullspace) and its orthogonal complement (= wrist tilt).
    # In coordinates of pos_null, M = J_z @ pos_null.T is 3x4, rank 2.
    M = j_z_np @ pos_null.T  # 3x4
    _, _, vh_M = np.linalg.svd(M, full_matrices=True)
    # Right singular vectors of M (4x4); last 2 rows = kernel of M = task nullspace coords.
    task_coords = vh_M[2:, :]   # 2x4 -> coefficients in pos_null basis
    tilt_coords = vh_M[:2, :]   # 2x4 -> wrist tilt coefficients in pos_null basis

    task_null = task_coords @ pos_null   # 2x7 in joint space
    wrist_tilt_2d = tilt_coords @ pos_null   # 2x7 in joint space

    # Elbow: in task_null, orthogonal to the pen-spin direction (~joint-7 axis).
    e7 = np.zeros(7, dtype=np.float64); e7[6] = 1.0
    spin_components = task_null @ e7   # 2-vector
    if np.linalg.norm(spin_components) > 1e-6:
        spin_dir = spin_components @ task_null
        spin_dir /= max(np.linalg.norm(spin_dir), 1e-12)
        v_elbow = task_null[0] - np.dot(task_null[0], spin_dir) * spin_dir
        if np.linalg.norm(v_elbow) < 1e-6:
            v_elbow = task_null[1] - np.dot(task_null[1], spin_dir) * spin_dir
        v_elbow /= max(np.linalg.norm(v_elbow), 1e-12)
    else:
        v_elbow = task_null[0] / max(np.linalg.norm(task_null[0]), 1e-12)

    # Wrist tilt: among the 2D wrist_tilt subspace, pick the direction whose
    # z-axis motion is aligned with the path tangent's projection onto T_z S^2.
    path_perp = path_tangent - np.dot(path_tangent, z_np) * z_np
    if np.linalg.norm(path_perp) < 1e-6:
        # Path is parallel to z (degenerate); fall back to any tilt direction.
        v_wrist_tilt = wrist_tilt_2d[0]
    else:
        path_perp /= np.linalg.norm(path_perp)
        Mw = j_z_np @ wrist_tilt_2d.T   # 3x2 mapping from beta-coords to z-tilt
        c, *_ = np.linalg.lstsq(Mw, path_perp, rcond=None)
        if np.linalg.norm(c) < 1e-9:
            v_wrist_tilt = wrist_tilt_2d[0]
        else:
            c /= np.linalg.norm(c)
            v_wrist_tilt = c @ wrist_tilt_2d
    # Calibrate beta so that ||J_z v_wrist_tilt|| = 1 rad z tilt per unit beta.
    z_dot_mag = np.linalg.norm(j_z_np @ v_wrist_tilt)
    if z_dot_mag > 1e-9:
        v_wrist_tilt = v_wrist_tilt / z_dot_mag
    return v_elbow.astype(np.float32), v_wrist_tilt.astype(np.float32)


def pos_only_ik_project(kin: BatchedFR3Kinematics,
                        q_seed: torch.Tensor,
                        p_target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    device = kin.device
    q = q_seed.clamp(kin.lmt_lo, kin.lmt_up).clone()
    n = q.shape[0]
    p_rep = p_target.unsqueeze(0).expand(n, 3)
    eye3 = torch.eye(3, device=device, dtype=q.dtype).expand(n, 3, 3)
    converged = torch.zeros(n, device=device, dtype=torch.bool)
    active = torch.ones(n, device=device, dtype=torch.bool)

    for _ in range(POS_IK_MAX_ITERS):
        p_tcp, _, J, _ = kin.tcp_fk_jac(q)
        delta_p = p_rep - p_tcp
        pos_err = delta_p.norm(dim=-1)
        in_limits = ((q >= kin.lmt_lo - 1e-5) & (q <= kin.lmt_up + 1e-5)).all(dim=-1)
        now_conv = (pos_err <= POS_IK_TOL) & in_limits
        newly = active & now_conv
        if newly.any():
            converged |= newly
            active &= ~newly
        if not active.any():
            break

        J_pos = J[:, :3, :]
        pos_scale = torch.where(pos_err > 0.1,
                                0.1 / pos_err.clamp_min(1e-12),
                                torch.ones_like(pos_err))
        delta_p_scaled = delta_p * pos_scale.unsqueeze(-1)
        A = J_pos @ J_pos.transpose(-1, -2) + (POS_IK_DAMPING ** 2) * eye3
        Jpinv = J_pos.transpose(-1, -2) @ torch.linalg.inv(A)
        delta_q = (Jpinv @ delta_p_scaled.unsqueeze(-1)).squeeze(-1)
        q_next = (q + delta_q).clamp(kin.lmt_lo, kin.lmt_up)
        q = torch.where(active.unsqueeze(-1), q_next, q)

    if active.any():
        p_tcp, _, _, _ = kin.tcp_fk_jac(q)
        pos_err = (p_rep - p_tcp).norm(dim=-1)
        in_limits = ((q >= kin.lmt_lo - 1e-5) & (q <= kin.lmt_up + 1e-5)).all(dim=-1)
        converged |= active & (pos_err <= POS_IK_TOL) & in_limits

    return q, converged


def build_motivation_grid(q_a: torch.Tensor,
                          v_elbow: np.ndarray,
                          v_wrist_tilt: np.ndarray,
                          device) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    alphas = np.linspace(-ELBOW_SPAN, ELBOW_SPAN, GRID_SIZE, dtype=np.float32)
    betas = np.linspace(-TILT_SPAN, TILT_SPAN, GRID_SIZE, dtype=np.float32)
    aa, bb = np.meshgrid(alphas, betas, indexing='xy')
    q0 = q_a.detach().cpu().numpy()
    q = (q0[None, :]
         + aa.reshape(-1, 1) * v_elbow[None, :]
         + bb.reshape(-1, 1) * v_wrist_tilt[None, :])
    return alphas, betas, as_tensor(q.astype(np.float32), device)


def strict_ok_mask(kin: BatchedFR3Kinematics,
                   q_proj: torch.Tensor,
                   p_anchor: torch.Tensor,
                   z_anchor: torch.Tensor,
                   ik_ok: torch.Tensor) -> np.ndarray:
    p_tcp, R_tcp, _, _ = kin.tcp_fk_jac(q_proj)
    pos_err = (p_tcp - p_anchor.unsqueeze(0)).norm(dim=-1)
    z_cur = R_tcp[:, :, 2]
    cos_th = (z_cur * z_anchor.unsqueeze(0)).sum(dim=-1).clamp(-1.0, 1.0)
    z_err = torch.acos(cos_th)
    ok = (ik_ok
          & (pos_err <= START_POS_TOL)
          & (z_err <= np.deg2rad(START_Z_TOL_DEG)))
    print('pos-only IK projection:')
    print(f'  IK ok fraction: {float(ik_ok.float().mean().item()):.3f}')
    print(f'  strict mask fraction: {float(ok.float().mean().item()):.3f}')
    if ok.any():
        print(f'  max strict pos err: {float(pos_err[ok].max().item()) * 1000:.2f} mm')
        print(f'  max strict z err: {float(torch.rad2deg(z_err[ok]).max().item()):.2f} deg')
    return ok.detach().cpu().numpy()


def _resample_to_signed_orient(alphas, betas_rad, L_grid, orient_grid, target_y_deg):
    n_alpha = len(alphas)
    n_y = len(target_y_deg)
    out = np.full((n_y, n_alpha), np.nan, dtype=np.float32)
    sign_per_row = np.sign(betas_rad).astype(np.float32)
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


def save_heatmap_with_overlay(out_path,
                              alphas: np.ndarray,
                              betas_rad: np.ndarray,
                              L_grid: np.ndarray,
                              orient_grid_deg: np.ndarray,
                              threshold_deg: float,
                              title: str,
                              xlabel: str,
                              ylabel: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(color='0.78')

    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    target_y = np.linspace(-60.0, 60.0, 121, dtype=np.float32)
    L_resampled = _resample_to_signed_orient(alphas, betas_rad, L_grid,
                                             orient_grid_deg, target_y)

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(
        L_resampled,
        origin='lower',
        extent=[alphas[0], alphas[-1], target_y[0], target_y[-1]],
        aspect='auto',
        vmin=0.0, vmax=1.0,
        cmap=cmap,
    )
    ax.axhline(+threshold_deg, color='red', linestyle='dashed', linewidth=1.2)
    ax.axhline(-threshold_deg, color='red', linestyle='dashed', linewidth=1.2)
    ax.scatter([0.0], [0.0], s=42, c='white', edgecolors='black',
               linewidths=0.8, zorder=5)
    ax.text(0.0, 0.0, ' q_A', color='white', fontsize=10,
            fontweight='bold', va='center', ha='left', zorder=6)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('L(q0) / L_max')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    fig.savefig(out_path.with_suffix('.pdf'))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    seed = int(args.seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kin = BatchedFR3Kinematics(device=device)

    out_png = OUT_DIR / f'v18_L_motivation_slice_seed{seed}.png'
    out_jsonl = OUT_DIR / f'v18_L_motivation_picks_seed{seed}.jsonl'
    print(f'seed={seed}')

    task = sample_line_task(rng, kin, l_range=LINE_L_RANGE)
    task = extend_task_path(task, TARGET_PATH_M)
    task_path = task['fine_path_pts']
    L_max = path_length(task_path)
    track_pts = as_tensor(task_path, device)
    plane_normal = as_tensor(task['plane_normal'], device)
    print(f'task: straight line, L_max={L_max:.3f}m')

    q_set = enumerate_start_iks(kin, rng, task, track_pts)
    L_start = rollout_lengths(kin, q_set, track_pts, plane_normal,
                              theta_max_rad=ROLLOUT_THETA_MAX,
                              enforce_init_pose=True,
                              pos_priority=True)
    q_a, _ = choose_branch_pair(q_set, L_start, L_max)

    p_anchor_t, R_anchor_t, _, _ = kin.tcp_fk_jac(q_a.unsqueeze(0))
    p_anchor = p_anchor_t[0]
    R_anchor = R_anchor_t[0]
    z_anchor = R_anchor[:, 2]

    path_tangent = (track_pts[1] - track_pts[0]).detach().cpu().numpy()
    path_tangent = path_tangent / max(np.linalg.norm(path_tangent), 1e-12)

    v_elbow, v_wrist_tilt = elbow_wristtilt_basis(kin, q_a, path_tangent)
    print(f'v_elbow norm: {np.linalg.norm(v_elbow):.3f} (unit in joint rad)')
    print(f'v_wrist_tilt joint-rad norm: {np.linalg.norm(v_wrist_tilt):.3f}'
          f' (calibrated so 1 beta = 1 z-tilt rad)')

    a_vals, b_vals, q_seed_grid = build_motivation_grid(q_a, v_elbow, v_wrist_tilt, device)
    q_proj, ik_ok = pos_only_ik_project(kin, q_seed_grid, p_anchor)
    valid_mask = strict_ok_mask(kin, q_proj, p_anchor, z_anchor, ik_ok)
    L_arr = rollout_lengths(kin, q_proj, track_pts, plane_normal,
                            theta_max_rad=ROLLOUT_THETA_MAX,
                            enforce_init_pose=True,
                            pos_priority=True) / L_max
    L_grid = L_arr.reshape(GRID_SIZE, GRID_SIZE).astype(np.float32)
    L_grid[~valid_mask.reshape(GRID_SIZE, GRID_SIZE)] = np.nan

    # Compute the true initial orient_err relative to the rollout's R_tgt for
    # segment 0. The y-axis beta is a first-order joint perturbation calibrated
    # at q_A; the actual z-tilt after IK projection differs, so the failure
    # boundary in L is not at beta = 30 deg. We overlay a contour at the true
    # 30 deg z-tilt to show the threshold is being enforced correctly.
    seg_dir_np = (task_path[1] - task_path[0])
    seg_dir_np = seg_dir_np / max(np.linalg.norm(seg_dir_np), 1e-12)
    rot_tgt_np = _build_R_from_normal_direction(task['plane_normal'], seg_dir_np)
    z_tgt = torch.as_tensor(rot_tgt_np[:, 2], device=device, dtype=torch.float32)
    _, R_proj, _, _ = kin.tcp_fk_jac(q_proj)
    cos_th = (R_proj[:, :, 2] * z_tgt.unsqueeze(0)).sum(dim=-1).clamp(-1.0, 1.0)
    init_orient_deg = torch.rad2deg(torch.acos(cos_th)).detach().cpu().numpy()
    init_orient_grid = init_orient_deg.reshape(GRID_SIZE, GRID_SIZE)
    print(f'  init orient_err vs R_tgt: min={init_orient_deg.min():.1f}'
          f' median={np.median(init_orient_deg):.1f}'
          f' max={init_orient_deg.max():.1f} deg')

    save_heatmap_with_overlay(
        out_png,
        a_vals,
        b_vals,
        L_grid,
        init_orient_grid,
        ROLLOUT_THETA_MAX_DEG,
        f'Motivation slice (seed={seed}): elbow swing vs wrist z-tilt',
        'alpha: elbow direction in task nullspace [rad in joint]',
        'signed z-axis tilt vs R_tgt [deg]',
    )
    print_grid_stats('Motivation slice', L_grid)

    # Serialize the entire grid to JSONL so an animation/replay script can
    # look up the post-IK q for any (alpha, beta) cell without re-running
    # the probe. First line is a meta record with the basis + task config;
    # subsequent lines are one entry per grid cell (GRID_SIZE^2 total).
    import json
    q_proj_np = q_proj.detach().cpu().numpy().reshape(GRID_SIZE, GRID_SIZE, 7)
    L_norm_grid = L_arr.reshape(GRID_SIZE, GRID_SIZE).astype(np.float32)
    L_abs_grid = L_norm_grid * float(L_max)
    valid_grid = valid_mask.reshape(GRID_SIZE, GRID_SIZE)
    ok_grid = ik_ok.detach().cpu().numpy().reshape(GRID_SIZE, GRID_SIZE)
    z_tgt_np = z_tgt.detach().cpu().numpy()
    plane_normal_np = task['plane_normal']

    meta = {
        'type': 'meta',
        'seed': seed,
        'L_max_m': float(L_max),
        'grid_size': int(GRID_SIZE),
        'elbow_span': float(ELBOW_SPAN),
        'tilt_span_deg': float(TILT_SPAN_DEG),
        'rollout_theta_max_deg': float(ROLLOUT_THETA_MAX_DEG),
        'q_a': [float(x) for x in q_a.detach().cpu().numpy()],
        'p_anchor': [float(x) for x in p_anchor.detach().cpu().numpy()],
        'z_anchor': [float(x) for x in z_anchor.detach().cpu().numpy()],
        'z_tgt': [float(x) for x in z_tgt_np],
        'v_elbow': [float(x) for x in v_elbow],
        'v_wrist_tilt': [float(x) for x in v_wrist_tilt],
        'path_tangent': [float(x) for x in path_tangent],
        'plane_normal': [float(x) for x in plane_normal_np],
        'task_path': [[float(x) for x in pt] for pt in task_path],
        'alphas': [float(x) for x in a_vals],
        'betas_rad': [float(x) for x in b_vals],
    }
    with open(out_jsonl, 'w') as f:
        f.write(json.dumps(meta) + '\n')
        for ai in range(GRID_SIZE):
            alpha_v = float(a_vals[ai])
            for bi in range(GRID_SIZE):
                beta_v = float(b_vals[bi])
                entry = {
                    'alpha_idx': ai,
                    'beta_idx': bi,
                    'alpha_rad': alpha_v,
                    'beta_rad': beta_v,
                    'beta_deg': float(np.rad2deg(beta_v)),
                    'q_init': [float(x) for x in q_proj_np[bi, ai]],
                    'length_m': float(L_abs_grid[bi, ai]) if np.isfinite(L_abs_grid[bi, ai]) else None,
                    'length_normalized': float(L_norm_grid[bi, ai]) if np.isfinite(L_norm_grid[bi, ai]) else None,
                    'init_orient_err_deg': float(init_orient_grid[bi, ai]),
                    'valid': bool(valid_grid[bi, ai]),
                    'ik_ok': bool(ok_grid[bi, ai]),
                }
                f.write(json.dumps(entry) + '\n')

    print(f'\nsaved: {out_png}')
    print(f'saved: {out_jsonl} ({GRID_SIZE * GRID_SIZE} grid entries + 1 meta)')


if __name__ == '__main__':
    main()

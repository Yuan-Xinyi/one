"""2D L(q0) landscape slices for the v18 straight-line task.

Run:
    python -m Yuan.RL.v18_landscape_probe

This script produces two fixed "money figure" probes:

1. Nullspace slice:
   q = q_A + alpha * v1 + beta * v2
   where v1, v2 are orthonormal directions in the TCP-position Jacobian
   nullspace at q_A. This shows how internal posture changes affect rollout
   length while preserving the same TCP start to first order.

2. IK branch switching slice:
   q = q_A + alpha * (q_B - q_A) + beta * v1
   where q_A and q_B are different start-IK branches for the same TCP point.
   Invalid initial TCP points are shown as gray.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import Yuan.RL.config as cfg
from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.RL.batched_rollout import batched_rollout_segment, _batched_ik_project
from Yuan.RL.v18_curve_eval import branch_signature, sample_curve_task
from Yuan.RL.v18_data_prep import _build_R_from_normal_direction, _dense_ik_at


SEED = 2026
N_CHECKPOINTS = 5
N_START_IK = 512

GRID_SIZE = 251
NULLSPACE_SPAN = 0.30
BRANCH_BETA_SPAN = 0.30
INIT_TCP_TOL = 0.02
START_POS_TOL = 0.002
START_Z_TOL_DEG = 1.0

V_PATH = 0.10
EPS_P = 0.05
CHUNK_SIZE = 1024

OUT_DIR = Path('/home/lqin/one/Yuan/RL/data')
OUT_NULLSPACE = OUT_DIR / 'v18_L_nullspace_slice'
OUT_BRANCH = OUT_DIR / 'v18_L_branch_switch_slice'


def as_tensor(x, device):
    return torch.as_tensor(x, device=device, dtype=torch.float32)


def sample_line_task(rng: np.random.Generator, kin: BatchedFR3Kinematics) -> dict:
    for _ in range(100):
        task = sample_curve_task(rng, kin, 'line', N_CHECKPOINTS)
        if task is not None:
            return task
    raise RuntimeError('failed to sample a feasible straight-line task')


def path_length(path: np.ndarray) -> float:
    return float(np.linalg.norm(path[1:] - path[:-1], axis=1).sum())


def start_rotation(task: dict, track_pts: torch.Tensor, device):
    tangent = track_pts[1] - track_pts[0]
    tangent = tangent / tangent.norm().clamp_min(1e-12)
    rot = _build_R_from_normal_direction(task['plane_normal'], tangent.detach().cpu().numpy())
    return as_tensor(rot, device)


def rollout_lengths(kin: BatchedFR3Kinematics,
                    q_batch: torch.Tensor,
                    track_pts: torch.Tensor,
                    plane_normal: torch.Tensor) -> np.ndarray:
    lengths = np.zeros(q_batch.shape[0], dtype=np.float32)
    for start in range(0, q_batch.shape[0], CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, q_batch.shape[0])
        lengths[start:end] = rollout_chunk(kin, q_batch[start:end], track_pts, plane_normal)
    return lengths


def rollout_chunk(kin: BatchedFR3Kinematics,
                  q_init: torch.Tensor,
                  track_pts: torch.Tensor,
                  plane_normal: torch.Tensor) -> np.ndarray:
    device = kin.device
    batch_size = q_init.shape[0]
    q = q_init.clone()
    alive = torch.ones(batch_size, device=device, dtype=torch.bool)
    lengths = torch.zeros(batch_size, device=device, dtype=torch.float32)
    branch_action = torch.tensor([1.0, 0.0, 1.0, 0.0], device=device,
                                 dtype=torch.float32).unsqueeze(0).expand(batch_size, 4)

    for idx in range(track_pts.shape[0] - 1):
        if not bool(alive.any().item()):
            break

        p0 = track_pts[idx]
        seg_vec = track_pts[idx + 1] - p0
        seg_len = float(seg_vec.norm().item())
        if seg_len < 1e-8:
            continue

        direction = seg_vec / seg_vec.norm().clamp_min(1e-12)
        rot_np = _build_R_from_normal_direction(
            plane_normal.detach().cpu().numpy(),
            direction.detach().cpu().numpy(),
        )
        n_steps = max(1, int(round(seg_len / (V_PATH * float(cfg.DT)))))

        out = batched_rollout_segment(
            q_init=q,
            R_tgt=as_tensor(rot_np, device).unsqueeze(0).expand(batch_size, 3, 3),
            branch_action=branch_action,
            p0=p0.unsqueeze(0).expand(batch_size, 3),
            d_dir=direction.unsqueeze(0).expand(batch_size, 3),
            v_path=torch.full((batch_size,), V_PATH, device=device, dtype=torch.float32),
            eps_p=torch.full((batch_size,), EPS_P, device=device, dtype=torch.float32),
            T_total=torch.full((batch_size,), n_steps, device=device, dtype=torch.long),
            start_step=0,
            end_step=n_steps,
            kin=kin,
            alive_mask=alive,
        )

        completed = out['lengths'].float() / float(n_steps) * seg_len
        lengths = torch.where(alive, lengths + completed, lengths)
        q = out['q_final']
        alive = out['alive_out']

    return lengths.detach().cpu().numpy()


def enumerate_start_iks(kin: BatchedFR3Kinematics,
                        rng: np.random.Generator,
                        task: dict,
                        track_pts: torch.Tensor) -> torch.Tensor:
    rot0 = start_rotation(task, track_pts, kin.device)
    q_set, _ = _dense_ik_at(kin, track_pts[0], rot0, N_START_IK, rng)
    if q_set.shape[0] == 0:
        raise RuntimeError('no IK found at line start')
    return q_set


def choose_branch_pair(q_set: torch.Tensor,
                       L_start: np.ndarray,
                       L_max: float) -> tuple[torch.Tensor, torch.Tensor]:
    q_np = q_set.detach().cpu().numpy()
    sigs = [branch_signature(q) for q in q_np]
    good = np.where(L_start >= 0.995 * L_max)[0]
    if len(good) == 0:
        good = np.array([int(np.argmax(L_start))])

    idx_a = int(good[np.argmax(L_start[good])])
    sig_a = sigs[idx_a]

    other_good = [idx for idx in good if sigs[idx] != sig_a]
    if other_good:
        idx_b = max(other_good, key=lambda idx: L_start[idx])
        return q_set[idx_a], q_set[int(idx_b)]

    other_branch = [idx for idx, sig in enumerate(sigs) if sig != sig_a]
    if other_branch:
        idx_b = max(other_branch, key=lambda idx: L_start[idx])
        return q_set[idx_a], q_set[int(idx_b)]

    dists = np.linalg.norm(q_np - q_np[idx_a], axis=1)
    idx_b = int(np.argmax(dists))
    return q_set[idx_a], q_set[idx_b]


def tcp_pos(kin: BatchedFR3Kinematics, q_batch: torch.Tensor) -> np.ndarray:
    p, _ = kin.fk_batch(q_batch)
    return p.detach().cpu().numpy()


def project_to_same_start_pose(kin: BatchedFR3Kinematics,
                               q_seed: torch.Tensor,
                               p_start: torch.Tensor,
                               R_start: torch.Tensor) -> tuple[torch.Tensor, np.ndarray]:
    p_rep = p_start.unsqueeze(0).expand(q_seed.shape[0], 3)
    R_rep = R_start.unsqueeze(0).expand(q_seed.shape[0], 3, 3)
    q_proj, ok, _ = _batched_ik_project(kin, q_seed, p_rep, R_rep, branch_action=None)

    p_tcp, R_tcp, _, _ = kin.tcp_fk_jac(q_proj)
    pos_err = (p_tcp - p_rep).norm(dim=-1)
    z_err = torch.acos((R_tcp[:, :, 2] * R_rep[:, :, 2]).sum(dim=-1).clamp(-1.0, 1.0))
    strict_ok = (ok
                 & (pos_err <= START_POS_TOL)
                 & (z_err <= np.deg2rad(START_Z_TOL_DEG)))

    print('same-start-pose projection:')
    print(f'  IK ok fraction: {float(ok.float().mean().item()):.3f}')
    print(f'  strict same TCP fraction: {float(strict_ok.float().mean().item()):.3f}')
    if strict_ok.any():
        print(f'  max strict pos err: {float(pos_err[strict_ok].max().item()) * 1000:.2f} mm')
        print(f'  max strict z err: {float(torch.rad2deg(z_err[strict_ok]).max().item()):.2f} deg')
    return q_proj, strict_ok.detach().cpu().numpy()


def nullspace_basis(kin: BatchedFR3Kinematics, q: torch.Tensor) -> np.ndarray:
    _, _, jac, _ = kin.tcp_fk_jac(q.unsqueeze(0))
    j_pos = jac[0, :3, :]
    _, _, vh = torch.linalg.svd(j_pos, full_matrices=True)
    basis = vh[3:, :].detach().cpu().numpy()
    return basis.astype(np.float32)


def build_nullspace_grid(q_a: torch.Tensor,
                         v1: np.ndarray,
                         v2: np.ndarray,
                         span: float,
                         device) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    alphas = np.linspace(-span, span, GRID_SIZE, dtype=np.float32)
    betas = np.linspace(-span, span, GRID_SIZE, dtype=np.float32)
    aa, bb = np.meshgrid(alphas, betas, indexing='xy')
    q0 = q_a.detach().cpu().numpy()
    q = q0[None, :] + aa.reshape(-1, 1) * v1[None, :] + bb.reshape(-1, 1) * v2[None, :]
    return alphas, betas, as_tensor(q.astype(np.float32), device)


def build_branch_grid(q_a: torch.Tensor,
                      q_b: torch.Tensor,
                      v_null: np.ndarray,
                      beta_span: float,
                      device) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    alphas = np.linspace(0.0, 1.0, GRID_SIZE, dtype=np.float32)
    betas = np.linspace(-beta_span, beta_span, GRID_SIZE, dtype=np.float32)
    aa, bb = np.meshgrid(alphas, betas, indexing='xy')
    q_a_np = q_a.detach().cpu().numpy()
    branch_dir = q_b.detach().cpu().numpy() - q_a_np
    q = q_a_np[None, :] + aa.reshape(-1, 1) * branch_dir[None, :] + bb.reshape(-1, 1) * v_null[None, :]
    return alphas, betas, as_tensor(q.astype(np.float32), device)


def evaluate_grid(kin: BatchedFR3Kinematics,
                  q_seed_grid: torch.Tensor,
                  valid: np.ndarray,
                  track_pts: torch.Tensor,
                  plane_normal: torch.Tensor,
                  L_max: float) -> np.ndarray:
    L = rollout_lengths(kin, q_seed_grid, track_pts, plane_normal)
    L_norm = L.reshape(GRID_SIZE, GRID_SIZE) / L_max
    L_norm = L_norm.astype(np.float32)
    L_norm[~valid.reshape(GRID_SIZE, GRID_SIZE)] = np.nan
    return L_norm


def save_heatmap(out_path: Path,
                 x_vals: np.ndarray,
                 y_vals: np.ndarray,
                 z: np.ndarray,
                 title: str,
                 xlabel: str,
                 ylabel: str,
                 markers: list[tuple[float, float, str]] | None = None):
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

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(
        z,
        origin='lower',
        extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]],
        aspect='auto',
        vmin=0.0,
        vmax=1.0,
        cmap=cmap,
    )
    valid = np.isfinite(z)
    if valid.any():
        z_contour = np.where(valid, z, -1.0)
        ax.contour(x_vals, y_vals, z_contour,
                   levels=[0.25, 0.50, 0.75, 0.98],
                   colors='white', linewidths=0.7, alpha=0.75)
    if markers:
        for x, y, label in markers:
            ax.scatter([x], [y], s=42, c='white', edgecolors='black',
                       linewidths=0.8, zorder=5)
            ax.text(x, y, f' {label}', color='white', fontsize=10,
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


def print_grid_stats(name: str, z: np.ndarray):
    valid = np.isfinite(z)
    vals = z[valid]
    if vals.size == 0:
        print(f'{name}: no valid initial TCP points')
        return
    jumps_x = np.abs(z[:, 1:] - z[:, :-1])
    jumps_y = np.abs(z[1:, :] - z[:-1, :])
    jumps = np.concatenate([jumps_x[np.isfinite(jumps_x)], jumps_y[np.isfinite(jumps_y)]])
    print(f'\n{name}')
    print(f'  valid fraction: {float(valid.mean()):.3f}')
    print(f'  success fraction among valid: {float((vals >= 0.995).mean()):.3f}')
    print(f'  mid fraction among valid: {float(((vals < 0.995) & (vals >= 0.8)).mean()):.3f}')
    print(f'  p10/p50/p90 L/Lmax: {np.percentile(vals, 10):.3f} / '
          f'{np.percentile(vals, 50):.3f} / {np.percentile(vals, 90):.3f}')
    if jumps.size:
        print(f'  edge jump p99: {np.percentile(jumps, 99):.3f}')


def save_npz(q_a: torch.Tensor,
             q_b: torch.Tensor,
             basis: np.ndarray,
             null_axes: tuple[np.ndarray, np.ndarray],
             null_grid: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
             branch_grid: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
             L_start: np.ndarray,
             task_path: np.ndarray,
             L_max: float):
    np.savez_compressed(
        OUT_DIR / 'v18_L_2d_slices.npz',
        q_a=q_a.detach().cpu().numpy().astype(np.float32),
        q_b=q_b.detach().cpu().numpy().astype(np.float32),
        nullspace_basis=basis.astype(np.float32),
        v1=null_axes[0].astype(np.float32),
        v2=null_axes[1].astype(np.float32),
        null_alphas=null_grid[0].astype(np.float32),
        null_betas=null_grid[1].astype(np.float32),
        L_null=null_grid[2].astype(np.float32),
        valid_null=null_grid[3],
        branch_alphas=branch_grid[0].astype(np.float32),
        branch_betas=branch_grid[1].astype(np.float32),
        L_branch=branch_grid[2].astype(np.float32),
        valid_branch=branch_grid[3],
        L_start=L_start.astype(np.float32),
        task_path=task_path.astype(np.float32),
        L_max=np.array(L_max, dtype=np.float32),
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kin = BatchedFR3Kinematics(device=device)

    task = sample_line_task(rng, kin)
    task_path = task['fine_path_pts']
    L_max = path_length(task_path)
    track_pts = as_tensor(task_path, device)
    plane_normal = as_tensor(task['plane_normal'], device)

    print(f'task: straight line, L_max={L_max:.3f}m')
    q_set = enumerate_start_iks(kin, rng, task, track_pts)
    print(f'evaluating start IK anchors: {q_set.shape[0]} candidates')
    L_start = rollout_lengths(kin, q_set, track_pts, plane_normal)
    q_a, q_b = choose_branch_pair(q_set, L_start, L_max)

    sig_a = branch_signature(q_a.detach().cpu().numpy())
    sig_b = branch_signature(q_b.detach().cpu().numpy())
    print(f'q_A branch={sig_a}, L/Lmax={rollout_lengths(kin, q_a.unsqueeze(0), track_pts, plane_normal)[0] / L_max:.3f}')
    print(f'q_B branch={sig_b}, L/Lmax={rollout_lengths(kin, q_b.unsqueeze(0), track_pts, plane_normal)[0] / L_max:.3f}')

    basis = nullspace_basis(kin, q_a)
    v1 = basis[0] / (np.linalg.norm(basis[0]) + 1e-12)
    v2 = basis[1] / (np.linalg.norm(basis[1]) + 1e-12)
    R_start = start_rotation(task, track_pts, device)

    a_vals, b_vals, q_null_seed = build_nullspace_grid(q_a, v1, v2, NULLSPACE_SPAN, device)
    q_null, valid_null = project_to_same_start_pose(kin, q_null_seed, track_pts[0], R_start)
    L_null = evaluate_grid(kin, q_null, valid_null, track_pts, plane_normal, L_max)
    save_heatmap(
        OUT_NULLSPACE.with_suffix('.png'),
        a_vals,
        b_vals,
        L_null,
        'Nullspace slice: same TCP start, different internal posture',
        'alpha along nullspace direction v1 [rad]',
        'beta along nullspace direction v2 [rad]',
        markers=[(0.0, 0.0, 'q_A')],
    )
    print_grid_stats('Nullspace slice', L_null)

    alpha_vals, beta_vals, q_branch_seed = build_branch_grid(q_a, q_b, v1, BRANCH_BETA_SPAN, device)
    q_branch, valid_branch = project_to_same_start_pose(kin, q_branch_seed, track_pts[0], R_start)
    L_branch = evaluate_grid(kin, q_branch, valid_branch, track_pts, plane_normal, L_max)
    save_heatmap(
        OUT_BRANCH.with_suffix('.png'),
        alpha_vals,
        beta_vals,
        L_branch,
        'IK branch switching slice: q_A to q_B plus nullspace offset',
        'alpha: linear interpolation q_A -> q_B',
        'beta along q_A nullspace direction v1 [rad]',
        markers=[(0.0, 0.0, 'q_A'), (1.0, 0.0, 'q_B')],
    )
    print_grid_stats('Branch switching slice', L_branch)

    save_npz(
        q_a,
        q_b,
        basis,
        (v1, v2),
        (a_vals, b_vals, L_null, valid_null.reshape(GRID_SIZE, GRID_SIZE)),
        (alpha_vals, beta_vals, L_branch, valid_branch.reshape(GRID_SIZE, GRID_SIZE)),
        L_start,
        task_path,
        L_max,
    )

    print(f'\nsaved: {OUT_NULLSPACE.with_suffix(".png")}')
    print(f'saved: {OUT_BRANCH.with_suffix(".png")}')
    print(f'saved: {OUT_DIR / "v18_L_2d_slices.npz"}')


if __name__ == '__main__':
    main()

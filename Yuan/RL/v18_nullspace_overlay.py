"""Render q_A's 2D task-nullspace sweep as overlaid transparent arms.

Run:
    python -m Yuan.RL.v18_nullspace_overlay
    python -m Yuan.RL.v18_nullspace_overlay --seed 3

Samples a GRID_N x GRID_N grid in the 5-DOF-task nullspace at q_A
(directions v1, v2 from `nullspace_basis`), IK-projects each seed back to
q_A's own FK pose, and draws each surviving configuration as a transparent
arm. Color encodes position in (alpha, beta) via 4-corner bilinear blend,
so neighboring postures look visually similar. q_A itself is more opaque
and gets a TCP frame.
"""
from __future__ import annotations

import argparse
import builtins

import numpy as np
import torch

import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw

from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.RL.batched_rollout import _batched_ik_project
from Yuan.RL.fr3_with_pen import attach_pen_visual, make_fr3_with_pen
from Yuan.RL.v18_landscape_probe import (
    SEED,
    as_tensor,
    choose_branch_pair,
    enumerate_start_iks,
    nullspace_basis,
    path_length,
    rollout_lengths,
    sample_line_task,
)


GRID_N = 3
SPAN = 0.7
ARM_ALPHA = 0.40
ANCHOR_ALPHA = 0.90
# Loose tolerance: this is a visualization, not a metric. Joint-posture
# differences are the point; cm-scale TCP drift is invisible vs the path arc.
START_POS_TOL = 0.05
START_Z_TOL_DEG = 10.0

C00 = np.array([0.85, 0.18, 0.18])   # low alpha, low beta
C10 = np.array([0.95, 0.78, 0.12])   # high alpha, low beta
C01 = np.array([0.18, 0.55, 0.92])   # low alpha, high beta
C11 = np.array([0.22, 0.72, 0.32])   # high alpha, high beta
ANCHOR_COLOR = (0.95, 0.95, 0.95)


def bilinear_colors(alpha_norm: np.ndarray, beta_norm: np.ndarray) -> np.ndarray:
    a = alpha_norm.reshape(-1, 1)
    b = beta_norm.reshape(-1, 1)
    return ((1 - a) * (1 - b) * C00
            + a * (1 - b) * C10
            + (1 - a) * b * C01
            + a * b * C11)


def add_task_path(base, task_path: np.ndarray):
    segs = np.stack([task_path[:-1], task_path[1:]], axis=1)
    ossop.linsegs(segs=segs, radius=0.0015,
                  srgbs=np.array([0.08, 0.08, 0.08]),
                  alpha=0.75).attach_to(base.scene)
    ossop.sphere(pos=tuple(task_path[0]), radius=0.012,
                 rgb=(0.05, 0.65, 0.20), alpha=0.95).attach_to(base.scene)
    ossop.sphere(pos=tuple(task_path[-1]), radius=0.014,
                 rgb=(0.85, 0.10, 0.10), alpha=0.95).attach_to(base.scene)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    seed = int(args.seed)

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kin = BatchedFR3Kinematics(device=device)

    task = sample_line_task(rng, kin)
    task_path = task['fine_path_pts']
    L_max = path_length(task_path)
    track_pts = as_tensor(task_path, device)
    plane_normal = as_tensor(task['plane_normal'], device)

    q_set = enumerate_start_iks(kin, rng, task, track_pts)
    L_start = rollout_lengths(kin, q_set, track_pts, plane_normal)
    q_a, _ = choose_branch_pair(q_set, L_start, L_max)
    print(f'seed={seed}, L_max={L_max:.3f}m, q_A picked')

    basis = nullspace_basis(kin, q_a)
    v1 = basis[0] / (np.linalg.norm(basis[0]) + 1e-12)
    v2 = basis[1] / (np.linalg.norm(basis[1]) + 1e-12)

    p_anchor_t, R_anchor_t, _, _ = kin.tcp_fk_jac(q_a.unsqueeze(0))
    p_anchor = p_anchor_t[0]
    R_anchor = R_anchor_t[0]

    alphas = np.linspace(-SPAN, SPAN, GRID_N, dtype=np.float32)
    betas = np.linspace(-SPAN, SPAN, GRID_N, dtype=np.float32)
    aa, bb = np.meshgrid(alphas, betas, indexing='xy')
    q0 = q_a.detach().cpu().numpy()
    q_seed_np = (q0[None, :]
                 + aa.reshape(-1, 1) * v1[None, :]
                 + bb.reshape(-1, 1) * v2[None, :])
    q_seed = as_tensor(q_seed_np, device)

    n = q_seed.shape[0]
    p_rep = p_anchor.unsqueeze(0).expand(n, 3)
    R_rep = R_anchor.unsqueeze(0).expand(n, 3, 3)
    q_proj, ok, _ = _batched_ik_project(kin, q_seed, p_rep, R_rep,
                                        branch_action=None, preserve_seed=True)

    p_tcp, R_tcp, _, _ = kin.tcp_fk_jac(q_proj)
    pos_err = (p_tcp - p_rep).norm(dim=-1)
    z_err = torch.acos((R_tcp[:, :, 2] * R_rep[:, :, 2]).sum(dim=-1).clamp(-1.0, 1.0))
    strict_ok = (ok
                 & (pos_err <= START_POS_TOL)
                 & (z_err <= np.deg2rad(START_Z_TOL_DEG)))
    valid_np = strict_ok.detach().cpu().numpy()
    print(f'grid {GRID_N}x{GRID_N}, valid arms: {int(valid_np.sum())}/{n}')
    if not valid_np.any():
        raise RuntimeError('no valid arms in the nullspace grid; lower SPAN')

    q_proj_np = q_proj.detach().cpu().numpy()
    span_a = alphas.max() - alphas.min()
    span_b = betas.max() - betas.min()
    alpha_norm = (aa - alphas.min()) / max(span_a, 1e-6)
    beta_norm = (bb - betas.min()) / max(span_b, 1e-6)
    colors = bilinear_colors(alpha_norm, beta_norm)
    aa_flat = aa.reshape(-1)
    bb_flat = bb.reshape(-1)

    base = ovw.World(cam_pos=(1.25, -1.65, 1.15),
                     cam_lookat_pos=(0.25, 0.0, 0.45),
                     toggle_auto_cam_orbit=False)
    builtins.base = base
    add_task_path(base, task_path)

    for idx in range(n):
        if not valid_np[idx]:
            continue
        is_anchor = (abs(aa_flat[idx]) < 1e-9) and (abs(bb_flat[idx]) < 1e-9)
        rgb = ANCHOR_COLOR if is_anchor else tuple(float(c) for c in colors[idx])
        alpha = ANCHOR_ALPHA if is_anchor else ARM_ALPHA
        arm, _ = make_fr3_with_pen(pos=np.array([0.0, 0.0, 0.0], dtype=np.float32))
        arm.attach_to(base.scene)
        arm.rgb = rgb
        arm.alpha = alpha
        attach_pen_visual(arm, rgb=rgb, alpha=0.95)
        if is_anchor:
            arm.toggle_tcp(length_scale=0.08, radius_scale=0.35)
        arm.fk(q_proj_np[idx])

    ossop.frame(length_scale=0.18, radius_scale=0.7).attach_to(base.scene)
    base.run()


if __name__ == '__main__':
    main()

"""Show good and bad start joint configurations in the ONE world viewer.

Run:
    python -m Yuan.RL.v18_landscape_world

Blue transparent arms are long-rollout starts. Red transparent arms fail early.
Configurations are sampled from the nullspace slice used by
``v18_landscape_probe.py`` and overlaid at the same robot base.
"""
from __future__ import annotations

import builtins
import numpy as np
import torch

import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw

from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.RL.fr3_with_pen import attach_pen_visual, make_fr3_with_pen
from Yuan.RL.v18_landscape_probe import (
    NULLSPACE_SPAN,
    EPS_P,
    V_PATH,
    as_tensor,
    enumerate_start_iks,
    nullspace_basis,
    path_length,
    rollout_lengths,
    sample_line_task,
    choose_branch_pair,
    SEED,
    start_rotation,
)
from Yuan.RL.batched_rollout import batched_rollout_segment, _batched_ik_project
from Yuan.RL.v18_data_prep import _build_R_from_normal_direction


N_BLUE = 3
N_RED = 3
VIEW_GRID_SIZE = 61
ARM_ALPHA = 0.32
START_POS_TOL = 0.002
START_Z_TOL_DEG = 1.0
N_RANDOM_IK_SEEDS = 4096


def build_viewer_nullspace_grid(q_a: torch.Tensor, v1: np.ndarray, v2: np.ndarray, device):
    vals = np.linspace(-NULLSPACE_SPAN, NULLSPACE_SPAN, VIEW_GRID_SIZE, dtype=np.float32)
    aa, bb = np.meshgrid(vals, vals, indexing='xy')
    q0 = q_a.detach().cpu().numpy()
    q = q0[None, :] + aa.reshape(-1, 1) * v1[None, :] + bb.reshape(-1, 1) * v2[None, :]
    return as_tensor(q.astype(np.float32), device)


def project_to_same_start_pose(kin: BatchedFR3Kinematics,
                               q_seed: torch.Tensor,
                               p_start: torch.Tensor,
                               R_start: torch.Tensor,
                               preserve_seed: bool = False) -> tuple[torch.Tensor, np.ndarray]:
    p_rep = p_start.unsqueeze(0).expand(q_seed.shape[0], 3)
    R_rep = R_start.unsqueeze(0).expand(q_seed.shape[0], 3, 3)
    q_proj, ok, _ = _batched_ik_project(kin, q_seed, p_rep, R_rep,
                                        branch_action=None, preserve_seed=preserve_seed)

    p_tcp, R_tcp, _, _ = kin.tcp_fk_jac(q_proj)
    pos_err = (p_tcp - p_rep).norm(dim=-1)
    z_err = torch.acos(
        (R_tcp[:, :, 2] * R_rep[:, :, 2]).sum(dim=-1).clamp(-1.0, 1.0))
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


def pick_examples(q_grid: torch.Tensor,
                  L_norm: np.ndarray,
                  valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid_idx = np.where(valid)[0]
    good_idx = valid_idx[L_norm[valid_idx] >= 0.995]
    bad_idx = valid_idx[L_norm[valid_idx] <= 0.05]

    if len(good_idx) == 0:
        good_idx = valid_idx[np.argsort(-L_norm[valid_idx])[:N_BLUE]]
    else:
        good_idx = good_idx[np.linspace(0, len(good_idx) - 1, min(N_BLUE, len(good_idx))).astype(int)]

    if len(bad_idx) > 0:
        bad_idx = bad_idx[np.linspace(0, len(bad_idx) - 1, min(N_RED, len(bad_idx))).astype(int)]

    return (q_grid[good_idx].detach().cpu().numpy(),
            q_grid[bad_idx].detach().cpu().numpy(),
            L_norm[good_idx],
            L_norm[bad_idx])


def random_joint_seeds(kin: BatchedFR3Kinematics,
                       rng: np.random.Generator,
                       n_seeds: int) -> torch.Tensor:
    lo = kin.lmt_lo.detach().cpu().numpy()
    hi = kin.lmt_up.detach().cpu().numpy()
    q = rng.uniform(lo[None, :], hi[None, :], size=(n_seeds, 7)).astype(np.float32)
    return as_tensor(q, kin.device)


def find_global_same_pose_examples(kin: BatchedFR3Kinematics,
                                   rng: np.random.Generator,
                                   p_start: torch.Tensor,
                                   R_start: torch.Tensor,
                                   track_pts: torch.Tensor,
                                   plane_normal: torch.Tensor,
                                   L_max: float):
    q_seed = random_joint_seeds(kin, rng, N_RANDOM_IK_SEEDS)
    q_proj, valid = project_to_same_start_pose(kin, q_seed, p_start, R_start)
    if not valid.any():
        return None

    L_norm = rollout_lengths(kin, q_proj, track_pts, plane_normal) / L_max
    good = np.where(valid & (L_norm >= 0.995))[0]
    bad = np.where(valid & (L_norm <= 0.05))[0]
    print('\nglobal same-start-pose search:')
    print(f'  valid projected IKs: {int(valid.sum())}/{len(valid)}')
    print(f'  good count: {len(good)}')
    print(f'  bad count: {len(bad)}')
    if len(good) == 0 or len(bad) == 0:
        return None

    good = good[np.linspace(0, len(good) - 1, min(N_BLUE, len(good))).astype(int)]
    bad = bad[np.linspace(0, len(bad) - 1, min(N_RED, len(bad))).astype(int)]
    return (q_proj[good].detach().cpu().numpy(),
            q_proj[bad].detach().cpu().numpy(),
            L_norm[good],
            L_norm[bad])


def diagnose_one(kin: BatchedFR3Kinematics,
                 q_np: np.ndarray,
                 track_pts: torch.Tensor,
                 plane_normal: torch.Tensor,
                 L_max: float) -> tuple[float, str, int, float, float]:
    device = kin.device
    q = as_tensor(q_np[None, :], device)
    alive = torch.ones(1, device=device, dtype=torch.bool)
    length = 0.0
    branch_action = torch.tensor([1.0, 0.0, 1.0, 0.0],
                                 device=device, dtype=torch.float32).view(1, 4)

    for seg_idx in range(track_pts.shape[0] - 1):
        p0 = track_pts[seg_idx]
        seg_vec = track_pts[seg_idx + 1] - p0
        seg_len = float(seg_vec.norm().item())
        direction = seg_vec / seg_vec.norm().clamp_min(1e-12)
        rot_np = _build_R_from_normal_direction(
            plane_normal.detach().cpu().numpy(),
            direction.detach().cpu().numpy(),
        )
        n_steps = max(1, int(round(seg_len / (V_PATH * 0.02))))
        out = batched_rollout_segment(
            q_init=q,
            R_tgt=as_tensor(rot_np, device).unsqueeze(0),
            branch_action=branch_action,
            p0=p0.view(1, 3),
            d_dir=direction.view(1, 3),
            v_path=torch.full((1,), V_PATH, device=device, dtype=torch.float32),
            eps_p=torch.full((1,), EPS_P, device=device, dtype=torch.float32),
            T_total=torch.full((1,), n_steps, device=device, dtype=torch.long),
            start_step=0,
            end_step=n_steps,
            kin=kin,
            alive_mask=alive,
        )
        progressed = int(out['lengths'][0].item())
        length += progressed / float(n_steps) * seg_len
        q = out['q_final']
        alive = out['alive_out']
        pos_err = float(out['last_pos_err'][0].item())
        ori_err = float(out['last_orient_err'][0].item())

        if not bool(alive[0].item()):
            if pos_err > EPS_P:
                reason = 'position error'
            elif ori_err > float(np.deg2rad(5.0)):
                reason = 'orientation error'
            else:
                near_lo = bool(((q[0] - kin.lmt_lo) < 1e-4).any().item())
                near_hi = bool(((kin.lmt_up - q[0]) < 1e-4).any().item())
                reason = 'joint limit / collision / threshold'
                if near_lo or near_hi:
                    reason = 'joint limit'
            return length / L_max, reason, seg_idx, pos_err, ori_err

    return length / L_max, 'complete', track_pts.shape[0] - 2, 0.0, 0.0


def print_diagnostics(kin: BatchedFR3Kinematics,
                      q_a: torch.Tensor,
                      blue_qs: np.ndarray,
                      red_qs: np.ndarray,
                      blue_L: np.ndarray,
                      red_L: np.ndarray,
                      track_pts: torch.Tensor,
                      plane_normal: torch.Tensor,
                      L_max: float):
    q_a_np = q_a.detach().cpu().numpy()
    print('\nSelected configuration diagnostics:')
    for i, (q, l_val) in enumerate(zip(blue_qs, blue_L)):
        dist_anchor = float(np.linalg.norm(q - q_a_np))
        p_err, z_err = start_pose_error(kin, q, track_pts[0], plane_normal)
        print(f'  blue {i}: L/Lmax={l_val:.3f}, ||q-q_A||={dist_anchor:.4f} rad, '
              f'start_pos_err={p_err * 1000:.2f}mm, start_z_err={z_err:.2f}deg')

    for i, (q, l_val) in enumerate(zip(red_qs, red_L)):
        dist_anchor = float(np.linalg.norm(q - q_a_np))
        nearest_blue = float(np.min(np.linalg.norm(blue_qs - q[None, :], axis=1)))
        p0_err, z0_err = start_pose_error(kin, q, track_pts[0], plane_normal)
        L_diag, reason, seg_idx, pos_err, ori_err = diagnose_one(
            kin, q, track_pts, plane_normal, L_max)
        print(
            f'  red  {i}: L/Lmax={l_val:.3f} (diag {L_diag:.3f}), '
            f'||q-q_A||={dist_anchor:.4f} rad, nearest-blue={nearest_blue:.4f} rad, '
            f'start_pos_err={p0_err * 1000:.2f}mm, start_z_err={z0_err:.2f}deg, '
            f'first_fail_seg={seg_idx}, reason={reason}, '
            f'pos_err={pos_err:.4f}m, ori_err={np.rad2deg(ori_err):.2f}deg'
        )


def start_pose_error(kin: BatchedFR3Kinematics,
                     q_np: np.ndarray,
                     p_start: torch.Tensor,
                     plane_normal: torch.Tensor) -> tuple[float, float]:
    q = as_tensor(q_np[None, :], kin.device)
    p_tcp, R_tcp, _, _ = kin.tcp_fk_jac(q)
    p_err = float((p_tcp[0] - p_start).norm().item())
    z_tgt = -plane_normal / plane_normal.norm().clamp_min(1e-12)
    z_err = torch.acos((R_tcp[0, :, 2] * z_tgt).sum().clamp(-1.0, 1.0))
    return p_err, float(torch.rad2deg(z_err).item())


def add_arm(base, q: np.ndarray, pos, rgb, alpha: float):
    arm, _ = make_fr3_with_pen(pos=np.asarray(pos, dtype=np.float32))
    arm.attach_to(base.scene)
    arm.rgb = rgb
    arm.alpha = alpha
    attach_pen_visual(arm, rgb=rgb, alpha=0.95)
    arm.toggle_tcp(length_scale=0.08, radius_scale=0.35)
    arm.fk(q)
    return arm


def add_overlay(base, qs: np.ndarray, rgb, alpha: float):
    for q in qs:
        add_arm(base, q, pos=(0.0, 0.0, 0.0), rgb=rgb, alpha=alpha)


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
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
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

    basis = nullspace_basis(kin, q_a)
    v1 = basis[0] / (np.linalg.norm(basis[0]) + 1e-12)
    v2 = basis[1] / (np.linalg.norm(basis[1]) + 1e-12)
    q_seed_grid = build_viewer_nullspace_grid(q_a, v1, v2, device)
    R_start = start_rotation(task, track_pts, device)
    q_grid, valid = project_to_same_start_pose(kin, q_seed_grid, track_pts[0], R_start,
                                               preserve_seed=True)
    L_norm = rollout_lengths(kin, q_grid, track_pts, plane_normal) / L_max

    blue_qs, red_qs, blue_L, red_L = pick_examples(q_grid, L_norm, valid)
    if red_qs.shape[0] == 0:
        print('\nNo true failed samples in the local projected nullspace grid.')
        print('Trying a larger random same-start-pose IK search...')
        found = find_global_same_pose_examples(
            kin, rng, track_pts[0], R_start, track_pts, plane_normal, L_max)
        if found is not None:
            blue_qs, red_qs, blue_L, red_L = found
        else:
            print('No same-start-pose failed samples found. Will show successful samples only.')
    print(f'blue examples: {blue_qs.shape[0]}  red examples: {red_qs.shape[0]}')
    print('blue = L/Lmax >= 0.995, red = L/Lmax <= 0.05')
    print_diagnostics(kin, q_a, blue_qs, red_qs, blue_L, red_L,
                      track_pts, plane_normal, L_max)

    base = ovw.World(cam_pos=(1.25, -1.65, 1.15),
                     cam_lookat_pos=(0.25, 0.0, 0.45),
                     toggle_auto_cam_orbit=False)
    builtins.base = base

    add_task_path(base, task_path)
    add_overlay(base, blue_qs, rgb=(0.05, 0.25, 1.0), alpha=ARM_ALPHA)
    add_overlay(base, red_qs, rgb=(0.95, 0.05, 0.04), alpha=ARM_ALPHA)
    ossop.frame(length_scale=0.18, radius_scale=0.7).attach_to(base.scene)

    base.run()


if __name__ == '__main__':
    main()

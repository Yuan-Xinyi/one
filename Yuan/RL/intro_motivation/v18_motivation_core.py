"""Shared helpers for the v18 motivation pipeline.

Three scripts make up the pipeline:

    Part 1  v18_branch_comparison.py
        Sample a feasible line task, extend it past the reachable workspace
        so every rollout fails partway, farthest-point-pick 16 IK anchors
        (A-P) spanning joint space, run an (elbow x wrist z-tilt) motivation
        slice per anchor, and save 16 per-panel JSONLs plus a 4x4 heatmap.

    Part 2  v18_motivation_animation.py
        Load ONE panel's JSONL, pick a handful of grid cells spanning that
        panel's L_norm range, replay each rollout in the ONE viewer with
        viridis coloring that matches the heatmap.

    Part 3  v18_motivation_overlay.py
        Load all 16 panel JSONLs, draw the 16 anchor poses overlaid in one
        ONE viewer (or grid-laid-out), color-matched to a 4x4 summary PNG.
        Optionally animate every anchor's rollout simultaneously.

This module owns every constant, geometry helper, IK projection, rollout
recorder, and viewer primitive the three scripts share.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import Yuan.RL.config as cfg
import one.scene.scene_object_primitive as ossop

from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.RL.batched_rollout import batched_rollout_segment
from Yuan.RL.v18_curve_eval import sample_curve_task
from Yuan.RL.v18_data_prep import _build_R_from_normal_direction, _dense_ik_at


# ---- task sampling / paths ----
SEED = 2026
N_CHECKPOINTS = 5
N_START_IK = 512
LINE_L_RANGE = (0.30, 0.40)
TARGET_PATH_M = 1.5  # extend the sampled path beyond reach so every rollout
                     # eventually dies; L per cell encodes how far each q got.

# ---- controller / rollout ----
V_PATH = 0.10
EPS_P = 0.05
CHUNK_SIZE = 1024
ROLLOUT_THETA_MAX_DEG = 30.0
ROLLOUT_THETA_MAX = np.deg2rad(ROLLOUT_THETA_MAX_DEG)

# ---- motivation slice geometry ----
ELBOW_SPAN = 0.25
TILT_SPAN_DEG = 60.0
TILT_SPAN = np.deg2rad(TILT_SPAN_DEG)

# ---- IK projection / validity tolerances ----
START_POS_TOL = 0.002
START_Z_TOL_DEG = 80.0
POS_IK_MAX_ITERS = 60
POS_IK_DAMPING = 1e-4
POS_IK_TOL = 1e-3

# ---- animation playback (Parts 2 & 3) ----
PLAYBACK_DT = 0.04
HOLD_AT_START_SEC = 2.0
HOLD_AT_END_SEC = 1.0

# ---- output ----
OUT_DIR = Path(__file__).resolve().parent / 'data'


def seed_dir(seed: int) -> Path:
    """Per-seed output subfolder under OUT_DIR. Created if missing.
    All Part 1 / 2 / 3 outputs for a given task seed live here, so
    batch runs with multiple seeds stay neatly separated."""
    d = OUT_DIR / f'seed{int(seed)}'
    d.mkdir(parents=True, exist_ok=True)
    return d


def as_tensor(x, device):
    return torch.as_tensor(x, device=device, dtype=torch.float32)


def path_length(path: np.ndarray) -> float:
    return float(np.linalg.norm(path[1:] - path[:-1], axis=1).sum())


def sample_line_task(rng: np.random.Generator,
                     kin: BatchedFR3Kinematics,
                     l_range: tuple[float, float] | None = None) -> dict:
    """Rejection-sample a feasible straight-line task. If l_range is None
    sample_curve_task's default is used."""
    for _ in range(100):
        if l_range is None:
            task = sample_curve_task(rng, kin, 'line', N_CHECKPOINTS)
        else:
            task = sample_curve_task(rng, kin, 'line', N_CHECKPOINTS, L_range=l_range)
        if task is not None:
            return task
    raise RuntimeError('failed to sample a feasible straight-line task')


def extend_task_path(task: dict, target_L: float) -> dict:
    """Stretch a feasible task's fine_path_pts along its direction so the
    total length equals target_L. Used to push the path past the reachable
    workspace; rollout L per cell then encodes "how far this q got"
    rather than just '1 = completed'."""
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


def _start_rotation(task: dict, track_pts: torch.Tensor, device) -> torch.Tensor:
    tangent = track_pts[1] - track_pts[0]
    tangent = tangent / tangent.norm().clamp_min(1e-12)
    rot = _build_R_from_normal_direction(task['plane_normal'],
                                         tangent.detach().cpu().numpy())
    return as_tensor(rot, device)


def enumerate_start_iks(kin: BatchedFR3Kinematics,
                        rng: np.random.Generator,
                        task: dict,
                        track_pts: torch.Tensor) -> torch.Tensor:
    """Return a tensor of all IK-feasible q at the task's start TCP."""
    rot0 = _start_rotation(task, track_pts, kin.device)
    q_set, _ = _dense_ik_at(kin, track_pts[0], rot0, N_START_IK, rng)
    if q_set.shape[0] == 0:
        raise RuntimeError('no IK found at line start')
    return q_set


def rollout_chunk(kin: BatchedFR3Kinematics,
                  q_init: torch.Tensor,
                  track_pts: torch.Tensor,
                  plane_normal: torch.Tensor,
                  theta_max_rad: float | None = None,
                  enforce_init_pose: bool = False,
                  pos_priority: bool = False) -> np.ndarray:
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
            theta_max_rad=theta_max_rad,
            enforce_init_pose=enforce_init_pose,
            pos_priority=pos_priority,
        )
        completed = out['lengths'].float() / float(n_steps) * seg_len
        lengths = torch.where(alive, lengths + completed, lengths)
        q = out['q_final']
        alive = out['alive_out']

    return lengths.detach().cpu().numpy()


def rollout_lengths(kin: BatchedFR3Kinematics,
                    q_batch: torch.Tensor,
                    track_pts: torch.Tensor,
                    plane_normal: torch.Tensor,
                    theta_max_rad: float | None = None,
                    enforce_init_pose: bool = False,
                    pos_priority: bool = False) -> np.ndarray:
    """Chunked rollout_lengths to keep peak memory bounded. Returns the
    total path length each q traversed before failing (in meters)."""
    lengths = np.zeros(q_batch.shape[0], dtype=np.float32)
    for start in range(0, q_batch.shape[0], CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, q_batch.shape[0])
        lengths[start:end] = rollout_chunk(kin, q_batch[start:end],
                                           track_pts, plane_normal,
                                           theta_max_rad=theta_max_rad,
                                           enforce_init_pose=enforce_init_pose,
                                           pos_priority=pos_priority)
    return lengths


def _skew(z: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(z[0])
    return torch.stack([
        torch.stack([zero, -z[2], z[1]]),
        torch.stack([z[2], zero, -z[0]]),
        torch.stack([-z[1], z[0], zero]),
    ])


def elbow_wristtilt_basis(kin: BatchedFR3Kinematics,
                          q_a: torch.Tensor,
                          path_tangent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (v_elbow, v_wrist_tilt). Both vectors live in the 4D kernel
    of the position Jacobian at q_a. v_elbow further lies in the 5-DOF
    task nullspace (preserves z-axis) and is orthogonalized against the
    pen-spin direction (~joint-7 axis). v_wrist_tilt is the
    pos-preserving direction whose z-axis motion aligns with the path
    tangent; it is scaled so that 1 unit of beta = 1 rad of z-axis tilt
    at q_a (first-order)."""
    _, R, jac, _ = kin.tcp_fk_jac(q_a.unsqueeze(0))
    j_pos = jac[0, :3, :]
    j_ang = jac[0, 3:, :]
    z = R[0, :, 2]
    j_z = -_skew(z) @ j_ang

    j_pos_np = j_pos.detach().cpu().numpy()
    j_z_np = j_z.detach().cpu().numpy()
    z_np = z.detach().cpu().numpy()

    _, _, vh_pos = np.linalg.svd(j_pos_np, full_matrices=True)
    pos_null = vh_pos[3:, :]                # 4 x 7

    M = j_z_np @ pos_null.T                 # 3 x 4, rank 2
    _, _, vh_M = np.linalg.svd(M, full_matrices=True)
    task_coords = vh_M[2:, :]               # 2 x 4
    tilt_coords = vh_M[:2, :]               # 2 x 4
    task_null = task_coords @ pos_null      # 2 x 7
    wrist_tilt_2d = tilt_coords @ pos_null  # 2 x 7

    e7 = np.zeros(7, dtype=np.float64); e7[6] = 1.0
    spin_components = task_null @ e7
    if np.linalg.norm(spin_components) > 1e-6:
        spin_dir = spin_components @ task_null
        spin_dir /= max(np.linalg.norm(spin_dir), 1e-12)
        v_elbow = task_null[0] - np.dot(task_null[0], spin_dir) * spin_dir
        if np.linalg.norm(v_elbow) < 1e-6:
            v_elbow = task_null[1] - np.dot(task_null[1], spin_dir) * spin_dir
        v_elbow /= max(np.linalg.norm(v_elbow), 1e-12)
    else:
        v_elbow = task_null[0] / max(np.linalg.norm(task_null[0]), 1e-12)

    path_perp = path_tangent - np.dot(path_tangent, z_np) * z_np
    if np.linalg.norm(path_perp) < 1e-6:
        v_wrist_tilt = wrist_tilt_2d[0]
    else:
        path_perp /= np.linalg.norm(path_perp)
        Mw = j_z_np @ wrist_tilt_2d.T
        c, *_ = np.linalg.lstsq(Mw, path_perp, rcond=None)
        if np.linalg.norm(c) < 1e-9:
            v_wrist_tilt = wrist_tilt_2d[0]
        else:
            c /= np.linalg.norm(c)
            v_wrist_tilt = c @ wrist_tilt_2d
    z_dot_mag = np.linalg.norm(j_z_np @ v_wrist_tilt)
    if z_dot_mag > 1e-9:
        v_wrist_tilt = v_wrist_tilt / z_dot_mag
    return v_elbow.astype(np.float32), v_wrist_tilt.astype(np.float32)


def pos_only_ik_project(kin: BatchedFR3Kinematics,
                        q_seed: torch.Tensor,
                        p_target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """DLS IK that only fixes TCP position; z-axis is left free. Used to
    anchor motivation-slice grid points to q_a's TCP without snapping
    z-axis back to z_anchor."""
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


def record_rollout(kin: BatchedFR3Kinematics,
                   q_init: torch.Tensor,
                   track_pts: torch.Tensor,
                   plane_normal_np: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Run the motivation rollout (pos-priority + dead-zone z + init pose
    check) for every row of q_init, recording the joint trajectory across
    all path segments. Returns (q_traj, fail_infos) where:
      q_traj: (T, B, 7) numpy array; dead rows freeze at last alive q.
      fail_infos: length-B list of dicts. Each dict has keys
        {'alive_end', 'segment', 'pos_err_m', 'orient_err_deg', 'reason',
         'near_joint_limit'}. 'reason' is a short tag inferred from the
        last alive step's errors."""
    device = kin.device
    batch_size = q_init.shape[0]
    q = q_init.clone()
    alive = torch.ones(batch_size, device=device, dtype=torch.bool)
    branch_action = torch.tensor([1.0, 0.0, 1.0, 0.0], device=device,
                                 dtype=torch.float32).unsqueeze(0).expand(batch_size, 4)
    q_traj_pieces = [q.unsqueeze(0).clone()]
    fail_info_list: list[dict | None] = [None] * batch_size

    for idx in range(track_pts.shape[0] - 1):
        p0 = track_pts[idx]
        seg_vec = track_pts[idx + 1] - p0
        seg_len = float(seg_vec.norm().item())
        if seg_len < 1e-8:
            continue
        direction = seg_vec / seg_vec.norm().clamp_min(1e-12)
        rot_np = _build_R_from_normal_direction(
            plane_normal_np, direction.detach().cpu().numpy())
        n_steps = max(1, int(round(seg_len / (V_PATH * float(cfg.DT)))))
        prev_alive_np = alive.detach().cpu().numpy().copy()
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
            theta_max_rad=ROLLOUT_THETA_MAX,
            enforce_init_pose=True,
            record_traj=True,
            pos_priority=True,
        )
        q_traj_pieces.append(out['q_record'][1:])
        q = out['q_final']
        alive = out['alive_out']
        new_alive_np = alive.detach().cpu().numpy()
        died_mask = prev_alive_np & ~new_alive_np
        if died_mask.any():
            last_pos = out['last_pos_err'].detach().cpu().numpy()
            last_ori = out['last_orient_err'].detach().cpu().numpy()
            q_np = q.detach().cpu().numpy()
            lo = kin.lmt_lo.detach().cpu().numpy()
            hi = kin.lmt_up.detach().cpu().numpy()
            # 0.05 rad (~3 deg) = roughly one controller step (qdot_max * DT).
            # The "last alive q" is one step before the death-triggering q_new,
            # so a joint within this margin almost certainly hit the limit
            # on the next step.
            joint_limit_margin = 0.05
            for i in np.where(died_mask)[0]:
                pos_err_m = float(last_pos[i])
                ori_err_deg = float(np.rad2deg(last_ori[i]))
                near_lo = ((q_np[i] - lo) < joint_limit_margin).any()
                near_hi = ((hi - q_np[i]) < joint_limit_margin).any()
                if ori_err_deg > ROLLOUT_THETA_MAX_DEG:
                    reason = 'orient_err_exceeded'
                elif pos_err_m > EPS_P:
                    reason = 'pos_err_exceeded'
                elif near_lo or near_hi:
                    reason = 'joint_limit'
                else:
                    reason = 'other'
                fail_info_list[int(i)] = {
                    'alive_end': False,
                    'segment': int(idx),
                    'pos_err_m': pos_err_m,
                    'orient_err_deg': ori_err_deg,
                    'reason': reason,
                    'near_joint_limit': bool(near_lo or near_hi),
                }

    for i in range(batch_size):
        if fail_info_list[i] is None:
            fail_info_list[i] = {
                'alive_end': True,
                'segment': int(track_pts.shape[0] - 2),
                'pos_err_m': 0.0,
                'orient_err_deg': 0.0,
                'reason': 'completed_path',
                'near_joint_limit': False,
            }

    q_traj_np = torch.cat(q_traj_pieces, dim=0).detach().cpu().numpy()
    return q_traj_np, fail_info_list


def add_task_path(base,
                  task_path: np.ndarray,
                  plane_normal: np.ndarray | None = None,
                  plane_size: float | None = None,
                  plane_rgb: tuple[float, float, float] = (0.82, 0.82, 0.86),
                  plane_alpha: float = 0.25):
    """Draw the task path line + endpoint spheres in the ONE viewer. If
    plane_normal is given, also draw a translucent plane (the "paper")
    auto-sized to cover the full path plus a margin."""
    if plane_normal is not None:
        plane_center = task_path.mean(axis=0).astype(np.float32)
        path_len = float(np.linalg.norm(task_path[-1] - task_path[0]))
        if plane_size is None:
            plane_size = max(2.0, path_len + 0.6)
        ossop.plane(pos=tuple(plane_center),
                    normal=tuple(plane_normal.astype(np.float32)),
                    size=(plane_size, plane_size),
                    thickness=2e-3,
                    rgb=plane_rgb,
                    alpha=plane_alpha).attach_to(base.scene)
    segs = np.stack([task_path[:-1], task_path[1:]], axis=1)
    ossop.linsegs(segs=segs, radius=0.0015,
                  srgbs=np.array([0.08, 0.08, 0.08]),
                  alpha=0.75).attach_to(base.scene)
    ossop.sphere(pos=tuple(task_path[0]), radius=0.012,
                 rgb=(0.05, 0.65, 0.20), alpha=0.95).attach_to(base.scene)
    ossop.sphere(pos=tuple(task_path[-1]), radius=0.014,
                 rgb=(0.85, 0.10, 0.10), alpha=0.95).attach_to(base.scene)

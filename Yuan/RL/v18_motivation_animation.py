"""Animate three rollouts loaded from a motivation-probe JSONL.

Run:
    python -m Yuan.RL.v18_motivation_animation
    python -m Yuan.RL.v18_motivation_animation --jsonl ... \
        --targets 1.0,0.6,0.5

The JSONL is produced by v18_motivation_probe and contains the full grid
of (alpha, beta) cells with their post-IK q_init and rollout lengths. For
each target normalized length, we find the closest grid cell and replay
its rollout in the ONE viewer. Three arms (yellow / blue / green) move
along the task path simultaneously; dead arms freeze at their last alive
joint state.
"""
from __future__ import annotations

import argparse
import builtins
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

import Yuan.RL.config as cfg
import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw

from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.RL.batched_rollout import batched_rollout_segment
from Yuan.RL.fr3_with_pen import attach_pen_visual, make_fr3_with_pen
from Yuan.RL.v18_data_prep import _build_R_from_normal_direction
from Yuan.RL.v18_landscape_probe import (
    EPS_P,
    OUT_DIR,
    V_PATH,
    as_tensor,
)
from Yuan.RL.v18_motivation_probe import ROLLOUT_THETA_MAX


ARM_ALPHA = 0.55
PLAYBACK_DT = 0.04
HOLD_AT_START_SEC = 2.0
HOLD_AT_END_SEC = 1.0
DEFAULT_JSONL = OUT_DIR / 'v18_branch_comparison_seed2026_K.jsonl'


def load_meta_and_entries(jsonl_path: Path) -> tuple[dict, list[dict]]:
    meta = None
    entries = []
    with open(jsonl_path) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get('type') == 'meta':
                meta = obj
            else:
                entries.append(obj)
    if meta is None:
        raise RuntimeError(f'no meta record in {jsonl_path}')
    print(f'loaded meta + {len(entries)} grid entries from {jsonl_path}')
    return meta, entries


def pick_representative_levels(entries: list[dict], n_arms: int,
                               min_L_norm: float = 0.1) -> list[dict]:
    """Pick up to n_arms entries whose length_normalized spans the panel's
    [min_L_norm, panel_max_L_norm] range as evenly as the actual data
    allows. Entries with L below min_L_norm are excluded (they freeze
    immediately). Sorted by L_norm ascending."""
    n_arms = max(1, min(int(n_arms), 10))
    ln_arr = np.array([e['length_normalized']
                       if e['length_normalized'] is not None else np.nan
                       for e in entries], dtype=np.float64)
    valid_mask = np.isfinite(ln_arr) & (ln_arr >= min_L_norm)
    if not valid_mask.any():
        raise RuntimeError(f'no entries with L_norm >= {min_L_norm}')
    valid_indices = np.where(valid_mask)[0]
    valid_ln = ln_arr[valid_indices]
    panel_max = float(valid_ln.max())
    targets = np.linspace(min_L_norm, panel_max, n_arms)
    print(f'panel L_norm range used: [{min_L_norm:.3f}, {panel_max:.3f}]')

    chosen = set()
    picks: list[dict] = []
    for tgt in targets:
        order = np.argsort(np.abs(valid_ln - tgt))
        for o in order:
            cand = int(valid_indices[o])
            if cand in chosen:
                continue
            chosen.add(cand)
            picks.append(entries[cand])
            break
    picks.sort(key=lambda e: e['length_normalized'])
    return picks


def record_rollout(kin: BatchedFR3Kinematics,
                   q_init: torch.Tensor,
                   track_pts: torch.Tensor,
                   plane_normal_np: np.ndarray,
                   theta_max_rad: float,
                   enforce_init_pose: bool) -> np.ndarray:
    device = kin.device
    batch_size = q_init.shape[0]
    q = q_init.clone()
    alive = torch.ones(batch_size, device=device, dtype=torch.bool)
    branch_action = torch.tensor([1.0, 0.0, 1.0, 0.0], device=device,
                                 dtype=torch.float32).unsqueeze(0).expand(batch_size, 4)
    q_traj_pieces = [q.unsqueeze(0).clone()]

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
            record_traj=True,
            pos_priority=True,
        )
        q_records = out['q_record'][1:]
        q_traj_pieces.append(q_records)
        q = out['q_final']
        alive = out['alive_out']

    q_traj = torch.cat(q_traj_pieces, dim=0)
    return q_traj.detach().cpu().numpy()


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
    parser.add_argument('--jsonl', type=str, default=str(DEFAULT_JSONL),
                        help='per-branch JSONL with full grid (default: branch F)')
    parser.add_argument('--n_arms', type=int, default=8,
                        help='number of arms to render (max 10)')
    parser.add_argument('--min_L_norm', type=float, default=0.1,
                        help='drop picks with L_norm below this (purple/dead arms)')
    parser.add_argument('--playback_speed', type=float, default=0.25,
                        help='rollout playback rate (1.0 = real-time, 0.25 = 4x slowmo)')
    args = parser.parse_args()

    meta, entries = load_meta_and_entries(Path(args.jsonl))
    picks = pick_representative_levels(entries, args.n_arms,
                                       min_L_norm=float(args.min_L_norm))
    cmap = matplotlib.colormaps['viridis']
    global_max = float(meta.get('global_max_L_norm') or 1.0)
    if global_max <= 1e-6:
        global_max = 1.0
    colors = [tuple(float(c) for c in cmap(min(p['length_normalized'] / global_max, 1.0))[:3])
              for p in picks]
    print(f'picked {len(picks)} arms (anchor={meta.get("anchor_label", "?")}, '
          f'branch={meta.get("branch_signature")}, global_max_L_norm={global_max:.3f}):')
    for p, rgb in zip(picks, colors):
        print(f'  L_norm={p["length_normalized"]:.3f}, '
              f'L_m={p["length_m"]:.3f}, '
              f'alpha={p["alpha_rad"]:+.3f} rad, '
              f'beta={p["beta_deg"]:+.2f} deg, '
              f'init_orient={p["init_orient_err_deg"]:.2f} deg, '
              f'rgb=({rgb[0]:.2f},{rgb[1]:.2f},{rgb[2]:.2f})')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kin = BatchedFR3Kinematics(device=device)
    task_path = np.array(meta['task_path'], dtype=np.float32)
    plane_normal_np = np.array(meta['plane_normal'], dtype=np.float32)
    track_pts = as_tensor(task_path, device)

    q_inits_np = np.stack(
        [np.array(p['q_init'], dtype=np.float32) for p in picks], axis=0)
    q_inits = as_tensor(q_inits_np, device)

    q_traj_np = record_rollout(kin, q_inits, track_pts, plane_normal_np,
                               theta_max_rad=ROLLOUT_THETA_MAX,
                               enforce_init_pose=True)
    print(f'recorded trajectory: {q_traj_np.shape[0]} frames, '
          f'shape={q_traj_np.shape}')

    base = ovw.World(cam_pos=(1.25, -1.65, 1.15),
                     cam_lookat_pos=(0.25, 0.0, 0.45),
                     toggle_auto_cam_orbit=False)
    builtins.base = base
    add_task_path(base, task_path)

    arms = []
    for i, rgb in enumerate(colors):
        arm, _ = make_fr3_with_pen(pos=np.array([0.0, 0.0, 0.0], dtype=np.float32))
        arm.attach_to(base.scene)
        arm.rgb = rgb
        arm.alpha = ARM_ALPHA
        attach_pen_visual(arm, rgb=rgb, alpha=0.95)
        arm.fk(q_traj_np[0, i])
        arms.append(arm)

    ossop.frame(length_scale=0.18, radius_scale=0.7).attach_to(base.scene)

    speed = max(0.01, float(args.playback_speed))
    rollout_steps_per_tick = speed * (PLAYBACK_DT / float(cfg.DT))
    n_frames = q_traj_np.shape[0]
    state = {'rollout_step_float': 0.0, 'hold_remaining': HOLD_AT_START_SEC}

    def animate(dt, *_args, **_kwargs):
        if state['hold_remaining'] > 0.0:
            state['hold_remaining'] -= dt
            frozen_idx = int(state['rollout_step_float'])
            for i, arm in enumerate(arms):
                arm.fk(q_traj_np[frozen_idx, i])
            return
        idx = int(state['rollout_step_float'])
        if idx >= n_frames:
            state['rollout_step_float'] = 0.0
            state['hold_remaining'] = HOLD_AT_END_SEC + HOLD_AT_START_SEC
            idx = 0
        for i, arm in enumerate(arms):
            arm.fk(q_traj_np[idx, i])
        state['rollout_step_float'] += rollout_steps_per_tick

    base.schedule_interval(animate, PLAYBACK_DT)
    base.run()


if __name__ == '__main__':
    main()

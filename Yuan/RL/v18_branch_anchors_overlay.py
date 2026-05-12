"""Overlay the 16 branch-comparison anchors in ONE viewer, colored by L_self.

Run:
    python -m Yuan.RL.v18_branch_anchors_overlay
    python -m Yuan.RL.v18_branch_anchors_overlay --animate
    python -m Yuan.RL.v18_branch_anchors_overlay --pattern '*_seed2026_*.jsonl'

Loads the meta records of v18_branch_comparison's per-panel JSONL files
(default: all 16 panels for seed 2026), reads each anchor's q_anchor and
L_self_normalized, and renders one transparent FR3 arm per anchor in the
same ONE world. Color = viridis(L_self / global_max_L_self) so the
"best" anchor is yellow and the "worst" is purple, exactly matching the
branch_comparison heatmap colorbar.

With --animate, the script also replays each anchor's rollout
simultaneously, so you can watch which anchors complete the path vs die
partway.
"""
from __future__ import annotations

import argparse
import builtins
import glob
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


ARM_ALPHA = 0.50
PLAYBACK_DT = 0.04
HOLD_AT_START_SEC = 2.0
HOLD_AT_END_SEC = 1.0


def load_anchor_metas(pattern: str) -> list[dict]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f'no JSONL files matched: {pattern}')
    metas = []
    for p in paths:
        with open(p) as f:
            first = next(f)
            obj = json.loads(first)
            if obj.get('type') != 'meta':
                continue
            obj['_path'] = p
            metas.append(obj)
    if not metas:
        raise SystemExit(f'no meta records found in {pattern}')
    return metas


def add_task_path(base, task_path: np.ndarray):
    segs = np.stack([task_path[:-1], task_path[1:]], axis=1)
    ossop.linsegs(segs=segs, radius=0.0015,
                  srgbs=np.array([0.08, 0.08, 0.08]),
                  alpha=0.75).attach_to(base.scene)
    ossop.sphere(pos=tuple(task_path[0]), radius=0.012,
                 rgb=(0.05, 0.65, 0.20), alpha=0.95).attach_to(base.scene)
    ossop.sphere(pos=tuple(task_path[-1]), radius=0.014,
                 rgb=(0.85, 0.10, 0.10), alpha=0.95).attach_to(base.scene)


def record_rollout(kin: BatchedFR3Kinematics,
                   q_init: torch.Tensor,
                   track_pts: torch.Tensor,
                   plane_normal_np: np.ndarray) -> np.ndarray:
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
            theta_max_rad=ROLLOUT_THETA_MAX,
            enforce_init_pose=True,
            record_traj=True,
            pos_priority=True,
        )
        q_records = out['q_record'][1:]
        q_traj_pieces.append(q_records)
        q = out['q_final']
        alive = out['alive_out']

    q_traj = torch.cat(q_traj_pieces, dim=0)
    return q_traj.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern', type=str,
                        default=str(OUT_DIR / 'v18_branch_comparison_seed2026_*.jsonl'),
                        help='glob pattern for per-panel JSONL files')
    parser.add_argument('--animate', action='store_true',
                        help='also play back each anchor rollout in the viewer')
    parser.add_argument('--playback_speed', type=float, default=0.25,
                        help='rollout playback rate (1.0 = real-time)')
    args = parser.parse_args()

    metas = load_anchor_metas(args.pattern)
    L_self_list = [float(m['L_self_normalized']) for m in metas]
    global_max = max(L_self_list) if max(L_self_list) > 1e-6 else 1.0
    cmap = matplotlib.colormaps['viridis']

    print(f'loaded {len(metas)} anchors:')
    for m, L_self in zip(metas, L_self_list):
        rel = L_self / global_max
        rgb = cmap(rel)[:3]
        print(f'  q_{m["anchor_label"]}: branch={tuple(m["branch_signature"])}, '
              f'L_self={L_self:.3f}  (rel {rel:.2f}), '
              f'rgb=({rgb[0]:.2f},{rgb[1]:.2f},{rgb[2]:.2f})')

    task_path = np.array(metas[0]['task_path'], dtype=np.float32)
    plane_normal_np = np.array(metas[0]['plane_normal'], dtype=np.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kin = BatchedFR3Kinematics(device=device)

    q_anchors = np.stack(
        [np.array(m['q_anchor'], dtype=np.float32) for m in metas], axis=0)

    q_traj_np = None
    n_frames = 1
    if args.animate:
        track_pts = as_tensor(task_path, device)
        q_init = as_tensor(q_anchors, device)
        q_traj_np = record_rollout(kin, q_init, track_pts, plane_normal_np)
        n_frames = q_traj_np.shape[0]
        print(f'recorded trajectory: {n_frames} frames for {len(metas)} anchors')

    base = ovw.World(cam_pos=(1.25, -1.65, 1.15),
                     cam_lookat_pos=(0.25, 0.0, 0.45),
                     toggle_auto_cam_orbit=False)
    builtins.base = base
    add_task_path(base, task_path)

    arms = []
    for m, L_self in zip(metas, L_self_list):
        rel = L_self / global_max
        rgb = tuple(float(c) for c in cmap(rel)[:3])
        arm, _ = make_fr3_with_pen(pos=np.array([0.0, 0.0, 0.0], dtype=np.float32))
        arm.attach_to(base.scene)
        arm.rgb = rgb
        arm.alpha = ARM_ALPHA
        attach_pen_visual(arm, rgb=rgb, alpha=0.95)
        arm.fk(np.array(m['q_anchor'], dtype=np.float32))
        arms.append(arm)

    ossop.frame(length_scale=0.18, radius_scale=0.7).attach_to(base.scene)

    if args.animate and q_traj_np is not None:
        speed = max(0.01, float(args.playback_speed))
        rollout_steps_per_tick = speed * (PLAYBACK_DT / float(cfg.DT))
        state = {'rollout_step_float': 0.0, 'hold_remaining': HOLD_AT_START_SEC}

        def animate(dt, *_args, **_kwargs):
            if state['hold_remaining'] > 0.0:
                state['hold_remaining'] -= dt
                idx0 = int(state['rollout_step_float'])
                for i, arm in enumerate(arms):
                    arm.fk(q_traj_np[idx0, i])
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

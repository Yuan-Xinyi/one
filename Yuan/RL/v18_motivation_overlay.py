"""Render the 16 branch-comparison anchors as overlaid arms in ONE viewer
plus a matching 4x4 PNG summary.

Run:
    python -m Yuan.RL.v18_motivation_overlay
    python -m Yuan.RL.v18_motivation_overlay --seed 3

Loads the meta records of v18_branch_comparison's per-panel JSONL files
for the given seed (16 panels A-P), reads each anchor's q_anchor +
L_self_normalized + branch_signature + global_max_L_norm, and renders
two views:

  - 4x4 PNG with each cell colored by viridis(L_self / global_max),
    annotated with anchor letter, branch tuple, and L_self.
  - ONE viewer with 16 transparent arms at their q_anchor poses, colored
    with the same viridis mapping so the PNG and viewer are visually
    synchronized.

Use this to compare the 16 distinct IK-branch starts at the same task TCP.
"""
from __future__ import annotations

import argparse
import builtins
import glob
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw

import Yuan.RL.config as cfg
from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.RL.batched_rollout import batched_rollout_segment
from Yuan.RL.fr3_with_pen import attach_pen_visual, make_fr3_with_pen
from Yuan.RL.v18_data_prep import _build_R_from_normal_direction
from Yuan.RL.v18_landscape_probe import (
    EPS_P,
    OUT_DIR,
    SEED,
    V_PATH,
    as_tensor,
)
from Yuan.RL.v18_motivation_probe import ROLLOUT_THETA_MAX


ARM_ALPHA = 0.45
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
    return metas


def save_summary_png(out_path: Path, metas: list[dict], global_max: float):
    matplotlib.use('Agg')
    cmap = plt.get_cmap('viridis').copy()

    n = len(metas)
    ncol = int(np.ceil(np.sqrt(n)))
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.2 * ncol, 2.2 * nrow), squeeze=False)
    plt.rcParams.update({'font.size': 9})

    for idx, m in enumerate(metas):
        r, c = idx // ncol, idx % ncol
        ax = axes[r][c]
        L_self = float(m['L_self_normalized'])
        rel = min(L_self / max(global_max, 1e-9), 1.0)
        rgb = cmap(rel)[:3]
        ax.set_facecolor((rgb[0], rgb[1], rgb[2]))
        ax.set_xticks([])
        ax.set_yticks([])
        text_color = 'white' if rel < 0.5 else 'black'
        ax.text(0.5, 0.75, f"q_{m['anchor_label']}",
                ha='center', va='center', transform=ax.transAxes,
                color=text_color, fontsize=14, fontweight='bold')
        ax.text(0.5, 0.5, f"branch={tuple(m['branch_signature'])}",
                ha='center', va='center', transform=ax.transAxes,
                color=text_color, fontsize=9)
        ax.text(0.5, 0.30, f"L_self={L_self:.3f}",
                ha='center', va='center', transform=ax.transAxes,
                color=text_color, fontsize=9)
        ax.text(0.5, 0.16, f"rel={rel:.2f}",
                ha='center', va='center', transform=ax.transAxes,
                color=text_color, fontsize=9)

    for idx in range(n, nrow * ncol):
        r, c = idx // ncol, idx % ncol
        axes[r][c].axis('off')

    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.7,
                 label=f'L_self / global_max  (global_max={global_max:.3f})')
    fig.suptitle('16-anchor summary: each square is one panel\'s anchor', y=1.02)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    fig.savefig(out_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


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


def add_task_path(base, task_path: np.ndarray,
                  plane_normal: np.ndarray | None = None,
                  plane_size: float | None = None,
                  plane_rgb: tuple[float, float, float] = (0.82, 0.82, 0.86),
                  plane_alpha: float = 0.25):
    if plane_normal is not None:
        plane_center = task_path.mean(axis=0).astype(np.float32)
        # Auto-size so the plane covers the full path + margin on both sides.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--pattern', type=str, default=None,
                        help='override glob (defaults to seed-based)')
    parser.add_argument('--n_arms', type=int, default=8,
                        help='how many anchors to render in viewer (1-16, '
                             'picked along the L_self spectrum). The PNG '
                             'summary still shows all 16.')
    parser.add_argument('--layout', type=str, choices=['stack', 'grid'],
                        default='stack',
                        help='stack: all at same base (default). grid: place '
                             'each arm at its own base in a 4x4 layout.')
    parser.add_argument('--grid_spacing', type=float, default=1.0,
                        help='spacing between bases when --layout=grid (m)')
    parser.add_argument('--animate', action='store_true',
                        help='replay each rendered anchor\'s rollout simultaneously')
    parser.add_argument('--playback_speed', type=float, default=0.25,
                        help='rollout playback rate (1.0 = real-time)')
    args = parser.parse_args()
    seed = int(args.seed)
    pattern = args.pattern or str(
        OUT_DIR / f'v18_branch_comparison_seed{seed}_*.jsonl')

    metas = load_anchor_metas(pattern)
    global_max = float(metas[0].get('global_max_L_norm') or
                       max(m['L_self_normalized'] for m in metas))
    cmap = matplotlib.colormaps['viridis']

    print(f'loaded {len(metas)} anchors (seed={seed}, global_max={global_max:.3f}):')
    for m in metas:
        L_self = float(m['L_self_normalized'])
        rel = min(L_self / max(global_max, 1e-9), 1.0)
        rgb = cmap(rel)[:3]
        print(f"  q_{m['anchor_label']}: branch={tuple(m['branch_signature'])}, "
              f"L_self={L_self:.3f} (rel {rel:.2f}), "
              f"rgb=({rgb[0]:.2f},{rgb[1]:.2f},{rgb[2]:.2f})")

    out_png = OUT_DIR / f'v18_motivation_overlay_summary_seed{seed}.png'
    save_summary_png(out_png, metas, global_max)
    print(f'\nsaved: {out_png}')

    # Pick a subset of metas spaced along L_self for the viewer.
    n_show = max(1, min(int(args.n_arms), len(metas)))
    metas_sorted = sorted(metas, key=lambda m: m['L_self_normalized'])
    if n_show >= len(metas):
        viewer_metas = metas_sorted
    else:
        idxs = np.linspace(0, len(metas_sorted) - 1, n_show).astype(int)
        viewer_metas = [metas_sorted[i] for i in idxs]
    print(f'viewer renders {len(viewer_metas)} of {len(metas)} anchors, '
          f'layout={args.layout}')

    task_path = np.array(metas[0]['task_path'], dtype=np.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BatchedFR3Kinematics(device=device)

    base = ovw.World(cam_pos=(1.25, -1.65, 1.15),
                     cam_lookat_pos=(0.25, 0.0, 0.45),
                     toggle_auto_cam_orbit=False)
    builtins.base = base

    plane_normal_np = np.array(metas[0]['plane_normal'], dtype=np.float32)
    if args.layout == 'stack':
        add_task_path(base, task_path, plane_normal=plane_normal_np)
        arm_positions = [(0.0, 0.0, 0.0)] * len(viewer_metas)
    else:
        ncol = int(np.ceil(np.sqrt(len(viewer_metas))))
        spacing = float(args.grid_spacing)
        arm_positions = []
        for idx in range(len(viewer_metas)):
            r = idx // ncol
            c = idx % ncol
            shift_x = (c - (ncol - 1) / 2.0) * spacing
            shift_y = (r - (ncol - 1) / 2.0) * spacing
            arm_positions.append((shift_x, shift_y, 0.0))
            shifted_path = task_path + np.array([shift_x, shift_y, 0.0],
                                                dtype=np.float32)
            add_task_path(base, shifted_path, plane_normal=plane_normal_np)

    arms = []
    for m, pos in zip(viewer_metas, arm_positions):
        L_self = float(m['L_self_normalized'])
        rel = min(L_self / max(global_max, 1e-9), 1.0)
        rgb = tuple(float(c) for c in cmap(rel)[:3])
        arm, _ = make_fr3_with_pen(pos=np.array(pos, dtype=np.float32))
        arm.attach_to(base.scene)
        arm.rgb = rgb
        arm.alpha = (ARM_ALPHA if args.layout == 'stack' else 0.92)
        attach_pen_visual(arm, rgb=rgb, alpha=0.95)
        arm.fk(np.array(m['q_anchor'], dtype=np.float32))
        arms.append(arm)

    ossop.frame(length_scale=0.18, radius_scale=0.7).attach_to(base.scene)

    if args.animate:
        track_pts = as_tensor(task_path, device)
        kin = BatchedFR3Kinematics(device=device)
        q_inits_np = np.stack(
            [np.array(m['q_anchor'], dtype=np.float32) for m in viewer_metas], axis=0)
        q_inits = as_tensor(q_inits_np, device)
        q_traj_np = record_rollout(kin, q_inits, track_pts, plane_normal_np)
        n_frames = q_traj_np.shape[0]
        print(f'recorded trajectory: {n_frames} frames for {len(viewer_metas)} arms')

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

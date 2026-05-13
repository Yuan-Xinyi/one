"""Part 3 - Across-branch anchor overlay + summary PNG.

Run:
    python -m Yuan.RL.v18_motivation_overlay
    python -m Yuan.RL.v18_motivation_overlay --seed 3
    python -m Yuan.RL.v18_motivation_overlay --n_arms 16 --layout grid
    python -m Yuan.RL.v18_motivation_overlay --animate

Loads the meta records of all 16 per-panel JSONLs produced by Part 1 for
a given seed. Two outputs:

  - 4x4 summary PNG with each cell colored by viridis(L_self/global_max),
    annotated with anchor letter / branch / L_self.
  - ONE viewer with `n_arms` of the 16 anchors rendered as transparent
    overlaid arms (--layout=stack, default) or spread across a 4x4 base
    grid (--layout=grid). Arm colors match the PNG cell colors.

With --animate, every rendered arm replays its rollout simultaneously;
high L_self arms walk far before freezing, low L_self arms fail early.
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

import Yuan.RL.config as cfg
import one.scene.scene_object_primitive as ossop
import one.viewer.world as ovw

from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.RL.fr3_with_pen import attach_pen_visual, make_fr3_with_pen
from Yuan.RL.intro_motivation.v18_motivation_core import (
    HOLD_AT_END_SEC,
    HOLD_AT_START_SEC,
    PLAYBACK_DT,
    SEED,
    add_task_path,
    as_tensor,
    record_rollout,
    seed_dir,
)


ARM_ALPHA = 0.45


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
        ax.text(0.5, 0.75, f"q_{m['anchor_label']}", ha='center', va='center',
                transform=ax.transAxes, color=text_color, fontsize=14, fontweight='bold')
        ax.text(0.5, 0.5, f"branch={tuple(m['branch_signature'])}",
                ha='center', va='center', transform=ax.transAxes,
                color=text_color, fontsize=9)
        ax.text(0.5, 0.30, f"L_self={L_self:.3f}", ha='center', va='center',
                transform=ax.transAxes, color=text_color, fontsize=9)
        ax.text(0.5, 0.16, f"rel={rel:.2f}", ha='center', va='center',
                transform=ax.transAxes, color=text_color, fontsize=9)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--pattern', type=str, default=None,
                        help='override glob (defaults to seed-based)')
    parser.add_argument('--n_arms', type=int, default=16,
                        help='how many anchors to render in viewer (1-16, '
                             'picked along the L_self spectrum). PNG still '
                             'always shows all 16.')
    parser.add_argument('--layout', type=str, choices=['stack', 'grid'],
                        default='stack',
                        help='stack: all at same base (default). '
                             'grid: separate bases in a 4x4 layout.')
    parser.add_argument('--grid_spacing', type=float, default=1.0,
                        help='spacing between bases when --layout=grid (m)')
    parser.add_argument('--animate', action='store_true',
                        help='replay each rendered anchor\'s rollout simultaneously')
    parser.add_argument('--playback_speed', type=float, default=0.25,
                        help='rollout playback rate (1.0 = real-time)')
    args = parser.parse_args()

    seed = int(args.seed)
    pattern = args.pattern or str(
        seed_dir(seed) / 'branch_comparison_*.jsonl')
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

    out_png = seed_dir(seed) / 'overlay_summary.png'
    save_summary_png(out_png, metas, global_max)
    print(f'\nsaved: {out_png}')

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
    plane_normal_np = np.array(metas[0]['plane_normal'], dtype=np.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kin = BatchedFR3Kinematics(device=device)

    base = ovw.World(cam_pos=(1.25, -1.65, 1.15),
                     cam_lookat_pos=(0.25, 0.0, 0.45),
                     toggle_auto_cam_orbit=False)
    builtins.base = base

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
        q_inits = as_tensor(np.stack(
            [np.array(m['q_anchor'], dtype=np.float32) for m in viewer_metas], axis=0),
            device)
        q_traj_np, fail_infos = record_rollout(kin, q_inits, track_pts, plane_normal_np)
        n_frames = q_traj_np.shape[0]
        print(f'recorded trajectory: {n_frames} frames for {len(viewer_metas)} arms')
        print('rollout outcomes:')
        for m, info in zip(viewer_metas, fail_infos):
            print(f'  q_{m["anchor_label"]} (L_self={m["L_self_normalized"]:.3f}): '
                  f'{info["reason"]}, seg={info["segment"]}, '
                  f'pos_err={info["pos_err_m"]*1000:.1f}mm, '
                  f'orient_err={info["orient_err_deg"]:.1f}deg'
                  + (' [near joint limit]' if info['near_joint_limit'] else ''))

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

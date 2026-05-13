"""Part 2 - Within-branch rollout animation.

Run:
    python -m Yuan.RL.v18_motivation_animation
    python -m Yuan.RL.v18_motivation_animation \
        --jsonl Yuan/RL/data/v18_branch_comparison_seed2026_K.jsonl

Loads ONE per-panel JSONL produced by Part 1 (v18_branch_comparison) and
picks `n_arms` grid cells spanning that panel's L_norm range. For each
pick we replay the rollout and animate in the ONE viewer; arm color is
viridis(L_norm / global_max_L_norm) so it matches the heatmap colorbar.

Dead arms freeze at their last alive joint state. Initial poses hold for
HOLD_AT_START_SEC before motion begins; end of trajectory holds for
HOLD_AT_END_SEC before looping.
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


ARM_ALPHA = 0.55
DEFAULT_ANCHOR = 'K'


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
    allows. Entries below min_L_norm are dropped (they freeze immediately
    in the viewer). Returns picks sorted by L_norm ascending."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=SEED,
                        help='task seed (selects the per-seed data subfolder)')
    parser.add_argument('--anchor', type=str, default=DEFAULT_ANCHOR,
                        help='anchor letter A-P (default: K)')
    parser.add_argument('--jsonl', type=str, default=None,
                        help='explicit JSONL path; overrides --seed/--anchor')
    parser.add_argument('--n_arms', type=int, default=8,
                        help='number of arms to render (max 10)')
    parser.add_argument('--min_L_norm', type=float, default=0.1,
                        help='drop picks below this L_norm (frozen at start)')
    parser.add_argument('--playback_speed', type=float, default=0.25,
                        help='rollout playback rate (1.0 = real-time)')
    args = parser.parse_args()

    if args.jsonl is None:
        jsonl_path = seed_dir(int(args.seed)) / f'branch_comparison_{args.anchor.upper()}.jsonl'
    else:
        jsonl_path = Path(args.jsonl)
    meta, entries = load_meta_and_entries(jsonl_path)
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
    q_inits = as_tensor(np.stack(
        [np.array(p['q_init'], dtype=np.float32) for p in picks], axis=0),
        device)
    q_traj_np, fail_infos = record_rollout(kin, q_inits, track_pts, plane_normal_np)
    n_frames = q_traj_np.shape[0]
    print(f'recorded trajectory: {n_frames} frames, shape={q_traj_np.shape}')
    print('rollout outcomes:')
    for i, (p, info) in enumerate(zip(picks, fail_infos)):
        print(f'  arm{i} (L_norm={p["length_normalized"]:.3f}): '
              f'{info["reason"]}, seg={info["segment"]}, '
              f'pos_err={info["pos_err_m"]*1000:.1f}mm, '
              f'orient_err={info["orient_err_deg"]:.1f}deg'
              + (' [near joint limit]' if info['near_joint_limit'] else ''))

    base = ovw.World(cam_pos=(1.25, -1.65, 1.15),
                     cam_lookat_pos=(0.25, 0.0, 0.45),
                     toggle_auto_cam_orbit=False)
    builtins.base = base
    add_task_path(base, task_path, plane_normal=plane_normal_np)

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

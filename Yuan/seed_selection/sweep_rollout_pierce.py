"""Sweep over all (task, label) pairs and record whether the label's rollout
ever pierces the bounded plane at ANY timestep.

Loads existing `<data>.plane_collision.npz`, runs batched rollout with
per-step plane-pierce tracking enabled, and writes back two new fields:
    seed_traj_pierces      (N,) bool   — q0_seed rollout ever pierces
    label_traj_pierces     (N, k) bool — each valid label's rollout ever pierces

Then defines:
    any_q_traj_pierces     (N,) = seed_traj_pierces OR any label
    any_label_traj_pierces (N,) = any label

`SeedSelectionDataset(..., plane_collision_scope='labels_only_traj')` (planned)
will use the trajectory pierce flag.

Usage:
    python -m Yuan.seed_selection.sweep_rollout_pierce
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from one.robots.manipulators.franka.fr3.sphere_collision import FR3SphereCollision
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.RL_controller.env.env import EnvConfig, NSRLBatchedEnv
from Yuan.seed_selection.batched_rollout import batched_rollout_many


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path,
                   default=Path('Yuan/seed_selection/runs/pilot_day5/pilot_20k.npz'))
    p.add_argument('--pc-path', type=Path, default=None,
                   help='plane_collision NPZ to write into (default: <data>.plane_collision.npz)')
    p.add_argument('--exclude-links', type=int, nargs='*', default=[0, 1])
    p.add_argument('--plane-extent-m', type=float, default=1.5)
    p.add_argument('--n-envs-rollout', type=int, default=64)
    p.add_argument('--config-yaml', default='Yuan/RL_controller/config.yaml')
    p.add_argument('--target-distance-m', type=float, default=1.5)
    p.add_argument('--device', default='cuda')
    p.add_argument('--max-tasks', type=int, default=None,
                   help='only process first N tasks (smoke testing).')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    z = np.load(args.data, allow_pickle=False)
    N_full = int(z['L_seed'].shape[0])
    k = int(z['labels_q0'].shape[1])
    N = min(int(args.max_tasks) if args.max_tasks else N_full, N_full)
    print(f'[sweep] N={N} (of {N_full}) tasks, k={k} label slots')

    pc_path = args.pc_path or args.data.with_suffix('.plane_collision.npz')
    if not pc_path.exists():
        raise SystemExit(f'plane_collision file not found at {pc_path}; '
                         f'run check_plane_collision.py first to create it.')
    pc = dict(np.load(pc_path, allow_pickle=False))
    print(f'[sweep] loaded existing flags from {pc_path}')

    with open(args.config_yaml, 'r') as f:
        cfg_yaml = yaml.safe_load(f)
    env_cfg = EnvConfig(**{**cfg_yaml['env'], 'n_envs': args.n_envs_rollout})
    env = NSRLBatchedEnv(env_cfg, line_dist=None, device=device)
    controller = ClassicalNullspaceController(env.kin)
    coll = FR3SphereCollision(device=device, dtype=env.kin.dtype)
    link_idx = coll.link_indices.detach().cpu().numpy().astype(np.int32)
    keep_mask = torch.from_numpy(~np.isin(link_idx, args.exclude_links)).to(device)

    # Allocate at FULL size so saved npz keeps shape consistency with existing fields.
    seed_traj = np.zeros(N_full, dtype=bool)
    label_traj = np.zeros((N_full, k), dtype=bool)

    # Batch all (task, label) pairs and run one big batched_rollout_many.
    # For 20k tasks × (1 seed + 3 labels) = 80k rollouts, but only valid label
    # slots (n_labels[i] > j) contribute. Process per-slot to keep mapping simple.
    cs_seed_list = []
    for i in range(N):
        cs_seed_list.append({
            'p0': torch.as_tensor(z['cs_p0'][i], device=device, dtype=env.kin.dtype),
            'line_dir': torch.as_tensor(z['cs_line_dir'][i], device=device, dtype=env.kin.dtype),
            'n_target': torch.as_tensor(z['cs_n_target'][i], device=device, dtype=env.kin.dtype),
        })

    # q0_seed rollouts
    print('[sweep] rolling out q0_seed for all tasks...')
    t0 = time.time()
    qs_seed = torch.as_tensor(z['q0_seeds'][:N], device=device, dtype=env.kin.dtype)
    res = batched_rollout_many(
        qs_seed, cs_seed_list, env=env, controller=controller,
        target_distance_m=args.target_distance_m,
        pierce_collision=coll, pierce_keep_mask=keep_mask,
        pierce_plane_extent_m=args.plane_extent_m,
    )
    seed_traj[:N] = res['ever_pierced']
    print(f'  done in {time.time()-t0:.1f}s; seed_traj_pierces: {int(seed_traj[:N].sum())}/{N} '
          f'({100*seed_traj[:N].mean():.1f}%)')

    # Label rollouts per slot
    for j in range(k):
        valid = (z['n_labels'][:N] > j)
        idx = np.where(valid)[0]
        if len(idx) == 0:
            continue
        print(f'[sweep] rolling out label_{j} for {len(idx)} tasks...')
        t0 = time.time()
        qs_lab = torch.as_tensor(z['labels_q0'][idx, j], device=device, dtype=env.kin.dtype)
        cs_lab = [cs_seed_list[i] for i in idx]
        res = batched_rollout_many(
            qs_lab, cs_lab, env=env, controller=controller,
            target_distance_m=args.target_distance_m,
            pierce_collision=coll, pierce_keep_mask=keep_mask,
            pierce_plane_extent_m=args.plane_extent_m,
        )
        label_traj[idx, j] = res['ever_pierced']
        print(f'  done in {time.time()-t0:.1f}s; label_{j}_traj_pierces: '
              f'{int(label_traj[idx, j].sum())}/{len(idx)} '
              f'({100*label_traj[idx, j].mean():.1f}%)')

    any_label_traj = label_traj.any(axis=1)
    any_q_traj = seed_traj | any_label_traj
    print(f'\nSummary (trajectory pierces, over first {N}):')
    print(f'  seed:        {int(seed_traj[:N].sum())}/{N} ({100*seed_traj[:N].mean():.1f}%)')
    print(f'  any label:   {int(any_label_traj[:N].sum())}/{N} ({100*any_label_traj[:N].mean():.1f}%)')
    print(f'  any (s|l):   {int(any_q_traj[:N].sum())}/{N} ({100*any_q_traj[:N].mean():.1f}%)')
    # Cross-reference with start-config pierce (over the same first N).
    if 'any_q_collides' in pc:
        start_collides = pc['any_q_collides'][:N].astype(bool)
        traj = any_q_traj[:N]
        only_traj = traj & ~start_collides
        only_start = start_collides & ~traj
        both = traj & start_collides
        print(f'\nCross-ref with start-config check (over first {N}):')
        print(f'  caught by both start AND trajectory: {int(both.sum())}')
        print(f'  caught only by trajectory check:     {int(only_traj.sum())} '
              f'(arm fine at start but rollout drifted into plane)')
        print(f'  caught only by start-config check:   {int(only_start.sum())} '
              f'(arm started straddling but rollout veered away)')

    pc['seed_traj_pierces'] = seed_traj
    pc['label_traj_pierces'] = label_traj
    pc['any_label_traj_pierces'] = any_label_traj
    pc['any_q_traj_pierces'] = any_q_traj
    np.savez(pc_path, **pc)
    print(f'\nSaved trajectory pierce flags → {pc_path}')


if __name__ == '__main__':
    main()

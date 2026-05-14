"""Scenario 3: JL gain robustness sweep.

Hypothesis: best q0 are robust to JL gain (curve is flat across gain);
worst q0 sensitive (curve drops at low gain because JL avoidance can't
catch them, AND may drop at high gain because secondary task poisons
primary tracking).

Method: for one hard task (high rel_spread), pick the top-3 / bottom-3 /
median L_self anchors. Sweep POS_PRIORITY_JLIMIT_GAIN from 0 to 8x
default (4.0). Run rollout, record L_self / L_max.

Usage:
    python -m Yuan.RL.intro_motivation.v18_jl_gain_sweep --seed 118
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import Yuan.RL.config as cfg
from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.RL.intro_motivation.v18_branch_comparison import farthest_point_pick
from Yuan.RL.intro_motivation.v18_motivation_core import (
    LINE_L_RANGE,
    ROLLOUT_THETA_MAX,
    TARGET_PATH_M,
    as_tensor,
    enumerate_start_iks,
    extend_task_path,
    path_length,
    rollout_lengths,
    sample_line_task,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=118)
    parser.add_argument('--n-gains', type=int, default=9)
    parser.add_argument('--max-gain-mult', type=float, default=4.0)
    parser.add_argument('--out-png', type=str,
                        default='Yuan/RL/intro_motivation/data/jl_gain_sweep.png')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kin = BatchedFR3Kinematics(device=device)

    seed = int(args.seed)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    task = sample_line_task(rng, kin, l_range=LINE_L_RANGE)
    task = extend_task_path(task, TARGET_PATH_M)
    task_path = task['fine_path_pts']
    L_max = path_length(task_path)
    track_pts = as_tensor(task_path, device)
    plane_normal_np = task['plane_normal']
    plane_normal_t = as_tensor(plane_normal_np, device)
    print(f'seed={seed}, L_max={L_max:.3f}')

    q_set = enumerate_start_iks(kin, rng, task, track_pts)
    lo_np = kin.lmt_lo.detach().cpu().numpy()
    hi_np = kin.lmt_up.detach().cpu().numpy()
    q_set_np = q_set.detach().cpu().numpy()
    inbounds = ((q_set_np - lo_np > 0.15)
                & (hi_np - q_set_np > 0.15)).all(axis=1)
    if int(inbounds.sum()) < 16:
        inbounds = np.ones(q_set.shape[0], dtype=bool)
    q_good = q_set[inbounds]
    q_good_np = q_good.detach().cpu().numpy()

    L_start_default = rollout_lengths(kin, q_good, track_pts, plane_normal_t,
                                      theta_max_rad=ROLLOUT_THETA_MAX,
                                      enforce_init_pose=True, pos_priority=True)
    seed_idx = int(np.argmax(L_start_default))
    picks = farthest_point_pick(q_good_np, min(16, q_good.shape[0]), seed_idx)
    q_picks = q_good[picks]
    L_picks_default = L_start_default[picks] / L_max
    order = np.argsort(-L_picks_default)
    top3 = order[:3]
    bot3 = order[-3:]
    mid3 = order[(len(order) // 2 - 1):(len(order) // 2 + 2)]
    chosen_idx = np.concatenate([top3, mid3, bot3])
    chosen_labels = (['top']*3 + ['mid']*3 + ['bot']*3)
    print(f'baseline gain={cfg.POS_PRIORITY_JLIMIT_GAIN}, '
          f'L_picks_default per chosen:')
    for k, b in enumerate(chosen_idx):
        print(f'  {chosen_labels[k]} q[{b}]: L_self={L_picks_default[b]:.3f}')

    default_gain = float(cfg.POS_PRIORITY_JLIMIT_GAIN)
    gains = np.concatenate([
        np.array([0.0]),
        np.linspace(default_gain * 0.25, default_gain * args.max_gain_mult,
                    args.n_gains - 1),
    ])
    print(f'sweeping JL gain over: {gains.tolist()}')

    L_grid = np.zeros((len(chosen_idx), len(gains)), dtype=np.float32)
    q_chosen = q_picks[chosen_idx]
    for gi, g in enumerate(gains):
        cfg.POS_PRIORITY_JLIMIT_GAIN = float(g)
        Ls = rollout_lengths(kin, q_chosen, track_pts, plane_normal_t,
                             theta_max_rad=ROLLOUT_THETA_MAX,
                             enforce_init_pose=True, pos_priority=True) / L_max
        L_grid[:, gi] = Ls
        print(f'  gain={g:.2f}: L per chosen = {[f"{x:.3f}" for x in Ls]}')
    cfg.POS_PRIORITY_JLIMIT_GAIN = default_gain

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {'top': '#1f77b4', 'mid': '#9467bd', 'bot': '#d62728'}
    seen_label = set()
    for k, b in enumerate(chosen_idx):
        cat = chosen_labels[k]
        lbl = f'{cat}-tier' if cat not in seen_label else None
        seen_label.add(cat)
        ax.plot(gains, L_grid[k], '-o', color=colors[cat], alpha=0.7, label=lbl,
                markersize=4)
    ax.axvline(default_gain, color='black', linestyle='--', alpha=0.5,
               label=f'default={default_gain}')
    ax.set_xlabel('POS_PRIORITY_JLIMIT_GAIN')
    ax.set_ylabel('L_self / L_max')
    ax.set_title(f'JL-gain sweep on seed {seed} (top-3 / mid-3 / bot-3 q0)')
    ax.grid(alpha=0.3)
    ax.legend()
    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved: {out_png}')


if __name__ == '__main__':
    main()

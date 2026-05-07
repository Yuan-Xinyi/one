"""Visualize per-joint trajectories for v18 sampled q_trajs.

For each task: 7 subplots (one per joint), x = checkpoint index, y = joint
angle. K lines = K different goal-IK samples. Colored by branch signature
(sign of J1, J4, J6) so different IK branches show as different colors.

Usage:
  python -m Yuan.RL.v18_viz_jointlines \
      --pkl Yuan/RL/data/v18_eval_viz.pkl \
      --task-idx 0 5 10 \
      --out Yuan/RL/diagnostics/v18_jointlines/
"""
from __future__ import annotations
import argparse, os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


JOINT_LIMITS = np.array([
    [-2.97, +2.97],   # J1
    [-1.83, +1.83],   # J2
    [-2.97, +2.97],   # J3
    [-3.05, -0.05],   # J4
    [-2.97, +2.97],   # J5
    [-0.27, +4.53],   # J6 (FR3 has ~ [-0.27, 4.53] when including pen)
    [-2.97, +2.97],   # J7
], dtype=np.float32)


def branch_color(sig: tuple[int, int, int]) -> str:
    """Map branch signature to a stable color."""
    palette = {
        (+1, -1, +1): 'tab:blue',
        (+1, -1, -1): 'tab:orange',
        (-1, -1, +1): 'tab:green',
        (-1, -1, -1): 'tab:red',
        (+1, +1, +1): 'tab:purple',
        (+1, +1, -1): 'tab:brown',
        (-1, +1, +1): 'tab:pink',
        (-1, +1, -1): 'tab:olive',
    }
    return palette.get(sig, 'tab:gray')


def plot_one_task(task: dict, out_path: str, task_idx: int):
    q_trajs = task['q_trajs']                          # (K, N+1, 7)
    sigs    = task['sigs_at_start']
    K, T_co, _ = q_trajs.shape
    tilt = task['tilt_deg']
    L = task['L']

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()
    joint_names = ['J1 (shoulder yaw)', 'J2 (shoulder pitch)', 'J3 (shoulder roll)',
                   'J4 (elbow)', 'J5 (wrist roll 1)', 'J6 (wrist pitch)',
                   'J7 (wrist roll 2)']
    x_ckpt = np.arange(T_co)

    for j in range(7):
        ax = axes[j]
        for k in range(K):
            color = branch_color(sigs[k])
            ax.plot(x_ckpt, q_trajs[k, :, j], '-o', color=color,
                    alpha=0.7, lw=1.5, markersize=3,
                    label=f"branch {sigs[k]}" if k < 3 else None)
        # joint limit lines
        ax.axhline(JOINT_LIMITS[j, 0], color='gray', ls='--', lw=0.5, alpha=0.5)
        ax.axhline(JOINT_LIMITS[j, 1], color='gray', ls='--', lw=0.5, alpha=0.5)
        ax.set_xlabel('checkpoint i')
        ax.set_ylabel(f'q_{j+1} (rad)')
        ax.set_title(joint_names[j])
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=7, loc='best')

    # branch signature summary panel (sub 7)
    ax = axes[7]
    unique_sigs = sorted(set(sigs))
    counts = {s: sigs.count(s) for s in unique_sigs}
    bars_x = list(range(len(unique_sigs)))
    bars_h = [counts[s] for s in unique_sigs]
    colors = [branch_color(s) for s in unique_sigs]
    ax.bar(bars_x, bars_h, color=colors)
    ax.set_xticks(bars_x)
    ax.set_xticklabels([str(s) for s in unique_sigs], rotation=30, fontsize=8)
    ax.set_ylabel('# samples')
    ax.set_title('q_0 branch signatures')

    # task info panel
    ax = axes[8]
    ax.axis('off')
    info = (f"Task index: {task_idx}\n"
            f"L = {L:.3f} m\n"
            f"tilt = {tilt:.1f}°\n"
            f"plane_normal = ({task['plane_normal'][0]:.2f}, "
            f"{task['plane_normal'][1]:.2f}, {task['plane_normal'][2]:.2f})\n"
            f"direction = ({task['direction'][0]:.2f}, "
            f"{task['direction'][1]:.2f}, {task['direction'][2]:.2f})\n"
            f"\n"
            f"K_goal = {K}\n"
            f"unique_q0 branches = {task['unique_q0_branches']}\n"
            f"oracle branches = {task['oracle_branches']}\n"
            f"\n"
            f"TCP err per sample (m):\n  "
            + "  ".join(f"{e:.4f}" for e in task['tcp_err_per_sample']))
    ax.text(0.05, 0.95, info, transform=ax.transAxes, va='top', ha='left',
            family='monospace', fontsize=9)

    plt.suptitle(f"v18 q-trajectories: task {task_idx} | L={L:.2f}m | tilt={tilt:.1f}°",
                 fontsize=12, y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="Yuan/RL/data/v18_eval_viz.pkl")
    ap.add_argument("--task-idx", type=int, nargs='+', default=None,
                    help="specific task indices; defaults to first 6 tasks "
                         "with multimodal q_0 (unique > 1)")
    ap.add_argument("--n-default", type=int, default=8)
    ap.add_argument("--out", default="Yuan/RL/diagnostics/v18_jointlines/")
    args = ap.parse_args()

    with open(args.pkl, 'rb') as f:
        tasks = pickle.load(f)
    print(f"loaded {len(tasks)} tasks from {args.pkl}")

    if args.task_idx:
        idxs = args.task_idx
    else:
        # pick first n_default with > 1 unique branches (= multimodal)
        idxs = [i for i, t in enumerate(tasks)
                if t['unique_q0_branches'] > 1][:args.n_default]
    print(f"plotting tasks: {idxs}")

    os.makedirs(args.out, exist_ok=True)
    for ti in idxs:
        out_path = os.path.join(args.out, f"task_{ti:03d}.png")
        plot_one_task(tasks[ti], out_path, ti)
        print(f"  saved {out_path}")


if __name__ == "__main__":
    main()

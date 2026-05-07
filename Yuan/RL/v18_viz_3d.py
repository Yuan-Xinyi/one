"""3D visualization of v18 sampled q-trajectories: arm sweep with branch colors.

For each task: render the FR3 arm at every checkpoint, for every K q_traj sample.
Different IK branches → different colors. Output PNG (or GIF for animation).

The arm is drawn as connected line segments between joint positions
(obtained via BatchedFR3Kinematics.link_transforms).
"""
from __future__ import annotations
import argparse, os, pickle
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D                  # noqa
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics


def branch_color(sig: tuple[int, int, int]) -> str:
    palette = {
        (+1, -1, +1): '#1f77b4',     # blue
        (+1, -1, -1): '#ff7f0e',     # orange
        (-1, -1, +1): '#2ca02c',     # green
        (-1, -1, -1): '#d62728',     # red
        (+1, +1, +1): '#9467bd',     # purple
        (+1, +1, -1): '#8c564b',     # brown
        (-1, +1, +1): '#e377c2',     # pink
        (-1, +1, -1): '#bcbd22',     # olive
    }
    return palette.get(sig, '#7f7f7f')


def arm_joint_positions(kin: BatchedFR3Kinematics, q: torch.Tensor) -> np.ndarray:
    """Return (n_links + 1, 3) array of arm joint positions (base → TCP)."""
    if q.dim() == 1:
        q = q.unsqueeze(0)
    link_tfs = kin.link_transforms(q)                    # (1, n_links, 4, 4)
    n_links = link_tfs.shape[1]
    base_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    joint_pos = link_tfs[0, :, :3, 3].cpu().numpy()
    # also include TCP (which is 0.2034m above the last joint along z)
    p_tcp, _ = kin.fk_batch(q)
    tcp_pos = p_tcp.squeeze(0).cpu().numpy()
    return np.vstack([base_pos[None, :], joint_pos, tcp_pos[None, :]])


def plot_one_task_3d(task: dict, kin: BatchedFR3Kinematics,
                     out_path: str, task_idx: int,
                     ckpts_to_show: list[int] | None = None,
                     alpha: float = 0.45):
    q_trajs = task['q_trajs']                          # (K, T_co, 7)
    sigs    = task['sigs_at_start']
    K, T_co, _ = q_trajs.shape
    path_pts = task['path_pts']                         # (T_co, 3)

    if ckpts_to_show is None:
        ckpts_to_show = list(range(T_co))

    fig = plt.figure(figsize=(15, 6))

    # ------- LEFT: ALL checkpoints overlay -------
    ax1 = fig.add_subplot(121, projection='3d')

    # plot the path itself (TCP target line)
    ax1.plot(path_pts[:, 0], path_pts[:, 1], path_pts[:, 2],
             '-', color='black', lw=2.5, alpha=0.9, label='target path')
    ax1.scatter(path_pts[0, 0], path_pts[0, 1], path_pts[0, 2],
                color='black', s=60, marker='o', label='start', zorder=5)
    ax1.scatter(path_pts[-1, 0], path_pts[-1, 1], path_pts[-1, 2],
                color='black', s=80, marker='*', label='goal', zorder=5)

    # plot each (k, ckpt) arm
    for k in range(K):
        color = branch_color(sigs[k])
        for ci in ckpts_to_show:
            q = torch.as_tensor(q_trajs[k, ci], dtype=torch.float32,
                                 device=kin.device)
            joints = arm_joint_positions(kin, q)
            ax1.plot(joints[:, 0], joints[:, 1], joints[:, 2],
                     '-', color=color, lw=1.2, alpha=alpha,
                     marker='o', markersize=2)

    ax1.scatter([0], [0], [0], color='red', s=80, marker='s',
                label='base', zorder=5)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(f'arm sweep, all {len(ckpts_to_show)} ckpts, K={K} samples')
    # legend with one entry per branch
    seen = {}
    for k in range(K):
        s = sigs[k]
        if s not in seen:
            seen[s] = branch_color(s)
    handles = [plt.Line2D([0], [0], color=c, lw=2, label=str(s))
               for s, c in seen.items()]
    handles += [plt.Line2D([0], [0], color='black', lw=2, label='TCP path')]
    ax1.legend(handles=handles, fontsize=8, loc='upper left')
    _set_equal_3d(ax1, joint_pos_lim=0.9)

    # ------- RIGHT: just start ckpt overlay (cleaner) -------
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(path_pts[:, 0], path_pts[:, 1], path_pts[:, 2],
             '-', color='black', lw=2.5, alpha=0.9)
    ax2.scatter(path_pts[0, 0], path_pts[0, 1], path_pts[0, 2],
                color='black', s=60, marker='o')
    ax2.scatter(path_pts[-1, 0], path_pts[-1, 1], path_pts[-1, 2],
                color='black', s=80, marker='*')

    for k in range(K):
        color = branch_color(sigs[k])
        q = torch.as_tensor(q_trajs[k, 0], dtype=torch.float32,
                             device=kin.device)
        joints = arm_joint_positions(kin, q)
        ax2.plot(joints[:, 0], joints[:, 1], joints[:, 2],
                 '-', color=color, lw=2.5, alpha=0.85,
                 marker='o', markersize=4)
    ax2.scatter([0], [0], [0], color='red', s=80, marker='s')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title(f'arm at START ckpt only (K={K} branches)')
    _set_equal_3d(ax2, joint_pos_lim=0.9)

    plt.suptitle(f"Task {task_idx} | L={task['L']:.2f}m | tilt={task['tilt_deg']:.1f}° "
                  f"| unique q_0 branches: {task['unique_q0_branches']}",
                  fontsize=12, y=0.99)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out_path


def _set_equal_3d(ax, joint_pos_lim=0.9):
    """Make 3D plot have equal aspect ratio with a reasonable bounding box."""
    ax.set_xlim([-joint_pos_lim, joint_pos_lim])
    ax.set_ylim([-joint_pos_lim, joint_pos_lim])
    ax.set_zlim([0, 1.4])
    try:
        ax.set_box_aspect([1, 1, 0.7])
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="Yuan/RL/data/v18_eval_viz.pkl")
    ap.add_argument("--task-idx", type=int, nargs='+', default=None)
    ap.add_argument("--n-default", type=int, default=8)
    ap.add_argument("--out", default="Yuan/RL/diagnostics/v18_3d/")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kin = BatchedFR3Kinematics(device=device)

    with open(args.pkl, 'rb') as f:
        tasks = pickle.load(f)
    print(f"loaded {len(tasks)} tasks from {args.pkl}")

    if args.task_idx:
        idxs = args.task_idx
    else:
        # tasks with exactly 2 unique branches and reasonable L
        idxs = [i for i, t in enumerate(tasks)
                if t['unique_q0_branches'] == 2 and t['L'] >= 0.40][:args.n_default]
    print(f"plotting tasks: {idxs}")

    os.makedirs(args.out, exist_ok=True)
    for ti in idxs:
        out_path = os.path.join(args.out, f"task_{ti:03d}.png")
        plot_one_task_3d(tasks[ti], kin, out_path, ti)
        print(f"  saved {out_path}")


if __name__ == "__main__":
    main()

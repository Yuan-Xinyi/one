"""Visualize v18 path tracking on curves: overlay actual FK(q_traj) on the
desired curve, post-manifold-snap. One subplot per curve type, one task each,
K samples colored differently.

For each sample we plot:
  - actual snapped FK at every checkpoint (markers)
  - continuous FK trace via linear q-interpolation between checkpoints (line)
The desired curve is drawn at high resolution from analytic params (so arcs
and S-curves render smoothly, not as polylines).
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D                    # noqa: F401

from Yuan.flow_connectivity.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.flow_connectivity.v18_cfm_model import CFMFlowModel, COND_DIM
from Yuan.flow_connectivity.v18_inference import backward_sample
from Yuan.flow_connectivity.v18_data_prep import _dense_ik_at
from Yuan.flow_connectivity.v18_curve_eval import sample_curve_task


def fk_at(kin, q_traj):
    p, _ = kin.fk_batch(q_traj)
    return p.detach().cpu().numpy()


def fk_qinterp(kin, q_traj, n_sub=12):
    """Linearly interp q in joint space between consecutive checkpoints,
    return continuous TCP position trace."""
    device = q_traj.device
    parts = []
    for i in range(q_traj.shape[0] - 1):
        a = torch.linspace(0.0, 1.0, n_sub + 1, device=device).unsqueeze(-1)
        seg = (1 - a) * q_traj[i].unsqueeze(0) + a * q_traj[i + 1].unsqueeze(0)
        if i > 0:
            seg = seg[1:]
        parts.append(seg)
    q_fine = torch.cat(parts, dim=0)
    p, _ = kin.fk_batch(q_fine)
    return p.detach().cpu().numpy()


def _equal_aspect_3d(ax, all_pts):
    rng_xyz = float(np.max(all_pts.max(0) - all_pts.min(0)))
    pad = 0.05 * rng_xyz
    mid = (all_pts.max(0) + all_pts.min(0)) / 2
    half = rng_xyz / 2 + pad
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="Yuan/flow_connectivity/checkpoints_v18_multi/best.pt")
    ap.add_argument("--n-checkpoints", type=int, default=5)
    ap.add_argument("--K-samples", type=int, default=4)
    ap.add_argument("--n-ode-steps", type=int, default=16)
    ap.add_argument("--snap-iters", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="Yuan/flow_connectivity/data/v18_curve_viz.png")
    ap.add_argument("--curve-types", nargs="+",
                    default=["line", "arc", "s_curve"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kin = BatchedFR3Kinematics(device=device)

    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    margs = state.get("args", {})
    model = CFMFlowModel(q_dim=7, cond_dim=COND_DIM,
                         hidden=margs.get("hidden", 512),
                         depth=margs.get("depth", 6)).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"loaded {args.ckpt}, epoch {state.get('epoch', '?')}")

    rng = np.random.default_rng(args.seed)
    fig = plt.figure(figsize=(7.0 * len(args.curve_types), 7.0))
    cmap = plt.get_cmap('tab10')

    for ax_i, ctype in enumerate(args.curve_types):
        task = None
        for _ in range(50):
            t = sample_curve_task(rng, kin, ctype, args.n_checkpoints)
            if t is not None:
                task = t
                break
        if task is None:
            print(f"  {ctype}: no task sampled, skipping")
            continue

        path_np = task['path_pts']
        fine_np = task['fine_path_pts']
        path_pts = torch.as_tensor(path_np, device=device, dtype=torch.float32)
        plane_normal = torch.as_tensor(task['plane_normal'], device=device, dtype=torch.float32)
        direction_axis = torch.as_tensor(task['direction_axis'], device=device, dtype=torch.float32)
        d_per_step = torch.as_tensor(task['d_per_step'], device=device, dtype=torch.float32)
        R_T = torch.as_tensor(task['R_target_at_goal'], device=device, dtype=torch.float32)
        x_T = path_pts[-1]

        q_Ts, _ = _dense_ik_at(kin, x_T, R_T, args.K_samples * 4, rng)
        if q_Ts.shape[0] == 0:
            print(f"  {ctype}: no IK at goal, skipping")
            continue
        if q_Ts.shape[0] > args.K_samples:
            idx = rng.permutation(q_Ts.shape[0])[:args.K_samples]
            q_Ts = q_Ts[idx]

        ax = fig.add_subplot(1, len(args.curve_types), ax_i + 1, projection='3d')

        # desired curve (analytic high-res)
        ax.plot(*fine_np.T, color='black', lw=2.4, alpha=0.85, label='desired curve')
        ax.scatter(*path_np.T, color='black', s=42, marker='o', zorder=6,
                   edgecolors='white', linewidths=0.6)
        ax.scatter(*path_np[0],  color='#15a040', s=140, marker='o',
                   label='x_0 (start)', zorder=7, edgecolors='black', linewidths=0.6)
        ax.scatter(*path_np[-1], color='#d12c2c', s=170, marker='X',
                   label='x_T (goal)',  zorder=7, edgecolors='black', linewidths=0.6)

        all_pts_for_aspect = [fine_np]
        max_err_sample = []
        for k in range(q_Ts.shape[0]):
            q_traj = backward_sample(
                model, kin, q_Ts[k], path_pts, plane_normal, direction_axis,
                n_ode_steps=args.n_ode_steps, snap_iters=args.snap_iters,
                direction_per_step=d_per_step)
            p_snap = fk_at(kin, q_traj)
            p_fine = fk_qinterp(kin, q_traj, n_sub=12)
            color = cmap(k % 10)
            ax.plot(*p_fine.T, color=color, lw=1.0, alpha=0.55)
            ax.scatter(*p_snap.T, color=color, s=22, alpha=0.95)
            all_pts_for_aspect.append(p_snap)
            err_mm = 1000.0 * np.linalg.norm(p_snap - path_np, axis=1)
            max_err_sample.append(err_mm.max())

        all_pts_for_aspect = np.concatenate(all_pts_for_aspect, axis=0)
        _equal_aspect_3d(ax, all_pts_for_aspect)

        max_err_arr = np.array(max_err_sample)
        ax.set_title(f"{ctype}   L={task['L']:.2f}m   K={q_Ts.shape[0]}\n"
                     f"per-sample max FK err (mm): "
                     f"{', '.join(f'{e:.1f}' for e in np.sort(max_err_arr))}",
                     fontsize=9)
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        if ax_i == 0:
            ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches='tight')
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

"""v17 inference: backward integrator with manifold snapping.

Algorithm (per task)
--------------------
Inputs:
  q_goal       (7,)      starting joint state at end of path (= goal)
  path_pts     (T, 3)    desired TCP positions at each step (start..end)
  R_target     (3, 3)    desired TCP rotation (constant for plane tracking)
  c            (9,)      task context for the flow model

Backward loop, t = T-1, T-2, ..., 1:
  J = Jacobian at q
  J† = damped pinv (Levenberg-Marquardt, λ=0.01)
  P  = I_7 - J† J                       (null-space projector)

  Δx_back = path_pts[t-1] - FK(q)       (6D: pos + rot — pos uses path,
                                         rot keeps R_target)
  Δq_task = J† @ Δx_back                (task-space backward step)

  v = model(q, t/T, c)                  (predicted null-space velocity)
  v_proj = P @ v                        (project, redundant-safe)

  q ← q + Δq_task + cfg_scale · v_proj  (Δq_raw)
  q ← q.clamp(joint_limits)

  # manifold snap: 1-2 Newton-Raphson IK steps to keep TCP on path
  for _ in range(snap_iters):
      Jn = Jacobian at q
      Δx = path_pts[t-1] - FK(q)        (residual)
      q ← q + damped_pinv(Jn) @ Δx
      q ← q.clamp(joint_limits)

  record q

Output: q_traj of length T, played back forward at deploy.

CFG scale
---------
If the flow model's output is too small (∥v_proj∥ ≈ 0), set cfg_scale > 1
to amplify the null-space guidance. Equivalent to classifier-free guidance
on the unconditional null direction.
"""
from __future__ import annotations
import argparse, time
import numpy as np
import torch

from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.RL.v17_flow_model import FlowModel, damped_pinv, null_projector


def _rotvec_between(R_cur: torch.Tensor, R_tgt: torch.Tensor) -> torch.Tensor:
    """Minimal axis-angle representation of R_tgt R_cur^T as a (3,) vector."""
    R_err = R_tgt @ R_cur.transpose(-1, -2)
    trace = R_err.diagonal(dim1=-2, dim2=-1).sum(-1).clamp(-1.0, 3.0)
    cos_th = (trace - 1.0) * 0.5
    cos_th = cos_th.clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(cos_th)
    sin_th = theta.sin().clamp_min(1e-8)
    rx = (R_err[..., 2, 1] - R_err[..., 1, 2]) / (2 * sin_th)
    ry = (R_err[..., 0, 2] - R_err[..., 2, 0]) / (2 * sin_th)
    rz = (R_err[..., 1, 0] - R_err[..., 0, 1]) / (2 * sin_th)
    return torch.stack([rx, ry, rz], dim=-1) * theta.unsqueeze(-1)


def backward_inference(model: FlowModel, kin: BatchedFR3Kinematics,
                       q_goal: torch.Tensor,
                       path_pts: torch.Tensor,         # (T, 3)
                       R_target: torch.Tensor,         # (3, 3) — kept for API
                       c: torch.Tensor,                # (9,)
                       cfg_scale: float = 1.0,
                       lam: float = 1e-3,
                       snap_iters: int = 2,
                       v_scale: float = 1.0) -> torch.Tensor:
    """Returns q_traj of shape (T, 7), q_traj[0] = start_q (computed),
    q_traj[T-1] = q_goal (input)."""
    device = q_goal.device
    T = path_pts.shape[0]
    q_traj = torch.zeros((T, 7), device=device, dtype=torch.float32)
    q_traj[T - 1] = q_goal
    q = q_goal.clone()
    R_target_b = R_target.unsqueeze(0)

    eye7 = torch.eye(7, device=device, dtype=torch.float32)

    for t in range(T - 1, 0, -1):
        if t == T - 1:
            print(f"  loop iter t={t}, q={q.shape}")
        # current FK + Jacobian (3D position only, matches dataset controller)
        q_b = q.unsqueeze(0)
        p_cur, R_cur, J_full, _ = kin.tcp_fk_jac(q_b)
        J_pos = J_full.squeeze(0)[:3, :]              # (3, 7)
        # damped pinv on 3D, leaving 4D null-space
        eye3 = torch.eye(3, device=device, dtype=torch.float32)
        JJT = J_pos @ J_pos.transpose(-1, -2) + (lam ** 2) * eye3
        J_pos_dag = J_pos.transpose(-1, -2) @ torch.linalg.inv(JJT)  # (7, 3)
        P = eye7 - J_pos_dag @ J_pos                  # (7, 7) null-space projector

        # backward 3D task-space step toward path_pts[t-1] (POSITION ONLY)
        delta_p = path_pts[t - 1] - p_cur.squeeze(0)  # (3,)
        delta_q_task = J_pos_dag @ delta_p             # (7,)

        # null-space guidance from deterministic flow model
        t_norm = torch.tensor([t / float(T)], device=device, dtype=torch.float32)
        v = model(q.unsqueeze(0), t_norm, c.unsqueeze(0)).squeeze(0)
        v = v / max(v_scale, 1e-8)
        v_proj = P @ v
        delta_q = delta_q_task + cfg_scale * v_proj

        # apply, clamp
        q = q + delta_q
        q = q.clamp(kin.lmt_lo, kin.lmt_up)

        # manifold snap: 3D position-only Newton steps
        for _ in range(snap_iters):
            q_b = q.unsqueeze(0)
            p_cur2, _, J_full2, _ = kin.tcp_fk_jac(q_b)
            J_pos2 = J_full2.squeeze(0)[:3, :]
            delta_p2 = path_pts[t - 1] - p_cur2.squeeze(0)
            JJT2 = J_pos2 @ J_pos2.transpose(-1, -2) + (lam ** 2) * eye3
            J_pos2_dag = J_pos2.transpose(-1, -2) @ torch.linalg.inv(JJT2)
            q = q + J_pos2_dag @ delta_p2
            q = q.clamp(kin.lmt_lo, kin.lmt_up)

        q_traj[t - 1] = q

    return q_traj


def evaluate_one(model, kin, q_goal, path_pts, R_target, c,
                 cfg_scale=1.0, lam=1e-3, snap_iters=2, v_scale=1.0,
                 verbose=False):
    """Run backward inference and report TCP tracking error along resulting q_traj."""
    q_traj = backward_inference(model, kin, q_goal, path_pts, R_target, c,
                                cfg_scale=cfg_scale, lam=lam,
                                snap_iters=snap_iters, v_scale=v_scale)
    # forward-compute TCP at each step
    p_pred, _ = kin.fk_batch(q_traj)
    tracking_err = (p_pred - path_pts).norm(dim=-1)   # (T,)
    if verbose:
        print(f"  tracking err: mean={tracking_err.mean():.5f}  "
              f"max={tracking_err.max():.5f}  "
              f"final={tracking_err[-1]:.5f}  start={tracking_err[0]:.5f}")
    return q_traj, tracking_err


def main():
    """Smoke test: pull one trajectory from dataset, use its q_goal + path,
    run backward inference, compare predicted q_traj vs ground-truth q_traj."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",  default="Yuan/RL/checkpoints_v17/best.pt")
    ap.add_argument("--hdf5",  default="Yuan/fr3_dit/data/pen_fr3_plane_trajectories_50k.hdf5")
    ap.add_argument("--traj-idx", type=int, default=0)
    ap.add_argument("--cfg-scale", type=float, default=1.0,
                    help="amplify null-space guidance (>1 = stronger)")
    ap.add_argument("--lam", type=float, default=1e-3,
                    help="DLS damping (matches dataset controller)")
    ap.add_argument("--snap-iters", type=int, default=2)
    args = ap.parse_args()

    import h5py
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kin = BatchedFR3Kinematics(device=device)

    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    margs = state.get("args", {})
    model = FlowModel(hidden=margs.get("hidden", 256),
                      depth=margs.get("depth", 4)).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    v_scale = float(state.get("v_scale", 1.0))
    print(f"loaded {args.ckpt}, epoch {state.get('epoch', '?')}, "
          f"v_scale={v_scale}")

    with h5py.File(args.hdf5, 'r') as f:
        keys = sorted(f.keys())
        g = f[keys[args.traj_idx]]
        q_full = np.asarray(g['q'])                    # (T, 7)
        tcp_pos = np.asarray(g['tcp_pos'])             # (T, 3)
        direction = np.asarray(g['direction'])
        plane_normal = np.asarray(g['plane_normal'])
        plane_point = np.asarray(g['plane_point'])
        traj_attrs = dict(g.attrs)
        traj_name = keys[args.traj_idx]

    T = q_full.shape[0]
    print(f"\ntraj {traj_name}  T={T}  attrs={traj_attrs}")

    q_goal = torch.as_tensor(q_full[T - 1], device=device, dtype=torch.float32)
    path_pts = torch.as_tensor(tcp_pos, device=device, dtype=torch.float32)
    c = torch.as_tensor(np.concatenate([direction, plane_normal, plane_point]),
                         device=device, dtype=torch.float32)

    # build R_target from plane_normal + direction (similar to RL pipeline)
    print("step 1: z_axis from plane_normal", flush=True)
    z_axis = torch.as_tensor(-plane_normal, device=device, dtype=torch.float32)
    z_axis = z_axis / z_axis.norm()
    print(f"  z_axis ok: {z_axis}", flush=True)
    # Use the dataset's ACTUAL orientation at q_goal as R_target. The
    # dataset's q_goal often ends in angle_violation (TCP rotated 30°+
    # off plane_normal), so the analytic R_target derived from
    # plane_normal+direction is INCONSISTENT with the recorded final pose.
    # Using FK(q_goal) keeps backward integration self-consistent with the
    # data — the model only ever needs to reproduce trajectories that
    # actually happen.
    _, R_at_goal = kin.fk_batch(q_goal.unsqueeze(0))
    R_target = R_at_goal.squeeze(0).contiguous()

    print(f"q_goal={q_goal.shape}  path_pts={path_pts.shape}  R_target={R_target.shape}  c={c.shape}")
    print(f"running backward inference  cfg={args.cfg_scale} ...")
    q_traj_pred, err = evaluate_one(
        model, kin, q_goal, path_pts, R_target, c,
        cfg_scale=args.cfg_scale, lam=args.lam,
        snap_iters=args.snap_iters, v_scale=v_scale, verbose=True)
    print("done backward inference")

    # compare to ground-truth q sequence
    q_gt = torch.as_tensor(q_full, device=device, dtype=torch.float32)
    q_diff = (q_traj_pred - q_gt).norm(dim=-1)
    print(f"\nq prediction error vs GT trajectory:")
    print(f"  mean={q_diff.mean():.4f}  median={q_diff.median():.4f}  "
          f"max={q_diff.max():.4f}")
    print(f"  at start (t=0): {q_diff[0]:.4f}  "
          f"at goal (t=T-1): {q_diff[-1]:.4f}")


if __name__ == "__main__":
    main()

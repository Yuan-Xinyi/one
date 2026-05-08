"""v18 inference: backward generative sampling of full q-trajectory.

Algorithm:
  q_T = q_goal (input)
  for i = T-1, T-2, ..., 0:
      cond = [q_{i+1}, x_i, x_{i+1}, plane_normal, direction]
      z = randn(7)
      q_i = ODE_integrate(z, τ ∈ [0, 1], cond)        ← CFM sample
      q_i = manifold_snap(q_i, x_i, R_target)           ← enforce FK = x_i
  return q_traj[0..T]

Each call gives ONE sampled q-trajectory; multiple calls produce diverse
trajectories sampling the connected feasible region.
"""
from __future__ import annotations
import argparse, time
import numpy as np
import torch

from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.RL.v18_cfm_model import CFMFlowModel, COND_DIM


def damped_pinv_pos(J_pos: torch.Tensor, lam: float = 1e-3) -> torch.Tensor:
    """J_pos: (..., 3, 7) -> J†: (..., 7, 3)"""
    sh = J_pos.shape
    eye3 = torch.eye(3, device=J_pos.device, dtype=J_pos.dtype).expand(*sh[:-2], 3, 3)
    JJT = J_pos @ J_pos.transpose(-1, -2) + (lam ** 2) * eye3
    return J_pos.transpose(-1, -2) @ torch.linalg.inv(JJT)


def manifold_snap(kin, q: torch.Tensor, x_target: torch.Tensor,
                  n_iters: int = 3, lam: float = 1e-3) -> torch.Tensor:
    """Pull q so FK(q) ≈ x_target via Newton on 3D position constraint."""
    for _ in range(n_iters):
        q_b = q.unsqueeze(0)
        p_cur, _, J_full, _ = kin.tcp_fk_jac(q_b)
        J_pos = J_full.squeeze(0)[:3, :]
        delta_p = x_target - p_cur.squeeze(0)
        J_dag = damped_pinv_pos(J_pos.unsqueeze(0), lam=lam).squeeze(0)
        q = q + J_dag @ delta_p
        q = q.clamp(kin.lmt_lo, kin.lmt_up)
    return q


def _z_ang_deg(kin, q, plane_normal_unit):
    """Angle (deg) between FK(q)'s TCP-z and -plane_normal."""
    _, R_tcp, _, _ = kin.tcp_fk_jac(q.unsqueeze(0))
    z_cur = R_tcp[0, :, 2]
    cos_v = (z_cur * (-plane_normal_unit)).sum().clamp(-1.0, 1.0)
    return float(torch.arccos(cos_v).item() * 180.0 / 3.14159265)


def backward_sample(model: CFMFlowModel, kin: BatchedFR3Kinematics,
                    q_goal: torch.Tensor,
                    path_pts: torch.Tensor,           # (T, 3)
                    plane_normal: torch.Tensor,       # (3,)
                    direction: torch.Tensor,          # (3,)
                    n_ode_steps: int = 16,
                    cfg_scale: float = 1.0,
                    snap_iters: int = 3,
                    direction_per_step: torch.Tensor | None = None,
                    plane_normal_per_step: torch.Tensor | None = None,
                    debug_orient: bool = False,
                    ) -> torch.Tensor:
    """One sampled q-trajectory of shape (T, 7).
    q_traj[T-1] = q_goal; q_traj[i-1] sampled by CFM conditioned on q_traj[i].

    If `direction_per_step` is given (shape (T-1, 3)), it overrides the
    global `direction` cond on a per-segment basis: at the backward step
    that emits q_traj[i-1] from q_traj[i], the direction fed into cond is
    direction_per_step[i-1] (i.e., the local tangent of segment i-1 → i).
    Used for zero-shot curved-path inference.

    If `debug_orient=True`, prints z-axis angle (TCP-z vs -plane_normal)
    before and after manifold_snap for every backward step. Cheap; useful
    to confirm whether snap is breaking orientation alignment.
    """
    device = q_goal.device
    T = path_pts.shape[0]
    q_traj = torch.zeros(T, 7, device=device, dtype=torch.float32)
    q_traj[T - 1] = q_goal
    q_next = q_goal

    if debug_orient:
        # diagnostic uses goal-segment normal as reference (not global)
        if plane_normal_per_step is not None:
            pn_ref = plane_normal_per_step[-1]
        else:
            pn_ref = plane_normal
        pn_unit = pn_ref / pn_ref.norm().clamp_min(1e-12)
        diag_rows = []
        ang_goal = _z_ang_deg(kin, q_goal, pn_unit)

    for i in range(T - 1, 0, -1):
        x_curr = path_pts[i - 1]                      # x_i in dataset terms
        x_next = path_pts[i]
        d_i = direction if direction_per_step is None else direction_per_step[i - 1]
        n_i = plane_normal if plane_normal_per_step is None else plane_normal_per_step[i - 1]
        cond = torch.cat([q_next, x_curr, x_next, n_i, d_i]
                         ).unsqueeze(0)                # (1, COND_DIM)
        # CFM sample
        q_curr = model.sample(cond, n_steps=n_ode_steps,
                              cfg_scale=cfg_scale).squeeze(0)
        q_curr = q_curr.clamp(kin.lmt_lo, kin.lmt_up)
        if debug_orient:
            ang_pre = _z_ang_deg(kin, q_curr, pn_unit)
        # snap to manifold
        if snap_iters > 0:
            q_curr = manifold_snap(kin, q_curr, x_curr, n_iters=snap_iters)
        if debug_orient:
            ang_post = _z_ang_deg(kin, q_curr, pn_unit)
            diag_rows.append((i - 1, ang_pre, ang_post))
        q_traj[i - 1] = q_curr
        q_next = q_curr

    if debug_orient:
        print(f"    [orient-diag] z-ang to -plane_normal (deg)  "
              f"goal={ang_goal:5.2f}°  THETA_MAX=5.00°")
        print(f"      ckpt :  pre-snap  →  post-snap")
        for (ci, pre, post) in diag_rows:
            warn = " ⚠ over-tol" if post > 5.0 else ""
            print(f"      {ci:>4d} :  {pre:6.2f}°   →   {post:6.2f}°{warn}")
    return q_traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",  default="Yuan/RL/checkpoints_v18/best.pt")
    ap.add_argument("--hdf5",  default="Yuan/fr3_dit/data/pen_fr3_plane_trajectories_50k.hdf5")
    ap.add_argument("--traj-idx", type=int, default=17)
    ap.add_argument("--n-samples", type=int, default=8,
                    help="how many CFM samples to draw (test diversity)")
    ap.add_argument("--n-ode-steps", type=int, default=16)
    ap.add_argument("--cfg-scale", type=float, default=1.0)
    ap.add_argument("--snap-iters", type=int, default=3)
    args = ap.parse_args()

    import h5py
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

    with h5py.File(args.hdf5, 'r') as f:
        keys = sorted(f.keys())
        g = f[keys[args.traj_idx]]
        q_full = np.asarray(g['q'])
        tcp_pos = np.asarray(g['tcp_pos'])
        direction = np.asarray(g['direction'])
        plane_normal = np.asarray(g['plane_normal'])

    T = q_full.shape[0]
    q_goal = torch.as_tensor(q_full[T - 1], device=device, dtype=torch.float32)
    path_pts = torch.as_tensor(tcp_pos, device=device, dtype=torch.float32)
    pn = torch.as_tensor(plane_normal, device=device, dtype=torch.float32)
    dr = torch.as_tensor(direction, device=device, dtype=torch.float32)

    # subsample path to coarser grid (CFM was trained on N_segments=3 spacing)
    N = max(3, T // 50)                                # ~50-step segments
    idx = torch.linspace(0, T - 1, N + 1, device=device).long()
    path_coarse = path_pts[idx]                        # (N+1, 3)
    print(f"\ntraj {keys[args.traj_idx]}  T={T}  "
          f"coarse N+1={path_coarse.shape[0]} checkpoints")

    # draw n_samples q-trajectories
    print(f"\ndrawing {args.n_samples} samples...")
    samples = []
    t0 = time.perf_counter()
    for s in range(args.n_samples):
        q_traj = backward_sample(
            model, kin, q_goal, path_coarse, pn, dr,
            n_ode_steps=args.n_ode_steps,
            cfg_scale=args.cfg_scale,
            snap_iters=args.snap_iters)
        samples.append(q_traj.cpu().numpy())
    wall = time.perf_counter() - t0
    samples = np.stack(samples, axis=0)                # (n_samples, N+1, 7)
    print(f"  wall = {wall:.2f}s  ({wall*1000/args.n_samples:.1f}ms / sample)")

    # ----- analysis -----
    # 1. TCP tracking error per sample
    print("\nTCP tracking error per sample (after manifold snap):")
    for s in range(min(args.n_samples, 5)):
        q_t = torch.as_tensor(samples[s], device=device, dtype=torch.float32)
        p_pred, _ = kin.fk_batch(q_t)
        err = (p_pred - path_coarse).norm(dim=-1)
        print(f"  sample {s}: mean={err.mean():.5f}  max={err.max():.5f}")

    # 2. Diversity: pairwise distance between samples
    print("\nSample diversity (pairwise q distance over N+1 checkpoints):")
    for ci in range(samples.shape[1]):
        qs_at_ci = samples[:, ci, :]                   # (n_samples, 7)
        # pairwise mean distance
        dists = []
        for a in range(qs_at_ci.shape[0]):
            for b in range(a + 1, qs_at_ci.shape[0]):
                dists.append(np.linalg.norm(qs_at_ci[a] - qs_at_ci[b]))
        dists = np.array(dists) if dists else np.array([0.0])
        print(f"  ckpt {ci}/{samples.shape[1]-1}  "
              f"pairwise q-dist  mean={dists.mean():.4f}  "
              f"max={dists.max():.4f}")

    # 3. q at start: how many distinct branches?
    starts = samples[:, 0, :]
    print(f"\nq at start (= reverse-derived feasible q_0):")
    print(f"  branch signatures (sign of J1, J4, J6):")
    for s in range(args.n_samples):
        q = starts[s]
        sig = (int(np.sign(q[0])), int(np.sign(q[3])), int(np.sign(q[5])))
        print(f"    sample {s}: q_0[:7]={np.round(q, 2)}  sig={sig}")


if __name__ == "__main__":
    main()

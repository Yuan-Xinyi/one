"""v17 data preprocessing: extract null-space backward velocity targets.

For each (q_t, q_{t-1}) transition in dataset trajectories, compute:
  J_t = Jacobian at q_t                     (6, 7)
  J†_t = damped pseudoinverse                (7, 6)
  P_t = I_7 - J†_t @ J_t                     (7, 7)  null-space projector
  Δq_back = q_{t-1} - q_t                    backward joint motion
  v* = P_t @ Δq_back                         null-space component

The model will learn v_θ(q_t, t_norm, c) ≈ v*. Task-space part is
deterministic (controller → path), null-space part is the model's job.

Output: NPZ with arrays (q, t_norm, c, v_star) — one row per transition.
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
import torch

from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics


def _damped_pinv_batch(J: torch.Tensor, lam: float = 0.01) -> torch.Tensor:
    """J: (B, 6, 7) → J†: (B, 7, 6).  J† = J^T (J J^T + λ²I)^-1."""
    B = J.shape[0]
    eye6 = torch.eye(6, device=J.device, dtype=J.dtype).expand(B, 6, 6)
    JJT = J @ J.transpose(-1, -2) + (lam ** 2) * eye6
    return J.transpose(-1, -2) @ torch.linalg.inv(JJT)


def process_trajectory(g, kin: BatchedFR3Kinematics, lam: float = 1e-3,
                       device='cuda'):
    """Returns (q_t, t_norm, c, v_star) arrays for one trajectory.

    Critical: use 3D POSITION Jacobian (j_pos) — matching the dataset
    generator's controller (Yuan/fr3_dit/data_generation/...). Their task
    constraint is 3D (TCP position only); the 4D null-space lets the
    controller use manipulability + joint-margin + angle gradients freely.
    Using a 6D J here would project away most of the actual null-space
    motion, leaving v* near zero and making the flow model useless.
    """
    q_full = np.asarray(g['q'])                          # (T, 7)
    direction    = np.asarray(g['direction'])            # (3,)
    plane_normal = np.asarray(g['plane_normal'])
    plane_point  = np.asarray(g['plane_point'])
    T = q_full.shape[0]
    if T < 2:
        return None

    q_t   = torch.as_tensor(q_full[1:],  device=device, dtype=torch.float32)
    q_prev= torch.as_tensor(q_full[:-1], device=device, dtype=torch.float32)
    # Full Jacobian, but take ONLY position rows (matches dataset controller)
    _, _, J_full, _ = kin.tcp_fk_jac(q_t)                # (T-1, 6, 7)
    J_pos = J_full[:, :3, :]                              # (T-1, 3, 7)
    # damped pinv + null-space projector  (3D task → 4D null-space)
    eye3 = torch.eye(3, device=device, dtype=torch.float32
                     ).expand(T - 1, 3, 3)
    JJT = J_pos @ J_pos.transpose(-1, -2) + (lam ** 2) * eye3
    J_pos_dag = J_pos.transpose(-1, -2) @ torch.linalg.inv(JJT)   # (T-1, 7, 3)
    eye7 = torch.eye(7, device=device, dtype=torch.float32
                     ).expand(T - 1, 7, 7)
    P = eye7 - J_pos_dag @ J_pos                          # (T-1, 7, 7)
    delta_q_back = q_prev - q_t                           # (T-1, 7)
    v_star = (P @ delta_q_back.unsqueeze(-1)).squeeze(-1)  # (T-1, 7)

    t_norm = np.arange(1, T, dtype=np.float32) / float(T)   # (T-1,)
    c_per_step = np.tile(np.concatenate(
        [direction, plane_normal, plane_point]).astype(np.float32),
        (T - 1, 1))                                       # (T-1, 9)

    return (q_full[1:].astype(np.float32),
            t_norm,
            c_per_step,
            v_star.cpu().numpy().astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", default="Yuan/fr3_dit/data/pen_fr3_plane_trajectories_50k.hdf5")
    ap.add_argument("--out",  default="Yuan/RL/data/v17_train.npz")
    ap.add_argument("--max-trajs", type=int, default=2000,
                    help="process this many trajectories from the start (None = all)")
    ap.add_argument("--lam", type=float, default=1e-3,
                    help="DLS damping for J† (matches dataset controller)")
    args = ap.parse_args()

    import h5py
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    kin = BatchedFR3Kinematics(device=device)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"loading {args.hdf5}")
    t0 = time.perf_counter()
    all_q, all_t, all_c, all_v = [], [], [], []
    with h5py.File(args.hdf5, 'r') as f:
        traj_keys = sorted(f.keys())
        if args.max_trajs is not None:
            traj_keys = traj_keys[:args.max_trajs]
        for i, k in enumerate(traj_keys):
            res = process_trajectory(f[k], kin, lam=args.lam, device=device)
            if res is None:
                continue
            q, t_norm, c, v = res
            all_q.append(q); all_t.append(t_norm); all_c.append(c); all_v.append(v)
            if (i + 1) % 200 == 0:
                n_pairs = sum(x.shape[0] for x in all_q)
                print(f"  {i+1}/{len(traj_keys)}  total pairs={n_pairs:>9d}  "
                      f"({time.perf_counter()-t0:.1f}s)")

    q_arr = np.concatenate(all_q, axis=0)
    t_arr = np.concatenate(all_t, axis=0)
    c_arr = np.concatenate(all_c, axis=0)
    v_arr = np.concatenate(all_v, axis=0)
    print(f"\ntotal pairs: {q_arr.shape[0]:,}")
    print(f"q  shape: {q_arr.shape}")
    print(f"t  shape: {t_arr.shape}")
    print(f"c  shape: {c_arr.shape}")
    print(f"v* shape: {v_arr.shape}")
    print(f"v* magnitude: mean={np.linalg.norm(v_arr, axis=1).mean():.4f}  "
          f"max={np.linalg.norm(v_arr, axis=1).max():.4f}")

    np.savez_compressed(args.out, q=q_arr, t=t_arr, c=c_arr, v=v_arr)
    print(f"\nsaved to {args.out}  size={os.path.getsize(args.out)/1e6:.1f} MB  "
          f"({time.perf_counter()-t0:.1f}s)")


if __name__ == "__main__":
    main()

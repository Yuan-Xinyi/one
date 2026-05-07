"""v18 max-reach finder: given a path direction, find max distance reachable
and the optimal initial joint configuration q_0.

Algorithm:
  Bisection on L, using v18 backward-sampling as feasibility oracle.
  At each L:
    1. Find K IK branches at goal_pos = start + L * direction
    2. For each, run v18 backward-sample to get a candidate q-trajectory
    3. Check joint limits (TCP is enforced by manifold snap)
    4. If any candidate is feasible, record (L, q_0)
  Return: max L found + the best q_0 (= one with largest joint margin)
"""
from __future__ import annotations
import argparse, time
import numpy as np
import torch

from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.RL.batched_rollout import _batched_ik_project, _branch_seed_bank
from Yuan.RL.v18_cfm_model import CFMFlowModel, COND_DIM
from Yuan.RL.v18_inference import backward_sample


def joint_margin_score(q_traj: torch.Tensor, lmt_lo: torch.Tensor,
                       lmt_up: torch.Tensor) -> float:
    """Min normalized distance to joint limits over the whole trajectory.
    Higher = safer (further from limits)."""
    span = (lmt_up - lmt_lo).clamp_min(1e-6)
    lo_d = (q_traj - lmt_lo) / span
    up_d = (lmt_up - q_traj) / span
    return float(torch.minimum(lo_d, up_d).min().item())


def is_feasible(q_traj: torch.Tensor, lmt_lo: torch.Tensor,
                lmt_up: torch.Tensor, margin: float = 0.02) -> bool:
    in_lim = ((q_traj > lmt_lo + margin) & (q_traj < lmt_up - margin)).all()
    return bool(in_lim.item())


def find_max_L_scan(model, kin, start_pos, direction, R_target, c,
                    L_grid=None, K=8, n_checkpoints=5,
                    n_ode_steps=16, snap_iters=8, margin=0.02,
                    n_samples_per_ik=4):
    """Linear scan over L_grid (does not assume monotonic feasibility).
    Returns (L_max, best_q_0, best_q_traj, n_feasible_at_L_max)."""
    device = kin.device
    a_dummy = torch.tensor([1.0, 0.0, 1.0, 0.0], device=device, dtype=torch.float32)
    pn = c[3:6]; dr = c[0:3]
    if L_grid is None:
        L_grid = np.linspace(0.10, 1.00, 19)            # 5cm step

    L_best = 0.0
    q_traj_best = None
    n_feas_best = 0
    feasibility_per_L = []                              # for analysis

    for L in L_grid:
        L = float(L)
        goal_pos = start_pos + L * direction
        # K IK at goal
        seeds = _branch_seed_bank(kin)[:K]
        p_rep = goal_pos.unsqueeze(0).expand(K, 3)
        R_rep = R_target.unsqueeze(0).expand(K, 3, 3)
        a_rep = a_dummy.unsqueeze(0).expand(K, 4)
        q_goal_K, ok_K, _ = _batched_ik_project(kin, seeds, p_rep, R_rep,
                                                branch_action=a_rep)
        path_pts = torch.stack([
            start_pos + (k / n_checkpoints) * L * direction
            for k in range(n_checkpoints + 1)
        ], dim=0)
        n_feas = 0
        n_total = 0
        best_score_at_L = -1.0
        best_q_traj_at_L = None
        for k in range(K):
            if not bool(ok_K[k].item()): continue
            for _ in range(n_samples_per_ik):
                q_traj = backward_sample(model, kin, q_goal_K[k], path_pts,
                                         pn, dr,
                                         n_ode_steps=n_ode_steps,
                                         snap_iters=snap_iters)
                n_total += 1
                if not is_feasible(q_traj, kin.lmt_lo, kin.lmt_up, margin):
                    continue
                score = joint_margin_score(q_traj, kin.lmt_lo, kin.lmt_up)
                n_feas += 1
                if score > best_score_at_L:
                    best_score_at_L = score
                    best_q_traj_at_L = q_traj
        feasibility_per_L.append((L, n_feas, n_total))
        if n_feas >= 1 and L > L_best:
            L_best = L
            q_traj_best = best_q_traj_at_L
            n_feas_best = n_feas

    if q_traj_best is None:
        return L_best, None, None, 0, feasibility_per_L
    return L_best, q_traj_best[0], q_traj_best, n_feas_best, feasibility_per_L


def find_max_L(model, kin, start_pos, direction, R_target, c,
               L_lo=0.05, L_hi=1.0, n_bisect=8, K=8,
               n_checkpoints=5, n_ode_steps=16, snap_iters=8,
               margin=0.02, n_samples_per_ik=4):
    """Returns (L_max, best_q_0, best_q_traj, n_feasible_at_L_max)."""
    device = kin.device
    a_dummy = torch.tensor([1.0, 0.0, 1.0, 0.0], device=device, dtype=torch.float32)
    pn = c[3:6]; dr = c[0:3]                         # cond reuses  c packing later
    # parse c: [direction(3), plane_normal(3), plane_point(3)] is the user's
    # eval framing; here we just pass plane_normal and direction as torch tensors
    # (c here is a 6-vector since we drop plane_point — start_pos plays its role).
    # For backward_sample we still need plane_normal + direction
    # We'll re-pack below.

    L_best = L_lo
    q_traj_best = None
    n_feas_best = 0

    for it in range(n_bisect):
        L = (L_lo + L_hi) / 2.0
        goal_pos = start_pos + L * direction

        # K IK branches at goal
        seeds = _branch_seed_bank(kin)[:K]
        p_rep = goal_pos.unsqueeze(0).expand(K, 3)
        R_rep = R_target.unsqueeze(0).expand(K, 3, 3)
        a_rep = a_dummy.unsqueeze(0).expand(K, 4)
        q_goal_K, ok_K, _ = _batched_ik_project(kin, seeds, p_rep, R_rep,
                                                branch_action=a_rep)

        # path discretization for v18 sampling
        path_pts = torch.stack([
            start_pos + (k / n_checkpoints) * L * direction
            for k in range(n_checkpoints + 1)
        ], dim=0)

        n_feas = 0
        best_score = -1.0
        best_q_traj_at_L = None
        for k in range(K):
            if not bool(ok_K[k].item()): continue
            for _ in range(n_samples_per_ik):
                q_traj = backward_sample(model, kin, q_goal_K[k], path_pts,
                                         pn, dr,
                                         n_ode_steps=n_ode_steps,
                                         snap_iters=snap_iters)
                if not is_feasible(q_traj, kin.lmt_lo, kin.lmt_up, margin):
                    continue
                score = joint_margin_score(q_traj, kin.lmt_lo, kin.lmt_up)
                n_feas += 1
                if score > best_score:
                    best_score = score
                    best_q_traj_at_L = q_traj

        if n_feas >= 1:
            L_best = L
            q_traj_best = best_q_traj_at_L
            n_feas_best = n_feas
            L_lo = L                                  # try further
        else:
            L_hi = L                                  # too far

    if q_traj_best is None:
        return L_best, None, None, 0
    return L_best, q_traj_best[0], q_traj_best, n_feas_best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="Yuan/RL/checkpoints_v18_50k/best.pt")
    ap.add_argument("--hdf5", default="Yuan/fr3_dit/data/pen_fr3_plane_trajectories_50k.hdf5")
    ap.add_argument("--n-tasks", type=int, default=20)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--n-bisect", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scan", action="store_true",
                    help="use linear scan over L grid instead of bisection")
    ap.add_argument("--n-grid", type=int, default=19,
                    help="grid points (linear scan only)")
    ap.add_argument("--N-samples-per-ik", type=int, default=4)
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

    rng = np.random.default_rng(args.seed)
    with h5py.File(args.hdf5, 'r') as f:
        keys = sorted(f.keys())
        idx_pool = list(range(min(20000, len(keys))))
        rng.shuffle(idx_pool)
        chosen = []
        for i in idx_pool:
            g = f[keys[i]]
            base_d = float(np.linalg.norm(np.asarray(g['plane_point'])))
            if 0.30 < base_d < 0.85 and float(g.attrs['total_projected_length']) > 0.30:
                chosen.append(i)
            if len(chosen) >= args.n_tasks:
                break

        print(f"\n{'task':>5}  {'L_dataset':>9}  {'L_v18':>7}  {'gain':>5}  "
              f"{'q_0':>32}  {'wall_ms':>8}")
        print('-' * 92)

        rows = []
        for traj_idx in chosen:
            g = f[keys[traj_idx]]
            plane_point = np.asarray(g['plane_point'])
            direction = np.asarray(g['direction'])
            plane_normal = np.asarray(g['plane_normal'])
            L_dataset = float(g.attrs['total_projected_length'])
            # R_target = FK(start_q) — matches training data's R definition
            # (data prep used the same). Using a different R at deploy makes
            # the IK candidates OOD for the trained CFM.
            start_q = torch.as_tensor(g['q'][0], device=device, dtype=torch.float32)
            _, R_at_start = kin.fk_batch(start_q.unsqueeze(0))
            R_target = R_at_start.squeeze(0).contiguous()

            sp = torch.as_tensor(plane_point, device=device, dtype=torch.float32)
            dr = torch.as_tensor(direction, device=device, dtype=torch.float32)
            pn = torch.as_tensor(plane_normal, device=device, dtype=torch.float32)
            c = torch.cat([dr, pn])    # plane_normal+direction for cond

            t0 = time.perf_counter()
            if args.scan:
                L_grid = np.linspace(0.10, 1.00, args.n_grid)
                L_max, q_0, q_traj, n_feas, feas_log = find_max_L_scan(
                    model, kin, sp, dr, R_target, c,
                    L_grid=L_grid, K=args.K,
                    n_samples_per_ik=args.N_samples_per_ik)
            else:
                L_max, q_0, q_traj, n_feas = find_max_L(
                    model, kin, sp, dr, R_target, c,
                    K=args.K, n_bisect=args.n_bisect,
                    n_samples_per_ik=args.N_samples_per_ik)
            wall_ms = (time.perf_counter() - t0) * 1000

            if q_0 is None:
                print(f"  {traj_idx:>3d}  {L_dataset:>9.3f}  {'FAIL':>7}  -    -")
                continue
            q_0_str = "[" + " ".join(f"{v:+.2f}" for v in q_0.cpu().numpy()) + "]"
            gain = (L_max - L_dataset) / L_dataset * 100
            print(f"  {traj_idx:>3d}  {L_dataset:>9.3f}  {L_max:>7.3f}  "
                  f"{gain:>+4.0f}%  {q_0_str:>32}  {wall_ms:>8.0f}")
            rows.append((L_dataset, L_max, n_feas, wall_ms))

        if rows:
            arr = np.array(rows)
            L_ds, L_v18, _, ws = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
            diff = L_v18 - L_ds
            print("\n" + "=" * 92)
            print(f"AGGREGATE over {len(rows)} tasks:")
            print(f"  L_dataset (forward roll) mean: {L_ds.mean():.3f} m")
            print(f"  L_v18 (max reach)        mean: {L_v18.mean():.3f} m  "
                  f"(+{(L_v18.mean() - L_ds.mean())*100:.0f} cm)")
            print(f"  L_v18 - L_dataset:  >0:{(diff>0).sum()}/{len(rows)}  "
                  f"≥+10cm:{(diff>0.10).sum()}/{len(rows)}")
            print(f"  mean bisection wall: {ws.mean():.0f} ms")


if __name__ == "__main__":
    main()

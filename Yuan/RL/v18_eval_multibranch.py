"""v18 multi-branch sample evaluation.

For each task, enumerate K IK branches at the path's goal point. Run v18
backward-sample for EACH branch. Result: K different q-trajectories,
each starting in a different IK branch at the goal.

Tests the framework's "joint-space connectivity sampling" claim:
  - Different goal-branches → different q_0 branches?
  - All sampled q_traj's track TCP on path? (manifold snap should ensure)
  - Branch coverage at start: how many distinct IK branches hit?

This is the strongest test: v17 cannot do this (deterministic single output);
v18 should give K distinct q_traj's that all faithfully complete the same
path.
"""
from __future__ import annotations
import argparse, time
import numpy as np
import torch

from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.RL.batched_rollout import _batched_ik_project, _branch_seed_bank
from Yuan.RL.v18_cfm_model import CFMFlowModel, COND_DIM
from Yuan.RL.v18_inference import backward_sample
from Yuan.RL.v18_data_prep import _dense_ik_at, _check_transitions_geometric


def enumerate_ik_branches_at_goal(kin, goal_pos, R_target, K=8,
                                   M_oversample=64, rng=None):
    """Use M random IK seeds → take K distinct converged solutions.
    Mix uniform + boundary-biased to match v18 training data IK enumeration."""
    if rng is None:
        rng = np.random.default_rng()
    device = kin.device
    lo = kin.lmt_lo.cpu().numpy(); hi = kin.lmt_up.cpu().numpy()
    span = hi - lo
    n_unif = M_oversample // 2
    n_edge = M_oversample - n_unif
    seeds_unif = rng.uniform(lo[None, :], hi[None, :], (n_unif, 7)).astype(np.float32)
    seeds_edge = rng.uniform(lo[None, :], hi[None, :], (n_edge, 7)).astype(np.float32)
    for i in range(n_edge):
        n_extreme = int(rng.integers(1, 3))
        for j in rng.choice(7, size=n_extreme, replace=False):
            if int(rng.integers(0, 2)) == 0:
                seeds_edge[i, j] = lo[j] + 0.03 * span[j]
            else:
                seeds_edge[i, j] = hi[j] - 0.03 * span[j]
    seeds_np = np.concatenate([seeds_unif, seeds_edge], axis=0)
    seeds = torch.as_tensor(seeds_np, device=device, dtype=torch.float32)
    M = seeds.shape[0]
    p_rep = goal_pos.unsqueeze(0).expand(M, 3)
    R_rep = R_target.unsqueeze(0).expand(M, 3, 3)
    q, ok, _ = _batched_ik_project(kin, seeds, p_rep, R_rep, branch_action=None)
    q_kept = q[ok]
    # diversify: dedupe by branch signature, take up to K
    if q_kept.shape[0] > K:
        # sort by random and take first K (or could cluster by branch sig)
        idx = rng.permutation(q_kept.shape[0])[:K]
        q_kept = q_kept[idx]
    return q_kept, ok[:q_kept.shape[0]]


def branch_signature(q):
    """Sign tuple (J1, J4, J6) — distinguishes IK branches."""
    return (int(np.sign(q[0])), int(np.sign(q[3])), int(np.sign(q[5])))


def oracle_max_branches_at_start(kin, plane_point, direction, R_target,
                                  L, n_seg=5, M_oversample=128, rng=None,
                                  tol_pos=0.05):
    """Run backward DP with dense IK + geometric connectivity to find ALL
    feasible q_0 branches for this task. Returns the number of unique
    branch signatures in S_0 (= the structural ceiling on q_0 multimodality)."""
    device = kin.device
    if rng is None:
        rng = np.random.default_rng(0)
    p0_t = torch.as_tensor(plane_point, device=device, dtype=torch.float32)
    d_t  = torch.as_tensor(direction,   device=device, dtype=torch.float32)
    R_t  = torch.as_tensor(R_target if isinstance(R_target, np.ndarray)
                            else R_target.cpu().numpy(),
                            device=device, dtype=torch.float32) \
           if not torch.is_tensor(R_target) else R_target

    checkpoints = [p0_t + (i / n_seg) * L * d_t for i in range(n_seg + 1)]
    Q = []
    for x_i in checkpoints:
        q_set, _ = _dense_ik_at(kin, x_i, R_t, M_oversample, rng)
        Q.append(q_set)
    if min(q.shape[0] for q in Q) == 0:
        return 0, set()
    in_S = [None] * (n_seg + 1)
    in_S[n_seg] = torch.ones(Q[n_seg].shape[0], device=device, dtype=torch.bool)
    for i in range(n_seg - 1, -1, -1):
        succ = _check_transitions_geometric(
            kin, Q[i], Q[i + 1],
            x_curr=checkpoints[i], x_next=checkpoints[i + 1],
            n_check=8, tol_pos=tol_pos)
        v_next = in_S[i + 1].unsqueeze(0).expand(Q[i].shape[0], -1)
        in_S[i] = (succ & v_next).any(dim=1)
    feas_q0 = Q[0][in_S[0]]
    sigs = set(branch_signature(q.cpu().numpy()) for q in feas_q0)
    return feas_q0.shape[0], sigs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="Yuan/RL/checkpoints_v18_50k/best.pt")
    ap.add_argument("--hdf5", default="Yuan/fr3_dit/data/pen_fr3_plane_trajectories_50k.hdf5")
    ap.add_argument("--n-tasks", type=int, default=10)
    ap.add_argument("--K-branches", type=int, default=8,
                    help="number of IK branches to enumerate at each goal")
    ap.add_argument("--n-checkpoints", type=int, default=5,
                    help="path discretization for backward sampling")
    ap.add_argument("--n-ode-steps", type=int, default=16)
    ap.add_argument("--snap-iters", type=int, default=3)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--with-oracle", action="store_true",
                    help="also run backward-DP oracle and report ceiling")
    ap.add_argument("--oracle-M", type=int, default=128)
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
        chosen_tasks = []
        for i in idx_pool:
            g = f[keys[i]]
            base_d = float(np.linalg.norm(np.asarray(g['plane_point'])))
            if 0.30 < base_d < 0.85 and float(g.attrs['total_projected_length']) > 0.30:
                chosen_tasks.append(i)
            if len(chosen_tasks) >= args.n_tasks:
                break

        all_branch_counts = []
        all_branches_at_start = []
        all_oracle_branches = []
        all_tcp_errs = []
        all_walls = []

        hdr = f"\n{'task':>5}  {'L':>5}  {'IK@goal':>8}  {'v18_uniq':>9}  {'tcp_err':>8}  {'ms':>5}"
        if args.with_oracle:
            hdr += f"  {'oracle_uniq':>11}  {'cov%':>5}"
        print(hdr)
        print('-' * (88 if args.with_oracle else 72))

        for ti, traj_idx in enumerate(chosen_tasks):
            g = f[keys[traj_idx]]
            plane_point = np.asarray(g['plane_point'])
            direction   = np.asarray(g['direction'])
            plane_normal = np.asarray(g['plane_normal'])
            L = float(g.attrs['total_projected_length'])

            # path discretization
            T_co = args.n_checkpoints + 1
            checkpoints_np = np.stack([
                plane_point + (k / args.n_checkpoints) * L * direction
                for k in range(T_co)
            ], axis=0)
            path_pts = torch.as_tensor(checkpoints_np, device=device, dtype=torch.float32)
            pn = torch.as_tensor(plane_normal, device=device, dtype=torch.float32)
            dr = torch.as_tensor(direction, device=device, dtype=torch.float32)

            # R_target = analytic from (plane_normal, direction) — matches the
            # multi-orientation training data convention
            z_ax = -torch.as_tensor(plane_normal, device=device, dtype=torch.float32)
            z_ax = z_ax / z_ax.norm().clamp_min(1e-8)
            x_raw = torch.as_tensor(direction, device=device, dtype=torch.float32)
            x_ax = x_raw - z_ax * (x_raw * z_ax).sum()
            x_ax = x_ax / x_ax.norm().clamp_min(1e-8)
            y_ax = torch.stack([
                z_ax[1]*x_ax[2] - z_ax[2]*x_ax[1],
                z_ax[2]*x_ax[0] - z_ax[0]*x_ax[2],
                z_ax[0]*x_ax[1] - z_ax[1]*x_ax[0],
            ])
            R_target = torch.stack([x_ax, y_ax, z_ax], dim=-1)

            goal_pos = path_pts[-1]
            q_goals, ok = enumerate_ik_branches_at_goal(
                kin, goal_pos, R_target, K=args.K_branches, rng=rng)
            n_ik = q_goals.shape[0]
            if n_ik == 0:
                print(f"  {traj_idx:>3d}  {L:.2f}  {0:>11d}  {'(no IK at goal)':>19}  -  -")
                continue

            # for each goal branch, run v18 backward sample
            t0 = time.perf_counter()
            q_trajs = []
            for k in range(n_ik):
                q_traj = backward_sample(
                    model, kin, q_goals[k], path_pts, pn, dr,
                    n_ode_steps=args.n_ode_steps,
                    snap_iters=args.snap_iters)
                q_trajs.append(q_traj.cpu().numpy())
            wall_ms = (time.perf_counter() - t0) * 1000.0 / n_ik
            q_trajs = np.stack(q_trajs, axis=0)         # (n_ik, T_co, 7)

            # branch signatures at start (q_0)
            sigs_at_start = [branch_signature(q_trajs[k, 0, :]) for k in range(n_ik)]
            unique_sigs = len(set(sigs_at_start))

            # TCP tracking error
            tcp_err_max = 0.0
            for k in range(n_ik):
                q_t = torch.as_tensor(q_trajs[k], device=device, dtype=torch.float32)
                p_pred, _ = kin.fk_batch(q_t)
                e = (p_pred - path_pts).norm(dim=-1).max().item()
                tcp_err_max = max(tcp_err_max, e)

            line = (f"  {traj_idx:>3d}  {L:.2f}  {n_ik:>8d}  "
                    f"{unique_sigs:>9d}  {tcp_err_max:>8.4f}  {wall_ms:>5.1f}")
            oracle_uniq = None
            if args.with_oracle:
                _, sigs = oracle_max_branches_at_start(
                    kin, plane_point, direction, R_target, L,
                    n_seg=args.n_checkpoints,
                    M_oversample=args.oracle_M, rng=rng)
                oracle_uniq = len(sigs)
                cov = (unique_sigs / max(oracle_uniq, 1)) * 100
                line += f"  {oracle_uniq:>11d}  {cov:>4.0f}%"
                all_oracle_branches.append(oracle_uniq)
            print(line)
            all_branch_counts.append(n_ik)
            all_branches_at_start.append(unique_sigs)
            all_tcp_errs.append(tcp_err_max)
            all_walls.append(wall_ms)

    # aggregate
    print("\n" + "=" * 88)
    print(f"AGGREGATE over {len(all_branch_counts)} tasks:")
    print(f"  mean IK branches at goal:      {np.mean(all_branch_counts):.1f}")
    print(f"  mean unique q_0 branches:      {np.mean(all_branches_at_start):.2f}")
    print(f"  TCP err: median={np.median(all_tcp_errs):.4f}  "
          f"p90={np.percentile(all_tcp_errs,90):.4f}  max={max(all_tcp_errs):.4f}")
    print(f"  mean wall per sample:          {np.mean(all_walls):.1f} ms")
    if args.with_oracle and all_oracle_branches:
        oracle = np.array(all_oracle_branches)
        v18 = np.array(all_branches_at_start)
        cov = v18 / np.maximum(oracle, 1)
        print(f"\n  --- ORACLE COMPARISON ---")
        print(f"  mean oracle q_0 branches (true ceiling):  {oracle.mean():.2f}")
        print(f"  mean v18 q_0 branches:                    {v18.mean():.2f}")
        print(f"  v18 / oracle coverage:                    "
              f"mean={cov.mean():.2f}  median={np.median(cov):.2f}")
        # histogram of (oracle, v18) pairs
        from collections import Counter
        ctr = Counter(zip(oracle.tolist(), v18.tolist()))
        print(f"\n  (oracle, v18) → count, top 10:")
        for (o, v), c in sorted(ctr.most_common(10)):
            print(f"    oracle={o}, v18={v}: {c} tasks")


if __name__ == "__main__":
    main()

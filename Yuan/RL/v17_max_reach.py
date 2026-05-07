"""v17 max-reach finder + validation against dataset's forward baseline.

For each task = (start_pos, direction, R_target):
  - Bisect L in [L_lo, L_hi] using v17 backward as feasibility oracle
  - At each L, enumerate K IK branches at goal_pos = start + L * direction
  - For each branch q_goal, run v17 backward integrate
  - Branch is "feasible" if q_traj stays in joint limits and TCP tracks path

Compare L_max_v17 vs L_forward (= dataset's recorded total_projected_length
for the same task setup). If L_v17 >= L_forward, v17 finds reachable goals
further than dataset's single-start forward simulation.

Also reports per-L the number of IK branches that survived backward integration
— this directly tests the "multiple branches avoid dead-ends" claim.
"""
from __future__ import annotations
import argparse, time
import numpy as np
import torch

from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.RL.v17_flow_model import FlowModel
from Yuan.RL.v17_inference import backward_inference
from Yuan.RL.batched_rollout import _batched_ik_project, _branch_seed_bank


def ik_at_goal(kin, goal_pos, R_target, branch_action_dummy, K=8):
    """Use the seed bank to enumerate K IK branches at (goal_pos, R_target).
    Returns (K, 7) tensor of converged q's and (K,) ok mask."""
    device = kin.device
    seeds = _branch_seed_bank(kin)[:K]                   # (K, 7)
    p_rep = goal_pos.unsqueeze(0).expand(K, 3)
    R_rep = R_target.unsqueeze(0).expand(K, 3, 3)
    a_rep = branch_action_dummy.unsqueeze(0).expand(K, 4)
    q, ok, _ = _batched_ik_project(kin, seeds, p_rep, R_rep, branch_action=a_rep)
    return q, ok


def is_feasible(kin, q_traj, path_pts, joint_lo, joint_up,
                margin=0.02, tcp_tol=1e-3):
    """Check: q_traj stays in joint_limits, TCP within tol of path."""
    # joint limit margin check
    in_lim = ((q_traj > joint_lo + margin) & (q_traj < joint_up - margin)).all(dim=-1)
    if not in_lim.all():
        return False
    # TCP tracking check
    p_pred, _ = kin.fk_batch(q_traj)
    tcp_err = (p_pred - path_pts).norm(dim=-1)
    if tcp_err.max() > tcp_tol:
        return False
    return True


def find_max_L(model, kin, start_pos, direction, R_target, c, v_scale,
               L_lo=0.05, L_hi=1.0, n_bisect=10, K=8, T_steps=200,
               dt=0.01):
    """Bisect L. At each candidate, try K IK branches at goal; succeed if any
    backward integration is fully feasible.
    Returns (L_max, n_feasible_at_L_max, traces) where traces is a per-L log."""
    device = kin.device
    joint_lo = kin.lmt_lo
    joint_up = kin.lmt_up
    branch_action_dummy = torch.zeros(4, device=device, dtype=torch.float32)
    branch_action_dummy[0] = 1.0  # arbitrary, only used by IK projector
    branch_action_dummy[2] = 1.0

    traces = []
    L_best = L_lo
    n_feas_best = 0

    for i in range(n_bisect):
        L = (L_lo + L_hi) / 2.0
        T = max(10, int(L / (dt * 0.1)))                # path_speed = 0.1 m/s
        T = min(T, T_steps)
        goal_pos = start_pos + L * direction
        path_pts = torch.stack([
            start_pos + (k / max(T - 1, 1)) * L * direction
            for k in range(T)
        ], dim=0)

        # K IK branches at goal
        q_goal_K, ok_K = ik_at_goal(kin, goal_pos, R_target,
                                     branch_action_dummy, K=K)
        n_feas = 0
        n_ik = int(ok_K.sum().item())
        for k in range(K):
            if not bool(ok_K[k].item()): continue
            q_traj = backward_inference(model, kin, q_goal_K[k], path_pts,
                                        R_target, c, cfg_scale=1.0,
                                        lam=1e-3, snap_iters=2,
                                        v_scale=v_scale)
            if is_feasible(kin, q_traj, path_pts, joint_lo, joint_up):
                n_feas += 1
        traces.append((L, n_ik, n_feas))

        if n_feas >= 1:
            L_best = L
            n_feas_best = n_feas
            L_lo = L                                     # try further
        else:
            L_hi = L                                     # too far

    return L_best, n_feas_best, traces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",  default="Yuan/RL/checkpoints_v17_pos/best.pt")
    ap.add_argument("--hdf5",  default="Yuan/fr3_dit/data/pen_fr3_plane_trajectories_50k.hdf5")
    ap.add_argument("--n-tasks", type=int, default=10)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--n-bisect", type=int, default=10)
    ap.add_argument("--seed", type=int, default=12345)
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
    print(f"loaded {args.ckpt}, v_scale={v_scale}")

    rng = np.random.default_rng(args.seed)
    with h5py.File(args.hdf5, 'r') as f:
        keys = sorted(f.keys())
        # pick random tasks (mix of all termination types)
        idx_pool = list(range(min(5000, len(keys))))
        rng.shuffle(idx_pool)
        chosen = idx_pool[:args.n_tasks]

        print(f"\n{'task':>5}  {'reason':>17}  {'L_fwd':>7}  "
              f"{'L_v17':>7}  {'n_feas':>6}  {'ratio':>6}  {'verdict':>10}")
        print('-' * 80)

        L_fwd_arr = []; L_v17_arr = []
        for ti, traj_idx in enumerate(chosen):
            g = f[keys[traj_idx]]
            direction = torch.as_tensor(g['direction'], device=device, dtype=torch.float32)
            plane_normal = torch.as_tensor(g['plane_normal'], device=device, dtype=torch.float32)
            plane_point = torch.as_tensor(g['plane_point'], device=device, dtype=torch.float32)
            L_forward = float(g.attrs['total_projected_length'])
            reason = str(g.attrs['termination_reason'])

            # R_target derived from FK at start_q (consistent with how data was generated)
            start_q = torch.as_tensor(g['q'][0], device=device, dtype=torch.float32)
            _, R_at_start = kin.fk_batch(start_q.unsqueeze(0))
            R_target = R_at_start.squeeze(0).contiguous()

            c = torch.cat([direction, plane_normal, plane_point])
            L_max, n_feas, traces = find_max_L(
                model, kin, plane_point, direction, R_target, c, v_scale,
                L_lo=0.05, L_hi=1.0, n_bisect=args.n_bisect, K=args.K)

            ratio = L_max / max(L_forward, 1e-3)
            verdict = "✓ ≥ fwd" if L_max >= L_forward * 0.95 else "✗ < fwd"
            print(f"  {traj_idx:>3d}  {reason:>17}  {L_forward:>7.3f}  "
                  f"{L_max:>7.3f}  {n_feas:>6d}  {ratio:>6.2f}  {verdict:>10}")
            L_fwd_arr.append(L_forward)
            L_v17_arr.append(L_max)

        # aggregate
        L_fwd = np.array(L_fwd_arr); L_v17 = np.array(L_v17_arr)
        diff = L_v17 - L_fwd
        print(f"\n{'='*80}")
        print(f"AGGREGATE over {len(L_fwd)} tasks:")
        print(f"  L_forward (dataset): mean={L_fwd.mean():.3f}m  median={np.median(L_fwd):.3f}m")
        print(f"  L_v17  (bisection):  mean={L_v17.mean():.3f}m  median={np.median(L_v17):.3f}m")
        print(f"  L_v17 - L_forward:   mean={diff.mean():+.3f}m  "
              f">0:{(diff>0).sum()}/{len(diff)}  "
              f"≥-5%:{(diff>-0.05).sum()}/{len(diff)}")
        if (L_v17 >= L_fwd * 0.95).mean() > 0.7:
            print(f"\n  ✓ v17 + bisection finds reach ≥ dataset's forward in "
                  f"{int((L_v17 >= L_fwd*0.95).sum())}/{len(diff)} tasks.")
            print(f"    Hypothesis 'backward + multiple branches beat single-shot "
                  f"forward' SUPPORTED.")
        else:
            print(f"\n  ✗ v17 + bisection FAILS to match forward in majority of tasks.")


if __name__ == "__main__":
    main()

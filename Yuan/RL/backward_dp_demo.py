"""Backward Reachability Recursion — concept-validation demo.

Hypothesis under test
---------------------
For redundant-arm path tracking, the standard bandit (= commit one (φ, ψ)
at t=0, run controller forward) is structurally suboptimal. Specifically:

    There exist starting configurations q₀ that LOOK feasible by naive
    forward preview from t=0, but actually fail because some intermediate
    checkpoint x_i has no IK branch reachable from q₀'s natural trajectory.

If this is true, then `backward_DP_value(q₀) ≠ naive_forward_value(q₀)` and
sequential decision-making (allowing branch switching at intermediate
checkpoints) has real value.

Method
------
Per task:
  1. Choose ONE (φ, ψ) — for the demo, we sweep K_init starting branches
     so we cover diverse t=0 configurations.
  2. Discretize path into N segments → N+1 checkpoints x_0..x_N.
  3. At each checkpoint, find K IK branches via multistart (different seeds).
     Each branch is a distinct q satisfying IK at (x_i, R_tgt).
  4. For each segment i and each (q_i^j, q_{i+1}^k):
        run controller from q_i^j with q_ref = q_{i+1}^k (K_NULL pulls
        toward target) for the segment's control steps. Success iff:
          - segment completes (no joint_limit / pos_err / orient_err / collision)
          - final q is "near" target q_{i+1}^k (within distance threshold)
        → transition[i, j, k] ∈ {0, 1}
  5. Backward DP:
        V[N, k] = 1 for all converged terminal branches
        V[i, j] = max_k transition[i, j, k] · V[i+1, k]
  6. Compare:
        V[0, j]   (= "this q₀ has a feasible sequence to end")
     vs naive_forward(q_0^j) (= "natural rollout from q₀ completes")
     If V[0,j] = 1 but naive[j] = 0 → branch switching saved this start
     If V[0,j] = 0 but naive[j] = 1 → noise (shouldn't happen often)
     If V[0,j] = naive[j] → DP redundant for this start

Output: per-task table, then aggregate stats. Looking for "saves" (V=1, naive=0).
"""
from __future__ import annotations
import argparse, time
import numpy as np
import torch

import Yuan.RL.config as cfg
from Yuan.RL.env import FarsightedSeedEnv
from Yuan.RL.batched_rollout import (
    batched_rollout, batched_rollout_segment,
    branch_project_multistart, build_branch_rotmat_batch,
    _batched_ik_project, _branch_seed_bank,
    _device_from_cfg, _load_fr3_sphere_collision_cls,
)
from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics


def _enumerate_ik_branches(kin, p_target, R_target, branch_action,
                           K: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """For B tasks × K seeds, run IK and return (B, K, 7) joint configs and
    (B, K) bool mask of which converged. Each (j, k) is the k-th seed's IK
    solution for task j's (p_target, R_target). Different seeds yield
    different IK branches (elbow up/down, wrist flip, etc.)."""
    B = p_target.shape[0]
    seeds = _branch_seed_bank(kin)                        # (S, 7)
    S = seeds.shape[0]
    K = min(K, S)
    seeds_K = seeds[:K]                                    # (K, 7)
    p_rep = p_target.unsqueeze(1).expand(B, K, 3).reshape(-1, 3)
    R_rep = R_target.unsqueeze(1).expand(B, K, 3, 3).reshape(-1, 3, 3)
    a_rep = branch_action.unsqueeze(1).expand(B, K, 4).reshape(-1, 4)
    q_seed = seeds_K.unsqueeze(0).expand(B, K, 7).reshape(-1, 7)
    q, ok, _ = _batched_ik_project(kin, q_seed, p_rep, R_rep, branch_action=a_rep)
    return q.view(B, K, 7), ok.view(B, K)


def _check_transitions_segment(kin,
                               q_curr_BK: torch.Tensor,        # (B, K, 7)
                               q_target_BK: torch.Tensor,      # (B, K, 7)
                               R_tgt_B: torch.Tensor,           # (B, 3, 3)
                               a_B: torch.Tensor,               # (B, 4)
                               p0_B: torch.Tensor,              # (B, 3)
                               d_dir_B: torch.Tensor,           # (B, 3)
                               v_path_B: torch.Tensor,          # (B,)
                               eps_p_B: torch.Tensor,           # (B,)
                               T_total_B: torch.Tensor,         # (B,)
                               start_step: int,
                               end_step: int,
                               sphere_cc,
                               q_dist_thresh: float = 0.5,
                               ) -> torch.Tensor:
    """Return (B, K, K) success mask. transition[b, j, k] = True iff controller
    can drive arm from q_curr_BK[b, j] to within q_dist_thresh of q_target_BK[b, k]
    while tracking path in segment [start_step, end_step]."""
    device = q_curr_BK.device
    B, K = q_curr_BK.shape[:2]
    # broadcast: (B, K, K) for all (j, k) pairs
    j_idx = torch.arange(K, device=device).view(1, K, 1).expand(B, K, K)
    k_idx = torch.arange(K, device=device).view(1, 1, K).expand(B, K, K)
    # flatten to (B*K*K, ...)
    BKK = B * K * K
    q_init_flat = q_curr_BK.gather(
        1, j_idx.reshape(B, K * K, 1).expand(B, K * K, 7)).reshape(BKK, 7)
    q_targ_flat = q_target_BK.gather(
        1, k_idx.reshape(B, K * K, 1).expand(B, K * K, 7)).reshape(BKK, 7)
    R_flat       = R_tgt_B.unsqueeze(1).expand(B, K * K, 3, 3).reshape(BKK, 3, 3)
    a_flat       = a_B.unsqueeze(1).expand(B, K * K, 4).reshape(BKK, 4)
    p0_flat      = p0_B.unsqueeze(1).expand(B, K * K, 3).reshape(BKK, 3)
    d_flat       = d_dir_B.unsqueeze(1).expand(B, K * K, 3).reshape(BKK, 3)
    v_flat       = v_path_B.unsqueeze(1).expand(B, K * K).reshape(BKK)
    eps_flat     = eps_p_B.unsqueeze(1).expand(B, K * K).reshape(BKK)
    T_flat       = T_total_B.unsqueeze(1).expand(B, K * K).reshape(BKK)

    # run segment with q_ref = q_target (K_NULL pulls there); FULL controller
    # (is_phantom=False) so the result reflects actual feasibility.
    out = batched_rollout_segment(
        q_init_flat, R_flat, a_flat, p0_flat, d_flat, v_flat, eps_flat, T_flat,
        start_step=start_step, end_step=end_step,
        preset_gains=None, alive_mask=None,
        sphere_cc=sphere_cc, kin=kin, is_phantom=False,
        q_ref=q_targ_flat)
    q_final = out['q_final']
    alive = out['alive_out']                                # (BKK,)
    # success = survived AND ended within q_dist_thresh of target
    q_diff = (q_final - q_targ_flat).norm(dim=-1)
    success = alive & (q_diff < q_dist_thresh)
    return success.view(B, K, K)


def _naive_forward(kin, q_init_BK, R_tgt_B, a_B, p0_B, d_dir_B, v_B, eps_B,
                   T_B, sphere_cc, max_T):
    """Run full-horizon controller from each (b, j) starting q. Returns
    (B, K) bool: did this starting q complete the path?"""
    B, K = q_init_BK.shape[:2]
    q_flat = q_init_BK.reshape(B * K, 7)
    R_flat = R_tgt_B.unsqueeze(1).expand(B, K, 3, 3).reshape(B * K, 3, 3)
    a_flat = a_B.unsqueeze(1).expand(B, K, 4).reshape(B * K, 4)
    p0_flat = p0_B.unsqueeze(1).expand(B, K, 3).reshape(B * K, 3)
    d_flat = d_dir_B.unsqueeze(1).expand(B, K, 3).reshape(B * K, 3)
    v_flat = v_B.unsqueeze(1).expand(B, K).reshape(B * K)
    eps_flat = eps_B.unsqueeze(1).expand(B, K).reshape(B * K)
    T_flat = T_B.unsqueeze(1).expand(B, K).reshape(B * K)
    out = batched_rollout_segment(
        q_flat, R_flat, a_flat, p0_flat, d_flat, v_flat, eps_flat, T_flat,
        start_step=0, end_step=max_T,
        preset_gains=None, alive_mask=None,
        sphere_cc=sphere_cc, kin=kin, is_phantom=False)
    alive = out['alive_out']
    lengths = out['lengths']
    return alive.view(B, K), lengths.view(B, K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tasks", type=int, default=50)
    ap.add_argument("--n-segments", type=int, default=5)
    ap.add_argument("--K-branches", type=int, default=8,
                    help="IK branches per checkpoint (max 16, the seed bank)")
    ap.add_argument("--n-phipsi", type=int, default=4,
                    help="how many distinct (φ, ψ) per task to sweep")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--q-dist-thresh", type=float, default=0.5,
                    help="rad: a transition counts as 'success' iff final q is "
                         "within this Euclidean distance of target q")
    ap.add_argument("--min-base-dist", type=float, default=0.30)
    ap.add_argument("--T-eval", type=int, default=None,
                    help="override per-task T (uses task T_target if not set). "
                         "Smaller T makes 'feasibility' less stringent.")
    args = ap.parse_args()

    print(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"n_tasks={args.n_tasks}  n_seg={args.n_segments}  "
          f"K_branches={args.K_branches}  n_phipsi={args.n_phipsi}")

    device = _device_from_cfg()
    kin = BatchedFR3Kinematics(device=device)
    sphere_cc = None
    if cfg.USE_COLLISION_CHECK and cfg.BATCHED_COLLISION_CHECK:
        sphere_cc = _load_fr3_sphere_collision_cls()(
            device=device, margin=cfg.BATCHED_COLLISION_MARGIN)

    env = FarsightedSeedEnv(seed=args.seed, randomize=False, contact_mode=False)
    tasks = env._sample_tasks(args.n_tasks)
    c_np = np.stack([t['c'] for t in tasks], axis=0).astype(np.float32)
    v_np = np.array([t['v_path'] for t in tasks], dtype=np.float32)
    e_np = np.array([t['eps_p']  for t in tasks], dtype=np.float32)
    T_np = np.array([t['T']      for t in tasks], dtype=np.int32)
    B = args.n_tasks
    K = args.K_branches
    N = args.n_segments

    # ----- Sample n_phipsi (φ, ψ) per task -----
    rng = np.random.default_rng(args.seed)
    phi_seq = rng.uniform(0, 2*np.pi, size=(args.n_phipsi, B)).astype(np.float32)
    psi_seq = rng.uniform(0, 2*np.pi, size=(args.n_phipsi, B)).astype(np.float32)
    a_PB = np.stack([np.cos(phi_seq), np.sin(phi_seq),
                     np.cos(psi_seq), np.sin(psi_seq)], axis=-1)   # (P, B, 4)

    # to torch
    c = torch.as_tensor(c_np, device=device, dtype=torch.float32)
    v_path = torch.as_tensor(v_np, device=device, dtype=torch.float32)
    eps_p  = torch.as_tensor(e_np, device=device, dtype=torch.float32)
    T_total= torch.as_tensor(T_np, device=device, dtype=torch.long)
    p0 = c[:, :3]; d_dir = c[:, 3:6]; n_out = c[:, 6:9]
    if args.T_eval is not None:
        T_total = torch.full_like(T_total, int(args.T_eval))
    max_T = int(T_total.max().item())
    print(f"per-task T (clipped) max={max_T}")

    # collect per-(P, B) backward V and naive forward outcome
    backward_V_PB = np.zeros((args.n_phipsi, B), dtype=np.int32)
    naive_alive_PB = np.zeros((args.n_phipsi, B), dtype=np.int32)
    naive_len_PB   = np.zeros((args.n_phipsi, B), dtype=np.int32)

    t0 = time.perf_counter()
    for p_idx in range(args.n_phipsi):
        a_t = torch.as_tensor(a_PB[p_idx], device=device, dtype=torch.float32)  # (B, 4)
        R_tgt_B = build_branch_rotmat_batch(d_dir, n_out, a_t)                  # (B, 3, 3)

        # ----- 1. Get K IK branches at each of N+1 checkpoints -----
        q_grid = []  # list of (B, K, 7) — one per checkpoint
        ok_grid = []
        for i in range(N + 1):
            t_step = i * max_T // N
            x_i = p0 + (t_step * float(cfg.DT)) * v_path.unsqueeze(-1) * d_dir
            q_i, ok_i = _enumerate_ik_branches(kin, x_i, R_tgt_B, a_t, K=K)
            q_grid.append(q_i)
            ok_grid.append(ok_i)
        ok_rates = [float(ok.float().mean().item()) for ok in ok_grid]
        print(f"  [(φ,ψ) {p_idx}] IK ok rate per ckpt: "
              + " ".join(f"{r:.2f}" for r in ok_rates))

        # ----- 2. Transition checks for each segment -----
        # transition_seg[i] is (B, K, K) bool. trans[b, j, k] = j → k feasible
        transitions = []
        trans_rates = []
        for i in range(N):
            t_start = i * max_T // N
            t_end   = (i + 1) * max_T // N
            trans_i = _check_transitions_segment(
                kin, q_grid[i], q_grid[i + 1], R_tgt_B, a_t,
                p0, d_dir, v_path, eps_p, T_total,
                start_step=t_start, end_step=t_end,
                sphere_cc=sphere_cc,
                q_dist_thresh=args.q_dist_thresh)
            # mask out invalid IK branches
            valid_curr = ok_grid[i].unsqueeze(2).expand(B, K, K)        # (B, K, K)
            valid_targ = ok_grid[i + 1].unsqueeze(1).expand(B, K, K)
            trans_i = trans_i & valid_curr & valid_targ
            transitions.append(trans_i)
            trans_rates.append(float(trans_i.float().mean().item()))
        print(f"  [(φ,ψ) {p_idx}] segment trans rate: "
              + " ".join(f"{r:.2f}" for r in trans_rates))

        # ----- 3. Backward DP -----
        # V[i] is (B, K) bool: from this q_i^j, is there a feasible sequence to terminal?
        V_terminal = ok_grid[N]                                              # (B, K)
        V = [None] * (N + 1)
        V[N] = V_terminal
        for i in range(N - 1, -1, -1):
            # V[i][b, j] = OR over k { transition[i][b, j, k] AND V[i+1][b, k] }
            t_i = transitions[i]                                              # (B, K, K)
            v_next = V[i + 1].unsqueeze(1).expand(B, K, K)                    # (B, K, K)
            V[i] = (t_i & v_next).any(dim=2)                                   # (B, K)
        # final: which q_0^j has a feasible sequence?
        backward_V_PB[p_idx] = V[0].sum(dim=1).cpu().numpy()  # how many valid starts per task

        # ----- 4. Naive forward: for each q_0^j, just run controller forward -----
        naive_alive_BK, naive_lens_BK = _naive_forward(
            kin, q_grid[0], R_tgt_B, a_t, p0, d_dir, v_path, eps_p, T_total,
            sphere_cc, max_T)
        naive_alive_PB[p_idx] = naive_alive_BK.sum(dim=1).cpu().numpy()
        naive_len_PB[p_idx]   = naive_lens_BK.max(dim=1).values.cpu().numpy()

        # ----- per-(p, b) detailed comparison -----
        # For each task, count: (V=1, naive=0) "saves" and (V=1, naive=1) "agree"
        v_starts = V[0].cpu().numpy()                        # (B, K) bool
        n_starts = naive_alive_BK.cpu().numpy()
        # also: for each starting q where naive failed, did backward DP find a save?
        if p_idx == 0:
            print(f"\n[(φ,ψ) sample {p_idx}] per-task:")
            saves = (v_starts & ~n_starts).sum(axis=1)
            losses = (~v_starts & n_starts).sum(axis=1)
            agree_ok = (v_starts & n_starts).sum(axis=1)
            agree_fail = (~v_starts & ~n_starts).sum(axis=1)
            for b in range(min(B, 10)):
                print(f"  task {b:3d}  saves={saves[b]}  agree_ok={agree_ok[b]}  "
                      f"agree_fail={agree_fail[b]}  losses={losses[b]}")

        wall = time.perf_counter() - t0
        print(f"[(φ,ψ) {p_idx+1}/{args.n_phipsi}] "
              f"backward valid starts/task = {backward_V_PB[p_idx].mean():.2f}/{K}  "
              f"naive valid starts/task   = {naive_alive_PB[p_idx].mean():.2f}/{K}  "
              f"({wall:.1f}s)")

    # ----- aggregate -----
    print("\n" + "=" * 70)
    print(f"AGGREGATE over n_phipsi={args.n_phipsi} × n_tasks={B} × K={K} starts")
    print("=" * 70)
    total = args.n_phipsi * B * K
    # need per-(p, b, j) booleans — re-compute one more time? store?
    # for aggregate we use the counts we have:
    print(f"\nMean valid starts per (φ,ψ, task):  "
          f"backward = {backward_V_PB.mean():.3f}/{K}  "
          f"naive    = {naive_alive_PB.mean():.3f}/{K}")
    diff = backward_V_PB - naive_alive_PB
    print(f"\nbackward - naive (per-task):  "
          f"mean = {diff.mean():+.3f}  "
          f">0: {(diff > 0).sum()}/{args.n_phipsi*B}  "
          f"=0: {(diff == 0).sum()}/{args.n_phipsi*B}  "
          f"<0: {(diff < 0).sum()}/{args.n_phipsi*B}")

    # interpretation
    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)
    if (diff > 0).sum() > (diff < 0).sum() * 2:
        print(f"  ✓ Backward DP finds MORE valid starts than naive forward.")
        print(f"    Sequential branch switching is recovering trajectories that")
        print(f"    naive single-(φ,ψ)+fixed-nullspace forward simulation kills.")
        print(f"  → Sequential RL has structural value. Proceed.")
    elif (diff > 0).sum() < (diff < 0).sum():
        print(f"  ✗ Backward DP finds FEWER valid starts than naive forward.")
        print(f"    Likely the q_dist_thresh is too tight, or the K_NULL pull is")
        print(f"    too aggressive and disrupts naturally-feasible trajectories.")
        print(f"  → Reconsider.")
    else:
        print(f"  ≈ Backward DP and naive forward give similar results.")
        print(f"    Sequential branch switching adds little; framework probably")
        print(f"    won't beat preview-bandit by much.")


if __name__ == "__main__":
    main()

"""Backward Constrained Projection — feasibility-region shrinkage demo.

Concrete validation of the user's hypothesis
--------------------------------------------
"Some q at x_0 satisfies IK at x_0 (= naive feasible), but cannot reach
ANY feasible q at x_N via the controller (= structurally infeasible).
A naive bandit picks from the naive-feasible set; backward projection
identifies the smaller, structurally-feasible subset."

If |S_0| << |Q_0| (where S_0 = backward-feasible, Q_0 = naive-feasible at
the start), the framework's premise is validated. The shrinkage AT EACH
CHECKPOINT shows where downstream constraints bite. NN's job (later) is to
fit the membership function of S_i.

Algorithm
---------
1. Per task, fix one (φ, ψ). Discretize path into N segments → N+1 checkpoints.
2. At each checkpoint x_i, dense-sample candidate q's via random-seed IK:
       Q_i = { q : random_seed → IK at (x_i, R_tgt) converges }
3. Backward projection:
       S_N = Q_N    (terminal: any IK-converged q is feasible)
       S_i = { q ∈ Q_i : ∃ q_next ∈ S_{i+1}.
                         controller(q, q_ref=q_next, segment) succeeds }
4. Shrinkage at each i = 1 − |S_i| / |Q_i|.

If shrinkage at i=0 is significant (>10%), framework valid.
"""
from __future__ import annotations
import argparse, time
import numpy as np
import torch

import Yuan.RL.config as cfg
from Yuan.RL.env import FarsightedSeedEnv
from Yuan.RL.batched_rollout import (
    batched_rollout_segment, branch_project_multistart,
    build_branch_rotmat_batch, _batched_ik_project,
    _device_from_cfg, _load_fr3_sphere_collision_cls, phantom_rollout,
)
from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics


def _dense_ik_at(kin, p_target, R_target, branch_action, M_oversample: int,
                 rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample M_oversample random joint configs as IK seeds, run IK, return
    (q_kept, ok_mask) where q_kept are the converged solutions."""
    device = kin.device
    lo = kin.lmt_lo.cpu().numpy()
    hi = kin.lmt_up.cpu().numpy()
    seeds_np = rng.uniform(lo[None, :], hi[None, :],
                           size=(M_oversample, 7)).astype(np.float32)
    q_seed = torch.as_tensor(seeds_np, device=device, dtype=torch.float32)
    p_rep = p_target.unsqueeze(0).expand(M_oversample, 3)
    R_rep = R_target.unsqueeze(0).expand(M_oversample, 3, 3)
    a_rep = branch_action.unsqueeze(0).expand(M_oversample, 4)
    q, ok, _ = _batched_ik_project(kin, q_seed, p_rep, R_rep, branch_action=a_rep)
    return q, ok


def _check_transitions_dense(kin,
                             q_curr: torch.Tensor,         # (Mc, 7)
                             q_target: torch.Tensor,       # (Mn, 7)
                             R_tgt: torch.Tensor,          # (3, 3)
                             a_t: torch.Tensor,            # (4,)
                             p0: torch.Tensor,             # (3,)
                             d_dir: torch.Tensor,          # (3,)
                             v_path_scalar: float,
                             eps_p_scalar: float,
                             T_total_scalar: int,
                             start_step: int,
                             end_step: int,
                             sphere_cc,
                             q_dist_thresh: float = 1.0,
                             chunk_size: int = 4096,
                             ) -> torch.Tensor:
    """Returns (Mc, Mn) bool: success[j, k] = controller can drive q_curr[j]
    to within q_dist_thresh of q_target[k] over segment [start_step, end_step]."""
    device = kin.device
    Mc = q_curr.shape[0]
    Mn = q_target.shape[0]
    if Mc == 0 or Mn == 0:
        return torch.zeros(Mc, Mn, device=device, dtype=torch.bool)
    # build (Mc * Mn) flat batch
    j_idx = torch.arange(Mc, device=device).view(Mc, 1).expand(Mc, Mn).reshape(-1)
    k_idx = torch.arange(Mn, device=device).view(1, Mn).expand(Mc, Mn).reshape(-1)
    q_init_flat = q_curr[j_idx]
    q_targ_flat = q_target[k_idx]
    n_pairs = Mc * Mn
    # broadcast scalars
    R_flat   = R_tgt.unsqueeze(0).expand(n_pairs, 3, 3)
    a_flat   = a_t.unsqueeze(0).expand(n_pairs, 4)
    p0_flat  = p0.unsqueeze(0).expand(n_pairs, 3)
    d_flat   = d_dir.unsqueeze(0).expand(n_pairs, 3)
    v_flat   = torch.full((n_pairs,), float(v_path_scalar), device=device, dtype=torch.float32)
    eps_flat = torch.full((n_pairs,), float(eps_p_scalar),  device=device, dtype=torch.float32)
    T_flat   = torch.full((n_pairs,), int(T_total_scalar),   device=device, dtype=torch.long)

    success_flat = torch.zeros(n_pairs, device=device, dtype=torch.bool)
    for s in range(0, n_pairs, chunk_size):
        e = min(s + chunk_size, n_pairs)
        out = batched_rollout_segment(
            q_init_flat[s:e], R_flat[s:e], a_flat[s:e],
            p0_flat[s:e], d_flat[s:e],
            v_flat[s:e], eps_flat[s:e], T_flat[s:e],
            start_step=start_step, end_step=end_step,
            preset_gains=None, alive_mask=None,
            sphere_cc=sphere_cc, kin=kin, is_phantom=False,
            q_ref=q_targ_flat[s:e])
        q_final = out['q_final']
        alive = out['alive_out']
        q_diff = (q_final - q_targ_flat[s:e]).norm(dim=-1)
        success_flat[s:e] = alive & (q_diff < q_dist_thresh)

    return success_flat.view(Mc, Mn)


def analyze_one_task(kin, sphere_cc, task: dict, args, rng):
    device = kin.device
    c = task['c']
    p0 = torch.as_tensor(c[:3], device=device, dtype=torch.float32)
    d_dir = torch.as_tensor(c[3:6], device=device, dtype=torch.float32)
    n_out = torch.as_tensor(c[6:9], device=device, dtype=torch.float32)
    v_path = float(task['v_path'])
    eps_p  = float(task['eps_p'])
    T_eff  = int(args.T_eval) if args.T_eval is not None else int(task['T'])

    # pick (φ, ψ): use preview-best from K random samples (mimics deploy's bandit)
    K_pp = 8
    phi = rng.uniform(0, 2*np.pi, size=K_pp).astype(np.float32)
    psi = rng.uniform(0, 2*np.pi, size=K_pp).astype(np.float32)
    a_cands = np.stack([np.cos(phi), np.sin(phi),
                        np.cos(psi), np.sin(psi)], axis=-1)              # (K_pp, 4)
    a_cands_flat = a_cands.astype(np.float32)
    c_rep = np.tile(c[None, :], (K_pp, 1)).astype(np.float32)
    v_rep = np.full(K_pp, v_path, dtype=np.float32)
    e_rep = np.full(K_pp, eps_p,  dtype=np.float32)
    T_rep = np.full(K_pp, T_eff,  dtype=np.int32)
    ph_out = phantom_rollout(a_cands_flat, c_rep, v_rep, e_rep, T_rep)
    L_ph   = np.asarray(ph_out['lengths'], dtype=np.int32)
    best_pp = int(L_ph.argmax())
    a_t = torch.as_tensor(a_cands[best_pp], device=device, dtype=torch.float32)
    R_tgt = build_branch_rotmat_batch(d_dir.unsqueeze(0), n_out.unsqueeze(0),
                                      a_t.unsqueeze(0)).squeeze(0)        # (3, 3)

    N = int(args.n_segments)
    M_over = int(args.M_oversample)
    q_dist_thresh = float(args.q_dist_thresh)

    # ----- 1. Dense Q at each checkpoint -----
    Q = []                 # list of (Mi, 7) tensors
    for i in range(N + 1):
        t_step = i * T_eff // N
        x_i = p0 + (t_step * float(cfg.DT)) * v_path * d_dir
        q_i, ok_i = _dense_ik_at(kin, x_i, R_tgt, a_t, M_over, rng)
        Q.append(q_i[ok_i])

    Q_sizes = [q.shape[0] for q in Q]
    print(f"  |Q_i| per checkpoint: " + " ".join(f"{s:3d}" for s in Q_sizes))

    # ----- 2. Backward projection -----
    S = [None] * (N + 1)
    S[N] = Q[N]                                    # terminal: all naive feasible
    for i in range(N - 1, -1, -1):
        t_start = i * T_eff // N
        t_end   = (i + 1) * T_eff // N
        if Q[i].shape[0] == 0 or S[i + 1].shape[0] == 0:
            S[i] = Q[i][:0]
            continue
        success = _check_transitions_dense(
            kin, Q[i], S[i + 1], R_tgt, a_t, p0, d_dir,
            v_path, eps_p, T_eff,
            start_step=t_start, end_step=t_end,
            sphere_cc=sphere_cc, q_dist_thresh=q_dist_thresh,
            chunk_size=args.chunk_size)
        # q in Q[i] is in S[i] iff it has any successful transition to S[i+1]
        in_S = success.any(dim=1)
        S[i] = Q[i][in_S]

    S_sizes = [s.shape[0] for s in S]
    shrinkage = [1.0 - (s / max(q, 1)) for s, q in zip(S_sizes, Q_sizes)]

    print(f"  |S_i| per checkpoint: " + " ".join(f"{s:3d}" for s in S_sizes))
    print(f"  shrink (1-S/Q):       " + " ".join(f"{s:.2f}" for s in shrinkage))

    return Q_sizes, S_sizes, shrinkage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tasks", type=int, default=10)
    ap.add_argument("--n-segments", type=int, default=5)
    ap.add_argument("--M-oversample", type=int, default=64,
                    help="random IK seeds per checkpoint (most won't converge; "
                         "we keep all that do)")
    ap.add_argument("--q-dist-thresh", type=float, default=1.0,
                    help="rad: transition success requires final q within this "
                         "Euclidean distance of target q_ref")
    ap.add_argument("--T-eval", type=int, default=80)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--min-base-dist", type=float, default=0.30)
    ap.add_argument("--M-sweep", type=str, default=None,
                    help="comma-list of M values for convergence test, "
                         "e.g. '32,64,128,256'. Overrides --M-oversample.")
    args = ap.parse_args()

    print(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"n_tasks={args.n_tasks}  N_seg={args.n_segments}  "
          f"M_over={args.M_oversample}  q_dist={args.q_dist_thresh}  T={args.T_eval}")

    device = _device_from_cfg()
    kin = BatchedFR3Kinematics(device=device)
    sphere_cc = None
    if cfg.USE_COLLISION_CHECK and cfg.BATCHED_COLLISION_CHECK:
        sphere_cc = _load_fr3_sphere_collision_cls()(
            device=device, margin=cfg.BATCHED_COLLISION_MARGIN)

    env = FarsightedSeedEnv(seed=args.seed, randomize=False, contact_mode=False)
    tasks = env._sample_tasks(args.n_tasks)
    base_dist = np.linalg.norm(np.stack([t['c'][:3] for t in tasks]), axis=-1)

    # M sweep mode: same task set, multiple M values, see if shrinkage converges
    if args.M_sweep is not None:
        M_list = [int(m) for m in args.M_sweep.split(",")]
        print(f"\n=== M-CONVERGENCE TEST  M_list={M_list} ===")
        eligible_tasks = []
        for b, task in enumerate(tasks):
            if base_dist[b] < args.min_base_dist or base_dist[b] > 0.85:
                continue
            eligible_tasks.append(task)
            if len(eligible_tasks) >= 5:
                break
        print(f"using {len(eligible_tasks)} fixed tasks for the sweep")
        sweep_results = {}
        for M in M_list:
            args.M_oversample = M
            rng = np.random.default_rng(args.seed)
            sh0_per_task = []
            t0 = time.perf_counter()
            for b, task in enumerate(eligible_tasks):
                Q_sz, S_sz, shr = analyze_one_task(kin, sphere_cc, task, args, rng)
                if min(Q_sz) == 0:
                    continue
                sh0_per_task.append((shr[0], Q_sz[0], S_sz[0]))
            wall = time.perf_counter() - t0
            sweep_results[M] = sh0_per_task
            mean_sh = np.mean([s for s, _, _ in sh0_per_task])
            mean_Q  = np.mean([q for _, q, _ in sh0_per_task])
            mean_S  = np.mean([s for _, _, s in sh0_per_task])
            print(f"\n  M={M:>4d}  wall={wall:.0f}s  "
                  f"|Q_0| mean={mean_Q:.1f}  |S_0| mean={mean_S:.1f}  "
                  f"shrink_0 mean={mean_sh:.3f}")
        # final summary table
        print("\n" + "=" * 60)
        print(f"{'M':>5}  {'|Q_0|':>6}  {'|S_0|':>6}  {'shrink_0':>9}")
        print("-" * 60)
        for M in M_list:
            r = sweep_results[M]
            mean_sh = np.mean([s for s, _, _ in r])
            mean_Q  = np.mean([q for _, q, _ in r])
            mean_S  = np.mean([s for _, _, s in r])
            print(f"{M:>5d}  {mean_Q:>6.1f}  {mean_S:>6.1f}  {mean_sh:>9.3f}")
        return

    rng = np.random.default_rng(args.seed)
    all_Q = []
    all_S = []
    all_shrink = []
    skipped_oob = 0
    skipped_oor = 0

    t0 = time.perf_counter()
    for b, task in enumerate(tasks):
        if base_dist[b] < args.min_base_dist:
            continue
        if base_dist[b] > 0.85:                  # outside FR3 nominal reach
            skipped_oor += 1
            continue
        print(f"\nTask {b:3d}  ‖p₀‖={base_dist[b]:.2f}")
        Q_sz, S_sz, shr = analyze_one_task(kin, sphere_cc, task, args, rng)
        # filter: drop tasks where ANY checkpoint has |Q|=0 (path leaves workspace)
        if min(Q_sz) == 0:
            skipped_oob += 1
            print(f"  -- skip: path leaves workspace (|Q|={Q_sz})")
            continue
        all_Q.append(Q_sz)
        all_S.append(S_sz)
        all_shrink.append(shr)
    wall = time.perf_counter() - t0
    print(f"\nSkipped {skipped_oor} out-of-reach + {skipped_oob} path-leaves-workspace")

    print("\n" + "=" * 70)
    print(f"AGGREGATE  (n={len(all_Q)} well-defined tasks, wall={wall:.0f}s)")
    print("=" * 70)
    Q_arr = np.array(all_Q, dtype=np.float64)        # (n_tasks, N+1)
    S_arr = np.array(all_S, dtype=np.float64)
    shr_arr = np.array(all_shrink, dtype=np.float64)

    print(f"\n{'checkpoint':>11}  {'|Q| mean':>10}  {'|S| mean':>10}  "
          f"{'shrink mean':>11}  {'shrink max':>10}")
    for i in range(Q_arr.shape[1]):
        print(f"  {i:>9d}  {Q_arr[:, i].mean():>10.1f}  "
              f"{S_arr[:, i].mean():>10.1f}  "
              f"{shr_arr[:, i].mean():>11.3f}  {shr_arr[:, i].max():>10.3f}")

    # KEY METRIC: shrinkage at i=0
    print(f"\nKEY: shrinkage at i=0 (= start)")
    s0 = shr_arr[:, 0]
    print(f"  mean = {s0.mean():.3f}  median = {np.median(s0):.3f}  "
          f"max = {s0.max():.3f}")
    print(f"  tasks with shrinkage > 0.10:  "
          f"{int((s0 > 0.10).sum())}/{len(s0)}  "
          f"({100*(s0>0.10).mean():.0f}%)")
    print(f"  tasks with shrinkage > 0.30:  "
          f"{int((s0 > 0.30).sum())}/{len(s0)}  "
          f"({100*(s0>0.30).mean():.0f}%)")

    print("\n" + "-" * 70)
    if s0.mean() > 0.10:
        print(f"  ✓ Significant shrinkage at start: mean {s0.mean():.2f}")
        print(f"    → {(s0>0.10).sum()}/{len(s0)} tasks have ≥10% of naive-feasible q's")
        print(f"      that are STRUCTURALLY infeasible. NN can learn this signal.")
        print(f"  → Backward Constrained Projection framework VALIDATED.")
    elif s0.mean() > 0.03:
        print(f"  ~ Mild shrinkage: mean {s0.mean():.2f}")
        print(f"    Some structure exists but signal may be weak.")
    else:
        print(f"  ✗ No significant shrinkage: mean {s0.mean():.2f}")
        print(f"    Almost every IK-feasible q at start can also reach the end.")
        print(f"  → Sequential framework adds little for THIS task type.")


if __name__ == "__main__":
    main()

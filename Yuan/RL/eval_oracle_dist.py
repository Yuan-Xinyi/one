"""Evaluate trained policy against a TASK-NULLSPACE oracle.

For each task, the branch_descriptor action (phi, psi) parameterizes the
2-D null space (elbow swivel + tool roll) — every (phi, psi) is a valid
seed that satisfies the task's IK constraint. We compare:
  1. Deterministic policy action  -> L_det
  2. K uniform (phi, psi) samples -> L_oracle distribution (policy-agnostic)

This is a tighter upper bound than the training-time policy-best-of-K
because it doesn't depend on the policy's exploration distribution.

Usage:
    python -m Yuan.RL.eval_oracle_dist --ckpt Yuan/RL/checkpoints_v11b_sampled_oracle_k8/ckpt_005000.pt
    python -m Yuan.RL.eval_oracle_dist --n-tasks 32 --k 1000
"""
from __future__ import annotations
import argparse, os
import numpy as np
import torch

import Yuan.RL.config as cfg
from Yuan.RL.env import FarsightedSeedEnv
from Yuan.RL.policy import make_policy
from Yuan.RL.batched_rollout import batched_rollout


def _load_policy(ckpt_path, env, device):
    q_mid = torch.as_tensor(env.action_mid, dtype=torch.float32, device=device)
    q_half = torch.as_tensor(env.action_half, dtype=torch.float32, device=device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy = make_policy(cfg.STATE_DIM, env.action_dim, q_mid, q_half,
                         policy_type=state.get("policy_type", "gaussian")).to(device)
    policy.load_state_dict(state["policy"])
    policy.eval()
    return policy, state.get("iter", -1)


def _rollout_chunked(actions_np, c_np, v_np, e_np, T_np, chunk=4096):
    """Run batched_rollout in chunks to bound peak GPU mem."""
    n = actions_np.shape[0]
    L_out = np.empty(n, dtype=np.int32)
    reasons_out = []
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        out = batched_rollout(actions_np[s:e], c_np[s:e], v_np[s:e],
                              e_np[s:e], T_np[s:e])
        L_out[s:e] = np.asarray(out["lengths"], dtype=np.int32)
        reasons_out.extend(out.get("reasons", []) or [""] * (e - s))
    return L_out, reasons_out


def evaluate_oracle_distribution(policy, env, n_tasks=32, K=1000, seed=12345):
    device = next(policy.parameters()).device
    rng = np.random.default_rng(seed)

    # ----- sample N tasks -----
    env.rng = np.random.default_rng(seed)
    tasks = env._sample_tasks(n_tasks)
    states = np.stack([env._state_vec(t) for t in tasks], axis=0).astype(np.float32)
    c_np = np.stack([t["c"] for t in tasks], axis=0).astype(np.float32)
    v_np = np.array([t["v_path"] for t in tasks], dtype=np.float32)
    e_np = np.array([t["eps_p"]  for t in tasks], dtype=np.float32)
    T_np = np.array([t["T"]      for t in tasks], dtype=np.int32)

    states_t = torch.as_tensor(states, dtype=torch.float32, device=device)

    # ----- 1) deterministic action per task -----
    with torch.no_grad():
        a_det, _ = policy.act(states_t, deterministic=True)
    a_det_np = a_det.cpu().numpy().astype(np.float32)
    L_det, _ = _rollout_chunked(a_det_np, c_np, v_np, e_np, T_np, chunk=4096)

    # ----- 2) K uniform (phi, psi) null-space samples per task -----
    # Sample phi, psi ~ U[0, 2pi) independently per (k, task); encode as
    # (cos phi, sin phi, cos psi, sin psi) — same parameterization the
    # policy outputs (see batched_rollout.build_branch_rotmat_batch).
    phi = rng.uniform(0.0, 2 * np.pi, size=(K, n_tasks)).astype(np.float32)
    psi = rng.uniform(0.0, 2 * np.pi, size=(K, n_tasks)).astype(np.float32)
    a_oracle = np.stack([np.cos(phi), np.sin(phi),
                         np.cos(psi), np.sin(psi)], axis=-1)   # (K, n_tasks, 4)
    a_oracle_np = a_oracle.reshape(K * n_tasks, 4).astype(np.float32)

    rep_c = np.tile(c_np, (K, 1))
    rep_v = np.tile(v_np, K)
    rep_e = np.tile(e_np, K)
    rep_T = np.tile(T_np, K)
    L_stoch_flat, _ = _rollout_chunked(a_oracle_np, rep_c, rep_v, rep_e, rep_T,
                                       chunk=4096)
    L_stoch = L_stoch_flat.reshape(K, n_tasks)            # (K, n_tasks)

    return {
        "tasks":   tasks,
        "T":       T_np,
        "L_det":   L_det,
        "L_stoch": L_stoch,
    }


def _summarize(out, K_eval=(1, 8, 32, 128, 1000),
               min_oracle_dist: float = 0.20):
    T = out["T"]
    L_det = out["L_det"]
    L_st  = out["L_stoch"]
    n_tasks = T.shape[0]
    K = L_st.shape[0]
    L_best = L_st.max(axis=0)

    # "well-defined" tasks: oracle (best-of-K_max) actually moves the TCP
    # at least min_oracle_dist meters along the path direction. Threshold
    # is absolute physical distance (meters), not relative to T. Distance
    # per oracle = L_best * DT * v_path.
    v_path = np.array([t["v_path"] for t in out["tasks"]], dtype=np.float64)
    oracle_dist = L_best.astype(np.float64) * float(cfg.DT) * v_path
    well = oracle_dist >= min_oracle_dist

    def _ratio_mean(num, den, mask=None):
        m = (den > 0) if mask is None else (mask & (den > 0))
        if not m.any():
            return float('nan')
        return float((num[m] / den[m]).mean())

    print(f"\n=== {n_tasks} tasks  (K={K} uniform (phi,psi) per task) ===")
    n_feas = int((L_best > 0).sum())
    n_well = int(well.sum())
    print(f"feasible       (L_best > 0)                    : {n_feas}/{n_tasks}")
    print(f"well-defined   (oracle TCP distance >= {min_oracle_dist*100:.0f} cm) : "
          f"{n_well}/{n_tasks}     ← used for stats below")

    # det / best-of-K ratio across K growth — the key signal.
    # K=1 is a single random shot (baseline, not oracle): ratio > 1 means
    # policy beats a random branch on average. K large is the true oracle.
    print("\npolicy_det / best-of-K_uniform per-task ratio stats (well-defined tasks):")
    print(f"  {'K':>5}  {'mean':>7}  {'std':>7}  {'min':>7}  "
          f"{'p25':>7}  {'p50':>7}  {'p75':>7}  {'max':>7}  {'role':<22}")
    for k in K_eval:
        if k > K:
            continue
        L_bestK = L_st[:k].max(axis=0).astype(np.float64)
        m = well & (L_bestK > 0)
        if not m.any():
            continue
        r = (L_det.astype(np.float64)[m] / L_bestK[m])
        if k == 1:
            role = "random-shot baseline"
        elif k >= 1000:
            role = "true upper bound"
        else:
            role = "intermediate oracle"
        print(f"  {k:>5d}  "
              f"{r.mean():>7.4f}  {r.std():>7.4f}  {r.min():>7.4f}  "
              f"{np.percentile(r, 25):>7.4f}  {np.percentile(r, 50):>7.4f}  "
              f"{np.percentile(r, 75):>7.4f}  {r.max():>7.4f}  "
              f"{role:<22}")

    # K-sample distribution shape, normalized to each task's L_best
    print("\nuniform-sample distribution (avg of L_pct/L_best across well-defined tasks):")
    print(f"  {'pct':>5}  {'L_pct/L_best':>14}")
    for p in [10, 25, 50, 75, 90, 99, 100]:
        vals = np.percentile(L_st, p, axis=0)
        rec = (vals[well] / L_best[well]).mean()
        print(f"  {p:>5d}  {rec:>14.4f}")

    # where does L_det land in the K-sample distribution per task?
    p50 = np.percentile(L_st, 50, axis=0)
    p90 = np.percentile(L_st, 90, axis=0)
    pmax = L_st.max(axis=0)
    print("\nper-task percentile rank of L_det in the K-sample distribution (well-defined):")
    print(f"  L_det >= median:  {(L_det[well] >= p50[well]).mean():.3f}")
    print(f"  L_det >= p90:     {(L_det[well] >= p90[well]).mean():.3f}")
    print(f"  L_det == max:     {(L_det[well] >= pmax[well]).mean():.3f}")

    # per-task table (top-and-bottom by ratio, well-defined only)
    safe_best = np.where(L_best > 0, L_best, 1)
    ratio_per_task = L_det.astype(np.float64) / safe_best
    ratio_per_task[L_best == 0] = float('nan')
    sort_idx = np.argsort(np.where(np.isnan(ratio_per_task), 1e9, ratio_per_task))
    print("\nworst 5 well-defined tasks (det/best):")
    print(f"  {'i':>3}  {'T':>4}  {'L_det':>5}  {'L_best':>6}  {'L_med':>6}  "
          f"{'orc_m':>6}  {'ratio':>6}")
    shown = 0
    for i in sort_idx:
        if not well[i] or shown >= 5:
            continue
        print(f"  {i:>3d}  {T[i]:>4d}  {L_det[i]:>5d}  {L_best[i]:>6d}  "
              f"{int(p50[i]):>6d}  {oracle_dist[i]:>6.3f}  "
              f"{ratio_per_task[i]:>6.3f}")
        shown += 1
    print("best 5 well-defined tasks (det/best):")
    shown = 0
    for i in sort_idx[::-1]:
        if not well[i] or shown >= 5:
            continue
        print(f"  {i:>3d}  {T[i]:>4d}  {L_det[i]:>5d}  {L_best[i]:>6d}  "
              f"{int(p50[i]):>6d}  {oracle_dist[i]:>6.3f}  "
              f"{ratio_per_task[i]:>6.3f}")
        shown += 1

    # list the dropped (oracle-distance-too-short) tasks for transparency
    dropped = ~well & (L_best > 0)
    if dropped.any():
        print(f"\ndropped {int(dropped.sum())} tasks "
              f"(oracle TCP distance < {min_oracle_dist*100:.0f} cm):")
        print(f"  {'i':>3}  {'T':>4}  {'L_best':>6}  {'v':>5}  {'orc_m':>6}")
        for i in np.where(dropped)[0]:
            print(f"  {i:>3d}  {T[i]:>4d}  {L_best[i]:>6d}  "
                  f"{v_path[i]:>5.2f}  {oracle_dist[i]:>6.3f}")
    if (L_best == 0).any():
        infeas = np.where(L_best == 0)[0]
        print(f"\ninfeasible {len(infeas)} tasks (L_best == 0): "
              f"{', '.join(str(i) for i in infeas)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str,
                    default=os.path.join(cfg.CKPT_DIR, "ckpt_005000.pt"))
    ap.add_argument("--n-tasks", type=int, default=32)
    ap.add_argument("--k", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--min-oracle-distance", type=float, default=0.20,
                    help="drop tasks where the oracle's best rollout TCP "
                         "distance (L_best * DT * v_path, meters) falls "
                         "below this — those tasks are intrinsically "
                         "infeasible and unfairly penalize the policy. "
                         "default 0.20 m (20 cm).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    env = FarsightedSeedEnv(seed=args.seed, randomize=True, use_collision=False)
    print(f"loading {args.ckpt}")
    policy, it = _load_policy(args.ckpt, env, device)
    print(f"  ckpt iter = {it}")

    out = evaluate_oracle_distribution(policy, env,
                                       n_tasks=args.n_tasks,
                                       K=args.k,
                                       seed=args.seed)
    _summarize(out, min_oracle_dist=args.min_oracle_distance)


if __name__ == "__main__":
    main()

"""Dump per-task policy_det L for the well-defined task subset.

For each well-defined task (passes both --min-oracle-distance and
--min-base-dist filters), prints:
    i, T (steps), L_det (steps), L_det_m (meters), v, ||p0||

Reuses cached oracle data if a recent eval_oracle_dist has been run with
the same n_tasks/seed; otherwise re-runs the oracle sweep.

Usage:
    python -m Yuan.RL.dump_well_defined \\
        --ckpt Yuan/RL/checkpoints_v12_pen_per_fixA_10k/ckpt_010000.pt
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
    return policy


def _rollout_chunked(actions_np, c_np, v_np, e_np, T_np, chunk=4096):
    n = actions_np.shape[0]
    L = np.empty(n, dtype=np.int32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        out = batched_rollout(actions_np[s:e], c_np[s:e], v_np[s:e],
                              e_np[s:e], T_np[s:e])
        L[s:e] = np.asarray(out["lengths"], dtype=np.int32)
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str,
                    default=os.path.join(cfg.CKPT_DIR, "ckpt_010000.pt"))
    ap.add_argument("--n-tasks", type=int, default=200)
    ap.add_argument("--k-oracle", type=int, default=1000,
                    help="K uniform samples used to compute L_best for filter")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--min-oracle-distance", type=float, default=0.20)
    ap.add_argument("--min-base-dist", type=float, default=0.30)
    ap.add_argument("--sort-by", choices=["i", "ratio", "L_det", "T"],
                    default="ratio")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    env = FarsightedSeedEnv(seed=args.seed, randomize=True, use_collision=False)
    policy = _load_policy(args.ckpt, env, device)

    env.rng = np.random.default_rng(args.seed)
    rng = np.random.default_rng(args.seed)
    tasks = env._sample_tasks(args.n_tasks)
    states = np.stack([env._state_vec(t) for t in tasks], axis=0).astype(np.float32)
    c_np = np.stack([t["c"] for t in tasks], axis=0).astype(np.float32)
    v_np = np.array([t["v_path"] for t in tasks], dtype=np.float32)
    e_np = np.array([t["eps_p"]  for t in tasks], dtype=np.float32)
    T_np = np.array([t["T"]      for t in tasks], dtype=np.int32)

    states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
    print("running policy_det rollouts...")
    with torch.no_grad():
        a_det, _ = policy.act(states_t, deterministic=True)
    L_det = _rollout_chunked(a_det.cpu().numpy().astype(np.float32),
                             c_np, v_np, e_np, T_np)

    print(f"running K={args.k_oracle} uniform oracle for filter...")
    K = args.k_oracle
    phi = rng.uniform(0.0, 2*np.pi, size=(K, args.n_tasks)).astype(np.float32)
    psi = rng.uniform(0.0, 2*np.pi, size=(K, args.n_tasks)).astype(np.float32)
    a_orc = np.stack([np.cos(phi), np.sin(phi),
                      np.cos(psi), np.sin(psi)], axis=-1).reshape(K*args.n_tasks, 4)
    rep_c = np.tile(c_np, (K, 1))
    rep_v = np.tile(v_np, K)
    rep_e = np.tile(e_np, K)
    rep_T = np.tile(T_np, K)
    L_flat = _rollout_chunked(a_orc.astype(np.float32),
                              rep_c, rep_v, rep_e, rep_T)
    L_best = L_flat.reshape(K, args.n_tasks).max(axis=0)

    # apply both filters
    DT = float(cfg.DT)
    oracle_m = L_best.astype(np.float64) * DT * v_np.astype(np.float64)
    p0 = c_np[:, :3]
    base_d = np.linalg.norm(p0, axis=-1)
    well = (oracle_m >= args.min_oracle_distance) & (base_d >= args.min_base_dist)

    L_det_m = L_det.astype(np.float64) * DT * v_np.astype(np.float64)
    ratio = np.where(L_best > 0,
                     L_det.astype(np.float64) / np.maximum(L_best, 1),
                     0.0)

    idx_well = np.where(well)[0]
    keys = {
        "i":     idx_well,
        "ratio": ratio[idx_well],
        "L_det": -L_det[idx_well].astype(float),       # descending
        "T":     -T_np[idx_well].astype(float),
    }
    order = np.argsort(keys[args.sort_by])
    sorted_idx = idx_well[order]

    n_well = len(idx_well)
    print(f"\n{n_well} well-defined tasks (sorted by {args.sort_by}):")
    print(f"  {'i':>3}  {'T':>4}  {'L_det':>5}  {'L_det_m':>7}  "
          f"{'L_best':>6}  {'orc_m':>6}  {'v':>5}  {'||p0||':>6}  {'ratio':>6}")
    for i in sorted_idx:
        print(f"  {i:>3d}  {T_np[i]:>4d}  {L_det[i]:>5d}  "
              f"{L_det_m[i]:>7.3f}  {L_best[i]:>6d}  {oracle_m[i]:>6.3f}  "
              f"{v_np[i]:>5.2f}  {base_d[i]:>6.3f}  {ratio[i]:>6.3f}")

    # quick stats
    L_d = L_det_m[idx_well]
    print(f"\nL_det meters across {n_well} well-defined tasks:")
    print(f"  mean={L_d.mean():.3f} m  std={L_d.std():.3f}  "
          f"min={L_d.min():.3f}  median={np.median(L_d):.3f}  "
          f"max={L_d.max():.3f}")
    L_b = oracle_m[idx_well]
    print(f"L_best (oracle) meters across {n_well} well-defined tasks:")
    print(f"  mean={L_b.mean():.3f} m  std={L_b.std():.3f}  "
          f"min={L_b.min():.3f}  median={np.median(L_b):.3f}  "
          f"max={L_b.max():.3f}")


if __name__ == "__main__":
    main()

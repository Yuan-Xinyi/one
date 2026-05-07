"""Phantom-rollout selector validation against the cached real-rollout L.

Hypothesis under test
---------------------
For path-tracking on FR3+pen, most rollout failures are geometric (joint
limit hits, IK chain breaks). If true, a *kinematic-only* phantom rollout
(no nullspace, no PD-controller fights) should approximately predict the
real L per candidate, and a phantom-argmax selector over K candidates
should approach the unif_oracle K=N ratio at a fraction of the wall time.

What this script does
---------------------
1. Loads an existing eval cache (with K_max real rollouts per task).
2. Recreates the same K_max uniform (φ, ψ) actions from the cache seed.
3. Runs `phantom_rollout` on those K_max*N actions (chunked for memory).
4. For each K, computes phantom_select_K = argmax_{k in 1..K} phantom_L_k,
   then looks up that pick's REAL L in the cache.
5. Reports phantom_select vs unif_oracle ratios alongside, plus timing.

Usage
-----
    python -m Yuan.RL.phantom_eval \\
        --ckpt Yuan/RL/checkpoints_v13_q_calibrated_10k/ckpt_010000.pt \\
        --n-tasks 1000 --k-list 1,2,4,8,16,32,128,1000
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
import torch

import Yuan.RL.config as cfg
from Yuan.RL.batched_rollout import phantom_rollout
from Yuan.RL.eval_heuristic_compare import _cache_path


def _phantom_chunked(actions_np, c_np, v_np, e_np, T_np,
                     chunk: int = 4096) -> np.ndarray:
    n = actions_np.shape[0]
    L = np.empty(n, dtype=np.int32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        out = phantom_rollout(actions_np[s:e], c_np[s:e], v_np[s:e],
                              e_np[s:e], T_np[s:e])
        L[s:e] = np.asarray(out['lengths'], dtype=np.int32)
    return L


def _ratio_mean(num, den, mask):
    num = num.astype(np.float64)
    den = den.astype(np.float64)
    base = mask & (den > 0)
    if not base.any():
        return float('nan')
    return float((num[base] / den[base]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-tasks", type=int, default=1000)
    ap.add_argument("--K-max", type=int, default=1000)
    ap.add_argument("--k-list", type=str,
                    default="1,2,4,8,16,32,128,1000")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--min-oracle-distance", type=float, default=0.20)
    ap.add_argument("--min-base-dist", type=float, default=0.30)
    ap.add_argument("--chunk", type=int, default=4096)
    args = ap.parse_args()

    K_list = [int(k) for k in args.k_list.split(',') if k.strip()]
    cache = _cache_path(args.ckpt, args.n_tasks, args.K_max, args.seed)
    if not os.path.exists(cache):
        raise SystemExit(f"cache not found: {cache}\n"
                         "run eval_heuristic_compare first to populate it.")
    print(f"[cache] loading {cache}")
    d = np.load(cache, allow_pickle=True)
    tasks = list(d["tasks"])
    T_np = np.asarray(d["T"], dtype=np.int32)
    L_unif = np.asarray(d["L_unif"], dtype=np.int32)
    L_pol  = np.asarray(d["L_pol"],  dtype=np.int32)
    L_det  = np.asarray(d["L_det"],  dtype=np.int32)
    K_max, n_tasks = L_unif.shape
    assert K_max == args.K_max and n_tasks == args.n_tasks, (
        f"cache shape {(K_max, n_tasks)} != requested "
        f"{(args.K_max, args.n_tasks)}")

    # well-defined mask
    L_top = L_unif.max(axis=0)
    v_path = np.array([t["v_path"] for t in tasks], dtype=np.float64)
    oracle_dist = L_top.astype(np.float64) * float(cfg.DT) * v_path
    p0 = np.stack([t["c"][:3] for t in tasks]).astype(np.float64)
    base_dist = np.linalg.norm(p0, axis=-1)
    well = (oracle_dist >= args.min_oracle_distance) & (base_dist >= args.min_base_dist)
    print(f"\n=== {n_tasks} tasks; K_max={K_max} ===")
    print(f"feasible     (L_top > 0)                                : "
          f"{int((L_top > 0).sum())}/{n_tasks}")
    print(f"well-defined (oracle TCP >= {args.min_oracle_distance*100:.0f}cm AND "
          f"||p0|| >= {args.min_base_dist*100:.0f}cm) : "
          f"{int(well.sum())}/{n_tasks}     ← used for stats")

    # ----- recreate the same uniform actions from the cache seed -----
    rng = np.random.default_rng(args.seed)
    # IMPORTANT: env.rng is also seeded but here we only care about uniform K
    # actions. The order of rng draws in precompute() is:
    #   1. env._sample_tasks (env.rng, separate)
    #   2. rng.uniform for phi (K_max, n_tasks)
    #   3. rng.uniform for psi (K_max, n_tasks)
    # So replicate that draw order.
    phi = rng.uniform(0.0, 2 * np.pi, size=(K_max, n_tasks)).astype(np.float32)
    psi = rng.uniform(0.0, 2 * np.pi, size=(K_max, n_tasks)).astype(np.float32)
    a_unif = np.stack([np.cos(phi), np.sin(phi),
                       np.cos(psi), np.sin(psi)], axis=-1)    # (K, N, 4)

    # task-broadcast contexts
    c_np = np.stack([t["c"] for t in tasks], axis=0).astype(np.float32)
    v_np = np.array([t["v_path"] for t in tasks], dtype=np.float32)
    e_np = np.array([t["eps_p"]  for t in tasks], dtype=np.float32)
    T_np_long = np.array([t["T"] for t in tasks], dtype=np.int32)

    a_unif_flat = a_unif.reshape(K_max * n_tasks, 4).astype(np.float32)
    rep_c = np.tile(c_np, (K_max, 1))
    rep_v = np.tile(v_np, K_max)
    rep_e = np.tile(e_np, K_max)
    rep_T = np.tile(T_np_long, K_max)

    # ----- run phantom -----
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    L_phant_flat = _phantom_chunked(a_unif_flat, rep_c, rep_v, rep_e, rep_T,
                                    chunk=args.chunk)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    phantom_wall = time.perf_counter() - t0
    L_phant = L_phant_flat.reshape(K_max, n_tasks)
    print(f"\n[phantom] ran K*N = {K_max*n_tasks} in {phantom_wall:.1f} s "
          f"({phantom_wall*1e6/(K_max*n_tasks):.1f} us / sample)")

    # sanity: phantom vs real correlation per task (Spearman not needed; show
    # rank-1 agreement and Pearson-like alignment)
    # how often phantom argmax == real argmax (per task)
    arange = np.arange(n_tasks)
    real_argmax = L_unif.argmax(axis=0)
    phant_argmax = L_phant.argmax(axis=0)
    rank1 = float((real_argmax == phant_argmax)[well].mean())
    # ratio of "real L of phantom argmax" vs "real L of real argmax"
    L_phant_pick_real = L_unif[phant_argmax, arange]    # (N,)
    r_phantom_full = _ratio_mean(L_phant_pick_real, L_top, mask=well)
    print(f"\n[phantom@K={K_max}] rank-1 agreement = {rank1:.3f}  "
          f"phantom-pick-vs-oracle ratio = {r_phantom_full:.4f}")

    # ----- main table: phantom_select vs uniform_oracle for each K -----
    print("\n=== Phantom selector vs uniform oracle (uniform K samples) ===")
    print(f"  {'K':>5}  {'unif_orc':>9}  {'phant_sel':>10}  {'gap_pp':>7}")
    print("  " + "-" * 38)
    for K in K_list:
        if K > K_max: continue
        Lu = L_unif[:K]
        Lp = L_phant[:K]
        # uniform_oracle K = max real L over first K
        L_orc_K = Lu.max(axis=0)
        # phantom_select K = real L of argmax phantom_L over first K
        L_psel_K = Lu[Lp.argmax(axis=0), arange]
        r_orc  = _ratio_mean(L_orc_K,  L_top, mask=well)
        r_psel = _ratio_mean(L_psel_K, L_top, mask=well)
        gap = (r_orc - r_psel) * 100.0
        print(f"  {K:>5d}  {r_orc:>9.4f}  {r_psel:>10.4f}  {gap:>7.2f}")

    # reference: how much real L can policy_det / cheap selectors get?
    r_pdet = _ratio_mean(L_det, L_top, mask=well)
    print(f"\n  (reference) policy_det = {r_pdet:.4f}")

    # ----- lower-bound stats at K=K_max -----
    print(f"\n=== Lower-bound stats at K={K_max} (well-defined n={int(well.sum())}) ===")
    print(f"  {'method':>26}  {'mean':>6}  {'std':>6}  {'min':>5}  "
          f"{'p10':>5}  {'p25':>5}  {'L=0':>4}  {'r<0.3':>5}")
    safe_top = np.where(L_top > 0, L_top, 1).astype(np.float64)
    L_orc_max = L_unif.max(axis=0)
    L_psel_max = L_unif[L_phant.argmax(axis=0), arange]
    for name, L in [("policy_det",                L_det),
                    ("uniform_oracle K_max",      L_orc_max),
                    ("phantom_select K_max",      L_psel_max)]:
        r = (L.astype(np.float64) / safe_top)[well]
        n0 = int((L[well] == 0).sum())
        nlo = int((r < 0.3).sum())
        print(f"  {name:>26}  {r.mean():>6.3f}  {r.std():>6.3f}  "
              f"{r.min():>5.3f}  {np.percentile(r,10):>5.3f}  "
              f"{np.percentile(r,25):>5.3f}  {n0:>4d}  {nlo:>5d}")
    # extended distribution
    print(f"\n=== Extended distribution stats (well-defined n={int(well.sum())}) ===")
    print(f"  {'method':>26}  {'mean':>6}  {'p50':>5}  {'p75':>5}  {'p90':>5}  "
          f"{'max':>5}  {'r<0.5':>6}  {'r<0.7':>6}  {'r<0.9':>6}  {'r≥0.99':>6}")
    for name, L in [("policy_det",                L_det),
                    ("uniform_oracle K_max",      L_orc_max),
                    ("phantom_select K_max",      L_psel_max)]:
        r = (L.astype(np.float64) / safe_top)[well]
        print(f"  {name:>26}  {r.mean():>6.3f}  "
              f"{np.percentile(r,50):>5.3f}  {np.percentile(r,75):>5.3f}  "
              f"{np.percentile(r,90):>5.3f}  {r.max():>5.3f}  "
              f"{int((r<0.5).sum()):>6d}  {int((r<0.7).sum()):>6d}  "
              f"{int((r<0.9).sum()):>6d}  {int((r>=0.99).sum()):>6d}")

    print(f"\n[speed] phantom: {phantom_wall*1e6/(K_max*n_tasks):.1f} us/sample")


if __name__ == "__main__":
    main()

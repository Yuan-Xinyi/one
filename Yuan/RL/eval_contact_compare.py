"""Sanity test for the contact-rollout pipeline.

Goal
----
Confirm that on contact-mode tasks (linear-spring surface contact + force
tracking), the phantom (kinematic) selector NO LONGER dominates the way it
does on the geometric task. If phantom_select still wins, the contact model
isn't biting and we need to tighten F_max / F_min or k_n.

What it does
------------
1. Sample N tasks (env in non-randomized mode for repeatability).
2. For each, sample K_max uniform (φ, ψ) candidates.
3. Run THREE rollout flavors per (task, sample):
     L_real_geo     = batched_rollout            (current geometric task)
     L_real_contact = batched_rollout_contact    (NEW: with force failures)
     L_phantom      = phantom_rollout            (NEW: kinematic, no nullspace)
4. Compare phantom-select vs uniform-oracle on EACH of the two real-rollout
   datasets. Phantom should still dominate on geo, but degrade on contact.

Usage
-----
    python -m Yuan.RL.eval_contact_compare --n-tasks 64 --K 32

This is a smoke test, not a full eval — meant to validate the contact model
shape before any retraining.
"""
from __future__ import annotations
import argparse, time
import numpy as np
import torch

import Yuan.RL.config as cfg
from Yuan.RL.env import FarsightedSeedEnv
from Yuan.RL.batched_rollout import (
    batched_rollout, batched_rollout_contact, phantom_rollout,
)


def _rollout_chunked(fn, actions_np, c_np, v_np, e_np, T_np, chunk=4096,
                     extra_keys=(), fn_kwargs=None):
    n = actions_np.shape[0]
    L = np.empty(n, dtype=np.int32)
    extras = {k: np.zeros(n, dtype=np.float32) for k in extra_keys}
    fn_kwargs = fn_kwargs or {}
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        out = fn(actions_np[s:e], c_np[s:e], v_np[s:e], e_np[s:e], T_np[s:e],
                 **fn_kwargs)
        L[s:e] = np.asarray(out['lengths'], dtype=np.int32)
        for k in extra_keys:
            extras[k][s:e] = np.asarray(out[k], dtype=np.float32)
    return L, extras


def _ratio_mean(num, den, mask):
    num = num.astype(np.float64)
    den = den.astype(np.float64)
    base = mask & (den > 0)
    if not base.any():
        return float('nan')
    return float((num[base] / den[base]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tasks", type=int, default=64)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--min-base-dist", type=float, default=0.30)
    ap.add_argument("--pen-target", type=float, default=None,
                    help="override CONTACT_PENETRATION_TARGET (m)")
    ap.add_argument("--pen-min", type=float, default=None,
                    help="override CONTACT_PEN_MIN (m)")
    ap.add_argument("--pen-max", type=float, default=None,
                    help="override CONTACT_PEN_MAX (m)")
    ap.add_argument("--k-n", type=float, default=None,
                    help="override CONTACT_K_N (N/m)")
    ap.add_argument("--use-dynamics", action="store_true",
                    help="enable v2 1-DOF tip mass-spring oscillator")
    ap.add_argument("--tip-mass", type=float, default=None)
    ap.add_argument("--grip-k",   type=float, default=None)
    ap.add_argument("--grip-c",   type=float, default=None)
    ap.add_argument("--n-substeps", type=int, default=None)
    args = ap.parse_args()

    print(f"device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    k_n   = args.k_n        if args.k_n        is not None else cfg.CONTACT_K_N
    p_tgt = args.pen_target if args.pen_target is not None else cfg.CONTACT_PENETRATION_TARGET
    p_min = args.pen_min    if args.pen_min    is not None else cfg.CONTACT_PEN_MIN
    p_max = args.pen_max    if args.pen_max    is not None else cfg.CONTACT_PEN_MAX
    print(f"contact knobs: K_n={k_n}  pen_target={p_tgt*1000:.1f}mm  "
          f"F_target={k_n*p_tgt:.1f}N  "
          f"F_range=[{k_n*p_min:.1f}, {k_n*p_max:.1f}]N  "
          f"grace={cfg.CONTACT_GRACE_STEPS}")
    contact_kwargs = dict(k_n=k_n, pen_target=p_tgt,
                          pen_min=p_min, pen_max=p_max)
    if args.use_dynamics:
        m_tip = args.tip_mass   if args.tip_mass   is not None else cfg.CONTACT_TIP_MASS
        kg    = args.grip_k     if args.grip_k     is not None else cfg.CONTACT_GRIP_K
        cg    = args.grip_c     if args.grip_c     is not None else cfg.CONTACT_GRIP_C
        nsub  = args.n_substeps if args.n_substeps is not None else cfg.CONTACT_N_SUBSTEPS
        omega_n = (kg / m_tip) ** 0.5
        zeta = cg / (2.0 * (kg * m_tip) ** 0.5)
        print(f"  v2 dyn ON: m={m_tip}kg K_grip={kg} C={cg}  "
              f"ω_n={omega_n:.0f} rad/s (T={2*3.14159/omega_n*1000:.1f}ms)  "
              f"ζ={zeta:.3f}  N_sub={nsub}")
        contact_kwargs.update(use_dynamics=True, tip_mass=m_tip,
                              grip_k=kg, grip_c=cg, n_substeps=nsub)

    env = FarsightedSeedEnv(seed=args.seed, randomize=False, contact_mode=False)
    tasks = env._sample_tasks(args.n_tasks)
    c_np = np.stack([t["c"] for t in tasks], axis=0).astype(np.float32)
    v_np = np.array([t["v_path"] for t in tasks], dtype=np.float32)
    e_np = np.array([t["eps_p"]  for t in tasks], dtype=np.float32)
    T_np = np.array([t["T"]      for t in tasks], dtype=np.int32)
    p0   = c_np[:, :3].astype(np.float64)
    base_dist = np.linalg.norm(p0, axis=-1)

    # uniform K candidates per task
    rng = np.random.default_rng(args.seed)
    phi = rng.uniform(0.0, 2 * np.pi, size=(args.K, args.n_tasks)).astype(np.float32)
    psi = rng.uniform(0.0, 2 * np.pi, size=(args.K, args.n_tasks)).astype(np.float32)
    a_unif = np.stack([np.cos(phi), np.sin(phi),
                       np.cos(psi), np.sin(psi)], axis=-1)         # (K, N, 4)
    a_flat = a_unif.reshape(args.K * args.n_tasks, 4).astype(np.float32)
    rep_c = np.tile(c_np, (args.K, 1))
    rep_v = np.tile(v_np, args.K)
    rep_e = np.tile(e_np, args.K)
    rep_T = np.tile(T_np, args.K)

    # ----- geometric (existing) rollout -----
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()
    L_geo_flat, _ = _rollout_chunked(batched_rollout, a_flat, rep_c, rep_v,
                                     rep_e, rep_T)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t_geo = time.perf_counter() - t0
    L_geo = L_geo_flat.reshape(args.K, args.n_tasks)

    # ----- contact rollout -----
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()
    L_con_flat, extras = _rollout_chunked(
        batched_rollout_contact, a_flat, rep_c, rep_v, rep_e, rep_T,
        extra_keys=('last_force',), fn_kwargs=contact_kwargs)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t_con = time.perf_counter() - t0
    L_con = L_con_flat.reshape(args.K, args.n_tasks)
    F_last = extras['last_force'].reshape(args.K, args.n_tasks)

    # ----- phantom rollout -----
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()
    L_phant_flat, _ = _rollout_chunked(phantom_rollout, a_flat, rep_c, rep_v,
                                       rep_e, rep_T)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t_phant = time.perf_counter() - t0
    L_phant = L_phant_flat.reshape(args.K, args.n_tasks)

    # ----- well-defined mask -----
    L_top_geo = L_geo.max(axis=0)
    L_top_con = L_con.max(axis=0)
    well = (base_dist >= args.min_base_dist) & (L_top_geo > 0) & (L_top_con > 0)
    print(f"\n=== {args.n_tasks} tasks, K={args.K} ===")
    print(f"feasible(geo, max>0): {int((L_top_geo>0).sum())}/{args.n_tasks}")
    print(f"feasible(con, max>0): {int((L_top_con>0).sum())}/{args.n_tasks}")
    print(f"well-defined (||p0||>={args.min_base_dist*100:.0f}cm AND both feas): "
          f"{int(well.sum())}/{args.n_tasks}")

    # contact-mode L typically << geo-mode L because force checks add failures
    geo_top_mean = float(L_top_geo[well].mean())
    con_top_mean = float(L_top_con[well].mean())
    print(f"\n[L_top mean over well-def] geo={geo_top_mean:.1f} steps  "
          f"con={con_top_mean:.1f} steps  "
          f"con/geo={con_top_mean/max(geo_top_mean,1e-6):.3f}")

    # rollout speed
    KN = args.K * args.n_tasks
    print(f"\n[speed] geo={t_geo*1e6/KN:.1f} us/sample  "
          f"con={t_con*1e6/KN:.1f}  phant={t_phant*1e6/KN:.1f}")

    # contact failure-mode breakdown (where contact > 0 but < geo)
    short_due_to_force = (L_con < L_geo).astype(np.float32).mean(axis=0)[well]
    print(f"[con vs geo] frac of K samples per task where con < geo: "
          f"{short_due_to_force.mean():.3f}")

    # ----- key result: phantom selector vs oracle, on each rollout flavor -----
    arange = np.arange(args.n_tasks)

    def _select_table(L_real, L_top, label):
        print(f"\n=== {label}: phantom selector vs uniform_oracle ===")
        print(f"  {'K':>5}  {'unif_orc':>9}  {'phant_sel':>10}  {'gap_pp':>7}")
        print("  " + "-" * 38)
        for K in [1, 2, 4, 8, 16, 32]:
            if K > args.K: continue
            Lr = L_real[:K]
            Lp = L_phant[:K]
            L_orc_K = Lr.max(axis=0)
            L_psel_K = Lr[Lp.argmax(axis=0), arange]
            r_orc  = _ratio_mean(L_orc_K,  L_top, mask=well)
            r_psel = _ratio_mean(L_psel_K, L_top, mask=well)
            gap = (r_orc - r_psel) * 100.0
            print(f"  {K:>5d}  {r_orc:>9.4f}  {r_psel:>10.4f}  {gap:>7.2f}")

    _select_table(L_geo, L_top_geo,
                  "GEO (existing kinematic task — phantom should win)")
    _select_table(L_con, L_top_con,
                  "CON (NEW contact task — phantom expected to lose)")

    # diagnostic: per-task force statistics
    print("\n[diag] last-step force stats over (K, N) well-def samples (N):")
    F_well = F_last[:, well].reshape(-1)
    print(f"  N={F_well.size}  mean={F_well.mean():.2f}N  "
          f"med={np.median(F_well):.2f}N  "
          f"p10={np.percentile(F_well,10):.2f}N  "
          f"p90={np.percentile(F_well,90):.2f}N  "
          f"max={F_well.max():.2f}N")


if __name__ == "__main__":
    main()

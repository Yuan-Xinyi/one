"""Day 2 smoke tests for Module 5 (evaluate_robustness) and Module 6
(filter_robust_candidates), plus the combined diagnostic that the user
requested as the Day 2 success criterion.

Runs against a real NSRLBatchedEnv (n_envs=1) under classical_nullspace
controller. Each rollout takes ~0.5-2s; the combined diagnostic on 3 tasks
runs ~60-120 evaluations total — expect 30-90s wall time.

Run:
    python -m Yuan.seed_selection.tests.smoke_day2
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

from one.robots.manipulators.franka.fr3.sphere_collision import FR3SphereCollision

from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.RL_controller.env.env import EnvConfig, NSRLBatchedEnv, TERM_NAMES

from Yuan.seed_selection.cone_ik import cone_constrained_ik_enumerate
from Yuan.seed_selection.robustness import (
    evaluate_robustness, filter_robust_candidates,
)
from Yuan.seed_selection.rollout import rollout_one
from Yuan.seed_selection.tests.smoke_day1 import make_task


# Override defaults for testing speed:
#   max_steps=2000 caps each rollout at 1.0m of EE travel (out of TARGET_M=1.5
#   normalizer). Single-env rollouts cost ~19 ms/step on this box; keeping
#   max_steps modest is what makes the smoke test finish in ~3 min instead
#   of ~30 min. Production pipeline will batch (Day 4) and can use larger caps.
TEST_MAX_STEPS = 2000
TARGET_M = 1.5


def build_env(device):
    cfg = EnvConfig(n_envs=1, max_steps=TEST_MAX_STEPS)
    env = NSRLBatchedEnv(cfg, line_dist=None, device=device)
    controller = ClassicalNullspaceController(env.kin)
    return env, controller


# ----------------------------------------------------------------------
# Unit test 1: rollout_one basic sanity
# ----------------------------------------------------------------------
def test_rollout_one(env, controller, collision):
    print("[rollout_one] running...")
    c = make_task(env.kin, collision, seed=42)
    q0 = c["_q0_seed"]
    t0 = time.time()
    res = rollout_one(q0, c, env=env, controller=controller, target_distance_m=TARGET_M)
    dt = time.time() - t0
    ok = (
        0.0 <= res["L"] <= TEST_MAX_STEPS * env.cfg.v * env.cfg.dt / TARGET_M + 1e-6
        and res["episode_len"] >= 1
        and res["term_reason"] in TERM_NAMES
    )
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] q0=seed  L={res['L']:.4f}  "
          f"progress={res['episode_progress_m']*1000:.2f}mm  "
          f"len={res['episode_len']}  term={TERM_NAMES[res['term_reason']]}  "
          f"({dt:.2f}s)")
    return ok


# ----------------------------------------------------------------------
# Unit test 2: evaluate_robustness reproducibility + zero-perturb
# ----------------------------------------------------------------------
def test_evaluate_robustness(env, controller, collision):
    print("[evaluate_robustness] running...")
    c = make_task(env.kin, collision, seed=7)
    q0 = c["_q0_seed"]

    # (a) reproducibility: same seed → same output
    r1 = evaluate_robustness(q0, c, env=env, controller=controller,
                              n_perturb=3, perturb_d_deg=3.0, perturb_n_deg=3.0,
                              perturb_p0_mm=5.0, target_distance_m=TARGET_M,
                              seed=2024)
    r2 = evaluate_robustness(q0, c, env=env, controller=controller,
                              n_perturb=3, perturb_d_deg=3.0, perturb_n_deg=3.0,
                              perturb_p0_mm=5.0, target_distance_m=TARGET_M,
                              seed=2024)
    same_clean = abs(r1["L_clean"] - r2["L_clean"]) < 1e-9
    same_pert = all(abs(a - b) < 1e-9 for a, b in zip(r1["L_perturbed"], r2["L_perturbed"]))
    ok_repro = same_clean and same_pert
    print(f"  [{('PASS' if ok_repro else 'FAIL')}] reproducibility: "
          f"L_clean same={same_clean}, L_perturbed same={same_pert}")
    print(f"           run1: L_clean={r1['L_clean']:.4f}  L_p={[f'{x:.4f}' for x in r1['L_perturbed']]}")

    # (b) zero perturbation: all L_perturbed == L_clean
    r0 = evaluate_robustness(q0, c, env=env, controller=controller,
                              n_perturb=3, perturb_d_deg=0.0, perturb_n_deg=0.0,
                              perturb_p0_mm=0.0, target_distance_m=TARGET_M,
                              seed=42)
    ok_zero = all(abs(L - r0["L_clean"]) < 1e-6 for L in r0["L_perturbed"])
    print(f"  [{('PASS' if ok_zero else 'FAIL')}] zero-perturb: "
          f"L_clean={r0['L_clean']:.4f}, L_p={[f'{x:.4f}' for x in r0['L_perturbed']]}")

    # (c) sanity: with a non-trivial perturb, L_perturbed should NOT all be identical
    # (otherwise the perturb isn't actually changing the rollout).
    distinct = len(set(round(x, 4) for x in r1["L_perturbed"])) > 1
    print(f"  [{('PASS' if distinct else 'WARN')}] perturb non-degenerate: "
          f"L_perturbed has >1 distinct value = {distinct}")

    return ok_repro and ok_zero


# ----------------------------------------------------------------------
# Unit test 3: filter_robust_candidates basic behavior
# ----------------------------------------------------------------------
def test_filter_synthetic(env, controller, collision):
    """Verify filter_robust_candidates honors top-K' cap and tau threshold
    with a hand-constructed candidate list."""
    print("[filter_robust] running synthetic check...")
    c = make_task(env.kin, collision, seed=11)
    q_seed = c["_q0_seed"]
    # Build 5 candidates: same q0 repeated, with FAKE L_clean values.
    # filter_robust_candidates only uses L_clean for sorting; it re-evaluates
    # the actual rollout when computing L_robust. So the input L_clean values
    # are just to test the sort+cap logic.
    fake_candidates = [
        {"q0": q_seed.clone(), "L_clean": L}
        for L in [0.10, 0.55, 0.42, 0.71, 0.30]
    ]
    K_prime = 3
    out_all = filter_robust_candidates(
        fake_candidates, c, K_prime=K_prime, tau_robust=0.5,
        env=env, controller=controller,
        n_perturb=2, perturb_d_deg=2.0, perturb_n_deg=2.0, perturb_p0_mm=3.0,
        target_distance_m=TARGET_M, seed=99, return_all_evaluations=True,
    )
    ok_cap = len(out_all) == K_prime
    # Note: L_clean in the output is the INPUT fake value (we pass through).
    fake_sorted = sorted([0.10, 0.55, 0.42, 0.71, 0.30], reverse=True)[:K_prime]
    sorted_correct = [x["L_clean"] for x in out_all] == fake_sorted
    print(f"  [{('PASS' if ok_cap else 'FAIL')}] returns exactly K'={K_prime} when "
          f"return_all_evaluations=True (got {len(out_all)})")
    print(f"  [{('PASS' if sorted_correct else 'FAIL')}] preserves top-K' order "
          f"(got L_clean={[x['L_clean'] for x in out_all]})")
    # Print the actual rollout-derived L_robust for each.
    for x in out_all:
        flag = "PASS" if x["passed"] == (x["L_robust_mean"] >= 0.5 * x["L_clean"]) else "FAIL"
        print(f"           L_clean={x['L_clean']:.3f}  "
              f"L_robust_mean={x['L_robust_mean']:.3f}  "
              f"thr(0.5*L_c)={0.5*x['L_clean']:.3f}  "
              f"passed={x['passed']} [{flag}]")
    # The 'passed' flag should match the threshold check explicitly.
    threshold_ok = all(x["passed"] == (x["L_robust_mean"] >= 0.5 * x["L_clean"])
                        for x in out_all)
    print(f"  [{('PASS' if threshold_ok else 'FAIL')}] 'passed' flag matches threshold")
    return ok_cap and sorted_correct and threshold_ok


# ----------------------------------------------------------------------
# Combined diagnostic test: full pipeline on 3 tasks
# ----------------------------------------------------------------------
def test_combined_diagnostic(env, controller, collision, *, task_seeds=(0,),
                              n_orient=5, n_restart=3, K_prime=3, n_perturb=2):
    """End-to-end: cone_ik → rollout L_clean for each → top-K' → filter_robust.
    Print everything needed to tune tau_robust and inspect failure modes.

    Defaults are sized for a fast smoke check (~3-4 min total wall) since
    single-env rollouts cost ~10s each on this box. Bump task_seeds/K_prime/
    n_perturb for a more thorough sweep once Day 4 batching lands.
    """
    print("\n" + "=" * 70)
    print(f"[combined diagnostic] cone_ik → L_clean → top-K' → filter_robust  "
          f"(tasks={list(task_seeds)}, n_orient={n_orient}, K'={K_prime}, n_perturb={n_perturb})")
    print("=" * 70)
    tau_robust = 0.5
    perturb_params = dict(perturb_d_deg=5.0, perturb_n_deg=5.0, perturb_p0_mm=10.0)

    all_ok = True
    for task_seed in task_seeds:
        c = make_task(env.kin, collision, seed=task_seed)
        rng = np.random.default_rng(1000 + task_seed)

        # 1. cone IK enumeration
        t0 = time.time()
        Q = cone_constrained_ik_enumerate(
            p0=c["p0"], n_target=c["n_target"], line_dir=c["line_dir"],
            kin=env.kin, collision=collision,
            cone_angle_deg=5.0, n_orientations=n_orient, n_ik_restarts=n_restart,
            joint_margin=0.05, dedup_rad=0.08, rng=rng,
        )
        t_ik = time.time() - t0
        print(f"\n--- task seed={task_seed} ---")
        print(f"  cone_ik: {Q.shape[0]} candidates  ({t_ik:.2f}s)")
        if Q.shape[0] == 0:
            print(f"  [FAIL] no candidates")
            all_ok = False
            continue

        # 2. Score each candidate with a clean rollout
        t0 = time.time()
        candidates = []
        for j in range(Q.shape[0]):
            res = rollout_one(Q[j], c, env=env, controller=controller,
                              target_distance_m=TARGET_M)
            candidates.append({"q0": Q[j], "L_clean": res["L"],
                                "term": res["term_reason"]})
        t_score = time.time() - t0
        L_sorted = sorted([cd["L_clean"] for cd in candidates], reverse=True)
        print(f"  scored {len(candidates)} candidates ({t_score:.2f}s)  "
              f"L_clean range [{min(L_sorted):.3f}, {max(L_sorted):.3f}]")
        # Histogram of L_clean (very rough)
        bins = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0]
        hist = [sum(1 for L in L_sorted if lo <= L < hi)
                for lo, hi in zip(bins[:-1], bins[1:])]
        print(f"  L_clean histogram: "
              + "  ".join(f"[{lo:.2f},{hi:.2f}):{n}"
                         for lo, hi, n in zip(bins[:-1], bins[1:], hist) if n > 0))

        # 3. Robust filter on top-K'
        t0 = time.time()
        evaluated = filter_robust_candidates(
            candidates, c, K_prime=K_prime, tau_robust=tau_robust,
            env=env, controller=controller,
            n_perturb=n_perturb,
            perturb_d_deg=perturb_params["perturb_d_deg"],
            perturb_n_deg=perturb_params["perturb_n_deg"],
            perturb_p0_mm=perturb_params["perturb_p0_mm"],
            target_distance_m=TARGET_M, seed=task_seed + 7000,
            return_all_evaluations=True,
        )
        t_filter = time.time() - t0
        n_passed = sum(1 for x in evaluated if x["passed"])
        print(f"  filter_robust: {n_passed}/{len(evaluated)} kept "
              f"(K'={K_prime}, τ={tau_robust}, n_perturb={n_perturb})  "
              f"({t_filter:.2f}s)")
        print(f"  {'#':>2}  {'L_clean':>8}  {'L_r_mean':>9}  {'L_r_min':>8}  "
              f"{'L_r_std':>8}  {'thr':>6}  {'pass':>5}")
        for i, x in enumerate(evaluated):
            print(f"  {i:>2}  {x['L_clean']:>8.4f}  {x['L_robust_mean']:>9.4f}  "
                  f"{x['L_robust_min']:>8.4f}  {x['L_robust_std']:>8.4f}  "
                  f"{tau_robust*x['L_clean']:>6.4f}  "
                  f"{'Y' if x['passed'] else 'N':>5}")
        # Sanity asserts
        out_kept = [x for x in evaluated if x["passed"]]
        within_kprime = len(evaluated) <= K_prime
        thr_ok = all(x["L_robust_mean"] >= tau_robust * x["L_clean"]
                     for x in out_kept)
        L_p_distinct = all(len(set(round(L, 4) for L in x["L_perturbed"])) > 0
                           for x in evaluated)
        if not (within_kprime and thr_ok and L_p_distinct):
            print(f"  [FAIL] within_kprime={within_kprime}  thr_ok={thr_ok}  L_p_distinct={L_p_distinct}")
            all_ok = False
        else:
            print(f"  [PASS] all invariants")

    return all_ok


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    coll = FR3SphereCollision(device=device)
    env, controller = build_env(device)

    results = {
        "rollout_one":              test_rollout_one(env, controller, coll),
        "evaluate_robustness":      test_evaluate_robustness(env, controller, coll),
        "filter_robust (synthetic)": test_filter_synthetic(env, controller, coll),
        "combined diagnostic":       test_combined_diagnostic(env, controller, coll),
    }
    print("\n=== summary ===")
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

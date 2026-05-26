"""Day 4 smoke test: batched_rollout_many + end-to-end label builder +
build_dataset round-trip.

Run:
    python -m Yuan.seed_selection.tests.smoke_day4

What it verifies:
    1. Parity: batched_rollout_many gives the same L (within float noise) as
       rollout_one for the same (q, c) pair.
    2. End-to-end: build_labels_for_one_task on 2 tasks completes and returns
       a valid label dict (status, n_labels, L_seed, threshold check, etc.).
    3. Round-trip: build_dataset on 3 tasks → NPZ on disk → re-run hits cache.

Expected wall time at n_envs=64: ~2-3 min total.
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from one.robots.manipulators.franka.fr3.sphere_collision import FR3SphereCollision

from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.RL_controller.env.env import EnvConfig, NSRLBatchedEnv

from Yuan.seed_selection.batched_rollout import batched_rollout_many
from Yuan.seed_selection.dataset_builder import build_dataset
from Yuan.seed_selection.label_builder import (
    STATUS_EDGE, STATUS_INFEASIBLE, STATUS_KEPT, STATUS_LOW_QUALITY,
    build_labels_for_one_task,
)
from Yuan.seed_selection.rollout import rollout_one
from Yuan.seed_selection.tests.smoke_day1 import make_task


TEST_MAX_STEPS = 2000
TARGET_M = 1.5
N_ENVS = 64

VALID_STATUS = {STATUS_KEPT, STATUS_EDGE, STATUS_INFEASIBLE, STATUS_LOW_QUALITY}


def build_env(device, n_envs):
    cfg = EnvConfig(n_envs=n_envs, max_steps=TEST_MAX_STEPS)
    env = NSRLBatchedEnv(cfg, line_dist=None, device=device)
    controller = ClassicalNullspaceController(env.kin)
    return env, controller


# ----------------------------------------------------------------------
# Test 1: batched vs single-env parity
# ----------------------------------------------------------------------
def test_parity(env_batched, controller_b, env_single, controller_s, collision):
    print("\n--- test 1: parity batched vs single-env ---", flush=True)
    c = make_task(env_batched.kin, collision, seed=42)
    # Build 3 candidate q's: the seed + 2 small perturbations of it.
    gen = torch.Generator(device=env_batched.device).manual_seed(7)
    q0 = c["_q0_seed"]
    qs = torch.stack([
        q0,
        q0 + 0.05 * torch.randn(7, device=env_batched.device, dtype=env_batched.kin.dtype, generator=gen),
        q0 + 0.10 * torch.randn(7, device=env_batched.device, dtype=env_batched.kin.dtype, generator=gen),
    ], dim=0)

    # Single-env: 3 calls
    t0 = time.time()
    Ls_single = []
    for j in range(3):
        res = rollout_one(qs[j], c, env=env_single, controller=controller_s,
                          target_distance_m=TARGET_M)
        Ls_single.append(res["L"])
    t_single = time.time() - t0

    # Batched: 1 call
    t0 = time.time()
    res_b = batched_rollout_many(qs, [c, c, c], env=env_batched, controller=controller_b,
                                  target_distance_m=TARGET_M)
    t_batched = time.time() - t0
    Ls_batched = list(res_b["L"])

    print(f"  single-env (3 calls): {t_single:.2f}s  L={[f'{L:.4f}' for L in Ls_single]}")
    print(f"  batched   (1 call):   {t_batched:.2f}s  L={[f'{L:.4f}' for L in Ls_batched]}")
    diffs = [abs(a - b) for a, b in zip(Ls_single, Ls_batched)]
    max_diff = max(diffs)
    speedup = t_single / max(t_batched, 1e-6)
    ok = max_diff < 5e-3   # rollout determinism is float32-iterated; small drift OK
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] max |L_single - L_batched| = {max_diff:.5f} (< 5e-3)  "
          f"speedup = {speedup:.1f}x")
    return ok


# ----------------------------------------------------------------------
# Test 2: end-to-end label builder
# ----------------------------------------------------------------------
def test_labels(env, controller, collision, *, task_seed=3):
    print(f"\n--- test 2: build_labels_for_one_task (task seed={task_seed}) ---", flush=True)
    c = make_task(env.kin, collision, seed=task_seed)
    q0_seed = c["_q0_seed"]
    t0 = time.time()
    out = build_labels_for_one_task(
        c, q0_seed,
        kin=env.kin, collision=collision,
        env=env, controller=controller,
        cone_angle_deg=5.0, n_orientations=5, n_ik_restarts=3,
        sample_per_branch=3,
        k=3, K_prime=4, tau_robust=0.5,
        n_perturb=3,
        perturb_d_deg=5.0, perturb_n_deg=5.0, perturb_p0_mm=10.0,
        L_min_abs=0.05, L_min_acceptable=0.30,
        target_distance_m=TARGET_M, seed=task_seed + 500,
        verbose=True,
    )
    dt = time.time() - t0
    print(f"  build_labels: {dt:.1f}s")
    print(f"  status            : {out['status']}")
    print(f"  n_labels          : {out['n_labels']}")
    print(f"  L_seed            : {out['L_seed']:.4f}")
    print(f"  labels_L_clean    : {[f'{x:.4f}' for x in out['labels_L_clean']]}")
    print(f"  labels_L_robust_mean: {[f'{x:.4f}' for x in out['labels_L_robust_mean']]}")
    print(f"  diagnostics       : {out['diagnostics']}")

    checks = {
        "status valid":           out["status"] in VALID_STATUS,
        "n_labels >= 1":           out["n_labels"] >= 1,
        "labels_q0 shape":         out["labels_q0"].shape == (out["n_labels"], 7),
        "L_seed finite":           np.isfinite(out["L_seed"]),
        "lists agree on length":   (len(out["labels_L_clean"]) == out["n_labels"]
                                     and len(out["labels_L_robust_mean"]) == out["n_labels"]
                                     and len(out["labels_L_robust_min"]) == out["n_labels"]),
    }
    if not out["fallback_used"]:
        tau = 0.5
        checks["robust threshold"] = all(
            rm >= tau * lc - 1e-9
            for lc, rm in zip(out["labels_L_clean"], out["labels_L_robust_mean"])
        )
    print("  invariants:")
    all_ok = True
    for name, ok in checks.items():
        print(f"    [{'PASS' if ok else 'FAIL'}]  {name}")
        all_ok = all_ok and ok
    return all_ok


# ----------------------------------------------------------------------
# Test 3: build_dataset round-trip
# ----------------------------------------------------------------------
def test_dataset(env, controller, collision):
    print("\n--- test 3: build_dataset round-trip ---", flush=True)
    out_dir = Path("Yuan/seed_selection/tests/_tmp_dataset")
    if out_dir.exists():
        shutil.rmtree(out_dir)

    # Tiny dataset: 3 c's.
    cs, q0_seeds = [], []
    for s in range(3):
        c = make_task(env.kin, collision, seed=100 + s)
        cs.append({"p0": c["p0"], "line_dir": c["line_dir"], "n_target": c["n_target"]})
        q0_seeds.append(c["_q0_seed"])

    label_kwargs = dict(
        cone_angle_deg=5.0, n_orientations=4, n_ik_restarts=2,
        sample_per_branch=2,
        k=2, K_prime=3, tau_robust=0.5, n_perturb=2,
        perturb_d_deg=5.0, perturb_n_deg=5.0, perturb_p0_mm=10.0,
        L_min_abs=0.05, L_min_acceptable=0.30,
        target_distance_m=TARGET_M,
    )
    hp = dict(
        label_kwargs=label_kwargs,
        env_max_steps=TEST_MAX_STEPS,
        target_distance_m=TARGET_M,
    )

    t0 = time.time()
    out_path = build_dataset(
        cs, q0_seeds,
        kin=env.kin, collision=collision,
        env=env, controller=controller,
        out_dir=out_dir,
        cache_name="smoke",
        hyperparams=hp,
        label_kwargs=label_kwargs,
        checkpoint_interval=2,
        verbose=True,
    )
    t_first = time.time() - t0
    print(f"  first build: {t_first:.1f}s  → {out_path}")

    # Re-run: should hit the cache.
    t0 = time.time()
    out_path2 = build_dataset(
        cs, q0_seeds,
        kin=env.kin, collision=collision,
        env=env, controller=controller,
        out_dir=out_dir,
        cache_name="smoke",
        hyperparams=hp,
        label_kwargs=label_kwargs,
        checkpoint_interval=2,
        verbose=True,
    )
    t_cached = time.time() - t0
    print(f"  cached:      {t_cached:.3f}s  (should be < 0.1s)")
    cache_hit_ok = (out_path2 == out_path) and (t_cached < 0.5)

    # Inspect the NPZ.
    z = np.load(out_path)
    print(f"  NPZ keys: {sorted(z.keys())}")
    shape_ok = (
        z["cs_p0"].shape == (3, 3)
        and z["q0_seeds"].shape == (3, 7)
        and z["labels_q0"].shape == (3, label_kwargs["k"], 7)
        and z["labels_L_clean"].shape == (3, label_kwargs["k"])
        and z["L_seed"].shape == (3,)
        and z["n_labels"].shape == (3,)
        and z["status"].shape == (3,)
        and z["fallback_used"].shape == (3,)
    )
    print(f"  shapes:           {dict((k, z[k].shape) for k in ['cs_p0','labels_q0','labels_L_clean','L_seed','n_labels'])}")
    print(f"  status per task:  {[str(s) for s in z['status']]}")
    print(f"  n_labels per task: {z['n_labels'].tolist()}")
    print(f"  L_seed per task:  {z['L_seed'].tolist()}")
    n_labels_ok = all(1 <= int(n) <= label_kwargs["k"] for n in z["n_labels"])
    status_ok = all(str(s) in VALID_STATUS for s in z["status"])

    # Clean up
    shutil.rmtree(out_dir)

    print(f"  [{'PASS' if cache_hit_ok else 'FAIL'}] cache hit on second call")
    print(f"  [{'PASS' if shape_ok else 'FAIL'}] NPZ shapes correct")
    print(f"  [{'PASS' if n_labels_ok else 'FAIL'}] n_labels in [1, k]")
    print(f"  [{'PASS' if status_ok else 'FAIL'}] status values valid")
    return cache_hit_ok and shape_ok and n_labels_ok and status_ok


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)
    coll = FR3SphereCollision(device=device)
    env_b, ctrl_b = build_env(device, n_envs=N_ENVS)
    env_s, ctrl_s = build_env(device, n_envs=1)

    results = {
        "parity":          test_parity(env_b, ctrl_b, env_s, ctrl_s, coll),
        "label builder":    test_labels(env_b, ctrl_b, coll, task_seed=3),
        "dataset builder":  test_dataset(env_b, ctrl_b, coll),
    }
    print("\n=== summary ===")
    for k, v in results.items():
        print(f"  [{'PASS' if v else 'FAIL'}]  {k}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

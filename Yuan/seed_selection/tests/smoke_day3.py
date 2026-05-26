"""Day 3 smoke test for Module 7 (build_labels_for_one_task).

This is a slow test: ~3 min per task on single-env rollouts (Day 4 batching
will fix that). It runs on a single fixed-seed task by default; bump
``N_TASKS`` for a wider sweep once batching lands.

Run:
    python -m Yuan.seed_selection.tests.smoke_day3
"""
from __future__ import annotations

import sys
import time

import torch

from one.robots.manipulators.franka.fr3.sphere_collision import FR3SphereCollision

from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.RL_controller.env.env import EnvConfig, NSRLBatchedEnv

from Yuan.seed_selection.label_builder import (
    STATUS_EDGE, STATUS_INFEASIBLE, STATUS_KEPT, STATUS_LOW_QUALITY,
    build_labels_for_one_task,
)
from Yuan.seed_selection.tests.smoke_day1 import make_task


TEST_MAX_STEPS = 2000        # caps each rollout at ~1.0 m EE travel
TARGET_M = 1.5
N_TASKS = 1                  # bump to 3 for wider coverage (~10 min total)


VALID_STATUS = {STATUS_KEPT, STATUS_EDGE, STATUS_INFEASIBLE, STATUS_LOW_QUALITY}


def build_env(device):
    cfg = EnvConfig(n_envs=1, max_steps=TEST_MAX_STEPS)
    env = NSRLBatchedEnv(cfg, line_dist=None, device=device)
    controller = ClassicalNullspaceController(env.kin)
    return env, controller


def _check_one(task_seed: int, env, controller, collision) -> bool:
    print(f"\n--- task seed={task_seed} ---", flush=True)
    c = make_task(env.kin, collision, seed=task_seed)
    q0_seed = c["_q0_seed"]
    t0 = time.time()
    out = build_labels_for_one_task(
        c, q0_seed,
        kin=env.kin, collision=collision,
        env=env, controller=controller,
        # cone IK (small for speed)
        cone_angle_deg=5.0, n_orientations=3, n_ik_restarts=2,
        # SMM walk (small for speed)
        sample_per_branch=2,
        # robust filter (small for speed)
        k=2, K_prime=3, tau_robust=0.5,
        n_perturb=2,
        perturb_d_deg=5.0, perturb_n_deg=5.0, perturb_p0_mm=10.0,
        # quality
        L_min_abs=0.05, L_min_acceptable=0.30,
        target_distance_m=TARGET_M, seed=task_seed + 500,
        verbose=True,
    )
    dt = time.time() - t0
    diag = out["diagnostics"]
    print(f"\n  build_labels: {dt:.1f}s")
    print(f"  status         : {out['status']}")
    print(f"  fallback_used  : {out['fallback_used']}")
    print(f"  n_labels       : {out['n_labels']}")
    print(f"  L_seed         : {out['L_seed']:.4f}")
    print(f"  labels_L_clean : {[f'{x:.4f}' for x in out['labels_L_clean']]}")
    print(f"  labels_L_robust_mean: {[f'{x:.4f}' for x in out['labels_L_robust_mean']]}")
    print(f"  diagnostics    :")
    for k, v in diag.items():
        print(f"    {k}: {v}")

    # Invariants
    checks = {
        "status valid":            out["status"] in VALID_STATUS,
        "n_labels >= 1":            out["n_labels"] >= 1,
        "labels_q0 shape OK":       out["labels_q0"].shape == (out["n_labels"], 7),
        "L_seed is finite":         out["L_seed"] >= 0.0,
        "labels_L_clean length":    len(out["labels_L_clean"]) == out["n_labels"],
        "labels_L_robust length":   len(out["labels_L_robust_mean"]) == out["n_labels"],
    }
    # If not fallback, robust threshold must be met by every label.
    if not out["fallback_used"]:
        tau = 0.5  # the value we passed
        checks["robust threshold OK"] = all(
            rm >= tau * lc - 1e-9
            for lc, rm in zip(out["labels_L_clean"], out["labels_L_robust_mean"])
        )

    all_ok = True
    print("\n  invariants:")
    for name, ok in checks.items():
        flag = "PASS" if ok else "FAIL"
        print(f"    [{flag}]  {name}")
        if not ok:
            all_ok = False
    return all_ok


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}", flush=True)
    coll = FR3SphereCollision(device=device)
    env, controller = build_env(device)

    all_ok = True
    for seed in range(N_TASKS):
        ok = _check_one(seed, env, controller, coll)
        all_ok = all_ok and ok

    print("\n=== summary ===")
    print("PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

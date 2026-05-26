"""Quick supplement to analyze_pilot.py: capture per-c SMM topology stats
(n_cone_ik, n_after_project_filter, n_branches, branch_lengths) which
``dataset_builder`` does NOT persist to the NPZ.

Re-runs only steps 1-3 of ``build_labels_for_one_task`` (cone IK + Newton
refine + SMM walk + branch grouping) — no rollouts. ~50s on the 100-c pilot.

Run:
    python -m Yuan.seed_selection.tests.analyze_topology
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from one.robots.manipulators.franka.fr3.sphere_collision import FR3SphereCollision

from Yuan.RL_controller.env.env import EnvConfig, NSRLBatchedEnv
from Yuan.RL_controller.env.line_distribution import LineDistribution

from Yuan.flow_connectivity.intro_motivation.v18_smm_core import (
    DEFAULT_H, DEDUP_RAD, JOINT_MARGIN,
    enumerate_branches, project_and_filter,
)
from Yuan.seed_selection.cone_ik import cone_constrained_ik_enumerate
from Yuan.seed_selection.label_builder import _build_R_target_strict


N_PILOT = 100
OUT_DIR = Path("Yuan/seed_selection/runs/pilot_day5")
CONFIG_PATH = Path("Yuan/RL_controller/config.yaml")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(CONFIG_PATH, "r") as f:
        cfg_yaml = yaml.safe_load(f)
    line_cfg = cfg_yaml["line_distribution"]
    env_cfg = EnvConfig(**cfg_yaml["env"])
    threshold_m = float(line_cfg["feasibility_threshold_m"])
    env = NSRLBatchedEnv(env_cfg, line_dist=None, device=device)
    collision = FR3SphereCollision(device=device)
    line_dist = LineDistribution.load_or_build(
        kin=env.kin, collision=collision,
        n_pool=line_cfg["n_pool"],
        n_target_noise_deg=line_cfg["n_target_noise_deg"],
        seed=line_cfg["train_seed"],
        env_cfg=env_cfg,
        feasibility_threshold_m=threshold_m,
    )
    valid_idx = torch.nonzero(line_dist.valid_mask, as_tuple=False).squeeze(-1)
    pick = valid_idx[:N_PILOT]
    qs = line_dist.q_pool[pick]
    line_dirs = line_dist.line_dir_pool[pick]
    n_targets = line_dist.n_target_pool[pick]
    p_tcps, _, _, _ = env.kin.tcp_fk_jac(qs)

    rows = []
    t0 = time.time()
    for i in range(N_PILOT):
        c = {"p0": p_tcps[i], "line_dir": line_dirs[i], "n_target": n_targets[i]}
        q0_seed = qs[i]
        rng = np.random.default_rng(i)
        # Step 1: cone IK
        Q_ik = cone_constrained_ik_enumerate(
            p0=c["p0"], n_target=c["n_target"], line_dir=c["line_dir"],
            kin=env.kin, collision=collision,
            cone_angle_deg=5.0, n_orientations=10, n_ik_restarts=5,
            joint_margin=JOINT_MARGIN, dedup_rad=0.08, rng=rng,
        )
        # Step 2: prepend q0_seed, refine to strict 6-DOF
        p0_np = c["p0"].detach().cpu().numpy().astype(np.float32)
        n_np = c["n_target"].detach().cpu().numpy().astype(np.float32)
        d_np = c["line_dir"].detach().cpu().numpy().astype(np.float32)
        R_tgt_np = _build_R_target_strict(n_np, d_np)
        q0_seed_np = q0_seed.detach().cpu().numpy().astype(np.float32)
        Q_pool = np.concatenate(
            [q0_seed_np[None], Q_ik.detach().cpu().numpy().astype(np.float32)], axis=0)
        lo_np = env.kin.lmt_lo.detach().cpu().numpy()
        hi_np = env.kin.lmt_up.detach().cpu().numpy()
        Q_clean = project_and_filter(
            env.kin, Q_pool, p0_np, R_tgt_np, lo_np, hi_np,
            joint_margin=JOINT_MARGIN, dedup_rad=DEDUP_RAD, verbose=False)
        # Step 3: walk
        if Q_clean.shape[0] == 0:
            branches = []
        else:
            branches, _ = enumerate_branches(
                env.kin, Q_clean, p0_np, R_tgt_np, h=DEFAULT_H)
        rows.append({
            "i": i,
            "n_cone_ik": int(Q_ik.shape[0]),
            "n_after_project_filter": int(Q_clean.shape[0]),
            "n_branches": len(branches),
            "branch_lengths": [int(b["traj"].shape[0]) for b in branches],
            "total_walk_points": int(sum(b["traj"].shape[0] for b in branches)),
        })
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{N_PILOT}  ({time.time() - t0:.1f}s)", flush=True)
    elapsed = time.time() - t0
    print(f"done: {N_PILOT} tasks in {elapsed:.1f}s", flush=True)

    # Aggregate.
    n_cone_ik = np.array([r["n_cone_ik"] for r in rows])
    n_clean = np.array([r["n_after_project_filter"] for r in rows])
    n_branches = np.array([r["n_branches"] for r in rows])
    total_walk = np.array([r["total_walk_points"] for r in rows])

    print("\n=== topology summary ===")
    for name, arr in [("n_cone_ik", n_cone_ik), ("n_after_project_filter", n_clean),
                       ("n_branches", n_branches), ("total_walk_points", total_walk)]:
        print(f"  {name:25s}  mean={arr.mean():.2f}  median={np.median(arr):.1f}  "
              f"p25={np.percentile(arr, 25):.1f}  p75={np.percentile(arr, 75):.1f}  "
              f"min={arr.min()}  max={arr.max()}")

    # Histogram of n_branches.
    print("\n  n_branches distribution:")
    for n in range(int(n_branches.max()) + 1):
        c = int((n_branches == n).sum())
        if c > 0:
            print(f"    {n} branches: {c} tasks ({100*c/N_PILOT:.0f}%)")

    # Save.
    out_path = OUT_DIR / "topology_stats.json"
    out_path.write_text(json.dumps({
        "per_task": rows,
        "summary": {
            "n_cone_ik_mean":       float(n_cone_ik.mean()),
            "n_cone_ik_median":     float(np.median(n_cone_ik)),
            "n_after_project_mean": float(n_clean.mean()),
            "n_branches_mean":      float(n_branches.mean()),
            "n_branches_median":    float(np.median(n_branches)),
            "n_branches_hist":      {int(n): int((n_branches == n).sum()) for n in range(int(n_branches.max()) + 1)},
            "total_walk_mean":      float(total_walk.mean()),
        },
    }, indent=2))
    print(f"\n  wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

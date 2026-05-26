"""Day 5 pilot: 100-c run with production config + default hyperparameters.

Reuses the cached LineDistribution pool (train_seed=0) so the (q0, c) pairs
are exactly the first 100 the RL training script sees, and the FR3 env
config matches `Yuan/RL_controller/config.yaml` so the rollout dynamics
match what eval uses.

Run:
    python -m Yuan.seed_selection.tests.pilot_day5

Outputs under Yuan/seed_selection/runs/pilot_day5/:
    pilot.npz           — labels per c, fixed-shape padded
    pilot.meta.json     — hyperparams, status counts, errors, wall time
    pilot.partial-*.npz — incremental checkpoints (auto-cleaned on success)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import yaml

from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.RL_controller.env.env import EnvConfig, NSRLBatchedEnv
from Yuan.RL_controller.env.line_distribution import LineDistribution

from Yuan.seed_selection.dataset_builder import build_dataset


N_PILOT = 1000                    # Day 6: scale up to 1000-c
N_ENVS_ROLLOUT = 64               # batched rollout chunk size
OUT_DIR = Path("Yuan/seed_selection/runs/pilot_day5")
CONFIG_PATH = Path("Yuan/RL_controller/config.yaml")


# Module 7 hyperparameters per user's Day-5/6 final spec:
#   - REMOVED L_seed_min (pilot v2 showed it cut out high-improvement weak-seed tasks)
#   - tau_robust=0.0  (pilot 1/v2 showed perturb 2-3°/5-8mm doesn't differentiate;
#                       robust filter is currently a no-op for this controller)
#   - perturb_* kept at 3/3/8 as documentation; with tau=0 they're not exercised
LABEL_KWARGS = dict(
    cone_angle_deg=5.0,
    n_orientations=10,
    n_ik_restarts=5,
    sample_per_branch=5,
    k=3,
    K_prime=6,
    tau_robust=0.0,
    n_perturb=4,
    perturb_d_deg=3.0,
    perturb_n_deg=3.0,
    perturb_p0_mm=8.0,
    L_min_abs=0.10,
    L_min_acceptable=0.20,
    target_distance_m=1.5,
    ik_dedup_rad=0.08,
    smm_dedup_rad=0.08,
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pilot] device={device}", flush=True)
    with open(CONFIG_PATH, "r") as f:
        cfg_yaml = yaml.safe_load(f)
    line_cfg = cfg_yaml["line_distribution"]
    train_env_cfg = EnvConfig(**cfg_yaml["env"])
    threshold_m = (float(line_cfg["feasibility_threshold_m"])
                   if line_cfg.get("feasibility_filter", False) else None)

    # Rollout env: n_envs=N_ENVS_ROLLOUT (batched), everything else from config.
    rollout_env_cfg = EnvConfig(**{**cfg_yaml["env"], "n_envs": N_ENVS_ROLLOUT})
    print(f"[pilot] building env  n_envs={rollout_env_cfg.n_envs}  "
          f"dt={rollout_env_cfg.dt}  v={rollout_env_cfg.v}  "
          f"max_steps={rollout_env_cfg.max_steps}  tcp_offset={rollout_env_cfg.tcp_offset}",
          flush=True)
    env = NSRLBatchedEnv(rollout_env_cfg, line_dist=None, device=device)
    controller = ClassicalNullspaceController(env.kin)

    # Load the same LineDistribution pool that train.py uses (cached).
    print(f"[pilot] loading line distribution (train_seed={line_cfg['train_seed']}, "
          f"n_pool={line_cfg['n_pool']})", flush=True)
    line_dist = LineDistribution.load_or_build(
        kin=env.kin, collision=env.collision,
        n_pool=line_cfg["n_pool"],
        n_target_noise_deg=line_cfg["n_target_noise_deg"],
        seed=line_cfg["train_seed"],
        env_cfg=train_env_cfg,
        feasibility_threshold_m=threshold_m,
    )
    valid_idx = torch.nonzero(line_dist.valid_mask, as_tuple=False).squeeze(-1)
    print(f"[pilot] line_dist has {valid_idx.shape[0]} valid tasks; "
          f"taking first {N_PILOT}", flush=True)
    pick = valid_idx[:N_PILOT]
    qs = line_dist.q_pool[pick]
    line_dirs = line_dist.line_dir_pool[pick]
    n_targets = line_dist.n_target_pool[pick]
    p_tcps, _, _, _ = env.kin.tcp_fk_jac(qs)

    cs = [
        {"p0": p_tcps[i].clone(),
         "line_dir": line_dirs[i].clone(),
         "n_target": n_targets[i].clone()}
        for i in range(N_PILOT)
    ]
    q0_seeds = [qs[i].clone() for i in range(N_PILOT)]

    # All hyperparams that affect dataset content go into the cache key.
    hyperparams = {
        "n_pilot": N_PILOT,
        "label_kwargs": LABEL_KWARGS,
        "env": {k: getattr(rollout_env_cfg, k) for k in
                ("dt", "v", "a_max", "cone_deg", "max_steps", "tcp_offset",
                 "lambda_0", "sigma_thr")},
        "line_distribution": {
            "n_pool": line_cfg["n_pool"],
            "n_target_noise_deg": line_cfg["n_target_noise_deg"],
            "train_seed": line_cfg["train_seed"],
            "feasibility_threshold_m": threshold_m,
        },
    }

    t0 = time.time()
    out_path = build_dataset(
        cs, q0_seeds,
        kin=env.kin, collision=env.collision,
        env=env, controller=controller,
        out_dir=OUT_DIR,
        cache_name="pilot_1k",   # 1000-c run with L_seed_min removed + tau_robust=0
        hyperparams=hyperparams,
        label_kwargs=LABEL_KWARGS,
        checkpoint_interval=100,  # save partial every 100 c's → trip wire
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\n[pilot] DONE  {N_PILOT} tasks in {elapsed:.1f}s "
          f"({elapsed / N_PILOT:.1f}s/task)", flush=True)
    print(f"  out: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

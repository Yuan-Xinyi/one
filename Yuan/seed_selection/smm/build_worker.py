"""Per-process dataset-build worker.

Used both as a standalone speed test (small N, single process) and as the
subprocess entry point invoked by ``smm.build_parallel`` for each shard of a
multi-process build. ``--n-tasks``, ``--seed``, and ``--cache-name`` together
select which slice of the cached LineDistribution this worker handles, so
sibling workers never collide on disk.

Usage (standalone):
    time python -m Yuan.seed_selection.smm.build_worker --n-tasks 100 --seed 0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.RL_controller.env.env import EnvConfig, NSRLBatchedEnv
from Yuan.RL_controller.env.line_distribution import LineDistribution
from Yuan.seed_selection.smm.dataset_builder import build_dataset


_REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = _REPO_ROOT / "Yuan/seed_selection/runs/pilot_20k"
CONFIG_PATH = _REPO_ROOT / "Yuan/RL_controller/config.yaml"


LABEL_KWARGS = dict(
    cone_angle_deg=5.0,
    n_orientations=10,
    n_ik_restarts=5,
    sample_per_branch=5,
    k=3,
    K_prime=64,   # retain ~all rolled-out candidates+L (kept-task mean 23, max ~61);
                  # tau_robust=0 so this is pure storage, zero extra rollout compute.
                  # Feeds ranker (full good/bad spread) + diffusion (within-3% positives).
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n-tasks', type=int, default=100)
    p.add_argument('--seed', type=int, default=0,
                   help='offset into the cached line_distribution.valid_idx pool')
    p.add_argument('--n-envs-rollout', type=int, default=64)
    p.add_argument('--cache-name', default=None,
                   help='override default cache name "speedtest_n{N}_s{seed}"')
    p.add_argument('--checkpoint-interval', type=int, default=None,
                   help='save partial NPZ every N tasks (default: max(N//4, 25))')
    p.add_argument('--label-seed-base', type=int, default=None,
                   help='per-task numpy seed = label_seed_base + i. Default: --seed '
                        '(so disjoint chunks of line_dist get disjoint per-task seeds).')
    p.add_argument('--task-npz', default=None,
                   help='if set, source tasks from this NPZ (keys q0_native, cs_p0, '
                        'cs_line_dir, cs_n_target) sliced [seed:seed+n_tasks] instead '
                        'of the LineDistribution valid_idx pool.')
    p.add_argument('--out-dir', default=None,
                   help='override the default output directory (pilot_20k).')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[speedtest] device={device}  n_tasks={args.n_tasks}  seed={args.seed}", flush=True)

    with open(CONFIG_PATH, "r") as f:
        cfg_yaml = yaml.safe_load(f)
    line_cfg = cfg_yaml["line_distribution"]
    train_env_cfg = EnvConfig(**cfg_yaml["env"])
    threshold_m = (float(line_cfg["feasibility_threshold_m"])
                   if line_cfg.get("feasibility_filter", False) else None)

    rollout_env_cfg = EnvConfig(**{**cfg_yaml["env"], "n_envs": args.n_envs_rollout})
    env = NSRLBatchedEnv(rollout_env_cfg, line_dist=None, device=device)
    controller = ClassicalNullspaceController(env.kin)

    lo = args.seed
    hi = lo + args.n_tasks
    if args.task_npz is not None:
        # Source tasks from an explicit NPZ (e.g. pipeline_v2 train split),
        # sliced [lo:hi]. q0_native is the seed config; cs_p0 is its FK TCP.
        import numpy as np
        z = np.load(args.task_npz)
        n_avail = int(z['q0_native'].shape[0])
        if hi > n_avail:
            raise SystemExit(f"--seed + --n-tasks = {hi} exceeds {n_avail} tasks in {args.task_npz}")
        dt = env.kin.dtype
        qs = torch.as_tensor(z['q0_native'][lo:hi], device=device, dtype=dt)
        line_dirs = torch.as_tensor(z['cs_line_dir'][lo:hi], device=device, dtype=dt)
        n_targets = torch.as_tensor(z['cs_n_target'][lo:hi], device=device, dtype=dt)
        p_tcps = torch.as_tensor(z['cs_p0'][lo:hi], device=device, dtype=dt)
    else:
        line_dist = LineDistribution.load_or_build(
            kin=env.kin, collision=env.collision,
            n_pool=line_cfg["n_pool"],
            n_target_noise_deg=line_cfg["n_target_noise_deg"],
            seed=line_cfg["train_seed"],
            env_cfg=train_env_cfg,
            feasibility_threshold_m=threshold_m,
        )
        valid_idx = torch.nonzero(line_dist.valid_mask, as_tuple=False).squeeze(-1)
        n_valid = int(valid_idx.shape[0])
        if hi > n_valid:
            raise SystemExit(f"--seed + --n-tasks = {hi} exceeds {n_valid} valid tasks")
        pick = valid_idx[lo:hi]
        qs = line_dist.q_pool[pick]
        line_dirs = line_dist.line_dir_pool[pick]
        n_targets = line_dist.n_target_pool[pick]
        p_tcps, _, _, _ = env.kin.tcp_fk_jac(qs)

    cs = [
        {"p0": p_tcps[i].clone(),
         "line_dir": line_dirs[i].clone(),
         "n_target": n_targets[i].clone()}
        for i in range(args.n_tasks)
    ]
    q0_seeds = [qs[i].clone() for i in range(args.n_tasks)]

    hyperparams = {
        "n_speedtest": args.n_tasks,
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

    cache_name = args.cache_name or f"speedtest_n{args.n_tasks}_s{args.seed}"
    ckpt_int = args.checkpoint_interval if args.checkpoint_interval is not None \
               else max(args.n_tasks // 4, 25)
    label_seed_base = args.label_seed_base if args.label_seed_base is not None else args.seed
    out_dir = Path(args.out_dir) if args.out_dir is not None else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    out_path = build_dataset(
        cs, q0_seeds,
        kin=env.kin, collision=env.collision,
        env=env, controller=controller,
        out_dir=out_dir,
        cache_name=cache_name,
        hyperparams=hyperparams,
        label_kwargs=LABEL_KWARGS,
        checkpoint_interval=ckpt_int,
        label_seed_base=label_seed_base,
        verbose=True,
    )
    elapsed = time.time() - t0
    if out_path is None:
        # Interrupted by SIGINT/SIGTERM; partial saved, caller can resume.
        print(f"\n[speedtest] INTERRUPTED after {elapsed:.1f}s; partial saved.", flush=True)
        sys.exit(130)
    print(f"\n[speedtest] DONE  {args.n_tasks} tasks in {elapsed:.1f}s "
          f"({elapsed / args.n_tasks:.2f}s/task)", flush=True)
    print(f"  out: {out_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)

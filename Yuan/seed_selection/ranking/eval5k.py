"""Self-contained 5k-test eval harness for the expert-iteration ablations.

Rolls a checkpoint (standalone / hybrid) or the classical controller from the
per-task DP seed `q0_seeds` on the frozen 5k test, and reports the ratio to the
cached reference `ell_ref` by bucket. Reuses the exact rollout used by the
paper tables (system_eval.rollout_controllers.rollout_seeds_batched), so the
numbers are consistent with pipeline_v2/final_tables.json.

Usage:
    python -m Yuan.seed_selection.ranking.eval5k --ckpt <dir> --mode standalone
    python -m Yuan.seed_selection.ranking.eval5k --classical   # validation
"""
import os, sys
_conda_lib = os.path.join(sys.prefix, "lib")
if _conda_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    new_env = dict(os.environ)
    new_env["LD_LIBRARY_PATH"] = _conda_lib + ":" + new_env.get("LD_LIBRARY_PATH", "")
    os.execvpe(sys.executable, [sys.executable] + sys.argv, new_env)

import argparse
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/home/lqin/one")
os.chdir("/home/lqin/one")
from Yuan.system_eval.rollout_controllers import (
    build_env, load_rl_agent, rollout_seeds_batched)
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController

PV2 = Path("Yuan/seed_selection/runs/pipeline_v2")
TESTSMM = PV2 / "test_smm_5k.npz"
PERTASK = PV2 / "eval_test_5k_perTask.npz"
CFG_REF = Path("Yuan/RL_controller/runs/p0_entfix_danger_floor_30M_0702/config.yaml")


def load_tasks():
    d = np.load(TESTSMM)
    p = np.load(PERTASK)
    return dict(
        q0=d["q0_seeds"].astype(np.float32),
        p0=d["cs_p0"].astype(np.float32),
        ld=d["cs_line_dir"].astype(np.float32),
        nt=d["cs_n_target"].astype(np.float32),
        ell_ref=p["ell_ref"].astype(np.float64),
        bucket=p["bucket"],
        l_cls=p["l_cls"].astype(np.float64),
        l_rl=p["l_rl"].astype(np.float64),
    )


def bucketed_ratio(length_m, ell_ref, bucket):
    """Mean of per-task ratios (%) over tasks with ell_ref>0, by bucket."""
    out = {}
    valid = ell_ref > 1e-9
    ratio = np.where(valid, 100.0 * length_m / np.maximum(ell_ref, 1e-9), np.nan)
    for name, mask in [("All", valid),
                       ("Easy", valid & (bucket == "Easy")),
                       ("Medium", valid & (bucket == "Medium")),
                       ("Difficult", valid & (bucket == "Difficult"))]:
        r = ratio[mask]
        out[name] = (float(np.mean(r)), float(np.median(r)),
                     float(np.mean(length_m[mask])), int(mask.sum()))
    return out


def run(mode, ckpt=None, tau_enter=float("inf"), tau_exit=float("inf"),
        n_envs=2500, device="cuda"):
    dev = torch.device(device)
    T = load_tasks()
    # env is built from the RL config (env section); classical needs the kin.
    cfg = ckpt + "/config.yaml" if (ckpt and (Path(ckpt) / "config.yaml").exists()) else str(CFG_REF)
    env = build_env(cfg, n_envs, dev)
    classical = ClassicalNullspaceController(env.kin)
    if mode == "classical":
        res = rollout_seeds_batched(
            T["q0"], T["p0"], T["ld"], T["nt"], env=env,
            controller="classical", classical=classical, progress_prefix="cls ")
    else:
        agent = load_rl_agent(ckpt, env, dev)
        te, tx = (float("inf"), float("inf")) if mode == "standalone" else (tau_enter, tau_exit)
        res = rollout_seeds_batched(
            T["q0"], T["p0"], T["ld"], T["nt"], env=env,
            controller="hybrid_variantB", classical=classical, agent=agent,
            tau_enter=te, tau_exit=tx, progress_prefix=f"{mode} ")
    length = res["episode_progress_m"].astype(np.float64)
    stats = bucketed_ratio(length, T["ell_ref"], T["bucket"])
    return length, stats, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--mode", default="standalone",
                    choices=["standalone", "hybrid", "classical"])
    ap.add_argument("--tau-enter", type=float, default=0.985)
    ap.add_argument("--tau-exit", type=float, default=0.96)
    ap.add_argument("--validate", action="store_true",
                    help="compare per-task length to cached l_cls/l_rl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    length, stats, T = run(args.mode, args.ckpt, args.tau_enter, args.tau_exit)
    print(f"\n=== mode={args.mode} ckpt={args.ckpt} "
          f"tau=({args.tau_enter},{args.tau_exit}) ===")
    for k, (m, med, lm, n) in stats.items():
        print(f"  {k:10s} n={n:5d}  ratio mean {m:6.2f}  median {med:6.2f}  len {lm:.3f}")

    if args.validate:
        ref = T["l_cls"] if args.mode == "classical" else T["l_rl"]
        valid = T["ell_ref"] > 1e-9
        diff = np.abs(length - ref)[valid]
        print(f"  [validate vs {'l_cls' if args.mode=='classical' else 'l_rl'}] "
              f"max|Δlen| {diff.max():.4f}  mean|Δlen| {diff.mean():.5f}  "
              f"cached mean-ratio "
              f"{np.mean(100*ref[valid]/T['ell_ref'][valid]):.2f}")

    if args.out:
        np.savez_compressed(args.out, length=length,
                            ell_ref=T["ell_ref"], bucket=T["bucket"])
        print(f"  saved -> {args.out}")


if __name__ == "__main__":
    main()

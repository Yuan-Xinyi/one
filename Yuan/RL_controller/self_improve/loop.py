"""Self-improvement outer loop: hybrid switching as a policy-improvement operator.

    pi_{k+1} = PPO+BC( verified rescue steps of hybrid(pi_k, classical) )

Round k (out_root/round{k}/):
  1. collect — hybrid(pi_k, classical) on fresh training-distribution tasks;
               task-level win filter keeps only rescue steps from tasks where
               hybrid strictly outlived pure pi_k (see collect.py).
  2. train   — warm-start from pi_k, joint objective
                   L = L_PPO(progress reward) + bc_coef * ||tanh(mu(s)) - a_cls||^2
               bc annealed within round; the buffer shrinks across rounds as
               rescues get internalized -> automatic outer annealing.
  3. eval    — 10k eval set (per repo convention, ratios only):
               L_pure/L_oracle, L_hyb/L_oracle, frac(hybrid>pure), switches.
               NB: "oracle" = classical-controller-optimal labels (max_label_L),
               not a true ceiling under other controllers.

Convergence = operator dry: frac(hybrid > pure) -> ~0 and L_pure/L_hyb -> 1.
The loop stops early when the collect round finds no winning tasks.

Usage:
    python -m Yuan.RL_controller.self_improve.loop \\
        --init-ckpt-dir Yuan/RL_controller/runs/p0_progress_only_30M_0520 \\
        --rounds 4
"""
from __future__ import annotations

# Self-relaunch with $CONDA_PREFIX/lib on LD_LIBRARY_PATH (same as train/eval).
# GUARDED to entry-point only: on plain import the exec would hijack the host
# process into `python -m <this module>` with default args (bit us 2026-07-02:
# an eval helper import silently restarted the whole RL+BC loop).
import os, sys
if __name__ == "__main__":
    _conda_lib = os.path.join(sys.prefix, "lib")
    if _conda_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
        new_env = dict(os.environ)
        new_env["LD_LIBRARY_PATH"] = _conda_lib + ":" + new_env.get("LD_LIBRARY_PATH", "")
        os.execvpe(sys.executable,
                   [sys.executable, "-m", __spec__.name] + sys.argv[1:]
                   if __spec__ is not None else [sys.executable] + sys.argv,
                   new_env)

import argparse
import copy
import dataclasses
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from Yuan.RL_controller.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.RL_controller.env.line_distribution import LineDistribution
from Yuan.RL_controller.algorithms.ppo import PPOConfig, train as ppo_train
from Yuan.RL_controller.self_improve.collect import collect_buffer, load_env_kw

EVAL_SET_10K = "Yuan/system_eval/runs/eval_10k_systematic/eval_set_10k.npz"


def _ppo_cfg_from_yaml(cfg_yaml: dict, **overrides) -> PPOConfig:
    valid = {f.name for f in dataclasses.fields(PPOConfig)}
    kw = {k: v for k, v in cfg_yaml["ppo"].items() if k in valid}
    kw.update(overrides)
    return PPOConfig(**kw)


def _write_round_config(cfg_yaml: dict, round_dir: Path, *,
                        steps: int, lr: float, si_meta: dict) -> None:
    """Round dir doubles as a standard ckpt_dir (config.yaml + agent.pt), so
    eval/hybrid.py, eval/rl_vs_classical.py and system_eval tools work on it."""
    cfg = copy.deepcopy(cfg_yaml)
    cfg["ppo"]["total_timesteps"] = int(steps)
    cfg["ppo"]["learning_rate"] = float(lr)
    cfg["self_improve"] = si_meta
    with open(round_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _make_log_fn(log_path: Path, t0: float):
    log_file = open(log_path, "w")

    def log_fn(d: dict):
        log_file.write(repr({"wall_s": time.time() - t0, **d}) + "\n")
        log_file.flush()
        if "update" in d and d["update"] % 20 == 0:
            print(f"  upd {d['update']:>4}  step {d['global_step']:>9}  "
                  f"r/prog {d.get('reward/progress', 0):+.3f}  "
                  f"bc_loss {d.get('train/bc_loss', float('nan')):.4f}  "
                  f"bc_coef {d.get('train/bc_coef', 0):.3f}  "
                  f"entropy {d.get('train/entropy', 0):.2f}", flush=True)

    return log_fn, log_file


@torch.no_grad()
def eval_ckpt_on_10k(ckpt_dir, out_path, *, eval_set=EVAL_SET_10K,
                     tau_enter: float = 0.98, tau_exit: float = 0.94,
                     n_envs_chunk: int = 4096, device=None) -> dict:
    """Roll pure pi and hybrid(pi, classical) over the 10k eval set; cache
    per-task results to `out_path` (never re-run for follow-up analysis)."""
    # Lazy import: system_eval depends on RL_controller, keep the reverse edge
    # runtime-only.
    from Yuan.system_eval.rollout_controllers import (
        build_env, load_rl_agent, rollout_seeds_batched)
    from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController

    ckpt_dir = Path(ckpt_dir)
    device = torch.device(device if device is not None
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    d = np.load(eval_set)
    qs, p0 = d["q0_seed"], d["cs_p0"]
    ld, nt = d["cs_line_dir"], d["cs_n_target"]
    L_oracle = d["max_label_L"]

    env = build_env(ckpt_dir / "config.yaml", n_envs_chunk, device)
    classical = ClassicalNullspaceController(env.kin)
    agent = load_rl_agent(ckpt_dir, env, device)

    # Pure policy = hybrid with tau_enter=inf (the switch never leaves RL).
    pure = rollout_seeds_batched(
        qs, p0, ld, nt, env=env, controller="hybrid_variantB",
        classical=classical, agent=agent,
        tau_enter=float("inf"), tau_exit=float("inf"),
        progress_prefix="pure   ")
    hyb = rollout_seeds_batched(
        qs, p0, ld, nt, env=env, controller="hybrid_variantB",
        classical=classical, agent=agent,
        tau_enter=tau_enter, tau_exit=tau_exit,
        progress_prefix="hybrid ")
    del env
    if device.type == "cuda":
        torch.cuda.empty_cache()

    valid = L_oracle > 1e-6
    r_pure = pure["L"][valid] / L_oracle[valid]
    r_hyb = hyb["L"][valid] / L_oracle[valid]
    len_pure, len_hyb = pure["episode_len"], hyb["episode_len"]
    metrics = {
        "ratio_pure_vs_oracle_mean": float(r_pure.mean()),
        "ratio_pure_vs_oracle_median": float(np.median(r_pure)),
        "ratio_hyb_vs_oracle_mean": float(r_hyb.mean()),
        "ratio_hyb_vs_oracle_median": float(np.median(r_hyb)),
        "ratio_pure_vs_hyb_mean": float(
            (pure["L"] / np.maximum(hyb["L"], 1e-6)).mean()),
        "frac_hyb_gt_pure": float((len_hyb > len_pure).mean()),
        "frac_pure_gt_hyb": float((len_pure > len_hyb).mean()),
        "mean_switches": float(hyb["switch_count"].mean()),
        "n_valid_oracle": int(valid.sum()),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        L_pure=pure["L"], L_hyb=hyb["L"], L_oracle=L_oracle,
        episode_len_pure=len_pure, episode_len_hyb=len_hyb,
        term_pure=pure["term_reason"], term_hyb=hyb["term_reason"],
        switch_count=hyb["switch_count"],
        tau_enter=np.float64(tau_enter), tau_exit=np.float64(tau_exit),
        ckpt_dir=np.str_(str(ckpt_dir)),
        **{f"metric_{k}": np.float64(v) for k, v in metrics.items()},
    )
    print(f"[eval] {ckpt_dir.name}: "
          f"L/L_oracle mean {metrics['ratio_pure_vs_oracle_mean']:.3f} "
          f"(hyb {metrics['ratio_hyb_vs_oracle_mean']:.3f})  "
          f"frac(hyb>pure) {100*metrics['frac_hyb_gt_pure']:.1f}%  "
          f"sw/ep {metrics['mean_switches']:.2f}  -> {out_path}")
    print("[eval] NB: oracle = classical-controller-optimal labels, "
          "not a true ceiling for other controllers.")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-ckpt-dir",
                        default="Yuan/RL_controller/runs/p0_progress_only_30M_0520")
    parser.add_argument("--out-root", default="Yuan/RL_controller/runs/self_improve")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--start-round", type=int, default=1,
                        help="resume the loop at this round (earlier round "
                             "dirs must already contain agent.pt)")
    parser.add_argument("--steps-per-round", type=int, default=5_000_000)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="fine-tune LR per round (original run used 3e-4)")
    parser.add_argument("--bc-coef", type=float, default=0.3)
    parser.add_argument("--no-bc-anneal", action="store_true")
    parser.add_argument("--n-collect-tasks", type=int, default=16384)
    parser.add_argument("--collect-chunk", type=int, default=4096)
    parser.add_argument("--collect-seed-base", type=int, default=7000,
                        help="collect seed = base + round (fresh tasks/round)")
    parser.add_argument("--tau-enter", type=float, default=0.98)
    parser.add_argument("--tau-exit", type=float, default=0.94)
    parser.add_argument("--min-frac-win", type=float, default=0.01,
                        help="stop when < this fraction of collect tasks "
                             "benefit from the switch (operator dry)")
    parser.add_argument("--eval-set", default=EVAL_SET_10K)
    parser.add_argument("--skip-eval", action="store_true",
                        help="skip the 10k evals (smoke runs)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device is not None
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    init_dir = Path(args.init_ckpt_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    with open(init_dir / "config.yaml") as f:
        init_cfg = yaml.safe_load(f)
    env_kw = load_env_kw(init_cfg)
    line_cfg = init_cfg["line_distribution"]
    threshold_m = (float(line_cfg["feasibility_threshold_m"])
                   if line_cfg.get("feasibility_filter", False) else None)

    # Train env + pool built once, reused across rounds (ppo_train resets it).
    train_env_cfg = EnvConfig(**env_kw)
    print(f"[loop] device={device}  building train env (n_envs={train_env_cfg.n_envs})")
    train_env = NSRLBatchedEnv(train_env_cfg, line_dist=None, device=device)
    train_env.line_dist = LineDistribution.load_or_build(
        kin=train_env.kin, collision=train_env.collision,
        n_pool=line_cfg["n_pool"],
        n_target_noise_deg=line_cfg["n_target_noise_deg"],
        seed=line_cfg["train_seed"],
        env_cfg=train_env_cfg,
        feasibility_threshold_m=threshold_m,
    )

    # Round-0 reference eval of pi_0 (cached; skipped if present).
    round0_eval = out_root / "round0_eval_10k.npz"
    if not args.skip_eval and not round0_eval.exists():
        print(f"[loop] round 0 reference eval of {init_dir.name} on 10k set")
        eval_ckpt_on_10k(init_dir, round0_eval, eval_set=args.eval_set,
                         tau_enter=args.tau_enter, tau_exit=args.tau_exit,
                         device=device)

    summary_rows = []
    prev_dir = (init_dir if args.start_round == 1
                else out_root / f"round{args.start_round - 1}")
    for k in range(args.start_round, args.rounds + 1):
        round_dir = out_root / f"round{k}"
        round_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[loop] ===== round {k}/{args.rounds}  (pi_{k-1} = {prev_dir}) =====")

        # ---- 1. collect (skipped if this round's buffer already exists) ----
        buffer_path = round_dir / "buffer.npz"
        if buffer_path.exists():
            print(f"[loop] buffer exists, skipping collect: {buffer_path}")
            buf = np.load(buffer_path)
            stats = {"n_kept_steps": int(buf["obs"].shape[0]),
                     "frac_win": float(buf["win"].mean()),
                     "n_win_tasks": int(buf["win"].sum()),
                     "n_tasks": int(buf["win"].shape[0])}
        else:
            stats = collect_buffer(
                prev_dir, buffer_path,
                n_tasks=args.n_collect_tasks,
                seed=args.collect_seed_base + k,
                tau_enter=args.tau_enter, tau_exit=args.tau_exit,
                chunk_size=args.collect_chunk, device=device)
            buf = np.load(buffer_path)

        if stats["n_kept_steps"] == 0 or stats["frac_win"] < args.min_frac_win:
            print(f"[loop] CONVERGED at round {k}: frac_win="
                  f"{100*stats['frac_win']:.2f}% (< {100*args.min_frac_win:.1f}%) "
                  f"— the switch has no verified rescues left to teach.")
            break

        # ---- 2. joint PPO + BC fine-tune ----
        _write_round_config(init_cfg, round_dir,
                            steps=args.steps_per_round, lr=args.lr,
                            si_meta={"round": k, "init_ckpt": str(prev_dir),
                                     "bc_coef": args.bc_coef,
                                     "bc_anneal": not args.no_bc_anneal,
                                     "tau_enter": args.tau_enter,
                                     "tau_exit": args.tau_exit,
                                     "collect": {kk: stats[kk] for kk in
                                                 ("n_tasks", "n_win_tasks",
                                                  "frac_win", "n_kept_steps")}})
        ppo_cfg = _ppo_cfg_from_yaml(init_cfg,
                                     total_timesteps=args.steps_per_round,
                                     learning_rate=args.lr)
        bc_obs = torch.from_numpy(buf["obs"])
        bc_actions = torch.from_numpy(buf["a_cls"])
        print(f"[loop] fine-tune: {args.steps_per_round} steps, lr={args.lr}, "
              f"bc_coef={args.bc_coef} on {bc_obs.shape[0]} rescue steps")
        log_fn, log_file = _make_log_fn(round_dir / "train.log", time.time())
        ppo_train(ppo_cfg, train_env, device=device,
                  eval_fn=None, log_fn=log_fn,
                  ckpt_path=str(round_dir / "agent.pt"),
                  resume_from_ckpt=str(prev_dir / "agent.pt"),
                  bc_obs=bc_obs, bc_actions=bc_actions,
                  bc_coef=args.bc_coef, bc_anneal=not args.no_bc_anneal)
        log_file.close()
        print(f"[loop] round {k} ckpt -> {round_dir / 'agent.pt'}")

        # ---- 3. 10k eval ----
        row = {"round": k, **{f"collect_{kk}": stats[kk] for kk in
                              ("frac_win", "n_kept_steps")}}
        if not args.skip_eval:
            metrics = eval_ckpt_on_10k(
                round_dir, round_dir / "eval_10k.npz", eval_set=args.eval_set,
                tau_enter=args.tau_enter, tau_exit=args.tau_exit, device=device)
            row.update(metrics)
        summary_rows.append(row)
        prev_dir = round_dir

    print("\n[loop] ===== summary =====")
    for r in summary_rows:
        line = (f"round {r['round']}:  collect frac_win "
                f"{100*r['collect_frac_win']:.1f}%  "
                f"kept {r['collect_n_kept_steps']}")
        if "ratio_pure_vs_oracle_mean" in r:
            line += (f"  |  L/L_or {r['ratio_pure_vs_oracle_mean']:.3f}  "
                     f"hyb {r['ratio_hyb_vs_oracle_mean']:.3f}  "
                     f"frac(hyb>pure) {100*r['frac_hyb_gt_pure']:.1f}%  "
                     f"sw/ep {r['mean_switches']:.2f}")
        print(line)


if __name__ == "__main__":
    main()

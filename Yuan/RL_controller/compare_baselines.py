"""Compare RL policy vs classical nullspace controller vs GPM baseline + init random.

Runs all four controllers on the SAME 200-line holdout, prints summary table.

Usage:
    python -m Yuan.RL_controller.compare_baselines \\
        --config Yuan/RL_controller/config.yaml \\
        --ckpt   Yuan/RL_controller/runs7/agent.pt
"""
from __future__ import annotations

import os, sys
_conda_lib = os.path.join(sys.prefix, "lib")
if _conda_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    new_env = dict(os.environ)
    new_env["LD_LIBRARY_PATH"] = _conda_lib + ":" + new_env.get("LD_LIBRARY_PATH", "")
    if __spec__ is not None and __spec__.name != "__main__":
        argv = [sys.executable, "-m", __spec__.name] + sys.argv[1:]
    else:
        argv = [sys.executable] + sys.argv
    os.execvpe(sys.executable, argv, new_env)

import argparse
import statistics

import torch
import yaml

from Yuan.RL_controller.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.RL_controller.env.line_distribution import LineDistribution, ScriptedLineDistribution
from Yuan.RL_controller.env.baseline_controller import (
    GPMBaselineController, rollout_first_episode, baseline_action_fn,
)
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController, cn_action_fn
from Yuan.RL_controller.ppo import Agent
from Yuan.RL_controller.eval import TERM_NAMES
_ScriptedLineDistribution = ScriptedLineDistribution  # local alias


def _summarize(name: str, stats: dict) -> dict:
    lens = stats["episode_len"].cpu().numpy()
    term = stats["term_reason"].cpu().numpy()
    progress = stats.get("episode_progress")
    if progress is not None:
        progress = progress.cpu().numpy()
    n = lens.shape[0]
    term_counts = {TERM_NAMES.get(int(t), "?"): 0 for t in TERM_NAMES}
    for t in term:
        term_counts[TERM_NAMES.get(int(t), "?")] += 1
    term_pct = {k: v / n for k, v in term_counts.items()}
    out = {
        "name": name,
        "n": n,
        "mean_len": float(lens.mean()),
        "median_len": float(statistics.median(lens)),
        "max_len": int(lens.max()),
        "term_pct": term_pct,
    }
    if progress is not None:
        out["mean_progress_m"] = float(progress.mean())
        out["median_progress_m"] = float(statistics.median(progress))
        out["max_progress_m"] = float(progress.max())
    return out


def _print_table(rows):
    init_len = rows[0]["mean_len"]
    init_prog = rows[0].get("mean_progress_m", 0.0)
    fmt = "{:<22} {:>6} {:>6} {:>6} {:>6} {:>9} {:>9} {:>9} {:>6} {:>6}"
    print(fmt.format("controller", "mean", "%init", "median", "max",
                     "mean(m)", "med(m)", "max(m)", "cone", "jl"))
    print("-" * 110)
    for r in rows:
        pct = 100 * r["mean_len"] / init_len if init_len > 0 else 0
        print(fmt.format(
            r["name"][:22], f"{r['mean_len']:.0f}", f"{pct:.0f}%",
            f"{r['median_len']:.0f}", f"{r['max_len']}",
            f"{r.get('mean_progress_m', 0):.3f}",
            f"{r.get('median_progress_m', 0):.3f}",
            f"{r.get('max_progress_m', 0):.3f}",
            f"{r['term_pct'].get('cone', 0):.2f}",
            f"{r['term_pct'].get('jl', 0):.2f}",
        ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config) as f:
        cfg_yaml = yaml.safe_load(f)
    eval_cfg = cfg_yaml["eval"]
    n_holdout = eval_cfg["n_holdout"]
    env_cfg = EnvConfig(**{**cfg_yaml["env"], "n_envs": n_holdout})

    # Build sampler env + draw deterministic 200-line spec
    proxy = NSRLBatchedEnv(env_cfg, line_dist=None, device=device)
    line_cfg = cfg_yaml["line_distribution"]
    threshold_m = (float(line_cfg["feasibility_threshold_m"])
                   if line_cfg.get("feasibility_filter", False) else None)
    sampler = LineDistribution.load_or_build(
        kin=proxy.kin, collision=proxy.collision,
        n_pool=line_cfg["n_pool"],
        n_target_noise_deg=line_cfg["n_target_noise_deg"],
        seed=eval_cfg["holdout_seed"],
        env_cfg=env_cfg,
        feasibility_threshold_m=threshold_m,
    )
    holdout = sampler.sample(n_holdout)
    print(f"[compare] holdout: {n_holdout} lines, seed={eval_cfg['holdout_seed']}")

    def fresh_env():
        e = NSRLBatchedEnv(env_cfg, line_dist=None, device=device)
        e.line_dist = _ScriptedLineDistribution({k: v.clone() for k, v in holdout.items()})
        return e

    rows = []

    # 1. Init random (untrained agent — uniform random Gaussian)
    print("[compare] running INIT RANDOM (untrained agent) ...")
    env = fresh_env()
    init_agent = Agent(env.obs_dim, env.act_dim,
                       hidden_dim=cfg_yaml["ppo"]["hidden_dim"],
                       init_log_std=cfg_yaml["ppo"]["init_log_std"]).to(device)
    init_agent.eval()
    # Use stochastic sampling for "random" baseline (matches first-eval behavior)
    def init_random_fn(env_):
        with torch.no_grad():
            a, _, _, _ = init_agent.get_action_and_value(env_.current_obs())
            return a.clamp(-1.0, 1.0)
    stats = rollout_first_episode(env, init_random_fn)
    rows.append(_summarize("init_random (σ≈0.6)", stats))

    # 2. RL policy (deterministic μ)
    print("[compare] running RL policy ...")
    env = fresh_env()
    agent = Agent(env.obs_dim, env.act_dim,
                  hidden_dim=cfg_yaml["ppo"]["hidden_dim"],
                  init_log_std=cfg_yaml["ppo"]["init_log_std"]).to(device)
    agent.load_state_dict(torch.load(args.ckpt, map_location=device))
    agent.eval()
    def rl_fn(env_):
        with torch.no_grad():
            return agent.actor_mean(env_.current_obs()).clamp(-1.0, 1.0)
    stats = rollout_first_episode(env, rl_fn)
    rows.append(_summarize(f"RL (runs7 final)", stats))

    # 3. GPM baseline
    print("[compare] running GPM baseline ...")
    env = fresh_env()
    gpm = GPMBaselineController(env.kin, k_jl=cfg_yaml["baseline"]["k_jl"])
    stats = rollout_first_episode(env, baseline_action_fn(gpm))
    rows.append(_summarize("GPM-JL baseline", stats))

    # 4. classical nullspace controller
    print("[compare] running classical nullspace controller ...")
    env = fresh_env()
    fc = ClassicalNullspaceController(env.kin)
    stats = rollout_first_episode(env, cn_action_fn(fc))
    rows.append(_summarize("classical_nullspace", stats))

    print()
    _print_table(rows)


if __name__ == "__main__":
    main()

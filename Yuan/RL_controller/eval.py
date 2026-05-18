"""Evaluate trained PPO policy vs GPM baseline on a fixed 200-line holdout.

Both controllers run with identical line specs (q_0, u_hat, n_target), identical
dt and termination logic. Uses `auto_reset=False` so each env runs exactly one
episode and finished envs freeze — no spec exhaustion.

Output:
    - CSV: line_id, T_rl, T_baseline, ratio, term_reason_rl, term_reason_baseline
      (T_* in steps; physical time = T_* · dt)
    - stdout: mean / median ratio, term_reason histograms, truncated-warning if > 5%

Usage:
    python -m Yuan.RL_controller.eval --config Yuan/RL_controller/config.yaml \\
        --ckpt path/to/agent.pt --out path/to/eval.csv
"""
from __future__ import annotations

# Self-relaunch with $CONDA_PREFIX/lib on LD_LIBRARY_PATH (see train.py).
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
import collections
import csv
from pathlib import Path

import torch
import yaml

from Yuan.RL_controller.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.RL_controller.env.line_distribution import LineDistribution, ScriptedLineDistribution
from Yuan.RL_controller.env.baseline_controller import (
    GPMBaselineController, rollout_first_episode, baseline_action_fn,
)
from Yuan.RL_controller.ppo import Agent


TERM_NAMES = {0: "alive", 2: "collision", 3: "cone", 4: "jl", 5: "truncated"}

# Backwards-compat alias for any external caller (the class lives in env/line_distribution.py now)
_ScriptedLineDistribution = ScriptedLineDistribution


def _rl_action_fn(agent: Agent):
    @torch.no_grad()
    def _fn(env: NSRLBatchedEnv) -> torch.Tensor:
        mean = agent.actor_mean(env.current_obs())
        return mean.clamp(-1.0, 1.0)
    return _fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="Yuan/RL_controller/runs/eval.csv")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config, "r") as f:
        cfg_yaml = yaml.safe_load(f)

    eval_cfg = cfg_yaml["eval"]
    line_cfg = cfg_yaml["line_distribution"]
    n_holdout = eval_cfg["n_holdout"]

    env_cfg = EnvConfig(**{**cfg_yaml["env"], "n_envs": n_holdout})

    # Build proxy env just to instantiate kin/collision and draw deterministic specs.
    proxy_env = NSRLBatchedEnv(env_cfg, line_dist=None, device=device)
    threshold_m = (float(line_cfg["feasibility_threshold_m"])
                   if line_cfg.get("feasibility_filter", False) else None)
    sampler = LineDistribution.load_or_build(
        kin=proxy_env.kin, collision=proxy_env.collision,
        n_pool=line_cfg["n_pool"],
        n_target_noise_deg=line_cfg["n_target_noise_deg"],
        seed=eval_cfg["holdout_seed"],
        env_cfg=env_cfg,
        feasibility_threshold_m=threshold_m,
    )
    holdout = sampler.sample(n_holdout)

    # RL rollout
    rl_env = proxy_env
    rl_env.line_dist = _ScriptedLineDistribution(
        {k: v.clone() for k, v in holdout.items()})
    agent = Agent(rl_env.obs_dim, rl_env.act_dim,
                  hidden_dim=cfg_yaml["ppo"]["hidden_dim"],
                  init_log_std=cfg_yaml["ppo"]["init_log_std"]).to(device)
    state_dict = torch.load(args.ckpt, map_location=device)
    agent.load_state_dict(state_dict)
    agent.eval()
    rl_stats = rollout_first_episode(rl_env, _rl_action_fn(agent))

    # Baseline rollout — fresh env to reset internal state
    base_env = NSRLBatchedEnv(env_cfg, line_dist=None, device=device)
    base_env.line_dist = _ScriptedLineDistribution(
        {k: v.clone() for k, v in holdout.items()})
    base_ctrl = GPMBaselineController(base_env.kin,
                                      k_jl=cfg_yaml["baseline"]["k_jl"])
    base_stats = rollout_first_episode(base_env, baseline_action_fn(base_ctrl))

    rl_len = rl_stats["episode_len"].cpu().numpy()
    base_len = base_stats["episode_len"].cpu().numpy()
    rl_term = rl_stats["term_reason"].cpu().numpy()
    base_term = base_stats["term_reason"].cpu().numpy()
    ratio = rl_len.astype(float) / base_len.astype(float).clip(min=1.0)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["line_id", "T_rl", "T_baseline", "ratio",
                    "term_reason_rl", "term_reason_baseline"])
        for i in range(n_holdout):
            w.writerow([i, int(rl_len[i]), int(base_len[i]), float(ratio[i]),
                        TERM_NAMES.get(int(rl_term[i]), "?"),
                        TERM_NAMES.get(int(base_term[i]), "?")])

    sorted_ratio = sorted(ratio)
    median = sorted_ratio[n_holdout // 2]
    print(f"[eval] wrote {n_holdout} rows → {out_path}")
    print(f"[eval] ratio  mean={ratio.mean():.3f}  median={median:.3f}")
    rl_hist = collections.Counter(TERM_NAMES.get(int(t), "?") for t in rl_term)
    base_hist = collections.Counter(TERM_NAMES.get(int(t), "?") for t in base_term)
    print(f"[eval] RL term_reason   : {dict(rl_hist)}")
    print(f"[eval] base term_reason : {dict(base_hist)}")
    rl_trunc_rate = rl_hist.get("truncated", 0) / n_holdout
    if rl_trunc_rate > 0.05:
        print(f"[eval] WARNING: RL truncated rate {rl_trunc_rate:.1%} > 5% — "
              "consider raising max_steps")


if __name__ == "__main__":
    main()

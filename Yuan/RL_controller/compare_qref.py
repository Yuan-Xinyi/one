"""One-off: classical nullspace with vs without q_ref pull (k_null).

Same env / sampler as `diagnose_p0_vs_baseline.py`; runs ClassicalNullspaceController
twice on the same N tasks — once with default k_null=0.5, once with k_null=0.0 —
and reports per-task ratio L_noqref / L_default plus term_reason histogram diffs.

Per repo convention, only ratios are reported (some tasks are infeasible, so absolute
episode-lengths and "success_rate" are not meaningful).

Usage:
    python -m Yuan.RL_controller.compare_qref \\
        --ckpt-dir Yuan/RL_controller/runs/p0_progress_only_30M_0520 \\
        --n 500 --seed 1000
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
import collections
import dataclasses
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from Yuan.RL_controller.env.env import NSRLBatchedEnv, EnvConfig, TERM_NAMES, TERM_TRUNCATED
from Yuan.RL_controller.env.line_distribution import LineDistribution, ScriptedLineDistribution
from Yuan.RL_controller.env.classical_nullspace import (
    ClassicalNullspaceController, cn_action_fn,
)


@torch.no_grad()
def rollout_lengths(env: NSRLBatchedEnv, action_fn) -> dict:
    cfg_max = env.max_steps
    n = env.n_envs
    episode_len = torch.full((n,), -1, dtype=torch.long, device=env.device)
    episode_term = torch.full((n,), -1, dtype=torch.long, device=env.device)
    finished = torch.zeros((n,), dtype=torch.bool, device=env.device)
    env.reset()
    for _ in range(cfg_max):
        a = action_fn(env)
        _, _, _, _, info = env.step(a, auto_reset=False)
        new_done = info["episode_done"]
        if new_done.any():
            episode_len[new_done] = env.t[new_done]
            episode_term[new_done] = info["term_reason"][new_done]
            finished = finished | new_done
        if bool(env.done_persistent.all().item()):
            break
    if (~finished).any():
        not_done = ~finished
        episode_len[not_done] = env.t[not_done]
        episode_term[not_done] = TERM_TRUNCATED
    return {"episode_len": episode_len.cpu().numpy(),
            "term_reason": episode_term.cpu().numpy()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    cfg_path = ckpt_dir / "config.yaml"
    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    with open(cfg_path, "r") as f:
        cfg_yaml = yaml.safe_load(f)
    line_cfg = cfg_yaml["line_distribution"]
    valid_keys = {f.name for f in dataclasses.fields(EnvConfig)}
    env_kw = {k: v for k, v in cfg_yaml["env"].items() if k in valid_keys}
    env_cfg = EnvConfig(**{**env_kw, "n_envs": args.n})

    proxy_env = NSRLBatchedEnv(env_cfg, line_dist=None, device=device)
    threshold_m = (float(line_cfg["feasibility_threshold_m"])
                   if line_cfg.get("feasibility_filter", False) else None)
    sampler = LineDistribution.load_or_build(
        kin=proxy_env.kin, collision=proxy_env.collision,
        n_pool=line_cfg["n_pool"],
        n_target_noise_deg=line_cfg["n_target_noise_deg"],
        seed=cfg_yaml["eval"]["holdout_seed"],
        env_cfg=env_cfg,
        feasibility_threshold_m=threshold_m,
    )
    gen = torch.Generator(device=device).manual_seed(int(args.seed))
    holdout = sampler.sample(args.n, generator=gen)

    print(f"[cmp] N={args.n}  seed={args.seed}  max_steps={env_cfg.max_steps}")
    results = {}
    for label, k_null in [("default(k_null=0.5)", 0.5), ("no_qref(k_null=0.0)", 0.0)]:
        env = NSRLBatchedEnv(env_cfg, line_dist=None, device=device)
        env.line_dist = ScriptedLineDistribution(
            {k: v.clone() for k, v in holdout.items()})
        ctrl = ClassicalNullspaceController(env.kin, k_null=k_null)
        t0 = time.time()
        out = rollout_lengths(env, cn_action_fn(ctrl))
        dt = time.time() - t0
        print(f"[cmp] {label}: {dt:.1f}s")
        results[label] = out
        del env
        torch.cuda.empty_cache()

    L_def = results["default(k_null=0.5)"]["episode_len"].astype(float)
    L_no = results["no_qref(k_null=0.0)"]["episode_len"].astype(float)

    # ratio (no_qref / default). >1 means no_qref survives longer; <1 means worse.
    ratio = L_no / np.clip(L_def, 1.0, None)

    def _q(x, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
        return {f"p{int(q*100)}": float(np.quantile(x, q)) for q in qs}

    print()
    print("# L_noqref / L_default (ratio per task; >1 means no_qref lasts longer)")
    print(f"  mean={ratio.mean():.3f}  "
          + "  ".join(f"{k}={v:.3f}" for k, v in _q(ratio).items()))

    n_better = int((L_no > L_def).sum())
    n_worse  = int((L_no < L_def).sum())
    n_tie    = int((L_no == L_def).sum())
    print(f"  per-task wins:  no_qref_better={n_better}  tie={n_tie}  "
          f"no_qref_worse={n_worse}  (N={args.n})")

    print()
    print("# term_reason histogram")
    for label in ("default(k_null=0.5)", "no_qref(k_null=0.0)"):
        hist = collections.Counter(
            TERM_NAMES.get(int(t), "?") for t in results[label]["term_reason"])
        print(f"  {label:<22}: {dict(hist)}")


if __name__ == "__main__":
    main()

"""Head-to-head evaluation: RL policy vs hand-tuned GPM-JL classical baseline.

Rolls out both controllers on the same N random tasks (default N=1000, seed
disjoint from training holdout_seed=42), shares env config with training, and
records per-task joint-angle trajectory for both — so follow-up analysis can
slice into specific tasks without re-running the rollout.

Output (under <ckpt-dir>/diag_10000_classical/):
  rollouts.npz   q_traj_rl  (T+1, N, 7) joint trajectory, frozen post-termination
                 q_traj_base (T+1, N, 7)
                 episode_len_rl (N,) steps until terminate/truncate
                 episode_len_base (N,)
                 term_reason_rl (N,) int code; TERM_NAMES from env.env
                 term_reason_base (N,)
                 q0 (N, 7), line_dir (N, 3), n_target (N, 3) task specs
                 max_steps, dt scalars
  per_task.csv   row per task: idx, T_rl, T_base, ratio, term_rl, term_base

Usage:
    python -m Yuan.RL_controller.eval.rl_vs_classical \\
        --ckpt-dir Yuan/RL_controller/runs/p0_progress_only_30M_0520 \\
        --n 1000 --seed 1000
"""
from __future__ import annotations

# Self-relaunch with $CONDA_PREFIX/lib on LD_LIBRARY_PATH (same as train/eval).
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

import numpy as np
import torch
import yaml

from Yuan.RL_controller.env.env import NSRLBatchedEnv, EnvConfig, TERM_NAMES, TERM_TRUNCATED
from Yuan.RL_controller.env.line_distribution import LineDistribution, ScriptedLineDistribution
from Yuan.RL_controller.env.classical_nullspace import (
    ClassicalNullspaceController, cn_action_fn,
)
from Yuan.RL_controller.algorithms.ppo import Agent


def _rl_action_fn(agent: Agent):
    @torch.no_grad()
    def _fn(env: NSRLBatchedEnv) -> torch.Tensor:
        mean = agent.actor_mean(env.current_obs())
        return mean.clamp(-1.0, 1.0)
    return _fn


@torch.no_grad()
def rollout_with_q_traj(env: NSRLBatchedEnv, action_fn,
                        max_steps: int | None = None) -> dict:
    """auto_reset=False rollout that also records env.q at every step.

    q_traj has shape (max_steps + 1, n_envs, 7):
      - q_traj[0] = q at reset (start of episode)
      - q_traj[t] = q after step t-1 (so q_traj[episode_len] is the final q
        right at termination; later entries hold the frozen post-term value
        since env freezes finished envs with auto_reset=False).
    """
    cfg_max = env.max_steps if max_steps is None else max_steps
    n = env.n_envs
    episode_len = torch.full((n,), -1, dtype=torch.long, device=env.device)
    episode_term = torch.full((n,), -1, dtype=torch.long, device=env.device)
    finished = torch.zeros((n,), dtype=torch.bool, device=env.device)

    q_traj = torch.zeros((cfg_max + 1, n, 7),
                         dtype=env.kin.dtype, device=env.device)

    env.reset()
    q_traj[0] = env.q.clone()

    for step_i in range(cfg_max):
        a = action_fn(env)
        _, _, _, _, info = env.step(a, auto_reset=False)
        q_traj[step_i + 1] = env.q.clone()
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
    return {"episode_len": episode_len, "term_reason": episode_term,
            "q_traj": q_traj}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True,
                        help="run dir containing agent.pt and config.yaml")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1000,
                        help="sample seed for the 1000 random tasks (disjoint "
                             "from holdout_seed=42)")
    parser.add_argument("--out-dir", default=None,
                        help="defaults to <ckpt-dir>/diag_<n>/")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    cfg_path = ckpt_dir / "config.yaml"
    ckpt_path = ckpt_dir / "agent.pt"
    out_dir = (Path(args.out_dir) if args.out_dir
               else ckpt_dir / f"diag_{args.n}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(cfg_path, "r") as f:
        cfg_yaml = yaml.safe_load(f)

    line_cfg = cfg_yaml["line_distribution"]
    # Drop stale shaping-reward keys EnvConfig no longer accepts (older P0
    # config pre-dates env.py simplification to progress-only).
    import dataclasses
    valid_keys = {f.name for f in dataclasses.fields(EnvConfig)}
    env_kw = {k: v for k, v in cfg_yaml["env"].items() if k in valid_keys}
    env_cfg = EnvConfig(**{**env_kw, "n_envs": args.n})

    # Build pool (cached) and draw N tasks with an explicit generator seeded
    # by --seed (deterministic across reruns).
    proxy_env = NSRLBatchedEnv(env_cfg, line_dist=None, device=device)
    threshold_m = (float(line_cfg["feasibility_threshold_m"])
                   if line_cfg.get("feasibility_filter", False) else None)
    sampler = LineDistribution.load_or_build(
        kin=proxy_env.kin, collision=proxy_env.collision,
        n_pool=line_cfg["n_pool"],
        n_target_noise_deg=line_cfg["n_target_noise_deg"],
        seed=cfg_yaml["eval"]["holdout_seed"],  # share pool with eval (same cache)
        env_cfg=env_cfg,
        feasibility_threshold_m=threshold_m,
    )
    sample_gen = torch.Generator(device=device).manual_seed(int(args.seed))
    holdout = sampler.sample(args.n, generator=sample_gen)

    # ---- RL rollout ----
    rl_env = proxy_env
    rl_env.line_dist = ScriptedLineDistribution(
        {k: v.clone() for k, v in holdout.items()})
    agent = Agent(rl_env.obs_dim, rl_env.act_dim,
                  hidden_dim=cfg_yaml["ppo"]["hidden_dim"],
                  init_log_std=cfg_yaml["ppo"]["init_log_std"]).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    agent.load_state_dict(state_dict)
    agent.eval()
    print(f"[diag] running RL rollout ({args.n} envs, max_steps={env_cfg.max_steps})")
    rl_stats = rollout_with_q_traj(rl_env, _rl_action_fn(agent))

    # ---- Classical 4-term hand-tuned baseline (fresh env) ----
    base_env = NSRLBatchedEnv(env_cfg, line_dist=None, device=device)
    base_env.line_dist = ScriptedLineDistribution(
        {k: v.clone() for k, v in holdout.items()})
    base_ctrl = ClassicalNullspaceController(base_env.kin)  # default hand-tuned gains
    print(f"[diag] running Classical hand-tuned baseline rollout "
          f"(manip={base_ctrl.manip_gain}, jl={base_ctrl.jl_gain}, "
          f"angle_b={base_ctrl.angle_boundary_gain}, k_null={base_ctrl.k_null})")
    base_stats = rollout_with_q_traj(base_env, cn_action_fn(base_ctrl))

    # ---- save npz ----
    rl_len = rl_stats["episode_len"].cpu().numpy()
    base_len = base_stats["episode_len"].cpu().numpy()
    rl_term = rl_stats["term_reason"].cpu().numpy()
    base_term = base_stats["term_reason"].cpu().numpy()
    q_traj_rl = rl_stats["q_traj"].cpu().numpy().astype(np.float32)
    q_traj_base = base_stats["q_traj"].cpu().numpy().astype(np.float32)

    npz_path = out_dir / "rollouts.npz"
    np.savez_compressed(
        npz_path,
        q_traj_rl=q_traj_rl,
        q_traj_base=q_traj_base,
        episode_len_rl=rl_len,
        episode_len_base=base_len,
        term_reason_rl=rl_term,
        term_reason_base=base_term,
        q0=holdout["q0"].cpu().numpy(),
        line_dir=holdout["line_dir"].cpu().numpy(),
        n_target=holdout["n_target"].cpu().numpy(),
        max_steps=np.int64(env_cfg.max_steps),
        dt=np.float64(env_cfg.dt),
        sample_seed=np.int64(args.seed),
    )
    print(f"[diag] saved trajectories → {npz_path} "
          f"({npz_path.stat().st_size / 1e6:.1f} MB)")

    # ---- per-task csv ----
    ratio = rl_len.astype(float) / base_len.astype(float).clip(min=1.0)
    csv_path = out_dir / "per_task.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "T_rl", "T_base", "ratio_rl_base",
                    "term_rl", "term_base"])
        for i in range(args.n):
            w.writerow([i, int(rl_len[i]), int(base_len[i]), float(ratio[i]),
                        TERM_NAMES.get(int(rl_term[i]), "?"),
                        TERM_NAMES.get(int(base_term[i]), "?")])
    print(f"[diag] saved per-task csv → {csv_path}")

    # ---- summary stats: only ratios (per repo convention; absolute lengths
    #      not meaningful since some tasks are intrinsically infeasible). ----
    def _q(x, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
        return {f"p{int(q*100)}": float(np.quantile(x, q)) for q in qs}
    rb_mean = float(ratio.mean())
    rb_q = _q(ratio)
    print()
    print(f"[diag] N = {args.n} tasks  (seed={args.seed})")
    print(f"[diag] L_rl / L_base — mean={rb_mean:.3f}  "
          + "  ".join(f"{k}={v:.3f}" for k, v in rb_q.items()))
    n_rl_worse = int((rl_len < base_len).sum())
    n_rl_better = int((rl_len > base_len).sum())
    n_tie = int((rl_len == base_len).sum())
    print(f"[diag] per-task wins:  RL_worse={n_rl_worse}  "
          f"tie={n_tie}  RL_better={n_rl_better}")

    rl_hist = collections.Counter(TERM_NAMES.get(int(t), "?") for t in rl_term)
    base_hist = collections.Counter(TERM_NAMES.get(int(t), "?") for t in base_term)
    print(f"[diag] RL term_reason   : {dict(rl_hist)}")
    print(f"[diag] base term_reason : {dict(base_hist)}")


if __name__ == "__main__":
    main()

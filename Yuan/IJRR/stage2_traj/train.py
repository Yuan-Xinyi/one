"""PPO training entry point.

Usage:
    python -m Yuan.IJRR.stage2_traj.train --config Yuan/IJRR/stage2_traj/config.yaml \\
        [--ckpt path/to/agent.pt] [--device cuda|cpu] \\
        [--wandb] [--wandb-project NSRL-FR3] [--wandb-run-name runs4]
"""
from __future__ import annotations

# Self-relaunch with $CONDA_PREFIX/lib on LD_LIBRARY_PATH so matplotlib (pulled
# in by `one/__init__.py`) finds the conda libstdc++. Must run before any
# import that triggers `one`. See Known Issues #8.
import os, sys
_conda_lib = os.path.join(sys.prefix, "lib")
if _conda_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    new_env = dict(os.environ)
    new_env["LD_LIBRARY_PATH"] = _conda_lib + ":" + new_env.get("LD_LIBRARY_PATH", "")
    # Preserve `python -m <module>` invocation; fall back to script mode.
    if __spec__ is not None and __spec__.name != "__main__":
        argv = [sys.executable, "-m", __spec__.name] + sys.argv[1:]
    else:
        argv = [sys.executable] + sys.argv
    os.execvpe(sys.executable, argv, new_env)

import argparse
import time
from pathlib import Path

import torch
import yaml

from Yuan.IJRR.env.env import NSRLBatchedEnv, EnvConfig, TERM_NAMES
from Yuan.IJRR.env.line_distribution import LineDistribution
from Yuan.IJRR.stage2_traj.ppo import PPOConfig, train as ppo_train, Agent
from Yuan.IJRR.stage2_traj.vertex_agent import VertexAgent, PriorVertexAgent


def _resolve_log_path(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "train.log"


def _make_eval_fn(eval_env: NSRLBatchedEnv):
    from Yuan.IJRR.env.rollout import rollout_first_episode

    @torch.no_grad()
    def _policy_action(env: NSRLBatchedEnv, agent: Agent) -> torch.Tensor:
        mean = agent.actor_mean(env.current_obs())
        return mean.clamp(-1.0, 1.0)

    @torch.no_grad()
    def _eval(agent: Agent) -> dict:
        def action_fn(env):
            return _policy_action(env, agent)
        stats = rollout_first_episode(eval_env, action_fn)
        term = stats["term_reason"].cpu().numpy()
        n = term.shape[0]
        # term_reason fractions for trend tracking (failure-mode panel)
        frac = {f"eval_term/{name}": float((term == code).sum()) / n
                for code, name in TERM_NAMES.items()}
        ep_progress = stats["episode_progress"].float()
        return {
            "eval/mean_progress_m": float(ep_progress.mean().item()),
            "eval/median_progress_m": float(ep_progress.median().item()),
            "eval/max_progress_m": float(ep_progress.max().item()),
            **frac,
        }

    return _eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None,
                        help="cuda or cpu; default auto-detect")
    parser.add_argument("--ckpt", default=None,
                        help="path to save agent.pt (default: <out-dir>/agent.pt)")
    parser.add_argument("--resume-from-ckpt", default=None,
                        help="load policy weights from this path before training")
    parser.add_argument("--out-dir", default="Yuan/IJRR/runs")
    parser.add_argument("--wandb", action="store_true",
                        help="enable wandb logging")
    parser.add_argument("--wandb-project", default="NSRL-FR3")
    parser.add_argument("--wandb-run-name", default=None,
                        help="defaults to basename of --out-dir")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg_yaml = yaml.safe_load(f)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    log_path = _resolve_log_path(out_dir)
    ckpt_path = args.ckpt or str(out_dir / "agent.pt")

    env_cfg = EnvConfig(**cfg_yaml["env"])
    ppo_cfg = PPOConfig(**cfg_yaml["ppo"])
    line_cfg = cfg_yaml["line_distribution"]
    eval_cfg = cfg_yaml["eval"]
    train_cfg = cfg_yaml["train"]

    threshold_m = (float(line_cfg["feasibility_threshold_m"])
                   if line_cfg.get("feasibility_filter", False) else None)

    print(f"[train] device={device}; building train env (n_envs={env_cfg.n_envs})")
    train_env = NSRLBatchedEnv(env_cfg, line_dist=None, device=device)
    train_env.line_dist = LineDistribution.load_or_build(
        kin=train_env.kin, collision=train_env.collision,
        n_pool=line_cfg["n_pool"],
        n_target_noise_deg=line_cfg["n_target_noise_deg"],
        seed=line_cfg["train_seed"],
        env_cfg=env_cfg,
        feasibility_threshold_m=threshold_m,
        swing_max_deg=line_cfg.get("swing_max_deg", 0.0),
        wavelen_range=tuple(line_cfg.get("wavelen_range", (0.4, 1.2))),
        min_radius_m=line_cfg.get("min_radius_m", 0.15),
    )


    eval_env_cfg = EnvConfig(**{**cfg_yaml["env"], "n_envs": eval_cfg["n_holdout"]})
    print(f"[train] building eval env (n_envs={eval_env_cfg.n_envs})")
    eval_env = NSRLBatchedEnv(eval_env_cfg, line_dist=None, device=device)
    eval_env.line_dist = LineDistribution.load_or_build(
        kin=eval_env.kin, collision=eval_env.collision,
        n_pool=line_cfg["n_pool"],
        n_target_noise_deg=line_cfg["n_target_noise_deg"],
        seed=eval_cfg["holdout_seed"],
        env_cfg=env_cfg,
        feasibility_threshold_m=threshold_m,
        swing_max_deg=line_cfg.get("swing_max_deg", 0.0),
        wavelen_range=tuple(line_cfg.get("wavelen_range", (0.4, 1.2))),
        min_radius_m=line_cfg.get("min_radius_m", 0.15),
    )

    log_file = open(log_path, "w")
    t0 = time.time()

    wandb_run = None
    if args.wandb:
        import wandb
        run_name = args.wandb_run_name or out_dir.name
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            dir=str(out_dir),
            config={
                "env": cfg_yaml["env"],
                "ppo": cfg_yaml["ppo"],
                "line_distribution": cfg_yaml["line_distribution"],
                "eval": cfg_yaml["eval"],
                "train": cfg_yaml["train"],
            },
            save_code=False,
        )
        # Group eval & term_reason under their own panels by using "/" naming
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")

    def log_fn(d: dict):
        d_with_time = {"wall_s": time.time() - t0, **d}
        log_file.write(repr(d_with_time) + "\n")
        log_file.flush()
        # Concise stdout: one line per PPO update with the metrics we care about
        if "update" in d:
            print(
                f"upd {d['update']:>4}  step {d['global_step']:>9}  "
                f"r/prog {d.get('reward/progress', 0):+.3f}  "
                f"v_loss {d.get('train/v_loss', 0):.4f}  "
                f"entropy {d.get('train/entropy', 0):.2f}",
                flush=True)
        elif "eval_at_step" in d:
            print(
                f"  eval @ {d['eval_at_step']:>9}  "
                f"progress(mean {d.get('eval/mean_progress_m', 0):.3f}m  med {d.get('eval/median_progress_m', 0):.3f}m  max {d.get('eval/max_progress_m', 0):.3f}m)",
                flush=True)
        if wandb_run is not None:
            # Map both per-update and eval logs onto global_step axis
            step = d.get("global_step", d.get("eval_at_step"))
            payload = {k: v for k, v in d_with_time.items() if k != "global_step"}
            if step is not None:
                wandb_run.log({"global_step": step, **payload})
            else:
                wandb_run.log(payload)



    eval_fn = _make_eval_fn(eval_env)
    n_updates = ppo_cfg.total_timesteps // (ppo_cfg.n_steps * env_cfg.n_envs)
    print(f"[train] starting PPO: total={ppo_cfg.total_timesteps}, updates={n_updates}")
    # The admissible-command box has its optimum at the vertices, so the
    # action space can be the vertex set itself; see vertex_agent.py.
    agent_obj = None
    kind = cfg_yaml.get("agent", {}).get("kind")
    if kind == "vertex":
        agent_obj = VertexAgent(obs_dim=train_env.obs_dim,
                                act_dim=train_env.act_dim,
                                hidden_dim=ppo_cfg.hidden_dim).to(device)
        print(f"[train] vertex action space: {agent_obj.n_actions} actions")
    elif kind == "vertex_prior":
        agent_obj = PriorVertexAgent(obs_dim=train_env.obs_dim,
                                     act_dim=train_env.act_dim,
                                     hidden_dim=ppo_cfg.hidden_dim).to(device)
        print(f"[train] vertex-prior action space: "
              f"{agent_obj.n_actions} actions, alpha init "
              f"{float(agent_obj.alpha):.1f}")

    agent = ppo_train(ppo_cfg, train_env, device=device, agent=agent_obj,
                      eval_fn=eval_fn,
                      eval_every=train_cfg["eval_every"],
                      log_fn=log_fn,
                      ckpt_path=ckpt_path,
                      ckpt_every_n_updates=train_cfg.get("ckpt_every_n_updates", 10),
                      resume_from_ckpt=args.resume_from_ckpt)
    log_file.close()
    if wandb_run is not None:
        wandb_run.finish()
    print(f"[train] done, ckpt → {ckpt_path}")


if __name__ == "__main__":
    main()

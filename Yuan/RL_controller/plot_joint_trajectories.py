"""Save joint-trajectory plots: RL vs classical controller on the same line.

For each of `--n-lines` sampled line specs, runs both controllers (no auto-reset),
records q_i(t) per joint, plots 7 subplots (one per joint) with both controllers
overlaid, JL bands as dashed horizontals, and termination markers.

X-axis is real time in seconds (= step · dt).

Usage:
    python -m Yuan.RL_controller.plot_joint_trajectories \\
        --config Yuan/RL_controller/config.yaml \\
        --ckpt   Yuan/RL_controller/runs11/agent.pt \\
        --n-lines 4 --seed 42 \\
        --out-dir Yuan/RL_controller/runs11/joint_traj
"""
from __future__ import annotations

# Self-relaunch with LD_LIBRARY_PATH
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
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; save to file only
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from Yuan.RL_controller.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.RL_controller.env.line_distribution import LineDistribution
from Yuan.RL_controller.env.classical_nullspace import (
    ClassicalNullspaceController, cn_action_fn,
)
from Yuan.RL_controller.ppo import Agent


# ----- CLI ------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--ckpt", required=True, help="RL agent ckpt")
parser.add_argument("--device", default="cpu")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--n-lines", type=int, default=4)
parser.add_argument("--out-dir", default=None,
                    help="output dir; defaults to <ckpt_dir>/joint_traj")
args = parser.parse_args()

with open(args.config) as f:
    cfg_yaml = yaml.safe_load(f)
device = torch.device(args.device)

out_dir = Path(args.out_dir) if args.out_dir else Path(args.ckpt).parent / "joint_traj"
out_dir.mkdir(parents=True, exist_ok=True)

# Pre-sample n_lines specs (one env to draw from)
env_cfg = EnvConfig(**{**cfg_yaml["env"], "n_envs": 1})
env_a = NSRLBatchedEnv(env_cfg, line_dist=None, device=device)
sampler = LineDistribution(
    kin=env_a.kin, collision=env_a.collision,
    n_pool=max(args.n_lines * 10, 200),
    n_target_noise_deg=cfg_yaml["line_distribution"]["n_target_noise_deg"],
    seed=args.seed,
)
if cfg_yaml["line_distribution"].get("feasibility_filter", False):
    sampler.filter_by_classical_controller(
        env_cfg,
        threshold_m=float(cfg_yaml["line_distribution"]["feasibility_threshold_m"]),
        verbose=False)
specs = sampler.sample(args.n_lines)
print(f"[plot] sampled {args.n_lines} line specs (seed={args.seed})")

# Build controllers
rl_agent = Agent(env_a.obs_dim, env_a.act_dim,
                 hidden_dim=cfg_yaml["ppo"]["hidden_dim"],
                 init_log_std=cfg_yaml["ppo"]["init_log_std"]).to(device)
rl_agent.load_state_dict(torch.load(args.ckpt, map_location=device))
rl_agent.eval()
cl_ctrl = ClassicalNullspaceController(env_a.kin)
cl_action_fn = cn_action_fn(cl_ctrl)


@torch.no_grad()
def _rl_action(env_):
    return rl_agent.actor_mean(env_.current_obs()).clamp(-1.0, 1.0)


def _rollout_one(env, action_fn, q0, line_dir, n_target, max_steps):
    """Returns (q_record (T+1, 7), term_step, term_reason)."""
    env.q[:] = q0.unsqueeze(0)
    env.line_dir[:] = line_dir.unsqueeze(0)
    env.n_target[:] = n_target.unsqueeze(0)
    env.t.zero_()
    env.a_prev.zero_()
    env.B_prev_valid.zero_()
    env.done_persistent.zero_()
    qs = [env.q[0].cpu().numpy().copy()]
    term_step, term_reason = None, None
    for step in range(max_steps):
        a = action_fn(env)
        _, _, _, _, info = env.step(a, auto_reset=False)
        qs.append(env.q[0].cpu().numpy().copy())
        if bool(env.done_persistent[0].item()):
            term_step = step + 1
            tr = int(info["term_reason"][0].item())
            term_reason = {0: "alive", 2: "collision", 3: "cone",
                           4: "jl", 5: "truncated"}.get(tr, "?")
            break
    return np.stack(qs, axis=0), term_step, term_reason


# Two envs: one each so internal state isolates per controller
env_rl = NSRLBatchedEnv(env_cfg, line_dist=None, device=device)
env_cl = NSRLBatchedEnv(env_cfg, line_dist=None, device=device)

# Joint limits for reference horizontals
lmt_lo = env_a.kin.lmt_lo.cpu().numpy()
lmt_up = env_a.kin.lmt_up.cpu().numpy()
joint_labels = [f"q{i+1}" for i in range(7)]

dt = float(env_cfg.dt)
max_steps = int(env_cfg.max_steps)

for i in range(args.n_lines):
    q0 = specs["q0"][i]
    u_hat = specs["line_dir"][i]
    n_target = specs["n_target"][i]

    q_rl, t_rl, r_rl = _rollout_one(env_rl, _rl_action, q0, u_hat, n_target, max_steps)
    q_cl, t_cl, r_cl = _rollout_one(env_cl, cl_action_fn, q0, u_hat, n_target, max_steps)

    times_rl = np.arange(q_rl.shape[0]) * dt
    times_cl = np.arange(q_cl.shape[0]) * dt

    fig, axes = plt.subplots(7, 1, figsize=(10, 12), sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(times_rl, q_rl[:, j], color="#E74C3C", lw=1.5,
                label=f"RL ({t_rl} steps, {r_rl})" if j == 0 else None)
        ax.plot(times_cl, q_cl[:, j], color="#3498DB", lw=1.5,
                label=f"classical ({t_cl} steps, {r_cl})" if j == 0 else None)
        ax.axhline(lmt_lo[j], color="gray", ls="--", lw=0.6, alpha=0.5)
        ax.axhline(lmt_up[j], color="gray", ls="--", lw=0.6, alpha=0.5)
        # Termination markers
        if t_rl is not None:
            ax.axvline(t_rl * dt, color="#E74C3C", ls=":", lw=0.8, alpha=0.6)
        if t_cl is not None:
            ax.axvline(t_cl * dt, color="#3498DB", ls=":", lw=0.8, alpha=0.6)
        ax.set_ylabel(joint_labels[j])
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(
        f"line {i+1}/{args.n_lines}   "
        f"u_hat={u_hat.tolist()}   n_target={n_target.tolist()}",
        fontsize=9, y=0.995)
    fig.tight_layout()

    out_path = out_dir / f"line_{i+1:02d}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] line {i+1}/{args.n_lines}  RL={t_rl}({r_rl})  "
          f"CL={t_cl}({r_cl})  → {out_path}")

print(f"\n[plot] done. {args.n_lines} figures saved under {out_dir}")

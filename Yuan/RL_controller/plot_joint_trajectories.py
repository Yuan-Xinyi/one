"""Plot per-joint + TCP-xyz trajectories for all eval cases: RL vs classical vs GPM.

Each eval case (one holdout line) yields one PNG with two columns:
    Left  — 7 stacked subplots, one per joint (q1..q7), joint limits as
            dotted grey reference lines.
    Right — 3 stacked subplots for TCP x/y/z (meters).

All three controllers overlay on every subplot:
    - RL policy (solid)
    - Classical 4-term nullspace baseline (dashed)
    - GPM-JL baseline (dashed)

X-axis is the env step index (each controller terminates independently, so
the three curves typically have different lengths).

Eval cases match `eval.py` exactly: same holdout_seed + same
ScriptedLineDistribution slice, so case i here = line i in eval.csv.

Caching:
    Rollouts (qs + ep_len + term_reason + tcp xyz) are cached to
    `<ckpt_dir>/joint_tcp_cache.npz`. If only the older qs-only cache
    `joint_traj_cache.npz` exists, this script bootstraps TCP from it via
    a single FK pass — no re-roll. Pass --force to re-roll from scratch.

Usage:
    python -m Yuan.RL_controller.plot_joint_trajectories \\
        --config Yuan/RL_controller/runs/framing_b_pd_10M_v4/config.yaml \\
        --ckpt   Yuan/RL_controller/runs/framing_b_pd_10M_v4/agent.pt
"""
from __future__ import annotations

# Self-relaunch with $CONDA_PREFIX/lib on LD_LIBRARY_PATH (matches train/eval).
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from Yuan.RL_controller.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.RL_controller.env.line_distribution import (
    LineDistribution, ScriptedLineDistribution,
)
from Yuan.RL_controller.env.baseline_controller import (
    GPMBaselineController, baseline_action_fn,
)
from Yuan.RL_controller.env.classical_nullspace import (
    ClassicalNullspaceController, cn_action_fn,
)
from Yuan.RL_controller.ppo import Agent


TERM_NAMES = {0: "alive", 2: "collision", 3: "cone", 4: "jl",
              5: "truncated", 6: "lateral"}
TERM_TRUNCATED = 5


def _rl_action_fn(agent: Agent):
    @torch.no_grad()
    def _fn(env: NSRLBatchedEnv) -> torch.Tensor:
        return agent.actor_mean(env.current_obs()).clamp(-1.0, 1.0)
    return _fn


@torch.no_grad()
def rollout_record(env: NSRLBatchedEnv, action_fn, max_steps: int):
    """Roll one episode per env (auto_reset=False), recording q after each step.

    Returns:
        qs:          (max_steps + 1, n_envs, 7) — qs[0] is reset state;
                     qs[t+1] is q after the t-th step. Frozen at the terminal
                     value for envs that finished early (env freezes done envs).
        ep_len:      (n_envs,) int — number of steps each env took.
        term_reason: (n_envs,) int — TERM_* code per env.
    """
    n = env.n_envs
    qs = np.zeros((max_steps + 1, n, 7), dtype=np.float32)
    ep_len = np.full(n, -1, dtype=np.int64)
    term_reason = np.full(n, -1, dtype=np.int64)
    env.reset()
    qs[0] = env.q.cpu().numpy()
    finished = torch.zeros(n, dtype=torch.bool, device=env.device)
    for step_i in range(max_steps):
        a = action_fn(env)
        _, _, _, _, info = env.step(a, auto_reset=False)
        qs[step_i + 1] = env.q.cpu().numpy()
        new_done = info["episode_done"]
        if new_done.any():
            mask = new_done.cpu().numpy()
            ep_len[mask] = step_i + 1
            term_reason[mask] = info["term_reason"].cpu().numpy()[mask]
            finished = finished | new_done
        if bool(env.done_persistent.all().item()):
            qs[step_i + 2:] = qs[step_i + 1]
            break
    not_done = ~finished.cpu().numpy()
    ep_len[not_done] = max_steps
    term_reason[not_done] = TERM_TRUNCATED
    return qs, ep_len, term_reason


def _build_env(cfg_yaml, n_envs, device):
    env_cfg = EnvConfig(**{**cfg_yaml["env"], "n_envs": n_envs})
    env = NSRLBatchedEnv(env_cfg, line_dist=None, device=device)
    return env, env_cfg


def _scripted(holdout):
    return ScriptedLineDistribution({k: v.clone() for k, v in holdout.items()})


@torch.no_grad()
def compute_tcp_positions(qs: np.ndarray, kin, device, batch: int = 8192) -> np.ndarray:
    """qs (T+1, n, 7) → tcp_xyz (T+1, n, 3) via batched FK."""
    qs_t = torch.from_numpy(qs).to(device=device, dtype=kin.dtype)
    T_plus, n, _ = qs_t.shape
    flat = qs_t.reshape(-1, 7)
    chunks = []
    for i in range(0, flat.shape[0], batch):
        p, _, _, _ = kin.tcp_fk_jac(flat[i:i + batch])
        chunks.append(p.cpu().numpy())
    return np.concatenate(chunks, axis=0).reshape(T_plus, n, 3)


def run_all_controllers(cfg_yaml, ckpt_path: str, device, n_holdout: int):
    """Mirror eval.py setup exactly so case ids align with eval.csv."""
    proxy, env_cfg = _build_env(cfg_yaml, n_holdout, device)
    line_cfg = cfg_yaml["line_distribution"]
    eval_cfg = cfg_yaml["eval"]
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
    max_steps = env_cfg.max_steps

    lmt_lo = proxy.lmt_lo.cpu().numpy()
    lmt_up = proxy.lmt_up.cpu().numpy()

    # RL — reuses proxy env (matches eval.py).
    rl_env = proxy
    rl_env.line_dist = _scripted(holdout)
    agent = Agent(rl_env.obs_dim, rl_env.act_dim,
                  hidden_dim=cfg_yaml["ppo"]["hidden_dim"],
                  init_log_std=cfg_yaml["ppo"]["init_log_std"]).to(device)
    agent.load_state_dict(torch.load(ckpt_path, map_location=device))
    agent.eval()
    print(f"[plot] rolling RL on {n_holdout} cases (max_steps={max_steps})...")
    rl = rollout_record(rl_env, _rl_action_fn(agent), max_steps)

    # Classical 4-term nullspace (hand-tuned).
    cls_env, _ = _build_env(cfg_yaml, n_holdout, device)
    cls_env.line_dist = _scripted(holdout)
    ctrl = ClassicalNullspaceController(cls_env.kin)
    print(f"[plot] rolling classical on {n_holdout} cases...")
    cls_ = rollout_record(cls_env, cn_action_fn(ctrl), max_steps)

    # GPM-JL.
    gpm_env, _ = _build_env(cfg_yaml, n_holdout, device)
    gpm_env.line_dist = _scripted(holdout)
    gpm = GPMBaselineController(
        gpm_env.kin,
        k_jl=cfg_yaml["baseline"]["k_jl"],
        k_dm=float(cfg_yaml["baseline"].get("k_dm", 0.0)),
        manip_damping=float(cfg_yaml["baseline"].get("manip_damping", 1e-3)),
    )
    print(f"[plot] rolling GPM on {n_holdout} cases...")
    gpm_ = rollout_record(gpm_env, baseline_action_fn(gpm), max_steps)

    data = {
        "rl":        {"qs": rl[0],   "ep_len": rl[1],   "term_reason": rl[2]},
        "classical": {"qs": cls_[0], "ep_len": cls_[1], "term_reason": cls_[2]},
        "gpm":       {"qs": gpm_[0], "ep_len": gpm_[1], "term_reason": gpm_[2]},
    }
    return data, lmt_lo, lmt_up


STYLES = {
    "rl":        {"color": "#1f77b4", "ls": "-",  "lw": 1.8, "label": "RL (ours)"},
    "classical": {"color": "#d62728", "ls": "--", "lw": 1.3, "label": "Classical (hand-tuned)"},
    "gpm":       {"color": "#2ca02c", "ls": "--", "lw": 1.3, "label": "GPM-JL"},
}


def plot_case(case_id: int, data: dict, lmt_lo, lmt_up, out_path: Path):
    """Left col: 7 joint subplots; right col: 3 TCP xyz subplots.

    Heights align via a 21-row gridspec: 7×3 (joints) vs 3×7 (TCP), so
    both columns span the same vertical extent.
    """
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(21, 2, hspace=0.4, wspace=0.18,
                          left=0.06, right=0.98, top=0.94, bottom=0.05)

    joint_axes = []
    for j in range(7):
        sharex = joint_axes[0] if j > 0 else None
        ax = fig.add_subplot(gs[j * 3:(j + 1) * 3, 0], sharex=sharex)
        if j < 6:
            ax.tick_params(labelbottom=False)
        joint_axes.append(ax)

    tcp_axes = []
    for k in range(3):
        sharex = tcp_axes[0] if k > 0 else None
        ax = fig.add_subplot(gs[k * 7:(k + 1) * 7, 1], sharex=sharex)
        if k < 2:
            ax.tick_params(labelbottom=False)
        tcp_axes.append(ax)

    for name, st in STYLES.items():
        d = data[name]
        T = int(d["ep_len"][case_id])
        xs = np.arange(T + 1)
        for j in range(7):
            joint_axes[j].plot(xs, d["qs"][:T + 1, case_id, j], **st)
        for k in range(3):
            tcp_axes[k].plot(xs, d["tcp"][:T + 1, case_id, k], **st)

    for j, ax in enumerate(joint_axes):
        ax.axhline(lmt_lo[j], color="grey", lw=0.6, ls=":")
        ax.axhline(lmt_up[j], color="grey", lw=0.6, ls=":")
        ax.set_ylabel(f"q{j+1} (rad)")
        ax.grid(True, alpha=0.3)

    for k, ax in enumerate(tcp_axes):
        ax.set_ylabel(f"tcp {'xyz'[k]} (m)")
        ax.grid(True, alpha=0.3)

    joint_axes[-1].set_xlabel("step")
    tcp_axes[-1].set_xlabel("step")
    joint_axes[0].legend(loc="upper right", fontsize=8)

    title_bits = [f"case {case_id}"]
    for name, d in data.items():
        T = int(d["ep_len"][case_id])
        reason = TERM_NAMES.get(int(d["term_reason"][case_id]), "?")
        title_bits.append(f"{name} T={T} ({reason})")
    fig.suptitle(" | ".join(title_bits), fontsize=10)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default=None,
                   help="default: <ckpt_dir>/joint_tcp_plots/")
    p.add_argument("--cache", default=None,
                   help="default: <ckpt_dir>/joint_tcp_cache.npz (includes TCP); "
                        "joint_traj_cache.npz (qs-only) is auto-bootstrapped if present")
    p.add_argument("--force", action="store_true",
                   help="ignore cache and re-roll from scratch")
    p.add_argument("--max-cases", type=int, default=None,
                   help="default: all eval.n_holdout cases")
    args = p.parse_args()

    device = (torch.device(args.device) if args.device is not None
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    with open(args.config, "r") as f:
        cfg_yaml = yaml.safe_load(f)

    ckpt_path = Path(args.ckpt)
    out_dir = (Path(args.out_dir) if args.out_dir
               else ckpt_path.parent / "joint_tcp_plots")
    cache = (Path(args.cache) if args.cache
             else ckpt_path.parent / "joint_tcp_cache.npz")
    legacy_cache = ckpt_path.parent / "joint_traj_cache.npz"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_holdout = cfg_yaml["eval"]["n_holdout"]

    def _save_cache(path, data, lmt_lo, lmt_up):
        np.savez(
            path,
            rl_qs=data["rl"]["qs"], rl_ep_len=data["rl"]["ep_len"],
            rl_term=data["rl"]["term_reason"], rl_tcp=data["rl"]["tcp"],
            classical_qs=data["classical"]["qs"],
            classical_ep_len=data["classical"]["ep_len"],
            classical_term=data["classical"]["term_reason"],
            classical_tcp=data["classical"]["tcp"],
            gpm_qs=data["gpm"]["qs"], gpm_ep_len=data["gpm"]["ep_len"],
            gpm_term=data["gpm"]["term_reason"], gpm_tcp=data["gpm"]["tcp"],
            lmt_lo=lmt_lo, lmt_up=lmt_up,
        )

    if cache.exists() and not args.force:
        print(f"[plot] loading cached trajectories+tcp from {cache}")
        npz = np.load(cache)
        data = {name: {"qs": npz[f"{name}_qs"],
                       "ep_len": npz[f"{name}_ep_len"],
                       "term_reason": npz[f"{name}_term"],
                       "tcp": npz[f"{name}_tcp"]}
                for name in ("rl", "classical", "gpm")}
        lmt_lo = npz["lmt_lo"]
        lmt_up = npz["lmt_up"]
    elif legacy_cache.exists() and not args.force:
        print(f"[plot] bootstrapping from qs-only cache {legacy_cache} "
              f"(computing TCP via FK; no re-roll)")
        npz = np.load(legacy_cache)
        data = {name: {"qs": npz[f"{name}_qs"],
                       "ep_len": npz[f"{name}_ep_len"],
                       "term_reason": npz[f"{name}_term"]}
                for name in ("rl", "classical", "gpm")}
        lmt_lo = npz["lmt_lo"]
        lmt_up = npz["lmt_up"]
        proxy, _ = _build_env(cfg_yaml, 1, device)
        for name in data:
            data[name]["tcp"] = compute_tcp_positions(
                data[name]["qs"], proxy.kin, device)
        _save_cache(cache, data, lmt_lo, lmt_up)
        print(f"[plot] cached trajectories+tcp → {cache}")
    else:
        data, lmt_lo, lmt_up = run_all_controllers(
            cfg_yaml, str(ckpt_path), device, n_holdout)
        proxy, _ = _build_env(cfg_yaml, 1, device)
        for name in data:
            data[name]["tcp"] = compute_tcp_positions(
                data[name]["qs"], proxy.kin, device)
        _save_cache(cache, data, lmt_lo, lmt_up)
        print(f"[plot] cached trajectories+tcp → {cache}")

    n_plot = (n_holdout if args.max_cases is None
              else min(args.max_cases, n_holdout))
    print(f"[plot] rendering {n_plot} cases → {out_dir}")
    for i in range(n_plot):
        plot_case(i, data, lmt_lo, lmt_up, out_dir / f"case_{i:03d}.png")
        if (i + 1) % 20 == 0 or (i + 1) == n_plot:
            print(f"[plot]   {i+1}/{n_plot}")
    print("[plot] done.")


if __name__ == "__main__":
    main()

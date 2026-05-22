"""Plot the joint trajectories for a "hybrid rescue" example task.

Picks one task from the cache where hybrid (te=0.99, tx=0.90) clearly beats
both pure RL and pure Classical, then re-runs the hybrid policy on JUST that
task to recover q_traj, and plots all three controllers' joint trajectories.

Usage:
    python -m Yuan.RL_controller.plot_hybrid_rescue \\
        --ckpt-dir Yuan/RL_controller/runs/p0_progress_only_30M_0520 \\
        --cache    Yuan/RL_controller/runs/p0_progress_only_30M_0520/diag_10000_classical/rollouts.npz \\
        --task-id 8316 --te 0.99 --tx 0.90 --out plot.png
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
import dataclasses
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Yuan.RL_controller.env.env import (
    NSRLBatchedEnv, EnvConfig, TERM_NAMES, TERM_TRUNCATED,
    build_task_aligned_basis,
)
from Yuan.RL_controller.env.line_distribution import ScriptedLineDistribution
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.RL_controller.ppo import Agent


def run_hybrid_one_task(env, agent, classical_ctrl, te, tx):
    """Single-batch hybrid rollout (n_envs already set to 1 by caller).
    Returns dict with q_traj (T+1, 7), episode_len (int), term_reason (int),
    switch_steps (list), using_rl_trace (T+1,) bool, q_ref_trace (T+1, 7)."""
    env.reset()
    q_mid = env.q_mid
    q_half = env.q_half

    def _max_abs_qn(q):
        return ((q - q_mid).abs() / q_half).max(dim=-1).values

    max_steps = env.max_steps
    q_traj = torch.zeros((max_steps + 1, 7), dtype=env.kin.dtype, device=env.device)
    using_rl_trace = torch.zeros((max_steps + 1,), dtype=torch.bool, device=env.device)
    q_ref_trace = torch.zeros((max_steps + 1, 7), dtype=env.kin.dtype, device=env.device)
    q_traj[0] = env.q[0]

    init_max_qn = _max_abs_qn(env.q)
    using_rl = init_max_qn < te
    q_ref = env.q.clone()
    using_rl_trace[0] = using_rl[0]
    q_ref_trace[0] = q_ref[0]
    switch_steps = []

    episode_len = -1
    episode_term = -1
    finished = False

    for step_i in range(max_steps):
        cur_max_qn = _max_abs_qn(env.q)
        new_using_rl = torch.where(using_rl,
                                   cur_max_qn < te,
                                   cur_max_qn < tx)
        if (new_using_rl != using_rl).any():
            switch_steps.append((step_i, bool(using_rl[0].item()), bool(new_using_rl[0].item())))
        rl_to_cls = using_rl & (~new_using_rl)
        if rl_to_cls.any():
            q_ref = torch.where(rl_to_cls.unsqueeze(-1), env.q, q_ref)
        using_rl = new_using_rl

        obs = env.current_obs()
        with torch.no_grad():
            rl_act = agent.actor_mean(obs).clamp(-1.0, 1.0)
            B_basis, _ = build_task_aligned_basis(
                env.kin, env.q, env.line_dir, env.n_target,
                env.kin.q_mid, env.q_half, env.cfg.manip_damping,
            )
        q_dot_raw = classical_ctrl.q_dot_null(env.q, env.line_dir, env.n_target, q_ref)
        with torch.no_grad():
            cls_act = (B_basis.transpose(-1, -2)
                       @ q_dot_raw.unsqueeze(-1)).squeeze(-1)
            cls_act = (cls_act / env.a_max).clamp(-1.0, 1.0)

        action = torch.where(using_rl.unsqueeze(-1), rl_act, cls_act)
        _, _, _, _, info = env.step(action, auto_reset=False)
        q_traj[step_i + 1] = env.q[0]
        using_rl_trace[step_i + 1] = using_rl[0]
        q_ref_trace[step_i + 1] = q_ref[0]

        new_done = info["episode_done"]
        if new_done.any() and not finished:
            episode_len = int(env.t[0].item())
            episode_term = int(info["term_reason"][0].item())
            finished = True
        if bool(env.done_persistent.all().item()):
            break

    if not finished:
        episode_len = int(env.t[0].item())
        episode_term = TERM_TRUNCATED

    return {
        "q_traj": q_traj.cpu().numpy(),
        "episode_len": episode_len,
        "term_reason": episode_term,
        "switch_steps": switch_steps,
        "using_rl_trace": using_rl_trace.cpu().numpy(),
        "q_ref_trace": q_ref_trace.cpu().numpy(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--te", type=float, default=0.99)
    parser.add_argument("--tx", type=float, default=0.90)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    cfg_path = ckpt_dir / "config.yaml"
    ckpt_path = ckpt_dir / "agent.pt"
    device = (torch.device(args.device) if args.device is not None
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    with open(cfg_path) as f:
        cfg_yaml = yaml.safe_load(f)

    cache = np.load(args.cache, allow_pickle=True)
    q0 = cache["q0"][args.task_id]
    line_dir = cache["line_dir"][args.task_id]
    n_target = cache["n_target"][args.task_id]
    q_traj_rl = cache["q_traj_rl"][:, args.task_id, :]      # (T+1, 7)
    q_traj_base = cache["q_traj_base"][:, args.task_id, :]  # (T+1, 7)
    T_rl = int(cache["episode_len_rl"][args.task_id])
    T_base = int(cache["episode_len_base"][args.task_id])
    term_rl = int(cache["term_reason_rl"][args.task_id])
    term_base = int(cache["term_reason_base"][args.task_id])
    dt = float(cache["dt"])

    # Build env with n_envs=1.
    valid_keys = {f.name for f in dataclasses.fields(EnvConfig)}
    env_kw = {k: v for k, v in cfg_yaml["env"].items() if k in valid_keys}
    env_cfg = EnvConfig(**{**env_kw, "n_envs": 1})
    env = NSRLBatchedEnv(env_cfg, line_dist=None, device=device)
    env.line_dist = ScriptedLineDistribution({
        "q0": torch.from_numpy(q0[None, :]).to(env.kin.dtype).to(device),
        "line_dir": torch.from_numpy(line_dir[None, :]).to(env.kin.dtype).to(device),
        "n_target": torch.from_numpy(n_target[None, :]).to(env.kin.dtype).to(device),
    })

    agent = Agent(env.obs_dim, env.act_dim,
                  hidden_dim=cfg_yaml["ppo"]["hidden_dim"],
                  init_log_std=cfg_yaml["ppo"]["init_log_std"]).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    agent.load_state_dict(state_dict)
    agent.eval()
    classical_ctrl = ClassicalNullspaceController(env.kin)

    print(f"[plot] running hybrid (te={args.te}, tx={args.tx}) on task {args.task_id}")
    hyb = run_hybrid_one_task(env, agent, classical_ctrl, args.te, args.tx)
    q_traj_hyb = hyb["q_traj"]
    T_hyb = hyb["episode_len"]
    term_hyb = hyb["term_reason"]
    print(f"[plot] T_rl={T_rl} ({TERM_NAMES.get(term_rl)})  "
          f"T_cls={T_base} ({TERM_NAMES.get(term_base)})  "
          f"T_hyb={T_hyb} ({TERM_NAMES.get(term_hyb)})  "
          f"switches={len(hyb['switch_steps'])}: {hyb['switch_steps']}")

    # --- Joint normalization ---
    q_mid = env.q_mid.cpu().numpy()
    q_half = env.q_half.cpu().numpy()
    def qn(q): return (q - q_mid) / q_half  # (..., 7)

    qn_rl = qn(q_traj_rl)
    qn_base = qn(q_traj_base)
    qn_hyb = qn(q_traj_hyb)

    # Stop curves at termination step so post-freeze flat segments don't mislead.
    def trim(arr, T):
        return arr[:T+1]

    qn_rl_t = trim(qn_rl, T_rl)
    qn_base_t = trim(qn_base, T_base)
    qn_hyb_t = trim(qn_hyb, T_hyb)
    using_rl_trace_t = hyb["using_rl_trace"][:T_hyb+1]

    # --- Single max|qn| panel ---
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 24,
        "axes.titlesize": 24,
        "axes.labelsize": 24,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 24,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#444",
        "axes.labelcolor": "#222",
        "xtick.color": "#444", "ytick.color": "#444",
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.7,
    })
    fig, ax_top = plt.subplots(figsize=(18, 10), facecolor="white")

    t_rl_axis = np.arange(T_rl + 1) * dt
    t_base_axis = np.arange(T_base + 1) * dt
    t_hyb_axis = np.arange(T_hyb + 1) * dt
    sw = hyb["switch_steps"]
    t_max = T_hyb * dt

    # Palette
    C_RL = "#d62728"; C_CLS = "#1f77b4"; C_HYB = "#2ca02c"
    C_TAU = "#ff7f0e"

    # ---- TOP: max|qn| panel ----
    mqn_rl = np.max(np.abs(qn_rl_t), axis=-1)
    mqn_cls = np.max(np.abs(qn_base_t), axis=-1)
    mqn_hyb = np.max(np.abs(qn_hyb_t), axis=-1)

    # Shade Classical-takeover regions on hybrid
    mask_cls = ~using_rl_trace_t
    if mask_cls.any():
        ax_top.fill_between(t_hyb_axis, 0, 1.05,
                            where=mask_cls, color=C_HYB, alpha=0.10,
                            step="mid", linewidth=0, zorder=0,
                            label="Hybrid in Classical mode")

    ax_top.axhline(1.0, color="#555", linestyle="-", lw=1.0, alpha=0.5,
                   label="JL limit (|q_norm| = 1)")
    if abs(args.te - args.tx) < 1e-6:
        ax_top.axhline(args.te, color=C_TAU, linestyle="--", lw=1.8,
                       label=f"τ = {args.te:.2f}")
    else:
        ax_top.axhline(args.te, color=C_TAU, linestyle="--", lw=1.8,
                       label=f"τ_enter = {args.te:.2f}")
        ax_top.axhline(args.tx, color="#9467bd", linestyle="--", lw=1.4,
                       label=f"τ_exit = {args.tx:.2f}")

    ax_top.plot(t_hyb_axis, mqn_hyb, color=C_HYB, lw=2.8, alpha=0.85,
                zorder=3, label=f"Hybrid  (T={T_hyb}, {TERM_NAMES.get(term_hyb)})")
    ax_top.plot(t_base_axis, mqn_cls, color=C_CLS, lw=1.8, alpha=0.85,
                zorder=4, label=f"Classical  (T={T_base}, {TERM_NAMES.get(term_base)})")
    ax_top.plot(t_rl_axis, mqn_rl, color=C_RL, lw=1.8, alpha=0.95,
                zorder=5, label=f"RL  (T={T_rl}, {TERM_NAMES.get(term_rl)})")

    # Terminal markers (RL / Cls only — hybrid endpoint is line tip)
    ax_top.scatter([T_rl * dt], [mqn_rl[T_rl]], color=C_RL, marker="X",
                   s=110, zorder=10, edgecolors="white", linewidths=1.5)
    ax_top.scatter([T_base * dt], [mqn_cls[T_base]], color=C_CLS, marker="X",
                   s=110, zorder=10, edgecolors="white", linewidths=1.5)
    # Switch markers on hybrid curve.
    for (s, was_rl, now_rl) in sw:
        if s > T_hyb:
            continue
        ax_top.scatter([s * dt], [mqn_hyb[s]], color="black", marker="o",
                       s=90, zorder=11, edgecolors="white", linewidths=1.5)
    # Single annotation for the (possibly multi-step) Classical handoff window.
    cls_steps = [s for (s, was_rl, _) in sw if was_rl] + \
                [s for (s, was_rl, _) in sw if not was_rl]
    if sw:
        s0 = sw[0][0]
        s1 = sw[-1][0]
        label = (f"Classical handoff\nstep {s0}–{s1}"
                 if s1 > s0 else f"Classical handoff\nstep {s0}")
        x_anchor = ((s0 + s1) / 2) * dt
        y_anchor = max(mqn_hyb[s0], mqn_hyb[s1])
        ax_top.annotate(label,
                        (x_anchor, y_anchor),
                        xytext=(40, 30),
                        textcoords="offset points",
                        fontsize=22, color="#222", ha="left", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.35",
                                  fc="white", ec="#666", lw=0.8, alpha=0.95),
                        arrowprops=dict(arrowstyle="-",
                                        color="#555", lw=1.0))

    ax_top.set_xlim(0, t_max * 1.02)
    ax_top.set_ylim(0, 1.08)
    ax_top.set_xlabel("time (s)")
    ax_top.set_ylabel("max |q_norm|")
    ax_top.legend(loc="lower right", framealpha=0.92,
                  ncol=2, columnspacing=1.2)

    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"[plot] saved → {args.out}")


if __name__ == "__main__":
    main()

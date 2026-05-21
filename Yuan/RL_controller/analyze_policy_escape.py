"""Test policy suboptimality: at t=0, does the trained policy pick the
nullspace action that escapes j* maximally, or does it under-use that DOF?

At t=0, escape rate on j* from a chosen action a ∈ [-1,1]^4 is
    e_rate(a) = e·(J_p^+ v u_hat)[j*]   +   e·(B_basis[j*, :] @ a) · a_max
                ────────task term────       ────────policy nullspace contribution────

Compare:
    e_policy   — produced by actor_mean(obs) at the t=0 state
    e_optimal  — produced by a_opt = e · sign(B_basis[j*, :])  (= max along escape dir)
    e_classical — produced by ClassicalNullspaceController hand-tuned 4-term action

Usage:
    python -m Yuan.RL_controller.analyze_policy_escape \\
        --diag-dir Yuan/RL_controller/runs/p0_progress_only_30M_0520/diag_5000_classical
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from Yuan.RL_controller.env.env import (
    NSRLBatchedEnv, EnvConfig, damped_pinv, build_task_aligned_basis,
)
from Yuan.RL_controller.env.line_distribution import ScriptedLineDistribution
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.RL_controller.ppo import Agent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--diag-dir", required=True)
    p.add_argument("--near-thr", type=float, default=0.9)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    diag_dir = Path(args.diag_dir)
    npz = np.load(diag_dir / "rollouts.npz")
    q0_np = npz["q0"]
    line_dir_np = npz["line_dir"]
    n_target_np = npz["n_target"]
    rl_len = npz["episode_len_rl"]
    base_len = npz["episode_len_base"]

    ckpt_dir = diag_dir.parent
    with open(ckpt_dir / "config.yaml") as f:
        cfg_yaml = yaml.safe_load(f)
    env_cfg_dict = cfg_yaml["env"]
    v = float(env_cfg_dict["v"])
    a_max = float(env_cfg_dict["a_max"])
    lambda_0 = float(env_cfg_dict["lambda_0"])
    sigma_thr = float(env_cfg_dict["sigma_thr"])
    manip_damping = float(env_cfg_dict["manip_damping"])
    dt = float(env_cfg_dict["dt"])

    device = (torch.device(args.device) if args.device is not None
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    N = q0_np.shape[0]

    valid_keys = {f.name for f in dataclasses.fields(EnvConfig)}
    env_kw = {k: vv for k, vv in env_cfg_dict.items() if k in valid_keys}
    env_cfg = EnvConfig(**{**env_kw, "n_envs": N})
    env = NSRLBatchedEnv(env_cfg, line_dist=None, device=device)
    scripted = ScriptedLineDistribution({
        "q0": torch.from_numpy(q0_np).to(device=device, dtype=env.kin.dtype),
        "line_dir": torch.from_numpy(line_dir_np).to(device=device, dtype=env.kin.dtype),
        "n_target": torch.from_numpy(n_target_np).to(device=device, dtype=env.kin.dtype),
    })
    env.line_dist = scripted
    env.reset()

    # Load agent
    agent = Agent(env.obs_dim, env.act_dim,
                  hidden_dim=cfg_yaml["ppo"]["hidden_dim"],
                  init_log_std=cfg_yaml["ppo"]["init_log_std"]).to(device)
    agent.load_state_dict(torch.load(ckpt_dir / "agent.pt", map_location=device))
    agent.eval()

    # Policy action at t=0
    with torch.no_grad():
        a_policy = agent.actor_mean(env.current_obs()).clamp(-1.0, 1.0)   # (N, 4)

    # FK + Jacobian + basis at t=0
    q = env.q
    u_hat = env.line_dir
    n_target = env.n_target
    _, _, J, _ = env.kin.tcp_fk_jac(q)
    J_p = J[:, :3, :]
    J_plus, _ = damped_pinv(J_p, lambda_0, sigma_thr)
    B_basis, _ = build_task_aligned_basis(
        env.kin, q, u_hat, n_target, env.kin.q_mid, env.q_half, manip_damping,
    )

    q_mid = env.kin.q_mid
    q_half = env.q_half
    qn = (q - q_mid) / q_half
    j_star = qn.abs().argmax(dim=-1)
    rows = torch.arange(N, device=device)
    qn_jstar = qn[rows, j_star]
    e = -torch.sign(qn_jstar)
    e = torch.where(e == 0, torch.ones_like(e), e)

    v_uhat = (v * u_hat).unsqueeze(-1)
    qdot_task = (J_plus @ v_uhat).squeeze(-1)
    task_on_j = qdot_task[rows, j_star]
    task_escape = e * task_on_j

    B_row = B_basis[rows, j_star, :]
    policy_ns = e * (B_row * a_policy).sum(-1) * a_max
    optimal_ns = B_row.abs().sum(-1) * a_max

    # Classical 4-term hand-tuned baseline action at t=0 (q_ref = q since t=0)
    cls_ctrl = ClassicalNullspaceController(env.kin)
    q_dot_raw = cls_ctrl.q_dot_null(q, u_hat, n_target, q_ref=q.clone())
    with torch.no_grad():
        a_cls = (B_basis.transpose(-1, -2) @ q_dot_raw.unsqueeze(-1)).squeeze(-1)
        a_cls = (a_cls / a_max).clamp(-1.0, 1.0)
    cls_ns = e * (B_row * a_cls).sum(-1) * a_max

    qh_j = q_half[j_star]
    pol_rate = ((task_escape + policy_ns) * dt / qh_j).cpu().numpy()
    opt_rate = ((task_escape + optimal_ns) * dt / qh_j).cpu().numpy()
    cls_rate = ((task_escape + cls_ns) * dt / qh_j).cpu().numpy()
    task_rate = (task_escape * dt / qh_j).cpu().numpy()
    init_abs = qn_jstar.abs().cpu().numpy()

    near = init_abs > args.near_thr
    rl_worse = (rl_len < base_len)
    rl_better = (rl_len > base_len)
    sub_w = near & rl_worse
    sub_b = near & rl_better

    def _summary(name, mask, x):
        vv = x[mask]
        if vv.size == 0:
            return f"{name:<30s} n=0"
        return (f"{name:<30s} n={vv.size:4d}  "
                f"mean={vv.mean():+.4f}  median={np.median(vv):+.4f}  "
                f"p10={np.quantile(vv, .1):+.4f}  p90={np.quantile(vv, .9):+.4f}")

    print(f"[policy] near-limit subset: RL_worse={int(sub_w.sum())}  RL_better={int(sub_b.sum())}")
    print()
    print("[policy] task-only escape rate (per step, no nullspace):")
    print("[policy] " + _summary("near ∩ RL_worse ", sub_w, task_rate))
    print("[policy] " + _summary("near ∩ RL_better", sub_b, task_rate))
    print()
    print("[policy] RL policy escape rate (task + policy's chosen nullspace):")
    print("[policy] " + _summary("near ∩ RL_worse ", sub_w, pol_rate))
    print("[policy] " + _summary("near ∩ RL_better", sub_b, pol_rate))
    print()
    print("[policy] optimal escape rate (task + ||B[j*]||_1·a_max):")
    print("[policy] " + _summary("near ∩ RL_worse ", sub_w, opt_rate))
    print("[policy] " + _summary("near ∩ RL_better", sub_b, opt_rate))
    print()
    print("[policy] Classical hand-tuned baseline escape rate:")
    print("[policy] " + _summary("near ∩ RL_worse ", sub_w, cls_rate))
    print("[policy] " + _summary("near ∩ RL_better", sub_b, cls_rate))

    gap = opt_rate - pol_rate
    print()
    print("[policy] policy suboptimality gap (optimal − actual policy rate):")
    print("[policy] " + _summary("near ∩ RL_worse ", sub_w, gap))
    print("[policy] " + _summary("near ∩ RL_better", sub_b, gap))

    print()
    for label, mask in [("near ∩ RL_worse ", sub_w), ("near ∩ RL_better", sub_b)]:
        n_pol_neg = (pol_rate[mask] < 0).sum()
        n_cls_neg = (cls_rate[mask] < 0).sum()
        n_opt_neg = (opt_rate[mask] < 0).sum()
        n_tot = int(mask.sum())
        print(f"[policy] {label}: frac<0 — policy={n_pol_neg}/{n_tot}={100*n_pol_neg/n_tot:.1f}%  "
              f"classical={n_cls_neg}/{n_tot}={100*n_cls_neg/n_tot:.1f}%  "
              f"optimal={n_opt_neg}/{n_tot}={100*n_opt_neg/n_tot:.1f}%")

    out_dir = diag_dir / "escape_geom"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    bins = np.linspace(-0.02, 0.04, 60)
    for ax, mask, title in [
        (axes[0], sub_w, f"near ∩ RL_worse (n={sub_w.sum()})"),
        (axes[1], sub_b, f"near ∩ RL_better (n={sub_b.sum()})"),
    ]:
        ax.hist(pol_rate[mask], bins=bins, alpha=0.55,
                label="RL policy", color="#1f77b4", density=True)
        ax.hist(cls_rate[mask], bins=bins, alpha=0.45,
                label="Classical baseline", color="#2ca02c", density=True)
        ax.hist(opt_rate[mask], bins=bins, alpha=0.3,
                label="optimal", color="#ff7f0e", density=True)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title(title + " — escape rate on j* per step")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = out_dir / "policy_vs_optimal.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print()
    print(f"[policy] saved histogram → {fig_path}")


if __name__ == "__main__":
    main()

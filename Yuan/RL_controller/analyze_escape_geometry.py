"""Test whether RL_worse-near-limit failures are 'escape-infeasible at t=0'.

At t=0, the joint velocity on the initially-saturated joint j* is:
    q_dot[j*] = (J_p^+ v u_hat)[j*]            # task-space contribution
              + (B_basis @ (a · a_max))[j*]    # nullspace contribution, a ∈ [-1,1]^4

Define escape direction e = -sign(qn[j*]). Best-case q_dot in escape direction
maximizes over a:
    max e·q_dot[j*] = e · (J_p^+ v u_hat)[j*]   +   ||B_basis[j*, :]||_1 · a_max
                      ────────────task term─────       ──────nullspace ceiling──────

If even this *best-achievable* escape rate ≤ 0, no policy can prevent j* from
moving deeper into its limit on step 1. That's the 'geometrically infeasible'
hypothesis for the 169 RL_worse-near-limit failures.

Usage:
    python -m Yuan.RL_controller.analyze_escape_geometry \\
        --diag-dir Yuan/RL_controller/runs/p0_progress_only_30M_0520/diag_5000
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
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from one.robots.manipulators.franka.fr3_pen.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.RL_controller.env.env import damped_pinv, build_task_aligned_basis


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
    dt = float(npz["dt"])

    # Env params from parent run config
    cfg_path = diag_dir.parent / "config.yaml"
    with open(cfg_path) as f:
        cfg_yaml = yaml.safe_load(f)
    env_cfg = cfg_yaml["env"]
    v = float(env_cfg["v"])
    a_max = float(env_cfg["a_max"])
    lambda_0 = float(env_cfg["lambda_0"])
    sigma_thr = float(env_cfg["sigma_thr"])
    manip_damping = float(env_cfg["manip_damping"])
    tcp_offset = float(env_cfg["tcp_offset"])

    device = (torch.device(args.device) if args.device is not None
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    kin = BatchedFR3Kinematics(device=device, tcp_offset=tcp_offset)
    q_mid = kin.q_mid
    q_half = 0.5 * (kin.lmt_up - kin.lmt_lo)

    N = q0_np.shape[0]
    q = torch.from_numpy(q0_np).to(device=device, dtype=kin.dtype)
    u_hat = torch.from_numpy(line_dir_np).to(device=device, dtype=kin.dtype)
    n_target = torch.from_numpy(n_target_np).to(device=device, dtype=kin.dtype)

    # FK + Jacobian at t=0
    _, _, J, _ = kin.tcp_fk_jac(q)
    J_p = J[:, :3, :]
    J_plus, _ = damped_pinv(J_p, lambda_0, sigma_thr)             # (N, 7, 3)
    B_basis, _ = build_task_aligned_basis(
        kin, q, u_hat, n_target, q_mid, q_half, manip_damping,
    )                                                              # (N, 7, 4)

    qn = (q - q_mid) / q_half                                      # (N, 7)
    j_star = qn.abs().argmax(dim=-1)                               # (N,)
    rows = torch.arange(N, device=device)
    qn_jstar = qn[rows, j_star]                                    # (N,)
    e = -torch.sign(qn_jstar)                                      # (N,) escape sign
    e = torch.where(e == 0, torch.ones_like(e), e)

    # Task contribution: (J_plus @ v u_hat)[j*] in rad/s
    v_uhat = (v * u_hat).unsqueeze(-1)                             # (N, 3, 1)
    qdot_task = (J_plus @ v_uhat).squeeze(-1)                      # (N, 7)
    task_on_j = qdot_task[rows, j_star]                            # (N,)
    task_escape = e * task_on_j                                    # signed; >0 = task helps escape

    # Nullspace ceiling on j*: ||B_basis[j*, :]||_1 * a_max (rad/s)
    B_row = B_basis[rows, j_star, :]                               # (N, 4)
    ns_max = B_row.abs().sum(-1) * a_max                           # (N,) always ≥ 0

    net_escape = task_escape + ns_max                              # rad/s; >0 = can escape

    # Convert to per-step |qn| velocity (units: |qn|/step)
    qh_j = q_half[j_star]                                          # (N,)
    task_escape_per_step = (task_escape * dt / qh_j).cpu().numpy()
    ns_max_per_step = (ns_max * dt / qh_j).cpu().numpy()
    net_escape_per_step = (net_escape * dt / qh_j).cpu().numpy()
    init_abs = qn_jstar.abs().cpu().numpy()

    # Subsets
    near = init_abs > args.near_thr
    rl_worse = (rl_len < base_len)
    rl_better = (rl_len > base_len)
    sub_w = near & rl_worse
    sub_b = near & rl_better

    def _summary(name, mask, x):
        v = x[mask]
        if v.size == 0:
            return f"{name:<28s} n=0"
        return (f"{name:<28s} n={v.size:4d}  "
                f"mean={v.mean():+.4f}  median={np.median(v):+.4f}  "
                f"p10={np.quantile(v, .1):+.4f}  p90={np.quantile(v, .9):+.4f}")

    print(f"[geom] near-limit (|qn[0]|>{args.near_thr}) subset breakdown")
    print(f"[geom]   near ∩ RL_worse  : n={int(sub_w.sum())}")
    print(f"[geom]   near ∩ RL_better : n={int(sub_b.sum())}")

    print()
    print("[geom] task-space escape rate (per step, +ve = task helps j* escape):")
    print("[geom] " + _summary("near ∩ RL_worse ", sub_w, task_escape_per_step))
    print("[geom] " + _summary("near ∩ RL_better", sub_b, task_escape_per_step))

    print()
    print("[geom] nullspace ceiling (||B[j*]||_1 · a_max · dt / q_half, max help avail):")
    print("[geom] " + _summary("near ∩ RL_worse ", sub_w, ns_max_per_step))
    print("[geom] " + _summary("near ∩ RL_better", sub_b, ns_max_per_step))

    print()
    print("[geom] NET best-case escape rate per step on j* (task + max nullspace):")
    print("[geom] " + _summary("near ∩ RL_worse ", sub_w, net_escape_per_step))
    print("[geom] " + _summary("near ∩ RL_better", sub_b, net_escape_per_step))

    # Headline: fraction of cases where even best-case can't escape (≤0)
    print()
    for thr_name, thr in [("≤0 (cannot escape)", 0.0),
                          ("≤0.001 (near zero)", 0.001),
                          ("≤0.005 (slow escape)", 0.005)]:
        w_frac = (net_escape_per_step[sub_w] <= thr).mean() if sub_w.sum() else 0.0
        b_frac = (net_escape_per_step[sub_b] <= thr).mean() if sub_b.sum() else 0.0
        print(f"[geom] frac with net_escape {thr_name}:"
              f"  RL_worse={w_frac*100:5.1f}%  RL_better={b_frac*100:5.1f}%")

    # Task contribution sign distribution
    task_negative_w = (task_escape_per_step[sub_w] < 0).mean() if sub_w.sum() else 0.0
    task_negative_b = (task_escape_per_step[sub_b] < 0).mean() if sub_b.sum() else 0.0
    print()
    print(f"[geom] frac with task term pushing INTO limit (task_escape<0):"
          f"  RL_worse={task_negative_w*100:5.1f}%  RL_better={task_negative_b*100:5.1f}%")

    # ---- histograms -----------------------------------------------------
    out_dir = diag_dir / "escape_geom"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=False)
    bins_net = np.linspace(-0.03, 0.06, 60)
    bins_task = np.linspace(-0.03, 0.06, 60)
    bins_ns = np.linspace(0, 0.06, 60)

    axes[0].hist(task_escape_per_step[sub_w], bins=bins_task, alpha=0.6,
                 label=f"RL_worse (n={sub_w.sum()})", color="#d62728", density=True)
    axes[0].hist(task_escape_per_step[sub_b], bins=bins_task, alpha=0.4,
                 label=f"RL_better (n={sub_b.sum()})", color="#2ca02c", density=True)
    axes[0].axvline(0, color="k", lw=0.6)
    axes[0].set_title("task-space escape rate on j* (per step), >0 = task helps")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(ns_max_per_step[sub_w], bins=bins_ns, alpha=0.6,
                 label="RL_worse", color="#d62728", density=True)
    axes[1].hist(ns_max_per_step[sub_b], bins=bins_ns, alpha=0.4,
                 label="RL_better", color="#2ca02c", density=True)
    axes[1].set_title("nullspace ceiling ||B[j*]||_1·a_max·dt/q_half (max help)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].hist(net_escape_per_step[sub_w], bins=bins_net, alpha=0.6,
                 label="RL_worse", color="#d62728", density=True)
    axes[2].hist(net_escape_per_step[sub_b], bins=bins_net, alpha=0.4,
                 label="RL_better", color="#2ca02c", density=True)
    axes[2].axvline(0, color="k", lw=0.6)
    axes[2].set_title("NET best-case escape rate (task + nullspace ceiling)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = out_dir / "escape_distributions.png"
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print()
    print(f"[geom] saved histogram → {fig_path}")


if __name__ == "__main__":
    main()

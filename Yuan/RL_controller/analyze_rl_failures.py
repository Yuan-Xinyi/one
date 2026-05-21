"""Analyze the RL-worse-than-baseline tasks from diag_1000/rollouts.npz.

Identifies the subset of tasks where T_rl < T_base, then:
  1. Aggregates termination reasons + which joint is closest to its limit at
     RL termination (argmax of |q_norm|, where q_norm = (q − q_mid) / q_half ∈ [-1,1]).
  2. Quantifies whether RL pushes the saturating joint TOWARD or AWAY from its
     limit relative to baseline (the discriminating mechanism: baseline has a
     ∇H_jl(q) joint-center attractor; P0's reward has w_progress=1.0 only).
  3. Plots per-joint trajectories for the top-K worst-ratio cases (RL vs baseline),
     with joint limits as dotted lines and term reasons annotated.

Reads cached data only — no re-rollout. Run after `diagnose_p0_vs_baseline.py`.

Usage:
    python -m Yuan.RL_controller.analyze_rl_failures \\
        --diag-dir Yuan/RL_controller/runs/p0_progress_only_30M_0520/diag_1000 \\
        --top-k 20
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
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from one.robots.manipulators.franka.fr3_pen.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.RL_controller.env.env import TERM_NAMES


def _qn(q, q_mid, q_half):
    """Joint norm ∈ [-1, 1] (limits at ±1)."""
    return (q - q_mid) / q_half


def _final_q(q_traj, ep_len, task_id):
    """q at the moment of termination (step = ep_len)."""
    return q_traj[int(ep_len[task_id]), task_id, :]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--diag-dir", required=True,
                   help="dir containing rollouts.npz from diagnose_p0_vs_baseline")
    p.add_argument("--top-k", type=int, default=20,
                   help="plot the K worst RL/base ratio cases among RL-losses")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    diag_dir = Path(args.diag_dir)
    npz = np.load(diag_dir / "rollouts.npz")
    q_traj_rl = npz["q_traj_rl"]          # (T+1, N, 7)
    q_traj_base = npz["q_traj_base"]
    rl_len = npz["episode_len_rl"]        # (N,)
    base_len = npz["episode_len_base"]
    rl_term = npz["term_reason_rl"]
    base_term = npz["term_reason_base"]
    q0 = npz["q0"]
    line_dir = npz["line_dir"]
    n_target = npz["n_target"]
    max_steps = int(npz["max_steps"])
    N = rl_len.shape[0]

    # Joint limits.
    device = (torch.device(args.device) if args.device is not None
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    kin = BatchedFR3Kinematics(device=device)
    lmt_lo = kin.lmt_lo.cpu().numpy()
    lmt_up = kin.lmt_up.cpu().numpy()
    q_mid = kin.q_mid.cpu().numpy()
    q_half = 0.5 * (lmt_up - lmt_lo)

    # Per-step q_norm for entire population.
    qn_init = _qn(q0, q_mid, q_half)                    # (N, 7)
    init_max_abs = np.max(np.abs(qn_init), axis=1)      # (N,) max |q_norm| at t=0

    rl_worse = np.where(rl_len < base_len)[0]
    rl_better = np.where(rl_len > base_len)[0]
    print(f"[analyze] N={N}  RL_worse={rl_worse.size}  RL_better={rl_better.size}  "
          f"tie={int((rl_len==base_len).sum())}")

    # ---- (1) Initial joint distance from limits: RL-worse vs RL-better. ----
    print()
    print("[analyze] initial max |q_norm| (closeness-to-limit @ reset):")
    def _q(x, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
        return "  ".join(f"p{int(q*100)}={np.quantile(x,q):.3f}" for q in qs)
    print(f"[analyze]   RL_worse   (n={rl_worse.size}): mean={init_max_abs[rl_worse].mean():.3f}  "
          + _q(init_max_abs[rl_worse]))
    print(f"[analyze]   RL_better  (n={rl_better.size}): mean={init_max_abs[rl_better].mean():.3f}  "
          + _q(init_max_abs[rl_better]))
    near_lim = 0.9
    print(f"[analyze]   frac with init max|q_norm| > {near_lim}: "
          f"RL_worse={int((init_max_abs[rl_worse]>near_lim).sum())}/{rl_worse.size}={100*(init_max_abs[rl_worse]>near_lim).mean():.1f}%  "
          f"RL_better={int((init_max_abs[rl_better]>near_lim).sum())}/{rl_better.size}={100*(init_max_abs[rl_better]>near_lim).mean():.1f}%")

    if rl_worse.size == 0:
        return

    # ---- (2) Termination reason histograms. ----
    term_hist = collections.Counter(TERM_NAMES.get(int(t), "?")
                                    for t in rl_term[rl_worse])
    print()
    print(f"[analyze] RL term_reason on losses : {dict(term_hist)}")
    print(f"[analyze] base term on same tasks  : "
          f"{dict(collections.Counter(TERM_NAMES.get(int(t),'?') for t in base_term[rl_worse]))}")

    # ---- (3) Argmax-|q_norm| joint at RL termination + how RL/baseline moved that joint. ----
    saturated_joint = np.zeros(rl_worse.size, dtype=np.int64)
    saturated_qn_end = np.zeros(rl_worse.size, dtype=np.float64)
    rl_dqn_on_satj = np.zeros(rl_worse.size, dtype=np.float64)   # Δ|qn| on that joint over RL rollout
    base_dqn_on_satj = np.zeros(rl_worse.size, dtype=np.float64) # same for baseline, over SAME number of steps as RL
    for k, i in enumerate(rl_worse):
        q_end = _final_q(q_traj_rl, rl_len, i)
        qn_end = _qn(q_end, q_mid, q_half)
        j = int(np.argmax(np.abs(qn_end)))
        saturated_joint[k] = j
        saturated_qn_end[k] = qn_end[j]

        # Δ|q_norm| on joint j from t=0 → t=T_rl for both controllers
        # (RL went into limit; did baseline pull the same joint AWAY from limit?)
        T_rl = int(rl_len[i])
        rl_q_j_init = q_traj_rl[0, i, j]
        rl_q_j_end = q_traj_rl[T_rl, i, j]
        base_q_j_init = q_traj_base[0, i, j]
        # baseline trajectory at the same step horizon (or its own end, whichever earlier)
        base_step = min(T_rl, int(base_len[i]))
        base_q_j_end = q_traj_base[base_step, i, j]
        rl_dqn_on_satj[k] = (abs((rl_q_j_end - q_mid[j]) / q_half[j])
                             - abs((rl_q_j_init - q_mid[j]) / q_half[j]))
        base_dqn_on_satj[k] = (abs((base_q_j_end - q_mid[j]) / q_half[j])
                               - abs((base_q_j_init - q_mid[j]) / q_half[j]))

    print()
    print("[analyze] argmax-|q_norm| joint at RL termination (RL-worse subset):")
    joint_hist = collections.Counter(saturated_joint.tolist())
    for j in range(7):
        n_j = joint_hist.get(j, 0)
        if n_j == 0:
            continue
        signs = saturated_qn_end[saturated_joint == j]
        n_pos = int((signs > 0).sum())
        n_neg = int((signs < 0).sum())
        n_atlimit = int((np.abs(signs) > 1.0).sum())
        print(f"[analyze]   q{j+1}: n={n_j:3d}  upper(+1)={n_pos}  lower(-1)={n_neg}  "
              f"<|qn_end|>={np.abs(signs).mean():.3f}  at-limit={n_atlimit}")

    # ---- (4) Did RL push the saturating joint TOWARD limit, while baseline pulls AWAY?
    print()
    print("[analyze] Δ|q_norm| on the saturating joint, t=0 → t=T_rl:")
    print(f"[analyze]   RL       : mean={rl_dqn_on_satj.mean():+.4f}  "
          f"median={np.median(rl_dqn_on_satj):+.4f}  "
          f"frac>0={(rl_dqn_on_satj>0).mean():.2%}  (positive = pushed TOWARD limit)")
    print(f"[analyze]   baseline : mean={base_dqn_on_satj.mean():+.4f}  "
          f"median={np.median(base_dqn_on_satj):+.4f}  "
          f"frac>0={(base_dqn_on_satj>0).mean():.2%}")
    diff = rl_dqn_on_satj - base_dqn_on_satj
    print(f"[analyze]   RL − base: mean={diff.mean():+.4f}  "
          f"frac>0={(diff>0).mean():.2%}  "
          f"(positive = RL moved that joint closer to limit than baseline)")

    # ---- per-term joint breakdown ----
    print()
    print("[analyze] per-term joint breakdown (RL-worse subset):")
    for term_code, term_name in TERM_NAMES.items():
        mask = rl_term[rl_worse] == term_code
        if mask.sum() == 0:
            continue
        sub = saturated_joint[mask]
        hist = collections.Counter(sub.tolist())
        print(f"[analyze]   term={term_name} (n={int(mask.sum())}): " +
              "  ".join(f"q{j+1}={hist.get(j,0)}" for j in range(7)))

    # ---- (5) plot top-K worst-by-ratio. ----
    ratio = rl_len.astype(float) / base_len.astype(float).clip(min=1.0)
    losses_by_ratio = rl_worse[np.argsort(ratio[rl_worse])]
    n_plot = min(args.top_k, losses_by_ratio.size)
    out_dir = diag_dir / "failure_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    print()
    print(f"[analyze] plotting top-{n_plot} worst-ratio RL losses → {out_dir}")
    for rank, i in enumerate(losses_by_ratio[:n_plot]):
        T_rl = int(rl_len[i])
        T_base = int(base_len[i])
        rl_traj = q_traj_rl[:T_rl + 1, i, :]
        base_traj = q_traj_base[:T_base + 1, i, :]
        x_rl = np.arange(T_rl + 1)
        x_base = np.arange(T_base + 1)

        fig, axes = plt.subplots(7, 1, figsize=(9, 12), sharex=True)
        for j in range(7):
            ax = axes[j]
            ax.plot(x_rl, rl_traj[:, j], color="#1f77b4", lw=1.8, label="RL")
            ax.plot(x_base, base_traj[:, j], color="#2ca02c", ls="--", lw=1.3,
                    label="baseline")
            ax.axhline(lmt_lo[j], color="grey", lw=0.6, ls=":")
            ax.axhline(lmt_up[j], color="grey", lw=0.6, ls=":")
            ax.axvline(T_rl, color="#1f77b4", lw=0.6, ls=":", alpha=0.5)
            ax.axvline(T_base, color="#2ca02c", lw=0.6, ls=":", alpha=0.5)
            ax.set_ylabel(f"q{j+1}")
            ax.grid(True, alpha=0.3)
        axes[0].legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("step")
        fig.suptitle(
            f"task {int(i)}  (rank {rank+1}/{n_plot} worst RL/base ratio)\n"
            f"T_rl={T_rl} ({TERM_NAMES.get(int(rl_term[i]),'?')})  "
            f"T_base={T_base} ({TERM_NAMES.get(int(base_term[i]),'?')})  "
            f"ratio={ratio[i]:.3f}  init max|qn|={init_max_abs[i]:.3f}",
            fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out_dir / f"rank{rank:02d}_task{int(i):04d}.png", dpi=110)
        plt.close(fig)
        if (rank + 1) % 5 == 0 or (rank + 1) == n_plot:
            print(f"[analyze]   {rank+1}/{n_plot}")
    print("[analyze] done.")


if __name__ == "__main__":
    main()

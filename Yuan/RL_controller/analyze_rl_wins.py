"""Inverse-failure analysis: in RL_better tasks that ALSO start near a joint
limit, what does RL do differently from RL_worse tasks?

Reads diag_<N>/rollouts.npz, restricts to near-limit subset (init max|qn|>0.9),
and compares per-task Δ|qn| on the initially-saturated joint between RL_worse
and RL_better. Also plots a few representative RL_better near-limit trajectories.

Usage:
    python -m Yuan.RL_controller.analyze_rl_wins \\
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

from one.robots.manipulators.franka.fr3_pen.batched_fr3_kin import BatchedFR3Kinematics
from Yuan.RL_controller.env.env import TERM_NAMES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--diag-dir", required=True)
    p.add_argument("--near-thr", type=float, default=0.9,
                   help="|q_norm[0]| threshold for 'near-limit' subset")
    p.add_argument("--plot-k", type=int, default=8,
                   help="how many RL_better near-limit cases to plot")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    diag_dir = Path(args.diag_dir)
    npz = np.load(diag_dir / "rollouts.npz")
    q_traj_rl = npz["q_traj_rl"]
    q_traj_base = npz["q_traj_base"]
    rl_len = npz["episode_len_rl"]
    base_len = npz["episode_len_base"]
    rl_term = npz["term_reason_rl"]
    base_term = npz["term_reason_base"]
    q0 = npz["q0"]
    N = rl_len.shape[0]

    device = (torch.device(args.device) if args.device is not None
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    kin = BatchedFR3Kinematics(device=device)
    lmt_lo = kin.lmt_lo.cpu().numpy()
    lmt_up = kin.lmt_up.cpu().numpy()
    q_mid = kin.q_mid.cpu().numpy()
    q_half = 0.5 * (lmt_up - lmt_lo)

    # initial qn for every task
    qn0 = (q0 - q_mid) / q_half               # (N, 7)
    init_argmax = np.argmax(np.abs(qn0), axis=1)   # joint index of initial max-|qn|
    init_abs = np.abs(qn0)[np.arange(N), init_argmax]  # |qn| at that joint, t=0

    near = init_abs > args.near_thr           # bool (N,)
    rl_worse = rl_len < base_len
    rl_better = rl_len > base_len

    sub_worse = near & rl_worse
    sub_better = near & rl_better
    print(f"[wins] near-limit subset (init |qn|>{args.near_thr}): n={int(near.sum())}/{N}")
    print(f"[wins]   ∩ RL_worse  : n={int(sub_worse.sum())}")
    print(f"[wins]   ∩ RL_better : n={int(sub_better.sum())}")

    def _delta_init_joint(traj, ep_len, horizon):
        """Δ|qn| on the initially-saturated joint j* over [0, horizon] per task."""
        T = horizon  # array (N,)
        # gather q at step T per task (broadcast along time)
        idx_t = np.minimum(T, ep_len)            # cap at own ep_len
        rows = np.arange(N)
        q_end = traj[idx_t, rows, init_argmax]   # (N,)
        q_init = traj[0, rows, init_argmax]
        end_abs = np.abs((q_end - q_mid[init_argmax]) / q_half[init_argmax])
        init_abs_ = np.abs((q_init - q_mid[init_argmax]) / q_half[init_argmax])
        return end_abs - init_abs_

    # Use min(T_rl, T_base) as comparable horizon per task (both controllers alive).
    horizon = np.minimum(rl_len, base_len)
    d_rl = _delta_init_joint(q_traj_rl, rl_len, horizon)
    d_base = _delta_init_joint(q_traj_base, base_len, horizon)

    def _stats(name, x):
        return (f"{name:<8s} mean={x.mean():+.4f}  median={np.median(x):+.4f}  "
                f"frac>0={(x>0).mean()*100:5.1f}%  "
                f"(>0 = moved TOWARD limit)")

    print()
    print("[wins] Δ|qn| on initial-saturated joint j*, over horizon=min(T_rl, T_base):")
    print("[wins] near-limit ∩ RL_worse:")
    print("[wins]  ", _stats("RL", d_rl[sub_worse]))
    print("[wins]  ", _stats("baseline", d_base[sub_worse]))
    print("[wins]  ", _stats("RL-base", (d_rl - d_base)[sub_worse]))
    print("[wins] near-limit ∩ RL_better:")
    print("[wins]  ", _stats("RL", d_rl[sub_better]))
    print("[wins]  ", _stats("baseline", d_base[sub_better]))
    print("[wins]  ", _stats("RL-base", (d_rl - d_base)[sub_better]))

    # Also: at the end of RL's own rollout, where is j*?
    # (long-horizon view, not capped by baseline)
    rows = np.arange(N)
    qrlT_init = q_traj_rl[rl_len, rows, init_argmax]
    end_abs_rl_full = np.abs((qrlT_init - q_mid[init_argmax]) / q_half[init_argmax])
    delta_rl_full = end_abs_rl_full - init_abs

    qbaseT_init = q_traj_base[base_len, rows, init_argmax]
    end_abs_base_full = np.abs((qbaseT_init - q_mid[init_argmax]) / q_half[init_argmax])
    delta_base_full = end_abs_base_full - init_abs

    print()
    print("[wins] Δ|qn| on initial-saturated joint j*, over each controller's OWN lifetime:")
    print("[wins] near-limit ∩ RL_worse :")
    print("[wins]  ", _stats("RL", delta_rl_full[sub_worse]))
    print("[wins]  ", _stats("baseline", delta_base_full[sub_worse]))
    print("[wins] near-limit ∩ RL_better:")
    print("[wins]  ", _stats("RL", delta_rl_full[sub_better]))
    print("[wins]  ", _stats("baseline", delta_base_full[sub_better]))

    # Distribution of which joint started saturated, per subset:
    import collections
    print()
    print("[wins] initial-saturated joint distribution:")
    for label, mask in [("near ∩ RL_worse", sub_worse), ("near ∩ RL_better", sub_better)]:
        h = collections.Counter(init_argmax[mask].tolist())
        line = "  ".join(f"q{j+1}={h.get(j,0)}" for j in range(7))
        print(f"[wins]   {label}: {line}")

    # ---- plot a few RL_better near-limit examples ----
    ratio = rl_len.astype(float) / base_len.astype(float).clip(min=1.0)
    # rank by largest ratio (best RL wins) among near-limit RL_better
    pool = np.where(sub_better)[0]
    if pool.size == 0:
        print("[wins] no RL_better near-limit cases to plot")
        return
    by_ratio = pool[np.argsort(-ratio[pool])]
    n_plot = min(args.plot_k, by_ratio.size)
    out_dir = diag_dir / "win_plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[wins] plotting top-{n_plot} biggest RL/base ratio (near-limit RL_better) → {out_dir}")
    for rank, i in enumerate(by_ratio[:n_plot]):
        T_rl = int(rl_len[i])
        T_base = int(base_len[i])
        rl_traj = q_traj_rl[:T_rl + 1, i, :]
        base_traj = q_traj_base[:T_base + 1, i, :]
        x_rl = np.arange(T_rl + 1)
        x_base = np.arange(T_base + 1)
        j_star = int(init_argmax[i])

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
            lbl = f"q{j+1}"
            if j == j_star:
                lbl += " *"
                ax.set_facecolor("#fffbe6")
            ax.set_ylabel(lbl)
            ax.grid(True, alpha=0.3)
        axes[0].legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("step")
        fig.suptitle(
            f"task {int(i)} (rank {rank+1}/{n_plot} biggest RL/base, near-limit RL_better)\n"
            f"T_rl={T_rl} ({TERM_NAMES.get(int(rl_term[i]),'?')})  "
            f"T_base={T_base} ({TERM_NAMES.get(int(base_term[i]),'?')})  "
            f"ratio={ratio[i]:.2f}  init|qn[q{j_star+1}]|={init_abs[i]:.3f}",
            fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out_dir / f"rank{rank:02d}_task{int(i):04d}.png", dpi=110)
        plt.close(fig)
    print("[wins] done.")


if __name__ == "__main__":
    main()

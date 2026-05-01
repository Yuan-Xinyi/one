"""Overlay the policy's stochastic action distribution onto the (phi, psi)
rollout-length heatmap of a single task.

Purpose: verify whether the policy's distribution covers the success
islands even when its deterministic mean lands in a fail pocket. If yes,
deploying with best-of-K stochastic sampling will recover those tasks
without retraining.

Usage:
    python -m Yuan.RL.diagnose_policy_dist --task-idx 14
    python -m Yuan.RL.diagnose_policy_dist --task-idx 14 --n-samples 1000
"""
from __future__ import annotations
import argparse, os, math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import Yuan.RL.config as cfg
from Yuan.RL.env import FarsightedSeedEnv
from Yuan.RL.policy import make_policy
from Yuan.RL.batched_rollout import batched_rollout


def _load_policy(ckpt_path, env, device):
    q_mid = torch.as_tensor(env.action_mid, dtype=torch.float32, device=device)
    q_half = torch.as_tensor(env.action_half, dtype=torch.float32, device=device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy = make_policy(cfg.STATE_DIM, env.action_dim, q_mid, q_half,
                         policy_type=state.get("policy_type", "gaussian")).to(device)
    policy.load_state_dict(state["policy"])
    policy.eval()
    return policy


def _rollout_chunked(actions_np, c_np, v_np, e_np, T_np, chunk=4096):
    n = actions_np.shape[0]
    L = np.empty(n, dtype=np.int32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        out = batched_rollout(actions_np[s:e], c_np[s:e], v_np[s:e],
                              e_np[s:e], T_np[s:e])
        L[s:e] = np.asarray(out["lengths"], dtype=np.int32)
    return L


def _action_to_phi_psi(a: np.ndarray):
    phi = np.arctan2(a[..., 1], a[..., 0]) % (2 * np.pi)
    psi = np.arctan2(a[..., 3], a[..., 2]) % (2 * np.pi)
    return phi, psi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str,
                    default=os.path.join(cfg.CKPT_DIR, "ckpt_005000.pt"))
    ap.add_argument("--task-idx", type=int, default=14)
    ap.add_argument("--n-tasks", type=int, default=32)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--n-samples", type=int, default=1000,
                    help="how many stochastic policy samples to draw")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    env = FarsightedSeedEnv(seed=args.seed, randomize=True, use_collision=False)
    policy = _load_policy(args.ckpt, env, device)

    env.rng = np.random.default_rng(args.seed)
    tasks = env._sample_tasks(args.n_tasks)
    task = tasks[args.task_idx]
    state = env._state_vec(task).astype(np.float32)
    c = task["c"].astype(np.float32)
    v_path = float(task["v_path"])
    eps_p  = float(task["eps_p"])
    T = int(task["T"])
    print(f"\ntask #{args.task_idx}: T={T}  v={v_path:.3f}  eps={eps_p*1000:.1f}mm")

    # --- background L heatmap (same as diagnose_phi_psi.py) ---
    G = int(args.grid)
    phi_grid = np.linspace(0.0, 2 * np.pi, G, endpoint=False, dtype=np.float32)
    psi_grid = np.linspace(0.0, 2 * np.pi, G, endpoint=False, dtype=np.float32)
    phi_mesh, psi_mesh = np.meshgrid(phi_grid, psi_grid, indexing="ij")
    a_grid = np.stack([
        np.cos(phi_mesh), np.sin(phi_mesh),
        np.cos(psi_mesh), np.sin(psi_mesh),
    ], axis=-1).reshape(G * G, 4).astype(np.float32)
    L_grid = _rollout_chunked(a_grid,
                              np.tile(c, (G * G, 1)),
                              np.full(G * G, v_path, dtype=np.float32),
                              np.full(G * G, eps_p,  dtype=np.float32),
                              np.full(G * G, T,      dtype=np.int32))
    L_grid = L_grid.reshape(G, G).astype(np.float32)
    L_best_grid = int(L_grid.max())

    # --- deterministic policy action ---
    with torch.no_grad():
        st = torch.as_tensor(state[None], dtype=torch.float32, device=device)
        a_det, _ = policy.act(st, deterministic=True)
    a_det_np = a_det.cpu().numpy().astype(np.float32).reshape(-1)
    phi_det, psi_det = _action_to_phi_psi(a_det_np)
    L_det = int(_rollout_chunked(a_det_np[None], c[None], np.array([v_path]),
                                 np.array([eps_p]), np.array([T]))[0])

    # --- stochastic samples ---
    N = int(args.n_samples)
    with torch.no_grad():
        st_rep = torch.as_tensor(np.tile(state, (N, 1)),
                                 dtype=torch.float32, device=device)
        a_stoch, _ = policy.act(st_rep, deterministic=False)
    a_stoch_np = a_stoch.cpu().numpy().astype(np.float32)
    phi_st, psi_st = _action_to_phi_psi(a_stoch_np)
    L_st = _rollout_chunked(a_stoch_np,
                            np.tile(c, (N, 1)),
                            np.full(N, v_path, dtype=np.float32),
                            np.full(N, eps_p,  dtype=np.float32),
                            np.full(N, T,      dtype=np.int32))

    # stats
    L_st = L_st.astype(np.int32)
    succ_threshold = max(1, int(0.5 * T))   # "successful" = at least half T
    n_good = int((L_st >= succ_threshold).sum())
    L_max  = int(L_st.max())
    L_mean = float(L_st.mean())
    L_p50  = float(np.percentile(L_st, 50))
    L_p90  = float(np.percentile(L_st, 90))
    print(f"  policy det: phi={phi_det:.3f}  psi={psi_det:.3f}  L={L_det}/{T}")
    print(f"  stochastic ({N} samples):  L_max={L_max}  L_p90={L_p90:.0f}  "
          f"L_p50={L_p50:.0f}  L_mean={L_mean:.1f}")
    print(f"  fraction of stochastic samples with L >= 0.5T = {n_good}/{N} ({n_good/N:.1%})")
    print(f"  best-of-K equivalent: K=8 best L ~ {int(np.sort(L_st)[-1])} (top-1) "
          f"or {int(np.sort(L_st)[-8:].max())} (top-8 = same)")

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    extent = [0, 2*np.pi, 0, 2*np.pi]

    # background L heatmap
    im = ax.imshow(L_grid.T, origin="lower", extent=extent,
                   cmap="viridis", vmin=0, vmax=T, aspect="auto", alpha=0.85)
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("L (steps), background = grid sweep")

    # stochastic samples colored by their L
    sc = ax.scatter(phi_st, psi_st, c=L_st, s=12, cmap="plasma",
                    vmin=0, vmax=T, edgecolors="black", linewidths=0.2,
                    alpha=0.85)
    cb2 = plt.colorbar(sc, ax=ax, location="bottom", pad=0.10, shrink=0.7)
    cb2.set_label(f"L of stochastic policy samples (N={N})")

    # deterministic mean
    ax.plot([phi_det], [psi_det], "*", color="red", markersize=22,
            markeredgecolor="white", markeredgewidth=1.5,
            label=f"policy det mean (L={L_det})")

    ax.set_xlabel("phi (elbow swivel target dir)")
    ax.set_ylabel("psi (tool roll)")
    ax.set_title(f"task #{args.task_idx}  T={T}  v={v_path:.2f}  "
                 f"eps={eps_p*1000:.1f}mm   "
                 f"(grid best L = {L_best_grid})")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 2*np.pi); ax.set_ylim(0, 2*np.pi)

    if args.out is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "diagnostics")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir,
                                f"diag_task{args.task_idx}_policy_dist.png")
    else:
        out_path = args.out
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
                    exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()

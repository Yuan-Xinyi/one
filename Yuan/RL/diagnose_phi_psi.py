"""Diagnose the policy on a single task by sweeping the (phi, psi) plane.

Builds a dense grid in (phi, psi) in [0, 2pi)^2, runs IK + rollout for every
grid cell, and plots:

  - IK feasibility heatmap (which cells the IK solver could converge on)
  - rollout-length heatmap L(phi, psi) (gray where IK infeasible)
  - policy's deterministic (phi, psi) point overlaid (red star)
  - best K=1000-uniform sample's (phi, psi) point overlaid (green dot)

Saves a PNG; also prints policy point + best-grid stats.

Usage:
    python -m Yuan.RL.diagnose_phi_psi --task-idx 14
    python -m Yuan.RL.diagnose_phi_psi --task-idx 28 --grid 96 --out /tmp/diag.png
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
    reasons = []
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        out = batched_rollout(actions_np[s:e], c_np[s:e], v_np[s:e],
                              e_np[s:e], T_np[s:e])
        L[s:e] = np.asarray(out["lengths"], dtype=np.int32)
        rs = out.get("reasons", [])
        if rs is None:
            rs = [""] * (e - s)
        reasons.extend(rs)
    return L, reasons


def _action_to_phi_psi(a: np.ndarray) -> tuple[float, float]:
    phi = math.atan2(a[1], a[0]) % (2 * math.pi)
    psi = math.atan2(a[3], a[2]) % (2 * math.pi)
    return phi, psi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str,
                    default=os.path.join(cfg.CKPT_DIR, "ckpt_005000.pt"))
    ap.add_argument("--task-idx", type=int, default=14,
                    help="task index in the 32-task batch (seed-controlled)")
    ap.add_argument("--n-tasks", type=int, default=32,
                    help="task batch size for the index lookup (must match eval)")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--grid", type=int, default=64,
                    help="grid resolution per dim (grid x grid total points)")
    ap.add_argument("--k-uniform", type=int, default=1000,
                    help="how many uniform random samples to compare against")
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
    print(f"  p0={c[:3]}  d={c[3:6]}  n={c[6:9]}")

    # policy deterministic action
    with torch.no_grad():
        st = torch.as_tensor(state[None], dtype=torch.float32, device=device)
        a_pol, _ = policy.act(st, deterministic=True)
    a_pol_np = a_pol.cpu().numpy().astype(np.float32).reshape(-1)
    phi_pol, psi_pol = _action_to_phi_psi(a_pol_np)
    L_pol, r_pol = _rollout_chunked(a_pol_np[None], c[None], np.array([v_path]),
                                    np.array([eps_p]), np.array([T]))
    L_pol = int(L_pol[0]); r_pol = r_pol[0] if r_pol else "?"
    print(f"  policy det: phi={phi_pol:.3f}  psi={psi_pol:.3f}  "
          f"L={L_pol}/{T}  reason={r_pol}")

    # build (phi, psi) grid
    G = int(args.grid)
    phi_grid = np.linspace(0.0, 2 * np.pi, G, endpoint=False, dtype=np.float32)
    psi_grid = np.linspace(0.0, 2 * np.pi, G, endpoint=False, dtype=np.float32)
    phi_mesh, psi_mesh = np.meshgrid(phi_grid, psi_grid, indexing="ij")
    a_grid = np.stack([
        np.cos(phi_mesh), np.sin(phi_mesh),
        np.cos(psi_mesh), np.sin(psi_mesh),
    ], axis=-1).reshape(G * G, 4).astype(np.float32)

    rep_c = np.tile(c, (G * G, 1))
    rep_v = np.full(G * G, v_path, dtype=np.float32)
    rep_e = np.full(G * G, eps_p, dtype=np.float32)
    rep_T = np.full(G * G, T, dtype=np.int32)

    print(f"  rolling out {G*G} grid cells ...")
    L_grid_flat, reasons_grid = _rollout_chunked(a_grid, rep_c, rep_v, rep_e, rep_T)
    L_grid = L_grid_flat.reshape(G, G).astype(np.float32)
    reasons_grid = np.asarray(reasons_grid, dtype=object).reshape(G, G)
    feasible_grid = reasons_grid != "init_ik_fail"

    n_feas = int(feasible_grid.sum())
    L_best_grid = int(L_grid[feasible_grid].max()) if n_feas else 0
    print(f"  IK feasible cells: {n_feas}/{G*G} ({n_feas/(G*G):.1%})")
    print(f"  best L on grid:    {L_best_grid}/{T}")

    # K-uniform comparison (independent random sampling, same seed for repro)
    rng = np.random.default_rng(args.seed)
    K = int(args.k_uniform)
    phi_u = rng.uniform(0.0, 2*np.pi, size=K).astype(np.float32)
    psi_u = rng.uniform(0.0, 2*np.pi, size=K).astype(np.float32)
    a_u = np.stack([np.cos(phi_u), np.sin(phi_u),
                    np.cos(psi_u), np.sin(psi_u)], axis=-1).astype(np.float32)
    L_u, r_u = _rollout_chunked(a_u, np.tile(c, (K, 1)),
                                np.full(K, v_path, dtype=np.float32),
                                np.full(K, eps_p, dtype=np.float32),
                                np.full(K, T, dtype=np.int32))
    feas_u = np.array([r != "init_ik_fail" for r in r_u])
    n_feas_u = int(feas_u.sum())
    if n_feas_u > 0:
        best_u_idx = int(np.argmax(L_u))
        phi_u_best = phi_u[best_u_idx]; psi_u_best = psi_u[best_u_idx]
        L_u_best = int(L_u[best_u_idx])
    else:
        phi_u_best = psi_u_best = float("nan"); L_u_best = 0
    print(f"  K={K} uniform:      feasible {n_feas_u}/{K},  best L={L_u_best}/{T}")

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    extent = [0, 2*np.pi, 0, 2*np.pi]
    # left: feasibility (binary) — also annotate failure-reason mode per cell
    feas_img = feasible_grid.astype(np.float32).T  # transpose so phi=x, psi=y
    axes[0].imshow(feas_img, origin="lower", extent=extent,
                   cmap="Greens", vmin=0, vmax=1, aspect="auto")
    axes[0].set_xlabel("phi (elbow swivel target dir)")
    axes[0].set_ylabel("psi (tool roll)")
    axes[0].set_title(f"IK feasibility  ({n_feas}/{G*G} cells)")
    # overlay: policy point
    axes[0].plot([phi_pol], [psi_pol], "*", color="red", markersize=18,
                 markeredgecolor="black", label=f"policy det (L={L_pol})")
    if not np.isnan(phi_u_best):
        axes[0].plot([phi_u_best], [psi_u_best], "o", color="cyan",
                     markersize=10, markeredgecolor="black",
                     label=f"best of K={K} (L={L_u_best})")
    axes[0].legend(loc="upper right")

    # right: rollout length, masked where infeasible
    L_plot = np.where(feasible_grid, L_grid, np.nan).T
    im = axes[1].imshow(L_plot, origin="lower", extent=extent,
                        cmap="viridis", vmin=0, vmax=T, aspect="auto")
    axes[1].set_xlabel("phi")
    axes[1].set_ylabel("psi")
    axes[1].set_title(f"rollout length L (T={T})  best={L_best_grid}")
    cb = plt.colorbar(im, ax=axes[1])
    cb.set_label("L (steps)")
    axes[1].plot([phi_pol], [psi_pol], "*", color="red", markersize=18,
                 markeredgecolor="white", label=f"policy det (L={L_pol})")
    if not np.isnan(phi_u_best):
        axes[1].plot([phi_u_best], [psi_u_best], "o", color="cyan",
                     markersize=10, markeredgecolor="white",
                     label=f"best of K={K} (L={L_u_best})")
    axes[1].legend(loc="upper right")

    fig.suptitle(f"task #{args.task_idx}  T={T}  v={v_path:.2f}  eps={eps_p*1000:.1f}mm")
    fig.tight_layout()

    if args.out is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "diagnostics")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"diag_task{args.task_idx}_grid{G}.png")
    else:
        out_path = args.out
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
                    exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()

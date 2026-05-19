"""Verify the task-aligned `build_task_aligned_basis` construction.

Samples N resets, builds B(q) via the new task-aligned MGS construction,
and reports per-column sign/projection statistics against three physical
gradients (∇w_u, ∇cos(z·n), ∇(−mean qn²)).

  (a) Distribution of sign(B[:, 0, k]) — surface-level sign of column 0 row 0.
      Has no physical meaning by itself; kept for output-schema continuity
      with the original SO(4)-gauge diagnostic.
  (b) Distribution of three physically-meaningful sign judges:
        sgn(B_k^T ∇w_u(q))           — "a_k>0 raises directional manipulability"
        sgn(B_k^T ∇cos(z, n_target)) — "a_k>0 improves cone alignment"
        sgn(B_k^T ∇(-q_norm²))       — "a_k>0 moves away from joint limits"
  (c) Agreement between (a) and (b).
  (d) Fallback trigger rate per e_k (SVD-column substitution in MGS).
  (e) Orthonormality residual max |B^T B - I|.

For the task-aligned basis, the diagonal anchor cells in (b) should be
≈ 100% +; strict lower-triangle cells should be ~0; upper-triangle cells
are unconstrained.

Usage:
    python -m Yuan.RL_controller.diagnose_sign_seed \\
        --config Yuan/RL_controller/config.yaml --n-total 2048
"""
from __future__ import annotations

# LD_LIBRARY_PATH self-relaunch (matches eval.py / train.py).
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

import torch
import yaml

from Yuan.RL_controller.env.env import (
    NSRLBatchedEnv, EnvConfig, build_task_aligned_basis,
)
from Yuan.RL_controller.env.line_distribution import LineDistribution


def _grad_w_u(kin, q: torch.Tensor, u_hat: torch.Tensor,
              manip_damping: float = 1e-3) -> torch.Tensor:
    """∇_q w_u(q, u_hat). (B, 7)."""
    with torch.enable_grad():
        q_eval = q.detach().clone().requires_grad_(True)
        _, _, J, _ = kin.tcp_fk_jac(q_eval)
        J_p = J[:, :3, :]
        eye3 = torch.eye(3, device=q.device, dtype=q.dtype).expand(q.shape[0], 3, 3)
        JJt_dmp = J_p @ J_p.transpose(-1, -2) + (manip_damping ** 2) * eye3
        u_col = u_hat.unsqueeze(-1)
        inv_quad = (u_col.transpose(-1, -2)
                    @ torch.linalg.inv(JJt_dmp) @ u_col
                    ).squeeze(-1).squeeze(-1).clamp_min(1e-12)
        w_u = inv_quad.pow(-0.5)
        grad = torch.autograd.grad(w_u.sum(), q_eval)[0]
    return grad.detach()


def _grad_cos(kin, q: torch.Tensor, n_target: torch.Tensor) -> torch.Tensor:
    """∇_q [z_tool(q) · n_target]. (B, 7)."""
    with torch.enable_grad():
        q_eval = q.detach().clone().requires_grad_(True)
        _, R, _, _ = kin.tcp_fk_jac(q_eval)
        z = R[:, :, 2]
        cos = (z * n_target).sum(-1)
        grad = torch.autograd.grad(cos.sum(), q_eval)[0]
    return grad.detach()


def _grad_neg_qnorm_sq(q: torch.Tensor, q_mid: torch.Tensor,
                       q_half: torch.Tensor) -> torch.Tensor:
    """∇_q [−mean((q − q_mid)/q_half)²]. (B, 7). Analytic (avoids autograd graph)."""
    qn = (q - q_mid) / q_half
    return -(2.0 / qn.shape[-1]) * qn / q_half


def _sign_stats(x: torch.Tensor, name: str, columns=4) -> str:
    """x: (N, columns) of signed scalars. Returns multi-line summary."""
    lines = [f"  {name}:"]
    for k in range(columns):
        v = x[:, k]
        pos = int((v > 0).sum().item())
        neg = int((v < 0).sum().item())
        zero = int((v == 0).sum().item())
        tot = pos + neg + zero
        pct_pos = 100.0 * pos / max(tot, 1)
        mean_abs = float(v.abs().mean().item())
        lines.append(f"    k={k}: + {pos:5d}  − {neg:5d}  0 {zero:3d}  "
                     f"({pct_pos:5.1f}% +) |mean|={mean_abs:.3e}")
    return "\n".join(lines)


def _agreement(a: torch.Tensor, b: torch.Tensor, columns=4) -> str:
    """Per-column agreement rate between two sign tensors (N, columns)."""
    lines = []
    for k in range(columns):
        sa = torch.sign(a[:, k])
        sb = torch.sign(b[:, k])
        nonzero = (sa != 0) & (sb != 0)
        if int(nonzero.sum().item()) == 0:
            lines.append(f"    k={k}: n/a")
            continue
        agree = ((sa == sb) & nonzero).sum().item() / int(nonzero.sum().item())
        lines.append(f"    k={k}: {100.0 * agree:5.1f}% agree "
                     f"(n={int(nonzero.sum().item())})")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--n-total", type=int, default=2048,
                        help="Total number of resets to sample.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config, "r") as f:
        cfg_yaml = yaml.safe_load(f)
    env_cfg = EnvConfig(**cfg_yaml["env"])
    line_cfg = cfg_yaml["line_distribution"]

    env = NSRLBatchedEnv(env_cfg, line_dist=None, device=device)
    sampler = LineDistribution.load_or_build(
        kin=env.kin, collision=env.collision,
        n_pool=line_cfg["n_pool"],
        n_target_noise_deg=line_cfg["n_target_noise_deg"],
        seed=line_cfg.get("train_seed", 0),
        env_cfg=env_cfg,
        # No feasibility filter for diagnostic — we want the actual reset distribution
        feasibility_threshold_m=None,
    )
    env.line_dist = sampler

    n_batches = (args.n_total + env.n_envs - 1) // env.n_envs

    seed_sign_all = []         # (N, 4)  sign(B_signed[:, 0, :])
    judge_dm_all = []          # (N, 4)  B_k^T ∇w_u
    judge_cone_all = []        # (N, 4)  B_k^T ∇cos
    judge_jl_all = []          # (N, 4)  B_k^T ∇(−q_norm²)
    fb_mask_all = []           # (N, 3)  fallback flag for e_0, e_1, e_2
    ortho_err_all = []         # (N,)    per-env max |B^T B - I| entry

    print(f"[diag] sampling {n_batches * env.n_envs} resets "
          f"(n_envs={env.n_envs}, n_batches={n_batches})…")

    for b_i in range(n_batches):
        env.reset()
        # Build the task-aligned basis (replaces old SVD+sign-seed construction).
        B_signed, fb_mask = build_task_aligned_basis(
            env.kin, env.q, env.line_dir, env.n_target,
            env.kin.q_mid, env.q_half, env_cfg.manip_damping,
        )
        sgn_seed = torch.sign(B_signed[:, 0, :])  # surface sign of column 0 row 0

        # Physical sign judges. Project each column of B_signed onto a physical
        # gradient direction in joint space.
        g_dm = _grad_w_u(env.kin, env.q, env.line_dir, env_cfg.manip_damping)
        g_cone = _grad_cos(env.kin, env.q, env.n_target)
        g_jl = _grad_neg_qnorm_sq(env.q, env.kin.q_mid, env.q_half)

        # B_signed: (B, 7, 4), g_*: (B, 7) → einsum over joint axis.
        proj_dm = torch.einsum("bjk,bj->bk", B_signed, g_dm)
        proj_cone = torch.einsum("bjk,bj->bk", B_signed, g_cone)
        proj_jl = torch.einsum("bjk,bj->bk", B_signed, g_jl)

        # Orthonormality check on this batch.
        BtB = B_signed.transpose(-1, -2) @ B_signed
        I4 = torch.eye(4, device=BtB.device, dtype=BtB.dtype).expand_as(BtB)
        per_env_max = (BtB - I4).abs().flatten(-2, -1).max(dim=-1).values

        seed_sign_all.append(sgn_seed.cpu())
        judge_dm_all.append(proj_dm.cpu())
        judge_cone_all.append(proj_cone.cpu())
        judge_jl_all.append(proj_jl.cpu())
        fb_mask_all.append(fb_mask.cpu())
        ortho_err_all.append(per_env_max.cpu())

    seed_sign = torch.cat(seed_sign_all, dim=0)
    judge_dm = torch.cat(judge_dm_all, dim=0)
    judge_cone = torch.cat(judge_cone_all, dim=0)
    judge_jl = torch.cat(judge_jl_all, dim=0)
    fb_mask = torch.cat(fb_mask_all, dim=0)
    ortho_err = torch.cat(ortho_err_all, dim=0)

    print(f"[diag] N={seed_sign.shape[0]} resets, 4 nullspace columns per reset.\n")

    print("(a) Current sign seed: sign(B_raw[:, 0, k])  "
          "— a coherent convention would NOT need to be ~50/50.")
    print(_sign_stats(seed_sign, "sign_seed"))
    print()

    print("(b) Physical sign judges on the SIGNED basis (what policy actually sees).")
    print("    If consistent, one direction should dominate (≫ 50%).")
    print(_sign_stats(judge_dm, "B_k^T ∇w_u(q)                 (dirmanip)"))
    print(_sign_stats(judge_cone, "B_k^T ∇cos(z, n_target)        (cone)"))
    print(_sign_stats(judge_jl, "B_k^T ∇(−mean q_norm²)         (JL margin)"))
    print()

    print("(c) Agreement: does the seed sign correlate with each physical judge?")
    print("    (If seed were physically meaningful, agreement ≫ 50%.)")
    print("    seed vs dirmanip-judge:")
    print(_agreement(seed_sign, judge_dm))
    print("    seed vs cone-judge:")
    print(_agreement(seed_sign, judge_cone))
    print("    seed vs JL-judge:")
    print(_agreement(seed_sign, judge_jl))
    print()

    print("(d) Fallback trigger rate per e_k (SVD-column substitution in MGS).")
    print(f"    e_0 (anchored to ∇w_u):    {100.0 * fb_mask[:, 0].float().mean().item():6.3f}%")
    print(f"    e_1 (anchored to ∇cos):    {100.0 * fb_mask[:, 1].float().mean().item():6.3f}%")
    print(f"    e_2 (anchored to ∇−qn²):   {100.0 * fb_mask[:, 2].float().mean().item():6.3f}%")
    print()

    print("(e) Orthonormality residual: max |B^T B - I| per env, aggregated.")
    print(f"    max  over {ortho_err.shape[0]} resets = {ortho_err.max().item():.2e}")
    print(f"    mean over {ortho_err.shape[0]} resets = {ortho_err.mean().item():.2e}")
    print()

    print("Interpretation:")
    print("  • (b) near 50/50 ⇒ action a_k has random physical sign at episode start.")
    print("  • (b) near 100/0 on at least one judge ⇒ that judge is a candidate")
    print("    deterministic sign convention. Pick one judge per column.")


if __name__ == "__main__":
    main()

"""MPC/lookahead teacher: exact action search in the deterministic simulator.

The ExIt loop's ceiling is set by the teacher's belt branch (classical's
local-gradient rescue). This module replaces that branch with SEARCH: for a
belt state, try K candidate nullspace actions (classical's, the policy's,
perturbations, uniforms), hold each for `hold_H` steps, then continue with
closed-loop classical to termination (capped), and score by the EXACT
discounted progress return. argmax is a certified-no-worse-than-classical
label (classical's own action is always a candidate).

Two entry points:
  validate — measure how much the searched action beats pure classical from
             the same states (uses the salvageable danger pool, which already
             carries exact classical returns G_cls). Gate for the pipeline:
             if the margin is negligible, a search teacher won't lift the
             distillation ceiling and the pipeline should stop here.
  label    — batch-label arbitrary belt states (used by mpc distill stage).

Usage:
    python -m Yuan.RL_controller.self_improve.mpc_teacher validate \\
        --policy-ckpt-dir Yuan/RL_controller/runs/distill_r5_warmstart
"""
from __future__ import annotations

# Self-relaunch with $CONDA_PREFIX/lib on LD_LIBRARY_PATH (entry-point only;
# see loop.py note, 2026-07-02).
import os, sys
if __name__ == "__main__":
    _conda_lib = os.path.join(sys.prefix, "lib")
    if _conda_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
        new_env = dict(os.environ)
        new_env["LD_LIBRARY_PATH"] = _conda_lib + ":" + new_env.get("LD_LIBRARY_PATH", "")
        os.execvpe(sys.executable,
                   [sys.executable, "-m", __spec__.name] + sys.argv[1:]
                   if __spec__ is not None else [sys.executable] + sys.argv,
                   new_env)

import argparse
from pathlib import Path

import numpy as np
import torch

from Yuan.RL_controller.env.env import (
    NSRLBatchedEnv, EnvConfig, build_task_aligned_basis)
from Yuan.RL_controller.env.line_distribution import ScriptedLineDistribution
from Yuan.RL_controller.env.classical_nullspace import (
    ClassicalNullspaceController, cn_action_fn)
from Yuan.RL_controller.self_improve.collect import load_agent, load_env_kw

GAMMA = 0.99


@torch.no_grad()
def _classical_action(env: NSRLBatchedEnv,
                      ctrl: ClassicalNullspaceController) -> torch.Tensor:
    B_basis, _ = build_task_aligned_basis(
        env.kin, env.q, env.line_dir, env.n_target,
        env.kin.q_mid, env.q_half, env.cfg.manip_damping)
    q_dot = ctrl.q_dot_null(env.q, env.line_dir, env.n_target)
    a = (B_basis.transpose(-1, -2) @ q_dot.unsqueeze(-1)).squeeze(-1)
    return torch.nan_to_num(a / env.a_max, nan=0.0).clamp(-1.0, 1.0)


def make_candidates(a_cls: torch.Tensor, a_pol: torch.Tensor, K: int,
                    gen: torch.Generator, sigma: float = 0.2) -> torch.Tensor:
    """(B,4)x2 -> (B,K,4). Slots: cls, pol, perturbations of each, uniforms."""
    B, A = a_cls.shape
    dev = a_cls.device
    n_pc = (K - 2) // 2          # perturbed classical
    n_pp = (K - 2) - n_pc - max((K - 2) // 4, 1)  # perturbed policy
    n_u = K - 2 - n_pc - n_pp    # uniform
    cands = [a_cls.unsqueeze(1), a_pol.unsqueeze(1)]
    if n_pc > 0:
        noise = torch.randn((B, n_pc, A), device=dev, generator=gen) * sigma
        cands.append((a_cls.unsqueeze(1) + noise).clamp(-1.0, 1.0))
    if n_pp > 0:
        noise = torch.randn((B, n_pp, A), device=dev, generator=gen) * sigma
        cands.append((a_pol.unsqueeze(1) + noise).clamp(-1.0, 1.0))
    if n_u > 0:
        u = torch.rand((B, n_u, A), device=dev, generator=gen) * 2.0 - 1.0
        cands.append(u)
    return torch.cat(cands, dim=1)  # (B, K, 4)


@torch.no_grad()
def mpc_label_states(q0: torch.Tensor, line_dir: torch.Tensor,
                     n_target: torch.Tensor, *, env_kw: dict,
                     policy, device, K: int = 16, hold_H: int = 10,
                     cont_cap: int = 240, sigma: float = 0.2,
                     seed: int = 0) -> dict:
    """Exact lookahead labeling of B states. Returns dict with:
        a_best (B,4), score_best (B,), score_cls_cand (B,), pick_idx (B,)
    score_cls_cand = score of classical's own action under the same
    hold-then-classical protocol (candidate 0)."""
    B = q0.shape[0]
    dev = device
    gen = torch.Generator(device=dev).manual_seed(seed)

    # A probe env (B envs) to compute per-state classical + policy actions.
    probe = NSRLBatchedEnv(EnvConfig(**{**env_kw, "n_envs": B}),
                           line_dist=None, device=dev)
    spec = {"q0": q0.to(dev, probe.kin.dtype),
            "line_dir": line_dir.to(dev, probe.kin.dtype),
            "n_target": n_target.to(dev, probe.kin.dtype)}
    probe.line_dist = ScriptedLineDistribution({k: v.clone() for k, v in spec.items()})
    probe.reset()
    ctrl_probe = ClassicalNullspaceController(probe.kin)
    a_cls = _classical_action(probe, ctrl_probe).float()
    a_pol = policy.actor_mean(probe.current_obs()).clamp(-1.0, 1.0).float()
    del probe
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    cands = make_candidates(a_cls, a_pol, K, gen, sigma)  # (B,K,4)

    # Tiled env: B*K rollouts (state i repeated K times, candidate j applied).
    env = NSRLBatchedEnv(EnvConfig(**{**env_kw, "n_envs": B * K}),
                         line_dist=None, device=dev)
    tiled = {k: v.repeat_interleave(K, dim=0) for k, v in spec.items()}
    env.line_dist = ScriptedLineDistribution(tiled)
    env.reset()
    ctrl = ClassicalNullspaceController(env.kin)
    flat_cands = cands.reshape(B * K, -1).to(env.kin.dtype)

    G = torch.zeros(B * K, dtype=torch.float64, device=dev)
    disc = 1.0
    for t in range(hold_H + cont_cap):
        if bool(env.done_persistent.all().item()):
            break
        a = flat_cands if t < hold_H else _classical_action(env, ctrl)
        _, r, _, _, _ = env.step(a, auto_reset=False)
        G += disc * r.double()
        disc *= GAMMA
    del env
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    G = G.float().view(B, K)
    pick = G.argmax(dim=1)
    a_best = cands[torch.arange(B, device=dev), pick]
    return {"a_best": a_best.cpu(), "score_best": G.max(dim=1).values.cpu(),
            "score_cls_cand": G[:, 0].cpu(), "pick_idx": pick.cpu(),
            "score_pol_cand": G[:, 1].cpu()}


def validate(policy_ckpt_dir, *, n_states: int = 2048, K: int = 16,
             hold_H: int = 10, chunk: int = 2048, device=None,
             pool_path="Yuan/RL_controller/runs/danger_starts/pool.npz",
             verbose=True) -> dict:
    """Gate: does search beat pure closed-loop classical from belt states?
    Compares score_best vs the pool's exact G_cls (pure classical rollouts,
    same gamma) AND vs score_cls_cand (protocol-internal baseline)."""
    device = torch.device(device if device is not None
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    policy, cfg_yaml = load_agent(policy_ckpt_dir, device)
    env_kw = load_env_kw(cfg_yaml)
    d = np.load(pool_path)
    n = min(n_states, d["q0"].shape[0])
    idx = np.random.default_rng(0).choice(d["q0"].shape[0], n, replace=False)
    q0 = torch.tensor(d["q0"][idx]); ld = torch.tensor(d["line_dir"][idx])
    nt = torch.tensor(d["n_target"][idx]); G_cls = torch.tensor(d["G_cls"][idx])

    outs = []
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        outs.append(mpc_label_states(q0[s:e], ld[s:e], nt[s:e], env_kw=env_kw,
                                     policy=policy, device=device, K=K,
                                     hold_H=hold_H, seed=s))
        if verbose:
            print(f"[mpc-val] {e}/{n} labeled", flush=True)
    best = torch.cat([o["score_best"] for o in outs])
    cls_cand = torch.cat([o["score_cls_cand"] for o in outs])
    pol_cand = torch.cat([o["score_pol_cand"] for o in outs])
    pick = torch.cat([o["pick_idx"] for o in outs])

    stats = {
        "n": n,
        "mean_G_cls_pure": float(G_cls.mean()),
        "mean_score_cls_cand": float(cls_cand.mean()),
        "mean_score_pol_cand": float(pol_cand.mean()),
        "mean_score_best": float(best.mean()),
        "gain_vs_pure_cls_pct": float(100 * (best.mean() - G_cls.mean())
                                      / max(G_cls.mean(), 1e-6)),
        "frac_best_is_cls": float((pick == 0).float().mean()),
        "frac_best_is_pol": float((pick == 1).float().mean()),
        "frac_best_is_search": float((pick >= 2).float().mean()),
    }
    if verbose:
        print(f"[mpc-val] pure classical G: {stats['mean_G_cls_pure']:.2f}   "
              f"search best: {stats['mean_score_best']:.2f}   "
              f"gain {stats['gain_vs_pure_cls_pct']:+.1f}%")
        print(f"[mpc-val] argmax split: cls {100*stats['frac_best_is_cls']:.0f}%  "
              f"pol {100*stats['frac_best_is_pol']:.0f}%  "
              f"searched {100*stats['frac_best_is_search']:.0f}%")
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["validate"])
    parser.add_argument("--policy-ckpt-dir", required=True)
    parser.add_argument("--n-states", type=int, default=2048)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--hold-H", type=int, default=10)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    validate(args.policy_ckpt_dir, n_states=args.n_states, K=args.K,
             hold_H=args.hold_H, device=args.device)


if __name__ == "__main__":
    main()

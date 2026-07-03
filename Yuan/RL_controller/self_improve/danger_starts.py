"""Danger-start episode pool: train where the headroom is.

The pure-vs-hybrid gap lives entirely in the ~23% of tasks that approach
joint limits (hard-task pure ratio 0.687 vs hybrid 0.995 on the 10k set).
Uniform task sampling starves that region of gradient signal. This module
builds a pool of SALVAGEABLE danger states — visited by a reference policy,
max|q_norm| in a near-limit belt, and verified by an exact classical rollout
to be survivable (G_cls >= min_G) — and a MixedLineDistribution that starts a
configurable fraction of training episodes from them.

Salvageability screen matters: starting from doomed states teaches nothing
and pollutes the reward scale.

Usage:
    python -m Yuan.RL_controller.self_improve.danger_starts \\
        --ckpt-dir Yuan/RL_controller/runs/p0_progress_only_30M_0520 \\
        --out Yuan/RL_controller/runs/danger_starts/pool.npz
"""
from __future__ import annotations

# Self-relaunch with $CONDA_PREFIX/lib on LD_LIBRARY_PATH (same as train/eval).
# GUARDED to entry-point only — on import the exec would hijack the host
# process (see loop.py note, 2026-07-02).
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

from Yuan.RL_controller.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.RL_controller.env.line_distribution import (
    LineDistribution, ScriptedLineDistribution)
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.RL_controller.self_improve.collect import (
    load_agent, load_env_kw, rl_action_fn)
from Yuan.RL_controller.self_improve.vcls import _classical_discounted_return


@torch.no_grad()
def _collect_belt_states(env: NSRLBatchedEnv, action_fn,
                         qn_lo: float, qn_hi: float) -> dict:
    """Roll `action_fn`; snapshot (q, line_dir, n_target) of active envs whose
    max|q_norm| is inside the [qn_lo, qn_hi] belt at that step."""
    out = {"q0": [], "line_dir": [], "n_target": []}
    env.reset()
    for _ in range(env.max_steps + 1):
        qn = ((env.q - env.q_mid).abs() / env.q_half).max(dim=-1).values
        m = (~env.done_persistent) & (qn >= qn_lo) & (qn <= qn_hi)
        if bool(m.any().item()):
            out["q0"].append(env.q[m].float().cpu())
            out["line_dir"].append(env.line_dir[m].float().cpu())
            out["n_target"].append(env.n_target[m].float().cpu())
        a = action_fn(env)
        env.step(a, auto_reset=False)
        if bool(env.done_persistent.all().item()):
            break
    return {k: (torch.cat(v) if v else torch.zeros((0, 7 if k == "q0" else 3)))
            for k, v in out.items()}


def build_danger_pool(ckpt_dir, out_path, *, n_tasks: int = 8192,
                      qn_lo: float = 0.90, qn_hi: float = 0.98,
                      min_G: float = 20.0, max_pool: int = 100_000,
                      chunk_size: int = 4096, seed: int = 8100,
                      device=None, verbose: bool = True) -> dict:
    """Build the salvageable danger-start pool from the reference policy's
    visitation. min_G=20 at gamma=0.99 ~ classical survives ~22+ steps."""
    ckpt_dir = Path(ckpt_dir)
    device = torch.device(device if device is not None
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    agent, cfg_yaml = load_agent(ckpt_dir, device)
    env_kw = load_env_kw(cfg_yaml)
    line_cfg = cfg_yaml["line_distribution"]
    threshold_m = (float(line_cfg["feasibility_threshold_m"])
                   if line_cfg.get("feasibility_filter", False) else None)

    proxy_cfg = EnvConfig(**{**env_kw, "n_envs": 1})
    proxy = NSRLBatchedEnv(proxy_cfg, line_dist=None, device=device)
    pool = LineDistribution.load_or_build(
        kin=proxy.kin, collision=proxy.collision,
        n_pool=line_cfg["n_pool"],
        n_target_noise_deg=line_cfg["n_target_noise_deg"],
        seed=line_cfg["train_seed"], env_cfg=proxy_cfg,
        feasibility_threshold_m=threshold_m, verbose=verbose)
    gen = torch.Generator(device=device).manual_seed(seed)
    tasks = pool.sample(n_tasks, generator=gen)
    del proxy

    # 1. belt states from reference-policy visitation
    parts = []
    for start in range(0, n_tasks, chunk_size):
        end = min(start + chunk_size, n_tasks)
        env = NSRLBatchedEnv(EnvConfig(**{**env_kw, "n_envs": end - start}),
                             line_dist=None, device=device)
        env.line_dist = ScriptedLineDistribution(
            {k: v[start:end].clone() for k, v in tasks.items()})
        parts.append(_collect_belt_states(env, rl_action_fn(agent), qn_lo, qn_hi))
        del env
        if device.type == "cuda":
            torch.cuda.empty_cache()
    states = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    n = states["q0"].shape[0]
    if verbose:
        print(f"[danger] {n} belt states collected "
              f"(max|qn| in [{qn_lo}, {qn_hi}], policy visitation)")
    if n > max_pool:
        pick = torch.randperm(n)[:max_pool]
        states = {k: v[pick] for k, v in states.items()}
        n = max_pool

    # 2. salvageability screen: exact classical return from each state
    gamma = 0.99
    G = torch.zeros(n)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        env = NSRLBatchedEnv(EnvConfig(**{**env_kw, "n_envs": end - start}),
                             line_dist=None, device=device)
        env.line_dist = ScriptedLineDistribution({
            k: states[k][start:end].to(device, env.kin.dtype)
            for k in ("q0", "line_dir", "n_target")})
        ctrl = ClassicalNullspaceController(env.kin)
        G[start:end] = _classical_discounted_return(env, ctrl, gamma).cpu()
        if verbose:
            print(f"[danger]   screened {end}/{n}", flush=True)
        del env
        if device.type == "cuda":
            torch.cuda.empty_cache()

    keep = G >= min_G
    n_keep = int(keep.sum().item())
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        q0=states["q0"][keep].numpy(),
        line_dir=states["line_dir"][keep].numpy(),
        n_target=states["n_target"][keep].numpy(),
        G_cls=G[keep].numpy(),
        qn_lo=np.float64(qn_lo), qn_hi=np.float64(qn_hi),
        min_G=np.float64(min_G), gamma=np.float64(gamma),
        src_ckpt=np.str_(str(ckpt_dir)), seed=np.int64(seed))
    stats = {"n_belt": n, "n_salvageable": n_keep,
             "frac_salvageable": n_keep / max(n, 1),
             "G_mean_kept": float(G[keep].mean()) if n_keep else 0.0}
    if verbose:
        print(f"[danger] salvageable {n_keep}/{n} ({100*stats['frac_salvageable']:.1f}%)  "
              f"G_cls(kept) mean {stats['G_mean_kept']:.1f}  -> {out_path}")
    return stats


class MixedLineDistribution:
    """With prob p_danger an episode starts from a salvageable danger state;
    otherwise from the base (feasibility-filtered) task distribution. Drop-in
    for LineDistribution.sample()."""

    def __init__(self, base, pool_path, p_danger: float,
                 device, dtype):
        d = np.load(pool_path)
        self.q0 = torch.tensor(d["q0"], device=device, dtype=dtype)
        self.line_dir = torch.tensor(d["line_dir"], device=device, dtype=dtype)
        self.n_target = torch.tensor(d["n_target"], device=device, dtype=dtype)
        self.n_pool = self.q0.shape[0]
        if self.n_pool == 0:
            raise ValueError(f"danger pool {pool_path} is empty")
        self.base = base
        self.p_danger = float(p_danger)
        self.device = device

    def sample(self, n: int, generator: torch.Generator | None = None
               ) -> dict[str, torch.Tensor]:
        spec = self.base.sample(n, generator=generator)
        use_d = (torch.rand(n, device=self.device, generator=generator)
                 < self.p_danger)
        if bool(use_d.any().item()):
            idx = torch.randint(0, self.n_pool, (n,), device=self.device,
                                generator=generator)
            m = use_d.unsqueeze(-1)
            spec = {
                "q0": torch.where(m, self.q0[idx], spec["q0"]),
                "line_dir": torch.where(m, self.line_dir[idx], spec["line_dir"]),
                "n_target": torch.where(m, self.n_target[idx], spec["n_target"]),
            }
        return spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-tasks", type=int, default=8192)
    parser.add_argument("--qn-lo", type=float, default=0.90)
    parser.add_argument("--qn-hi", type=float, default=0.98)
    parser.add_argument("--min-G", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=8100)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    build_danger_pool(args.ckpt_dir, args.out, n_tasks=args.n_tasks,
                      qn_lo=args.qn_lo, qn_hi=args.qn_hi, min_G=args.min_G,
                      seed=args.seed, device=args.device)


if __name__ == "__main__":
    main()

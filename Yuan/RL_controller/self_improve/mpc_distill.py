"""Distill the MPC-hybrid teacher: pi0 in the safe region, exact-search MPC
actions in the danger belt (soft-blended in between), into one policy net.

Why this composition (2026-07-03 overnight run):
  - The classical-teacher ExIt loop converged at ~1.045 (fidelity maxed at
    val MSE 0.0013; frac(hyb>pure) down to 9%). Its ceiling is classical's
    local-gradient rescue quality.
  - MPC validation gate: exact lookahead search beats pure classical by +8.6%
    from belt states, and 94% of argmax actions are searched ones — real
    headroom above the local gradient.
  - Safe labels come from pi0 directly (not the student lineage) to cut the
    inherited easy-layer loss (students: 1.067-1.071 vs pi0's 1.079).

Usage:
    python -m Yuan.RL_controller.self_improve.mpc_distill \\
        --out-dir Yuan/RL_controller/runs/distill_r7_mpc
"""
from __future__ import annotations

# Self-relaunch preamble (entry-point only; see loop.py note, 2026-07-02).
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
import yaml

from Yuan.RL_controller.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.RL_controller.env.line_distribution import (
    LineDistribution, ScriptedLineDistribution)
from Yuan.RL_controller.self_improve.collect import load_agent, load_env_kw
from Yuan.RL_controller.self_improve.distill import fit_actor, TARGET_CLAMP
from Yuan.RL_controller.self_improve.mpc_teacher import mpc_label_states


@torch.no_grad()
def _collect_full_rollout(env: NSRLBatchedEnv, policy) -> dict:
    """Closed-loop rollout of `policy` (deterministic mean); record
    (obs, q, line_dir, n_target) of every active step."""
    out = {"obs": [], "q": [], "ld": [], "nt": []}
    env.reset()
    for _ in range(env.max_steps + 1):
        active = ~env.done_persistent
        if bool(active.any().item()):
            obs = env.current_obs()
            out["obs"].append(obs[active].float().cpu())
            out["q"].append(env.q[active].float().cpu())
            out["ld"].append(env.line_dir[active].float().cpu())
            out["nt"].append(env.n_target[active].float().cpu())
        a = policy.actor_mean(env.current_obs()).clamp(-1.0, 1.0)
        env.step(a, auto_reset=False)
        if bool(env.done_persistent.all().item()):
            break
    return {k: torch.cat(v) for k, v in out.items()}


def mpc_distill(out_dir, *,
                behavior_ckpt="Yuan/RL_controller/runs/distill_r6_soft",
                pi0_ckpt="Yuan/RL_controller/runs/p0_progress_only_30M_0520",
                n_tasks: int = 12288, dagger_rounds: int = 1,
                tau_hi: float = 0.975, band: float = 0.02,
                K: int = 16, hold_H: int = 10, mpc_chunk: int = 2048,
                chunk_size: int = 4096, seed: int = 8700,
                epochs: int = 80, device=None, verbose: bool = True) -> dict:
    """Teacher labels:  qn < tau_hi-band        -> pi0 action
                        qn in [tau_hi-band, tau_hi] -> linear blend
                        qn >= tau_hi            -> MPC searched action
    Student warm-starts from `behavior_ckpt` and provides DAgger behavior."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device if device is not None
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    student, cfg_yaml = load_agent(behavior_ckpt, device)
    pi0, _ = load_agent(pi0_ckpt, device)
    env_kw = load_env_kw(cfg_yaml)
    line_cfg = cfg_yaml["line_distribution"]
    threshold_m = (float(line_cfg["feasibility_threshold_m"])
                   if line_cfg.get("feasibility_filter", False) else None)

    proxy = NSRLBatchedEnv(EnvConfig(**{**env_kw, "n_envs": 1}),
                           line_dist=None, device=device)
    pool = LineDistribution.load_or_build(
        kin=proxy.kin, collision=proxy.collision,
        n_pool=line_cfg["n_pool"],
        n_target_noise_deg=line_cfg["n_target_noise_deg"],
        seed=line_cfg["train_seed"], env_cfg=EnvConfig(**{**env_kw, "n_envs": 1}),
        feasibility_threshold_m=threshold_m, verbose=verbose)
    del proxy

    gen = torch.Generator(device=device).manual_seed(seed)
    obs_all, act_all = [], []
    stats = {"rounds": []}

    for rnd in range(dagger_rounds + 1):
        tasks = pool.sample(n_tasks, generator=gen)
        parts = []
        for start in range(0, n_tasks, chunk_size):
            end = min(start + chunk_size, n_tasks)
            env = NSRLBatchedEnv(EnvConfig(**{**env_kw, "n_envs": end - start}),
                                 line_dist=None, device=device)
            env.line_dist = ScriptedLineDistribution(
                {k: v[start:end].clone() for k, v in tasks.items()})
            parts.append(_collect_full_rollout(env, student))
            del env
            if device.type == "cuda":
                torch.cuda.empty_cache()
        states = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
        n_states = states["obs"].shape[0]
        qn = states["obs"][:, :7].abs().max(dim=-1).values

        # Safe labels from pi0 (GPU-chunked). no_grad is essential: a label
        # tensor carrying pi0's autograd graph poisons fit_actor's backward.
        labels = torch.zeros((n_states, 4))
        with torch.no_grad():
            for s in range(0, n_states, 65536):
                e = min(s + 65536, n_states)
                labels[s:e] = pi0.actor_mean(
                    states["obs"][s:e].to(device)).clamp(-1, 1).float().cpu()

        # Belt labels from exact MPC search.
        belt = qn >= (tau_hi - band)
        n_belt = int(belt.sum())
        if n_belt > 0:
            bidx = torch.nonzero(belt, as_tuple=False).squeeze(-1)
            mpc_a = torch.zeros((n_belt, 4))
            for s in range(0, n_belt, mpc_chunk):
                e = min(s + mpc_chunk, n_belt)
                sel = bidx[s:e]
                r = mpc_label_states(
                    states["q"][sel], states["ld"][sel], states["nt"][sel],
                    env_kw=env_kw, policy=student, device=device,
                    K=K, hold_H=hold_H, seed=seed + rnd * 1000 + s)
                mpc_a[s:e] = r["a_best"]
                if verbose:
                    print(f"[mpc-distill]   round {rnd}: MPC-labeled {e}/{n_belt}",
                          flush=True)
            w = ((qn[bidx] - (tau_hi - band)) / band).clamp(0, 1).unsqueeze(-1)
            labels[bidx] = (1 - w) * labels[bidx] + w * mpc_a

        if verbose:
            print(f"[mpc-distill] round {rnd}: {n_states} states "
                  f"({n_belt} belt / {100*n_belt/max(n_states,1):.1f}%)")
        obs_all.append(states["obs"])
        act_all.append(labels.detach().clamp(-TARGET_CLAMP, TARGET_CLAMP))
        obs_cat, act_cat = torch.cat(obs_all), torch.cat(act_all)
        # Checkpoint the dataset BEFORE fitting — MPC labels are the
        # expensive part; a fit-stage crash must not lose them.
        np.savez_compressed(out_dir / f"dataset_round{rnd}.npz",
                            obs=states["obs"].numpy(),
                            act=labels.detach().numpy())
        val = fit_actor(student, obs_cat, act_cat, device,
                        epochs=epochs, verbose=verbose)
        stats["rounds"].append({"round": rnd, "n_states": n_states,
                                "n_belt": n_belt, "val_mse": val})

    torch.save(student.state_dict(), out_dir / "agent.pt")
    cfg_out = dict(cfg_yaml)
    cfg_out["distill"] = {"teacher": "pi0(safe) + MPC-search(belt)",
                          "behavior_ckpt": str(behavior_ckpt),
                          "pi0_ckpt": str(pi0_ckpt), "tau_hi": tau_hi,
                          "band": band, "K": K, "hold_H": hold_H,
                          "n_tasks_per_round": n_tasks, "seed": seed}
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg_out, f, sort_keys=False)
    np.savez_compressed(out_dir / "distill_dataset.npz",
                        obs=obs_cat.numpy(), act=act_cat.numpy())
    if verbose:
        print(f"[mpc-distill] saved -> {out_dir}/agent.pt")
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--behavior-ckpt",
                        default="Yuan/RL_controller/runs/distill_r6_soft")
    parser.add_argument("--pi0-ckpt",
                        default="Yuan/RL_controller/runs/p0_progress_only_30M_0520")
    parser.add_argument("--n-tasks", type=int, default=12288)
    parser.add_argument("--dagger-rounds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=8700)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    mpc_distill(args.out_dir, behavior_ckpt=args.behavior_ckpt,
                pi0_ckpt=args.pi0_ckpt, n_tasks=args.n_tasks,
                dagger_rounds=args.dagger_rounds, seed=args.seed,
                device=args.device)


if __name__ == "__main__":
    main()

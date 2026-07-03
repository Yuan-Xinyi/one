"""Distill hybrid(pi0, classical) into a single policy network (+ DAgger).

Diagnosis behind this (2026-07-02): the classical rescue action field is
easily representable from the 31-dim obs (supervised fit: R^2=0.907, per-dim
RMSE 0.06), but PPO cannot DISCOVER it — in the danger belt every from-scratch
attempt dies within a few steps, so advantages compare noise with noise (the
0.82 hard-task plateau of the entfix runs). Sequential plan:

  Phase 1 (this module): pure-supervised distillation of the FULL hybrid
      behavior map — pi0's action in the safe region, classical's in the
      danger belt — into one Agent-compatible actor. DAgger rounds fix the
      distribution shift. Gate: 10k rollout of pi_D should land ~1.02-1.05
      (matching the hybrid system as a single net).
  Phase 2 (train.py resume): entropy-fixed PPO fine-tune from pi_D. The
      policy starts ON the survival manifold, so danger-belt advantages
      compare "live 60 vs 80 steps" instead of "die at 5 vs 7" — the regime
      where PPO can refine. Target: beat the hybrid line.

Expert = MEMORYLESS single-threshold hybrid: a*(s) = classical(s) if
max|q_norm(s)| >= tau else pi0(s). No hysteresis — chattering is an execution
artifact, irrelevant to a distilled smooth function (te=tx=0.98 variant also
scores +0.4pp over the hysteresis one per the 2026-05-30 sweep).

Usage:
    python -m Yuan.RL_controller.self_improve.distill \\
        --pi0-ckpt-dir Yuan/RL_controller/runs/p0_progress_only_30M_0520 \\
        --out-dir Yuan/RL_controller/runs/distill_hybrid_0702
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
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml

from Yuan.RL_controller.env.env import (
    NSRLBatchedEnv, EnvConfig, OBS_DIM, ACT_DIM, build_task_aligned_basis)
from Yuan.RL_controller.env.line_distribution import (
    LineDistribution, ScriptedLineDistribution)
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.RL_controller.algorithms.ppo import Agent
from Yuan.RL_controller.self_improve.collect import load_agent, load_env_kw

TARGET_CLAMP = 0.999  # keep tanh-regression targets strictly inside (-1, 1)


@torch.no_grad()
def _expert_step(env: NSRLBatchedEnv, pi0: Agent,
                 classical: ClassicalNullspaceController, tau: float,
                 soft_band: float = 0.0):
    """Expert label at the env's CURRENT state for every env.

    soft_band > 0 makes the teacher CONTINUOUS: labels blend linearly from
    the policy action to the classical action over qn in [tau-soft_band, tau]
    (a hard switch is a discontinuity a tanh-MLP can never fit exactly —
    the boundary layer was a measurable chunk of the residual fidelity gap).

    Returns (obs, a_expert, use_cls_mask)."""
    obs = env.current_obs()
    rl_act = pi0.actor_mean(obs).clamp(-1.0, 1.0)
    B_basis, _ = build_task_aligned_basis(
        env.kin, env.q, env.line_dir, env.n_target,
        env.kin.q_mid, env.q_half, env.cfg.manip_damping)
    q_dot = classical.q_dot_null(env.q, env.line_dir, env.n_target)
    cls_act = (B_basis.transpose(-1, -2) @ q_dot.unsqueeze(-1)).squeeze(-1)
    cls_act = torch.nan_to_num(cls_act / env.a_max, nan=0.0).clamp(-1.0, 1.0)
    qn = ((env.q - env.q_mid).abs() / env.q_half).max(dim=-1).values
    if soft_band > 0.0:
        w = ((qn - (tau - soft_band)) / soft_band).clamp(0.0, 1.0).unsqueeze(-1)
        a = (1.0 - w) * rl_act + w * cls_act
    else:
        a = torch.where((qn >= tau).unsqueeze(-1), cls_act, rl_act)
    use_cls = qn >= tau
    return obs, a, use_cls


@torch.no_grad()
def gen_labeled_rollout(env: NSRLBatchedEnv, behavior_fn, pi0: Agent,
                        classical: ClassicalNullspaceController, tau: float,
                        soft_band: float = 0.0) -> dict:
    """Roll `behavior_fn(env)->(B,4)` from reset; at every step record the
    EXPERT label at the visited state (DAgger). behavior_fn=None means the
    expert itself is the behavior (round-0 data)."""
    obs_l, act_l, cls_l = [], [], []
    env.reset()
    for _ in range(env.max_steps + 1):
        obs, a_exp, use_cls = _expert_step(env, pi0, classical, tau, soft_band)
        active = ~env.done_persistent
        if bool(active.any().item()):
            obs_l.append(obs[active].float().cpu())
            act_l.append(a_exp[active].float().cpu())
            cls_l.append(use_cls[active].cpu())
        a_beh = a_exp if behavior_fn is None else behavior_fn(env)
        env.step(a_beh, auto_reset=False)
        if bool(env.done_persistent.all().item()):
            break
    return {"obs": torch.cat(obs_l), "act": torch.cat(act_l),
            "use_cls": torch.cat(cls_l)}


def _student_fn(student: Agent):
    @torch.no_grad()
    def _fn(env):
        return student.actor_mean(env.current_obs()).clamp(-1.0, 1.0)
    return _fn


def fit_actor(student: Agent, obs: torch.Tensor, act: torch.Tensor,
              device, epochs: int = 40, bs: int = 4096, lr: float = 1e-3,
              verbose: bool = True) -> float:
    """Supervised regression of the student's deterministic actor
    (tanh(mean_head(trunk))) onto expert actions. Critic/log_std untouched.
    Cosine LR anneal over epochs (fidelity is the binding constraint of the
    ExIt loop — the final low-lr epochs buy the last fraction of val MSE)."""
    params = (list(student._actor_trunk.parameters())
              + list(student._mean_head.parameters()))
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = obs.shape[0]
    n_val = max(n // 20, 1)
    perm = torch.randperm(n)
    tr, va = perm[n_val:], perm[:n_val]
    obs_tr, act_tr = obs[tr].to(device), act[tr].to(device)
    obs_va, act_va = obs[va].to(device), act[va].to(device)
    val = float("nan")
    for ep in range(epochs):
        order = torch.randperm(obs_tr.shape[0], device=device)
        for s in range(0, len(order), bs):
            i = order[s:s + bs]
            pred = torch.tanh(student._mean_head(student._actor_trunk(obs_tr[i])))
            loss = ((pred - act_tr[i]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if verbose and (ep + 1) % 10 == 0:
            with torch.no_grad():
                pred = torch.tanh(student._mean_head(student._actor_trunk(obs_va)))
                val = float(((pred - act_va) ** 2).mean().item())
            print(f"[distill]   epoch {ep+1}/{epochs}  val MSE {val:.5f}")
    return val


def distill(pi0_ckpt_dir, out_dir, *, n_tasks: int = 16384,
            dagger_rounds: int = 2, tau: float = 0.98,
            soft_band: float = 0.0, init_from: str | None = None,
            chunk_size: int = 4096, seed: int = 8200,
            epochs: int = 40, hidden_dim: int | None = None,
            device=None, verbose: bool = True) -> dict:
    pi0_ckpt_dir = Path(pi0_ckpt_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device if device is not None
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    pi0, cfg_yaml = load_agent(pi0_ckpt_dir, device)
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
    del proxy

    # Student: full Agent so the ckpt is PPO-resumable (critic random-init,
    # log_std at config init — Phase 2 trains those). hidden_dim may be
    # widened beyond the teacher's: distillation fidelity is the binding
    # constraint of the ExIt loop, and capacity buys fidelity.
    student_hidden = int(hidden_dim or cfg_yaml["ppo"]["hidden_dim"])
    student = Agent(OBS_DIM, ACT_DIM,
                    hidden_dim=student_hidden,
                    init_log_std=cfg_yaml["ppo"]["init_log_std"]).to(device)
    if init_from is not None:
        # Warm start: safe-region behavior starts correct, the fit budget
        # concentrates on the belt (recovers most of the easy-layer loss).
        student.load_state_dict(torch.load(Path(init_from) / "agent.pt",
                                           map_location=device))
        if verbose:
            print(f"[distill] student warm-started from {init_from}")

    obs_all: list[torch.Tensor] = []
    act_all: list[torch.Tensor] = []
    stats = {"rounds": []}
    gen = torch.Generator(device=device).manual_seed(seed)

    for rnd in range(dagger_rounds + 1):
        behavior = None if rnd == 0 else _student_fn(student)
        tag = "expert(round0)" if rnd == 0 else f"student(dagger{rnd})"
        tasks = pool.sample(n_tasks, generator=gen)
        n_cls = n_steps = 0
        for start in range(0, n_tasks, chunk_size):
            end = min(start + chunk_size, n_tasks)
            env = NSRLBatchedEnv(EnvConfig(**{**env_kw, "n_envs": end - start}),
                                 line_dist=None, device=device)
            env.line_dist = ScriptedLineDistribution(
                {k: v[start:end].clone() for k, v in tasks.items()})
            classical = ClassicalNullspaceController(env.kin)
            d = gen_labeled_rollout(env, behavior, pi0, classical, tau, soft_band)
            obs_all.append(d["obs"])
            act_all.append(d["act"].clamp(-TARGET_CLAMP, TARGET_CLAMP))
            n_cls += int(d["use_cls"].sum()); n_steps += d["obs"].shape[0]
            del env
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if verbose:
            print(f"[distill] round {rnd} ({tag}): +{n_steps} states "
                  f"({100*n_cls/max(n_steps,1):.1f}% classical-labeled), "
                  f"dataset total {sum(o.shape[0] for o in obs_all)}")
        obs_cat, act_cat = torch.cat(obs_all), torch.cat(act_all)
        val = fit_actor(student, obs_cat, act_cat, device,
                        epochs=epochs, verbose=verbose)
        stats["rounds"].append({"round": rnd, "n_new_states": n_steps,
                                "frac_cls": n_cls / max(n_steps, 1),
                                "val_mse": val})

    # Save as a standard ckpt_dir (config.yaml + agent.pt) for eval tools and
    # Phase-2 PPO resume.
    torch.save(student.state_dict(), out_dir / "agent.pt")
    cfg_out = dict(cfg_yaml)
    # Persist the student's actual width so eval tools / PPO resume build
    # the right architecture from this ckpt_dir.
    cfg_out["ppo"] = dict(cfg_yaml["ppo"], hidden_dim=student_hidden)
    cfg_out["distill"] = {"pi0_ckpt": str(pi0_ckpt_dir), "tau": tau,
                          "soft_band": soft_band, "init_from": str(init_from),
                          "n_tasks_per_round": n_tasks,
                          "dagger_rounds": dagger_rounds, "seed": seed,
                          "note": "agent.pt: actor distilled from hybrid; "
                                  "critic RANDOM-INIT (Phase-2 PPO trains it)"}
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg_out, f, sort_keys=False)
    np.savez_compressed(out_dir / "distill_dataset.npz",
                        obs=obs_cat.numpy(), act=act_cat.numpy())
    if verbose:
        print(f"[distill] saved -> {out_dir}/agent.pt "
              f"(dataset {obs_cat.shape[0]} states cached)")
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pi0-ckpt-dir",
                        default="Yuan/RL_controller/runs/p0_progress_only_30M_0520")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-tasks", type=int, default=16384)
    parser.add_argument("--dagger-rounds", type=int, default=2)
    parser.add_argument("--tau", type=float, default=0.98)
    parser.add_argument("--seed", type=int, default=8200)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--hidden-dim", type=int, default=None,
                        help="student width override (default: teacher's)")
    parser.add_argument("--soft-band", type=float, default=0.0)
    parser.add_argument("--init-from", default=None,
                        help="warm-start student from this ckpt_dir")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    distill(args.pi0_ckpt_dir, args.out_dir, n_tasks=args.n_tasks,
            dagger_rounds=args.dagger_rounds, tau=args.tau,
            seed=args.seed, epochs=args.epochs, hidden_dim=args.hidden_dim,
            soft_band=args.soft_band, init_from=args.init_from,
            device=args.device)


if __name__ == "__main__":
    main()

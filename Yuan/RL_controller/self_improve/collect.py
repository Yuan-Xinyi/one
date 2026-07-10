"""Round-k rescue-step collection for the self-improvement loop.

Runs hybrid(pi_k, classical, hysteresis tau) on training-distribution tasks,
recording every classical-rescue step (obs, a_cls). Then runs pure pi_k on the
SAME tasks and applies a task-level win filter: only rescue steps from tasks
where hybrid strictly outlived pure RL survive. The surviving pairs are the
*verified* improvement signal that the next round's PPO internalizes via the
BC auxiliary loss (see algorithms/ppo.py `bc_*` args).

Usage (standalone; loop.py calls collect_buffer() directly):
    python -m Yuan.RL_controller.self_improve.collect \\
        --ckpt-dir Yuan/RL_controller/runs/p0_progress_only_30M_0520 \\
        --n-tasks 16384 --seed 7001 \\
        --out Yuan/RL_controller/runs/self_improve/round1/buffer.npz
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
import dataclasses
from pathlib import Path

import numpy as np
import torch
import yaml

from Yuan.RL_controller.env.env import (
    NSRLBatchedEnv, EnvConfig, OBS_DIM, ACT_DIM, build_task_aligned_basis)
from Yuan.RL_controller.env.line_distribution import (
    LineDistribution, ScriptedLineDistribution)
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.RL_controller.env.rollout import rollout_first_episode
from Yuan.RL_controller.algorithms.ppo import Agent

# BC targets clamped strictly inside (-1, 1): an exactly-saturated target asks
# tanh(mu) -> +-1 i.e. mu -> +-inf (vanishing-but-never-satisfied gradient).
BC_TARGET_CLAMP = 0.999


def load_env_kw(cfg_yaml: dict) -> dict:
    """Filter the yaml env section down to EnvConfig fields (older run configs
    may carry stale shaping keys, same guard as eval/rl_vs_classical.py)."""
    valid = {f.name for f in dataclasses.fields(EnvConfig)}
    return {k: v for k, v in cfg_yaml["env"].items() if k in valid}


def load_agent(ckpt_dir: Path, device: torch.device) -> tuple[Agent, dict]:
    """Build Agent from <ckpt_dir>/{config.yaml, agent.pt}. Returns (agent, cfg_yaml)."""
    ckpt_dir = Path(ckpt_dir)
    with open(ckpt_dir / "config.yaml") as f:
        cfg_yaml = yaml.safe_load(f)
    agent = Agent(OBS_DIM, ACT_DIM,
                  hidden_dim=cfg_yaml["ppo"]["hidden_dim"],
                  init_log_std=cfg_yaml["ppo"]["init_log_std"]).to(device)
    agent.load_state_dict(torch.load(ckpt_dir / "agent.pt", map_location=device))
    agent.eval()
    return agent, cfg_yaml


def rl_action_fn(agent: Agent):
    @torch.no_grad()
    def _fn(env: NSRLBatchedEnv) -> torch.Tensor:
        return agent.actor_mean(env.current_obs()).clamp(-1.0, 1.0)
    return _fn


@torch.no_grad()
def hybrid_rollout_collect(env: NSRLBatchedEnv, agent: Agent,
                           classical: ClassicalNullspaceController,
                           tau_enter: float, tau_exit: float) -> dict:
    """Variant-B hysteresis hybrid rollout that additionally records
    (obs, a_cls, env_idx) at every classical-controlled step of active envs.

    Mirrors eval/hybrid.py run_hybrid_rollout; buffers are returned on CPU
    as float32 so multi-chunk collection doesn't hold GPU memory.
    """
    n = env.n_envs
    device = env.device
    q_mid, q_half = env.q_mid, env.q_half

    def _max_abs_qn(q):
        return ((q - q_mid).abs() / q_half).max(dim=-1).values

    env.reset()
    using_rl = _max_abs_qn(env.q) < tau_enter
    switch_count = torch.zeros(n, dtype=torch.long, device=device)
    episode_len = torch.full((n,), -1, dtype=torch.long, device=device)
    finished = torch.zeros(n, dtype=torch.bool, device=device)
    env_arange = torch.arange(n, device=device)

    obs_chunks: list[torch.Tensor] = []
    act_chunks: list[torch.Tensor] = []
    idx_chunks: list[torch.Tensor] = []

    for _ in range(env.max_steps + 1):
        cur_qn = _max_abs_qn(env.q)
        new_using_rl = torch.where(
            using_rl,
            cur_qn < tau_enter,   # RL: stay while under enter threshold
            cur_qn < tau_exit,    # Cls: come back only once past exit threshold
        )
        active = ~env.done_persistent
        switch_count = switch_count + ((new_using_rl != using_rl) & active).long()
        using_rl = new_using_rl

        obs = env.current_obs()
        rl_act = agent.actor_mean(obs).clamp(-1.0, 1.0)

        # Classical action: q_dot_null projected onto the shared task-aligned basis.
        B_basis, _ = build_task_aligned_basis(
            env.kin, env.q, env.line_dir, env.n_target,
            env.kin.q_mid, env.q_half, env.cfg.manip_damping,
        )
        q_dot = classical.q_dot_null(env.q, env.line_dir, env.n_target)
        cls_act = (B_basis.transpose(-1, -2) @ q_dot.unsqueeze(-1)).squeeze(-1)
        cls_act = (cls_act / env.a_max).clamp(-1.0, 1.0)

        # Rescue step = active env currently under classical control.
        rescue = active & ~using_rl
        if bool(rescue.any().item()):
            obs_chunks.append(obs[rescue].float().cpu())
            act_chunks.append(cls_act[rescue].float().cpu())
            idx_chunks.append(env_arange[rescue].cpu())

        a = torch.where(using_rl.unsqueeze(-1), rl_act, cls_act)
        _, _, _, _, info = env.step(a, auto_reset=False)
        new_done = info["episode_done"]
        if bool(new_done.any().item()):
            episode_len[new_done] = env.t[new_done]
            finished = finished | new_done
        if bool(env.done_persistent.all().item()):
            break

    episode_len[~finished] = env.t[~finished]

    if obs_chunks:
        obs_buf = torch.cat(obs_chunks)
        act_buf = torch.cat(act_chunks)
        idx_buf = torch.cat(idx_chunks)
    else:
        obs_buf = torch.zeros((0, OBS_DIM))
        act_buf = torch.zeros((0, ACT_DIM))
        idx_buf = torch.zeros((0,), dtype=torch.long)
    return {"episode_len": episode_len, "switch_count": switch_count,
            "obs": obs_buf, "a_cls": act_buf, "env_idx": idx_buf}


def collect_buffer(ckpt_dir, out_path, *, n_tasks: int = 16384,
                   seed: int = 7001, tau_enter: float = 0.98,
                   tau_exit: float = 0.94, chunk_size: int = 4096,
                   device: torch.device | str | None = None,
                   restrict_idx=None,
                   verbose: bool = True) -> dict:
    """Collect the win-filtered rescue buffer for one self-improvement round.

    Tasks come from the training-distribution pool (train_seed, feasibility-
    filtered, disk-cached) sampled with an explicit `seed` generator — use a
    fresh seed per round so each round's operator acts on new tasks.

    Saves `out_path` (npz) and returns a stats dict.
    """
    ckpt_dir = Path(ckpt_dir)
    device = torch.device(device if device is not None
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    agent, cfg_yaml = load_agent(ckpt_dir, device)
    env_kw = load_env_kw(cfg_yaml)
    line_cfg = cfg_yaml["line_distribution"]
    threshold_m = (float(line_cfg["feasibility_threshold_m"])
                   if line_cfg.get("feasibility_filter", False) else None)

    # Pool bound to a throwaway env's kin/collision (cache-key matches training).
    pool_env_cfg = EnvConfig(**{**env_kw, "n_envs": 1})
    proxy = NSRLBatchedEnv(pool_env_cfg, line_dist=None, device=device)
    pool = LineDistribution.load_or_build(
        kin=proxy.kin, collision=proxy.collision,
        n_pool=line_cfg["n_pool"],
        n_target_noise_deg=line_cfg["n_target_noise_deg"],
        seed=line_cfg["train_seed"],
        env_cfg=pool_env_cfg,
        feasibility_threshold_m=threshold_m,
        verbose=verbose,
    )
    if restrict_idx is not None:
        # Leak-safety: restrict the pool to a task-index subset (e.g. the 95k
        # train split) so collected DAgger data never touches held-out test tasks.
        m = torch.zeros(pool.valid_mask.shape[0], dtype=torch.bool, device=pool.valid_mask.device)
        m[torch.as_tensor(restrict_idx, device=m.device, dtype=torch.long)] = True
        pool.valid_mask = pool.valid_mask & m
        if verbose:
            print(f"[collect] restricted pool to {int(pool.valid_mask.sum())} tasks (train-split only)")
    gen = torch.Generator(device=device).manual_seed(int(seed))
    tasks = pool.sample(n_tasks, generator=gen)
    del proxy

    T_hyb = torch.zeros(n_tasks, dtype=torch.long)
    T_pure = torch.zeros(n_tasks, dtype=torch.long)
    switches = torch.zeros(n_tasks, dtype=torch.long)
    obs_all: list[torch.Tensor] = []
    act_all: list[torch.Tensor] = []
    task_all: list[torch.Tensor] = []

    for start in range(0, n_tasks, chunk_size):
        end = min(start + chunk_size, n_tasks)
        chunk_n = end - start
        spec = {k: v[start:end].clone() for k, v in tasks.items()}
        env_cfg = EnvConfig(**{**env_kw, "n_envs": chunk_n})
        env = NSRLBatchedEnv(env_cfg, line_dist=None, device=device)
        classical = ClassicalNullspaceController(env.kin)

        # Hybrid pass (records rescue steps).
        env.line_dist = ScriptedLineDistribution({k: v.clone() for k, v in spec.items()})
        hyb = hybrid_rollout_collect(env, agent, classical, tau_enter, tau_exit)
        T_hyb[start:end] = hyb["episode_len"].cpu()
        switches[start:end] = hyb["switch_count"].cpu()
        obs_all.append(hyb["obs"])
        act_all.append(hyb["a_cls"])
        task_all.append(hyb["env_idx"] + start)

        # Pure pi_k pass on the same tasks (win-filter reference).
        env.line_dist = ScriptedLineDistribution({k: v.clone() for k, v in spec.items()})
        pure = rollout_first_episode(env, rl_action_fn(agent))
        T_pure[start:end] = pure["episode_len"].cpu()

        if verbose:
            print(f"[collect] {end}/{n_tasks} tasks  "
                  f"(rescue steps so far: {sum(o.shape[0] for o in obs_all)})",
                  flush=True)
        del env
        if device.type == "cuda":
            torch.cuda.empty_cache()

    obs = torch.cat(obs_all)
    a_cls = torch.cat(act_all).clamp(-BC_TARGET_CLAMP, BC_TARGET_CLAMP)
    task_idx = torch.cat(task_all)

    # Task-level win filter: keep rescue steps only from tasks where hybrid
    # STRICTLY outlived pure RL — verified improvement, not classical worship.
    win = T_hyb > T_pure
    keep = win[task_idx]
    obs_kept, act_kept, task_kept = obs[keep], a_cls[keep], task_idx[keep]

    n_win = int(win.sum().item())
    stats = {
        "n_tasks": n_tasks,
        "n_win_tasks": n_win,
        "frac_win": n_win / n_tasks,
        "n_rescue_steps": int(obs.shape[0]),
        "n_kept_steps": int(obs_kept.shape[0]),
        "mean_switches": float(switches.float().mean().item()),
        "mean_len_ratio_hyb_pure": float(
            (T_hyb.double() / T_pure.double().clamp(min=1)).mean().item()),
    }
    if verbose:
        print(f"[collect] win tasks {n_win}/{n_tasks} ({100*stats['frac_win']:.1f}%)  "
              f"rescue steps {stats['n_rescue_steps']} -> kept {stats['n_kept_steps']}  "
              f"mean_switches {stats['mean_switches']:.2f}  "
              f"T_hyb/T_pure {stats['mean_len_ratio_hyb_pure']:.3f}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        obs=obs_kept.numpy(),
        a_cls=act_kept.numpy(),
        task_idx=task_kept.numpy(),
        T_hybrid=T_hyb.numpy(),
        T_pure=T_pure.numpy(),
        win=win.numpy(),
        switch_count=switches.numpy(),
        q0=tasks["q0"].cpu().numpy().astype(np.float32),
        line_dir=tasks["line_dir"].cpu().numpy().astype(np.float32),
        n_target=tasks["n_target"].cpu().numpy().astype(np.float32),
        tau_enter=np.float64(tau_enter),
        tau_exit=np.float64(tau_exit),
        seed=np.int64(seed),
        ckpt_dir=np.str_(str(ckpt_dir)),
    )
    if verbose:
        print(f"[collect] saved buffer -> {out_path} "
              f"({out_path.stat().st_size / 1e6:.1f} MB)")
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-tasks", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=7001)
    parser.add_argument("--tau-enter", type=float, default=0.98)
    parser.add_argument("--tau-exit", type=float, default=0.94)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    collect_buffer(args.ckpt_dir, args.out, n_tasks=args.n_tasks,
                   seed=args.seed, tau_enter=args.tau_enter,
                   tau_exit=args.tau_exit, chunk_size=args.chunk_size,
                   device=args.device)


if __name__ == "__main__":
    main()

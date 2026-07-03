"""Classical value function V_cls: exact lower bound for the PPO critic.

Since the classical controller is always executable, V*(s) >= V_cls(s) for
every state. The env is a deterministic batched simulator, so V_cls labels are
EXACT: from any state (q, line_dir, n_target), roll the classical controller
to termination and accumulate the discounted progress reward.

Pipeline (main / build_vcls):
  1. collect states along classical AND policy rollouts on train-distribution
     tasks (both visitation distributions, subsampled every `stride` steps);
  2. label every state by a fresh classical rollout from it (episode re-rooted
     at the state: p_start := FK(q_s); lateral drift is negligible so the
     re-rooting bias is immaterial);
  3. fit a small MLP obs(31) -> G_cls (targets standardized; mean/std stored);
  4. save vcls.pt {state_dict, target_mean, target_std, gamma} + dataset npz.

At PPO time (algorithms/ppo.py `critic_floor_*` args) the frozen net provides
    L_critic += coef * relu(V_cls(s)/reward_scale - V(s))^2
so the critic never writes off states that classical could survive — the
advantage of classical-like rescue actions in the danger zone stays positive
even in unprotected curriculum episodes.

Usage:
    python -m Yuan.RL_controller.self_improve.vcls \\
        --ckpt-dir Yuan/RL_controller/runs/p0_progress_only_30M_0520 \\
        --out Yuan/RL_controller/runs/vcls/vcls.pt
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
import torch.nn as nn

from Yuan.RL_controller.env.env import NSRLBatchedEnv, EnvConfig, OBS_DIM
from Yuan.RL_controller.env.line_distribution import (
    LineDistribution, ScriptedLineDistribution)
from Yuan.RL_controller.env.classical_nullspace import (
    ClassicalNullspaceController, cn_action_fn)
from Yuan.RL_controller.self_improve.collect import (
    load_agent, load_env_kw, rl_action_fn)


class VClsNet(nn.Module):
    """obs(31) -> standardized G_cls. Use load_vcls() for raw-unit queries."""

    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_vcls(path, device):
    """Load a fitted V_cls net; returns callable obs -> conservative floor in
    RAW reward units: de-standardized prediction minus the stored margin.

    Floor errors are asymmetric — an overestimated floor pulls the policy into
    states classical cannot actually salvage, an underestimated one merely
    loosens the bound. The net is therefore quantile-fitted (pinball, low tau)
    and a validation-RMSE margin is subtracted on top.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = VClsNet().to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    mean, std = float(ckpt["target_mean"]), float(ckpt["target_std"])
    margin = float(ckpt.get("margin", 0.0))

    @torch.no_grad()
    def _fn(obs: torch.Tensor) -> torch.Tensor:
        return net(obs) * std + mean - margin

    return _fn


@torch.no_grad()
def _snapshot_rollout(env: NSRLBatchedEnv, action_fn, stride: int) -> dict:
    """auto_reset=False rollout; every `stride` steps snapshot (obs, q, task)
    of still-active envs. Returns cpu tensors."""
    obs_l, q_l, ld_l, nt_l = [], [], [], []
    env.reset()
    for step_i in range(env.max_steps + 1):
        if step_i % stride == 0:
            active = ~env.done_persistent
            if bool(active.any().item()):
                obs_l.append(env.current_obs()[active].float().cpu())
                q_l.append(env.q[active].float().cpu())
                ld_l.append(env.line_dir[active].float().cpu())
                nt_l.append(env.n_target[active].float().cpu())
        a = action_fn(env)
        env.step(a, auto_reset=False)
        if bool(env.done_persistent.all().item()):
            break
    return {"obs": torch.cat(obs_l), "q": torch.cat(q_l),
            "line_dir": torch.cat(ld_l), "n_target": torch.cat(nt_l)}


@torch.no_grad()
def _classical_discounted_return(env: NSRLBatchedEnv,
                                 ctrl: ClassicalNullspaceController,
                                 gamma: float) -> torch.Tensor:
    """Exact G_cls for the freshly scripted env batch: roll classical to
    termination, accumulate sum_t gamma^t r_t (frozen envs contribute 0)."""
    action_fn = cn_action_fn(ctrl)
    env.reset()
    n = env.n_envs
    G = torch.zeros(n, dtype=torch.float64, device=env.device)
    disc = 1.0
    for _ in range(env.max_steps + 1):
        a = action_fn(env)
        _, r, _, _, _ = env.step(a, auto_reset=False)
        G += disc * r.double()
        disc *= gamma
        if bool(env.done_persistent.all().item()):
            break
    return G.float()


def fit_vcls_from_dataset(dataset_path, out_path, *, quantile: float = 0.25,
                          margin_rmse_mult: float = 1.0, fit_epochs: int = 40,
                          device=None, verbose: bool = True) -> dict:
    """(Re)fit a conservative V_cls net from a saved .dataset.npz.

    Pinball loss at `quantile` (< 0.5 biases the estimate low), then stores
    margin = margin_rmse_mult * val RMSE to be subtracted at query time.
    Reports the post-margin overestimation rate on held-out states — the
    number that must stay small for the floor to be safe.
    """
    device = torch.device(device if device is not None
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    d = np.load(dataset_path)
    obs_all = torch.from_numpy(d["obs"]).float()
    labels = torch.from_numpy(d["G_cls"]).float()
    gamma = float(d["gamma"])
    n_states = obs_all.shape[0]

    t_mean, t_std = float(labels.mean()), float(labels.std().clamp(min=1e-6))
    targets = (labels - t_mean) / t_std
    n_val = max(n_states // 20, 1)
    perm = torch.randperm(n_states)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    net = VClsNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    obs_tr, tgt_tr = obs_all[tr_idx].to(device), targets[tr_idx].to(device)
    obs_val, tgt_val = obs_all[val_idx].to(device), targets[val_idx].to(device)
    bs = 4096

    def _pinball(pred, tgt):
        diff = tgt - pred
        return torch.max(quantile * diff, (quantile - 1.0) * diff).mean()

    for ep in range(fit_epochs):
        order = torch.randperm(obs_tr.shape[0], device=device)
        for s in range(0, obs_tr.shape[0], bs):
            idx = order[s:s + bs]
            loss = _pinball(net(obs_tr[idx]), tgt_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        if verbose and (ep + 1) % 10 == 0:
            with torch.no_grad():
                val_rmse = (((net(obs_val) - tgt_val) ** 2).mean().sqrt()
                            * t_std).item()
            print(f"[vcls] epoch {ep+1}/{fit_epochs}  val RMSE {val_rmse:.2f} raw")

    with torch.no_grad():
        pred_val_raw = net(obs_val) * t_std + t_mean
        tgt_val_raw = tgt_val * t_std + t_mean
        val_rmse = ((pred_val_raw - tgt_val_raw) ** 2).mean().sqrt().item()
        margin = margin_rmse_mult * val_rmse
        floor_val = pred_val_raw - margin
        overest_frac = float((floor_val > tgt_val_raw).float().mean().item())
        overest_mean_gap = float(
            torch.relu(floor_val - tgt_val_raw).mean().item())

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "target_mean": t_mean,
                "target_std": t_std, "gamma": gamma, "margin": margin,
                "quantile": quantile, "n_states": n_states,
                "val_rmse": val_rmse}, out_path)
    stats = {"n_states": n_states, "val_rmse_raw": val_rmse, "margin": margin,
             "floor_overest_frac": overest_frac,
             "floor_overest_mean_gap": overest_mean_gap}
    if verbose:
        print(f"[vcls] saved -> {out_path}")
        print(f"[vcls] conservative floor check (held-out): "
              f"overestimation rate {100*overest_frac:.1f}%  "
              f"mean overshoot {overest_mean_gap:.2f} raw units "
              f"(labels ~{t_mean:.0f}±{t_std:.0f}, margin {margin:.2f})")
    return stats


def build_vcls(ckpt_dir, out_path, *, n_tasks: int = 8192, stride: int = 4,
               gamma: float = 0.99, chunk_size: int = 4096,
               max_states: int = 200_000, seed: int = 8000,
               fit_epochs: int = 40, device=None, verbose: bool = True) -> dict:
    ckpt_dir = Path(ckpt_dir)
    out_path = Path(out_path)
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

    # ---- 1. states from classical + policy visitation ----
    parts = []
    for name, fn_builder in (("classical", None), ("policy", agent)):
        for start in range(0, n_tasks, chunk_size):
            end = min(start + chunk_size, n_tasks)
            spec = {k: v[start:end].clone() for k, v in tasks.items()}
            env = NSRLBatchedEnv(EnvConfig(**{**env_kw, "n_envs": end - start}),
                                 line_dist=None, device=device)
            env.line_dist = ScriptedLineDistribution(spec)
            action_fn = (cn_action_fn(ClassicalNullspaceController(env.kin))
                         if fn_builder is None else rl_action_fn(fn_builder))
            parts.append(_snapshot_rollout(env, action_fn, stride))
            del env
            torch.cuda.empty_cache() if device.type == "cuda" else None
        if verbose:
            n_so_far = sum(p["obs"].shape[0] for p in parts)
            print(f"[vcls] {name} states collected (running total {n_so_far})")

    states = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    n_states = states["obs"].shape[0]
    if n_states > max_states:
        pick = torch.randperm(n_states)[:max_states]
        states = {k: v[pick] for k, v in states.items()}
        n_states = max_states
    if verbose:
        print(f"[vcls] labeling {n_states} states with exact classical returns")

    # ---- 2. exact labels by classical rollout from each state ----
    labels = torch.zeros(n_states)
    for start in range(0, n_states, chunk_size):
        end = min(start + chunk_size, n_states)
        env = NSRLBatchedEnv(EnvConfig(**{**env_kw, "n_envs": end - start}),
                             line_dist=None, device=device)
        env.line_dist = ScriptedLineDistribution({
            "q0": states["q"][start:end].to(device, env.kin.dtype),
            "line_dir": states["line_dir"][start:end].to(device, env.kin.dtype),
            "n_target": states["n_target"][start:end].to(device, env.kin.dtype)})
        ctrl = ClassicalNullspaceController(env.kin)
        labels[start:end] = _classical_discounted_return(env, ctrl, gamma).cpu()
        if verbose and (end // chunk_size) % 5 == 0 or end == n_states:
            print(f"[vcls]   {end}/{n_states} labeled", flush=True)
        del env
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # ---- 3. save dataset, then conservative quantile fit ----
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path = out_path.with_suffix(".dataset.npz")
    np.savez_compressed(dataset_path,
                        obs=states["obs"].numpy(), G_cls=labels.numpy(),
                        gamma=np.float64(gamma))
    if verbose:
        print(f"[vcls] dataset saved -> {dataset_path}")
    return fit_vcls_from_dataset(dataset_path, out_path,
                                 fit_epochs=fit_epochs, device=device,
                                 verbose=verbose)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True,
                        help="policy used for the visitation-state half of the dataset")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-tasks", type=int, default=8192)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max-states", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=8000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--refit-from", default=None,
                        help="skip collection/labeling; refit from an existing "
                             ".dataset.npz (labeling is the expensive part)")
    parser.add_argument("--quantile", type=float, default=0.25)
    parser.add_argument("--margin-rmse-mult", type=float, default=1.0)
    args = parser.parse_args()
    if args.refit_from is not None:
        fit_vcls_from_dataset(args.refit_from, args.out,
                              quantile=args.quantile,
                              margin_rmse_mult=args.margin_rmse_mult,
                              device=args.device)
    else:
        build_vcls(args.ckpt_dir, args.out, n_tasks=args.n_tasks,
                   stride=args.stride, gamma=args.gamma,
                   max_states=args.max_states, seed=args.seed,
                   device=args.device)


if __name__ == "__main__":
    main()

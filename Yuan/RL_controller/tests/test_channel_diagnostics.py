"""Reward-shaping channel diagnostics.

Test 1 (dimensional): per-channel raw delta magnitudes under zero-nullspace
baseline. Reports mean / std / |mean| / |p50| / |p99| / |max| for each of the
four reward channels. Progress is per-step (Δp·u_hat, raw — pre-clip,
pre-weight); jl/cone/dm use N=10 lookback to match env's `delta_lookback`.
This is the magnitudes the policy gradient actually sees through advantage.

Test 2 (ranking): episode-cumulative reward under three baselines —
zero-nullspace, classical (Yoshikawa + JL + cone + q_ref), random. Expect
classical > zero > random. Any other ordering means the reward function
ranks bad-policy above good-policy and no amount of PPO will save it.

Run:
    python -m Yuan.RL_controller.tests.test_channel_diagnostics --test all
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
import math

import torch
import yaml

from Yuan.RL_controller.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.RL_controller.env.line_distribution import LineDistribution
from Yuan.RL_controller.env.baseline_controller import zero_nullspace_action_fn
from Yuan.RL_controller.env.classical_nullspace import (
    ClassicalNullspaceController, cn_action_fn,
)


def _compute_metrics(env: NSRLBatchedEnv, q: torch.Tensor):
    """Return (qsq_mean, cos_angle, w_u, p_tcp) at q under env's u_hat/n_target.
    Mirrors env.step()'s internal metric definitions exactly."""
    p, R, J, _ = env.kin.tcp_fk_jac(q)
    z = R[:, :, 2]
    q_norm = (q - env.q_mid) / env.q_half
    qsq = (q_norm * q_norm).mean(-1)
    cos_a = (z * env.n_target).sum(-1).clamp(-1.0, 1.0)
    J_p = J[:, :3, :]
    B = q.shape[0]
    eye3 = torch.eye(3, device=q.device, dtype=q.dtype).expand(B, 3, 3)
    JJt = J_p @ J_p.transpose(-1, -2) + (env.cfg.manip_damping ** 2) * eye3
    u_col = env.line_dir.unsqueeze(-1)
    sol = torch.linalg.solve(JJt, u_col)
    inv_quad = (u_col.transpose(-1, -2) @ sol).squeeze(-1).squeeze(-1)
    inv_quad = torch.nan_to_num(inv_quad, nan=1.0, posinf=1e12, neginf=1e-12).clamp_min(1e-12)
    w_u = inv_quad.pow(-0.5).clamp_min(1e-6)
    return qsq, cos_a, w_u, p


def _random_action_fn():
    def _fn(env: NSRLBatchedEnv) -> torch.Tensor:
        return (torch.rand(env.n_envs, env.act_dim,
                           device=env.device, dtype=env.kin.dtype) * 2.0 - 1.0)
    return _fn


def _load_env(cfg_path: str, n_envs_override: int | None,
              device: str, seed: int) -> NSRLBatchedEnv:
    with open(cfg_path) as f:
        y = yaml.safe_load(f)
    env_kwargs = dict(y["env"])
    if n_envs_override is not None:
        env_kwargs["n_envs"] = n_envs_override
    cfg = EnvConfig(**env_kwargs)
    env = NSRLBatchedEnv(cfg, line_dist=None, device=device)
    line_cfg = y["line_distribution"]
    thr = (float(line_cfg["feasibility_threshold_m"])
           if line_cfg.get("feasibility_filter", False) else None)
    env.line_dist = LineDistribution.load_or_build(
        kin=env.kin, collision=env.collision,
        n_pool=line_cfg["n_pool"],
        n_target_noise_deg=line_cfg["n_target_noise_deg"],
        seed=seed, env_cfg=cfg,
        feasibility_threshold_m=thr,
    )
    return env


# --------------------------------------------------------------- Test 1

def _print_stats_row(name: str, vals: torch.Tensor):
    v = vals.double()
    a = v.abs()
    print(f"  {name:>16}  n={v.numel():>6}  "
          f"mean={v.mean().item():+.5f}  "
          f"std={v.std().item():.5f}  "
          f"|mean|={a.mean().item():.5f}  "
          f"|p50|={a.median().item():.5f}  "
          f"|p99|={a.quantile(0.99).item():.5f}  "
          f"|max|={a.max().item():.5f}")


def test1_dimensional_diagnostics(cfg_path: str, device: str,
                                  n_envs: int, n_steps: int) -> None:
    """Per-channel raw delta magnitudes under zero-nullspace baseline.

    progress: per-step Δp·u_hat (raw, no clip, no weight)
    qsq/cos/wu: N=delta_lookback (default 10) lookback delta to match env.
    Signed convention matches env reward (positive = improvement).
    """
    env = _load_env(cfg_path, n_envs_override=n_envs, device=device, seed=0)
    env.reset()
    action_fn = zero_nullspace_action_fn()
    N = int(env.cfg.delta_lookback)
    print(f"\n[test1] dimensional diagnostics: zero-nullspace, "
          f"n_envs={n_envs}, n_steps={n_steps}, lookback N={N}")
    print(f"        dt={env.cfg.dt}, v={env.cfg.v}, "
          f"per-step v·dt={env.cfg.v*env.cfg.dt:.4f} m (progress upper bound)")
    print(f"        delta_scale K={env.cfg.delta_scale}, weights "
          f"(prog,jl,cone,dm)=({env._w_progress:.3f},{env._w_jl:.3f},"
          f"{env._w_cone:.3f},{env._w_dm:.3f})")

    # Per-step progress logged separately (always N=1).
    dp_log: list[torch.Tensor] = []
    # Lookback deltas: ring buffer of (n_envs, N) per metric.
    qsq_hist = torch.full((env.n_envs, N), float("nan"),
                          device=env.device, dtype=env.kin.dtype)
    cos_hist = torch.full((env.n_envs, N), float("nan"),
                          device=env.device, dtype=env.kin.dtype)
    wu_hist  = torch.full((env.n_envs, N), float("nan"),
                          device=env.device, dtype=env.kin.dtype)
    # "alive-N" mask per env, used to gate logging to only fully-clean N-step windows.
    age = torch.zeros((env.n_envs,), device=env.device, dtype=torch.long)

    dqsq_log: list[torch.Tensor] = []
    dcos_log: list[torch.Tensor] = []
    dwu_log:  list[torch.Tensor] = []

    qsq_prev, cos_prev, wu_prev, p_prev = _compute_metrics(env, env.q)
    line_dir = env.line_dir.clone()

    for step_i in range(n_steps):
        active_before = ~env.done_persistent
        if not bool(active_before.any().item()):
            break
        a = action_fn(env)
        _, _, _, _, info = env.step(a, auto_reset=False)
        # Per-step progress (always valid when env was active before)
        qsq_now, cos_now, wu_now, p_now = _compute_metrics(env, env.q)
        dp = ((p_now - p_prev) * line_dir).sum(-1)
        dp_log.append(dp[active_before].detach().cpu())

        # Lookback: read slot (step_i % N) BEFORE writing new value
        slot = step_i % N
        old_qsq = qsq_hist[:, slot]
        old_cos = cos_hist[:, slot]
        old_wu  = wu_hist[:, slot]
        # Valid if env was active for full N+1 step window: age must be ≥ N (i.e. we've
        # written this slot at least once and env never died in between).
        valid = active_before & (age >= N)
        if bool(valid.any().item()):
            dqsq_log.append((old_qsq - qsq_now)[valid].detach().cpu())
            dcos_log.append((cos_now - old_cos)[valid].detach().cpu())
            dwu_log .append((wu_now  - old_wu )[valid].detach().cpu())
        # Update ring buffer for active envs; dead envs keep stale data but won't be sampled
        qsq_hist[active_before, slot] = qsq_now[active_before]
        cos_hist[active_before, slot] = cos_now[active_before]
        wu_hist[active_before, slot]  = wu_now[active_before]
        age = torch.where(active_before, age + 1, age)

        p_prev, qsq_prev, cos_prev, wu_prev = p_now, qsq_now, cos_now, wu_now

    print(f"  channel              n     mean       std       |mean|     "
          f"|p50|      |p99|      |max|")
    _print_stats_row("delta_progress (N=1)", torch.cat(dp_log))
    if dqsq_log:
        _print_stats_row(f"delta_qsq (N={N})", torch.cat(dqsq_log))
        _print_stats_row(f"delta_cos (N={N})", torch.cat(dcos_log))
        _print_stats_row(f"delta_wu  (N={N})", torch.cat(dwu_log))
    else:
        print(f"  (no envs survived ≥ {N} steps under zero-nullspace; can't compute lookback delta)")

    # Per-step contribution to reward = weight * (raw delta) [* K for lookback channels]
    if dqsq_log:
        K = env.cfg.delta_scale
        wp, wjl, wcone, wdm = env._w_progress, env._w_jl, env._w_cone, env._w_dm
        # Progress reward contribution upper bound = wp * 1.0 (after clip)
        contrib = {
            "progress (wp · clip(dp/v·dt,0,1))": wp * 1.0,  # ceiling
            "jl       (wjl · K · |dqsq|)":   wjl   * K * torch.cat(dqsq_log).abs().mean().item(),
            "cone     (wcone· K · |dcos|)":  wcone * K * torch.cat(dcos_log).abs().mean().item(),
            "dm       (wdm  · K · |dwu|)":   wdm   * K * torch.cat(dwu_log).abs().mean().item(),
        }
        print(f"\n  per-step |reward contribution| under zero-nullspace "
              f"(progress = upper bound = wp · 1.0):")
        for name, v in contrib.items():
            print(f"    {name:<38} {v:.5f}")


# --------------------------------------------------------------- Test 2

def test2_reward_ranking(cfg_path: str, device: str, n_envs: int) -> None:
    """Episode-cumulative reward ranking: zero vs classical vs random.
    Expected classical > zero > random. Violations = reward ranks bad policy
    above good policy → reward design is broken regardless of PPO quality."""
    print(f"\n[test2] reward ranking: cumulative episode reward, "
          f"n_envs={n_envs} per policy (one full episode each)")
    results: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, mkfn in [
        ("zero",      lambda env: zero_nullspace_action_fn()),
        ("classical", lambda env: cn_action_fn(ClassicalNullspaceController(env.kin))),
        ("random",    lambda env: _random_action_fn()),
    ]:
        env = _load_env(cfg_path, n_envs_override=n_envs, device=device, seed=42)
        env.reset()
        action_fn = mkfn(env)
        ep_r_list: list[torch.Tensor] = []
        ep_l_list: list[torch.Tensor] = []
        cfg_max = env.max_steps
        for _ in range(cfg_max + 1):
            a = action_fn(env)
            _, _, _, _, info = env.step(a, auto_reset=False)
            done = info["episode_done"]
            if bool(done.any().item()):
                ep_r_list.append(env.episode_reward[done].detach().cpu())
                ep_l_list.append(env.episode_steps[done].detach().cpu())
            if bool(env.done_persistent.all().item()):
                break
        ep_r = torch.cat(ep_r_list).double() if ep_r_list else torch.tensor([])
        ep_l = torch.cat(ep_l_list).double() if ep_l_list else torch.tensor([])
        results[name] = (ep_r, ep_l)

    print(f"  {'policy':>10}  {'n':>4}  {'reward (mean ± std)':>25}  "
          f"{'[p25, p75]':>20}  {'len (mean)':>10}")
    for name in ("classical", "zero", "random"):
        r, l = results[name]
        q25 = r.quantile(0.25).item()
        q75 = r.quantile(0.75).item()
        print(f"  {name:>10}  {r.numel():>4}  "
              f"{r.mean().item():>+12.4f} ± {r.std().item():>8.4f}  "
              f"[{q25:>+7.4f}, {q75:>+7.4f}]  {l.mean().item():>10.1f}")
    r_cls = results["classical"][0].mean().item()
    r_zer = results["zero"][0].mean().item()
    r_rnd = results["random"][0].mean().item()
    ok = r_cls > r_zer > r_rnd
    print(f"\n  expected: classical > zero > random")
    print(f"  actual:   {r_cls:+.4f} {'>' if r_cls > r_zer else '≤'} "
          f"{r_zer:+.4f} {'>' if r_zer > r_rnd else '≤'} {r_rnd:+.4f}   "
          f"→ {'PASS' if ok else 'FAIL — reward ranks bad policy above good policy'}")


# --------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="Yuan/RL_controller/config.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--test", choices=["1", "2", "all"], default="all")
    ap.add_argument("--n-envs", type=int, default=128)
    ap.add_argument("--n-steps", type=int, default=1000,
                    help="test 1: number of env.step() calls")
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[diag] device={device}")
    if args.test in ("1", "all"):
        test1_dimensional_diagnostics(args.config, device=device,
                                      n_envs=args.n_envs, n_steps=args.n_steps)
    if args.test in ("2", "all"):
        test2_reward_ranking(args.config, device=device, n_envs=args.n_envs)


if __name__ == "__main__":
    main()

"""Unit tests for the length-based shaping reward.

Reward formula (signed per-step deltas; cumulative telescopes cleanly):
    r_t = w_progress · clip(Δp·u_hat / (v·dt), 0, 1)
        + w_jl   · K · (q_norm_sq_prev_mean - q_norm_sq_curr_mean)
        + w_cone · K · (cos_curr - cos_prev)
        + w_dm   · K · (w_u_curr - w_u_prev)

Weights are normalized at runtime to sum to 1. First step after reset → 0 delta.

Run:
    python -m Yuan.RL_controller.tests.test_reward
"""
from __future__ import annotations

import os, sys

import torch

_conda_lib = os.path.join(sys.prefix, "lib")
if _conda_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    new_env = dict(os.environ)
    new_env["LD_LIBRARY_PATH"] = _conda_lib + ":" + new_env.get("LD_LIBRARY_PATH", "")
    if __spec__ is not None and __spec__.name != "__main__":
        argv = [sys.executable, "-m", __spec__.name] + sys.argv[1:]
    else:
        argv = [sys.executable] + sys.argv
    os.execvpe(sys.executable, argv, new_env)

from Yuan.RL_controller.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.RL_controller.env.line_distribution import LineDistribution


def _build_env(n_envs=4, max_steps=100, seed=0, **overrides):
    cfg = EnvConfig(**{"n_envs": n_envs, "max_steps": max_steps, **overrides})
    env = NSRLBatchedEnv(cfg, line_dist=None, device="cpu")
    env.line_dist = LineDistribution(
        env.kin, env.collision, n_pool=200, seed=seed, batch_size=256)
    env.reset()
    return env


def _w_u_ref(env: NSRLBatchedEnv, q: torch.Tensor) -> torch.Tensor:
    """Reference w_u(q, u_hat) computation."""
    _, _, J, _ = env.kin.tcp_fk_jac(q)
    J_p = J[:, :3, :]
    eye3 = torch.eye(3, device=q.device, dtype=q.dtype).expand(q.shape[0], 3, 3)
    JJt = J_p @ J_p.transpose(-1, -2) + (env.cfg.manip_damping ** 2) * eye3
    u_col = env.line_dir.unsqueeze(-1)
    inv_quad = (u_col.transpose(-1, -2) @ torch.linalg.inv(JJt) @ u_col
                ).squeeze(-1).squeeze(-1).clamp_min(1e-12)
    return inv_quad.pow(-0.5)


# ---------- tests ----------------------------------------------------------

def test_weights_normalize_to_one():
    """Weights should be renormalized to sum=1 if user provides anything else."""
    env = _build_env(w_progress=2.0, w_jl=2.0, w_cone=2.0, w_dm=2.0)  # sum=8
    s = env._w_progress + env._w_jl + env._w_cone + env._w_dm
    assert abs(s - 1.0) < 1e-5, f"weights not normalized; sum={s}"
    assert abs(env._w_progress - 0.25) < 1e-5
    print(f"[ok] weight normalization: w_progress={env._w_progress:.4f}, sum={s:.4f}")


def test_first_step_delta_is_zero():
    """First step after reset: all deltas must be 0 (no spurious signal from NaN prev)."""
    # Run with only delta terms, no progress — first step reward should be ~0
    env = _build_env(w_progress=0.0, w_jl=1.0, w_cone=1.0, w_dm=1.0)
    # Force terminal penalties to 0 (default already 0)
    a = torch.zeros(env.n_envs, env.act_dim)
    _, rew, _, _, _ = env.step(a, auto_reset=False)
    assert torch.allclose(rew, torch.zeros_like(rew), atol=1e-5), \
        f"first-step rew should be 0 (w_progress=0, all deltas=0); got {rew}"
    print(f"[ok] first-step delta = 0; rew = {rew[0].item():.6f}")


def test_progress_only_full_speed():
    """w_progress=1, others 0, zero nullspace action: when far from singularity
    EE tracks line at full speed → per-step reward ≈ 1.0. Allow tolerance for
    damped-pinv shortfall near singular configs."""
    env = _build_env(w_progress=1.0, w_jl=0.0, w_cone=0.0, w_dm=0.0)
    a = torch.zeros(env.n_envs, env.act_dim)
    rewards = []
    for _ in range(15):
        _, r, _, _, _ = env.step(a, auto_reset=False)
        rewards.append(r.mean().item())
    avg = sum(rewards) / len(rewards)
    # Damped pinv may give < 1.0 near singular; LineDistribution.sample picks
    # reasonable configs so average should be high.
    assert 0.5 < avg <= 1.0 + 1e-4, f"progress-only mean expected ~1.0; got {avg}"
    print(f"[ok] progress-only mean reward (full speed): {avg:.6f}")


def test_telescoping_property():
    """Under signed per-step delta with delta_lookback=1, cumulative jl reward
    over T steps EXACTLY telescopes to K · w_jl · (q_norm_sq_after_step_1 −
    q_norm_sq_after_step_T). Modulo float precision the diff should be ~0; we
    allow < 0.1 to leave headroom for fp32 accumulation noise."""
    env = _build_env(w_progress=0.0, w_jl=1.0, w_cone=0.0, w_dm=0.0,
                     delta_lookback=1)

    torch.manual_seed(0)
    cumulative = torch.zeros(env.n_envs)
    q_norm_sq_after_step1 = None
    final_q_norm_sq_mean = None
    for i in range(10):
        a = torch.randn(env.n_envs, env.act_dim) * 0.3
        _, r, _, _, _ = env.step(a, auto_reset=False)
        cumulative = cumulative + r
        q_norm = (env.q - env.q_mid) / env.q_half
        q_norm_sq_mean = (q_norm * q_norm).mean(-1)
        if i == 0:
            q_norm_sq_after_step1 = q_norm_sq_mean.clone()
        final_q_norm_sq_mean = q_norm_sq_mean

    K = env.cfg.delta_scale
    # Step 1 contributes r=0 (NaN history sentinel); steps 2..T form a telescoping
    # sum that collapses to (qsq_after_step1 − qsq_after_stepT).
    expected = env._w_jl * K * (q_norm_sq_after_step1 - final_q_norm_sq_mean)
    diff = (cumulative - expected).abs()
    assert torch.all(diff < 0.1), \
        f"telescoping violated by > 0.1: diff = {diff}"
    print(f"[ok] telescoping: |cum − K·w·(qsq_step1 − qsq_final)| < 0.1, "
          f"got max={diff.max():.6f}")


def test_signed_delta_can_be_negative():
    """Regression test for the original positive-only-clamp bug.

    Under the old code, delta_jl/cone/dm were clamped at 0, so per-step shaping
    reward was always ≥ 0 — the policy could oscillate the metric and only
    accrue the "improvement" half of every cycle. Under signed delta, per-step
    rewards must be allowed to be negative when the metric degrades.

    We drive the env with random nullspace actions; q_norm_sq fluctuates in
    both directions; at least one step must produce a strictly negative reward
    (impossible under the old clamped code)."""
    env = _build_env(n_envs=16, w_progress=0.0, w_jl=1.0, w_cone=0.0, w_dm=0.0,
                     delta_lookback=1, max_steps=10000, seed=0)

    torch.manual_seed(1)
    a = torch.zeros(env.n_envs, env.act_dim)
    saw_negative = False
    saw_positive = False
    min_reward = float("inf")
    max_reward = float("-inf")
    for _ in range(30):
        a = torch.randn_like(a) * 0.5
        _, r, _, _, _ = env.step(a, auto_reset=False)
        active = ~env.done_persistent
        if not active.any():
            break
        r_active = r[active]
        min_reward = min(min_reward, float(r_active.min().item()))
        max_reward = max(max_reward, float(r_active.max().item()))
        if (r_active < -1e-4).any():
            saw_negative = True
        if (r_active > 1e-4).any():
            saw_positive = True

    assert saw_negative, (
        "Under signed delta some per-step rewards must be NEGATIVE (regression "
        f"test for positive-only-clamp bug). Observed reward range: "
        f"[{min_reward:.4f}, {max_reward:.4f}]")
    assert saw_positive, (
        f"Sanity: some per-step rewards must also be POSITIVE. "
        f"Observed range: [{min_reward:.4f}, {max_reward:.4f}]")
    print(f"[ok] signed delta: per-step reward spans "
          f"[{min_reward:.4f}, {max_reward:.4f}] (both signs present)")


def test_reset_clears_prev_caches():
    """After auto-reset of an env, history ring buffers are NaN, next step delta = 0 for that env."""
    env = _build_env(w_progress=0.0, w_jl=1.0, w_cone=1.0, w_dm=1.0, max_steps=3)
    a = torch.zeros(env.n_envs, env.act_dim)
    for _ in range(3):
        _, _, _, _, info = env.step(a, auto_reset=True)
        if info["episode_done"].any():
            break
    # After auto-reset (e.g., truncation), ring-buffer history should be NaN
    assert torch.any(torch.isnan(env.q_norm_sq_hist)) or \
           torch.any(torch.isnan(env.cos_angle_hist)) or \
           torch.any(torch.isnan(env.w_u_hist)), \
        f"some history buffers should be NaN after reset"
    # Next step's reward should be 0 (no delta contribution because of NaN sentinel)
    _, rew, _, _, _ = env.step(a, auto_reset=True)
    expected = torch.zeros_like(rew)
    assert torch.allclose(rew, expected, atol=1e-4), \
        f"first-step-after-reset reward should be 0 (alive=0, deltas=0); got {rew}"
    print(f"[ok] reset clears caches; first post-reset step delta = 0")


def test_baseline_k_dm_actually_adds_term():
    """Sanity preserved from previous reward design."""
    from Yuan.RL_controller.env.baseline_controller import GPMBaselineController
    from Yuan.RL_controller.env.env import build_task_aligned_basis
    env = _build_env(n_envs=4)
    B_basis, _ = build_task_aligned_basis(
        env.kin, env.q, env.line_dir, env.n_target,
        env.kin.q_mid, env.q_half, env.cfg.manip_damping,
    )
    ctrl_weak = GPMBaselineController(env.kin, k_jl=1.0, k_dm=0.0)
    ctrl_strong = GPMBaselineController(env.kin, k_jl=1.0, k_dm=1.0)
    a_weak = ctrl_weak.action(env.q, B_basis, u_hat=env.line_dir)
    a_strong = ctrl_strong.action(env.q, B_basis, u_hat=env.line_dir)
    diff = (a_strong - a_weak).norm(dim=-1)
    assert torch.all(diff > 1e-6), f"k_dm=1 should change action; diff={diff}"
    print(f"[ok] baseline k_dm adds non-trivial term")


if __name__ == "__main__":
    test_weights_normalize_to_one()
    test_first_step_delta_is_zero()
    test_progress_only_full_speed()
    test_telescoping_property()
    test_signed_delta_can_be_negative()
    test_reset_clears_prev_caches()
    test_baseline_k_dm_actually_adds_term()
    print("\n=== all tests passed ===")

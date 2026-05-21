"""Joint-space limit audit for a trained PPO policy.

Rolls out the policy deterministically (action = tanh(μ)) on the eval line set,
records q each step, finite-differences to q̇ and q̈, then compares per-joint
extremes to FR3 limits:

    q_lo / q_up   : env.lmt_lo / env.lmt_up                 (from kinematics)
    q̇_max         : (2.62, 2.62, 2.62, 2.62, 3.14, 3.14, 3.14)  rad/s — kin.qdot_max
    q̈_max         : (15, 15, 15, 15, 20, 20, 20)               rad/s² — FR3 datasheet

Run:
    python -m Yuan.RL_controller.tests.eval_kinematic_limits \\
        --ckpt /tmp/p0_snapshot_HHMM.pt \\
        --config Yuan/RL_controller/runs/p0_progress_only_30M_0520/config.yaml
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

import torch
import yaml

from Yuan.RL_controller.env.env import NSRLBatchedEnv, EnvConfig, TERM_NAMES
from Yuan.RL_controller.env.line_distribution import LineDistribution
from Yuan.RL_controller.ppo import Agent


# FR3 acceleration datasheet limits (rad/s²). Not stored in repo; surface here
# so the limit-check is transparent. Source: Franka Robotics FR3 datasheet.
QDDOT_MAX = (15.0, 15.0, 15.0, 15.0, 20.0, 20.0, 20.0)


def _build_env(cfg_path: str, n_envs: int, device: str, seed: int) -> NSRLBatchedEnv:
    with open(cfg_path) as f:
        y = yaml.safe_load(f)
    env_kwargs = dict(y["env"])
    env_kwargs["n_envs"] = n_envs
    cfg = EnvConfig(**env_kwargs)
    env = NSRLBatchedEnv(cfg, line_dist=None, device=device)
    line_cfg = y["line_distribution"]
    thr = (float(line_cfg["feasibility_threshold_m"])
           if line_cfg.get("feasibility_filter", False) else None)
    eval_seed = y.get("eval", {}).get("holdout_seed", 42)
    env.line_dist = LineDistribution.load_or_build(
        kin=env.kin, collision=env.collision,
        n_pool=line_cfg["n_pool"],
        n_target_noise_deg=line_cfg["n_target_noise_deg"],
        seed=eval_seed, env_cfg=cfg,
        feasibility_threshold_m=thr,
    )
    return env


@torch.no_grad()
def rollout_and_record(env: NSRLBatchedEnv, agent: Agent) -> dict:
    """Run one full eval pass (auto_reset=False) and record per-step q for
    each env. Returns (n_envs, T_max, 7) tensor of q, plus per-env ep lengths
    and term reasons."""
    n = env.n_envs
    T = env.max_steps
    q_seq = torch.full((n, T + 1, 7), float("nan"),
                       device=env.device, dtype=env.kin.dtype)
    ep_len = torch.full((n,), -1, dtype=torch.long, device=env.device)
    ep_term = torch.full((n,), -1, dtype=torch.long, device=env.device)
    finished = torch.zeros((n,), dtype=torch.bool, device=env.device)

    env.reset()
    q_seq[:, 0] = env.q
    for t in range(T):
        obs = env.current_obs()
        a = agent.actor_mean(obs)
        _, _, _, _, info = env.step(a, auto_reset=False)
        # store post-step q for envs that were active at start of this step
        active_now = ~env.done_persistent  # currently active (post-step)
        # write q_seq[:, t+1] = env.q for envs that took a step this round
        q_seq[:, t + 1] = env.q
        new_done = info["episode_done"]
        if new_done.any():
            ep_len[new_done] = env.t[new_done]
            ep_term[new_done] = info["term_reason"][new_done]
            finished = finished | new_done
        if bool(env.done_persistent.all().item()):
            break
    if (~finished).any():
        nd = ~finished
        ep_len[nd] = env.t[nd]
        ep_term[nd] = 5  # TRUNCATED

    return {"q_seq": q_seq, "ep_len": ep_len, "ep_term": ep_term}


def analyze(q_seq: torch.Tensor, ep_len: torch.Tensor,
            kin, dt: float) -> dict:
    """Per-joint max-abs / p99 / p50 for q, q̇, q̈ over all valid transitions."""
    n, Tplus1, d = q_seq.shape
    # Build a per-env validity mask: time t is valid if t ≤ ep_len[env].
    # q[t] is valid for t in [0, ep_len]; q̇[t] = (q[t+1]-q[t])/dt valid for
    # t in [0, ep_len-1]; q̈[t] = (q̇[t+1]-q̇[t])/dt valid for t in [0, ep_len-2].
    t_idx = torch.arange(Tplus1, device=q_seq.device).unsqueeze(0)  # (1, T+1)
    valid_q = t_idx <= ep_len.unsqueeze(-1)  # (n, T+1)
    # Finite diffs
    qdot = (q_seq[:, 1:] - q_seq[:, :-1]) / dt        # (n, T, 7)
    qddot = (qdot[:, 1:] - qdot[:, :-1]) / dt         # (n, T-1, 7)
    valid_qd = (t_idx[:, :-1] <= (ep_len - 1).unsqueeze(-1))   # (n, T)
    valid_qdd = (t_idx[:, :-2] <= (ep_len - 2).unsqueeze(-1))  # (n, T-1)
    # Mask NaN to zero magnitude and clip via .where to count only valid
    def collect(arr: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        a = arr.abs()
        # gather only valid entries per joint → list of (m, 7)
        flat = a.reshape(-1, 7)
        m = mask.reshape(-1)
        return flat[m]
    q_abs = collect(q_seq, valid_q)
    qd_abs = collect(qdot, valid_qd)
    qdd_abs = collect(qddot, valid_qdd)
    # For position, use signed values vs lo/up.
    q_signed = q_seq.reshape(-1, 7)[valid_q.reshape(-1)]
    return {
        "q_signed": q_signed,
        "qd_abs": qd_abs,
        "qdd_abs": qdd_abs,
    }


def _pct(x: torch.Tensor, q: float) -> float:
    return float(x.quantile(q).item())


def report(stats: dict, kin) -> None:
    q = stats["q_signed"].cpu()
    qd = stats["qd_abs"].cpu()
    qdd = stats["qdd_abs"].cpu()
    lo = kin.lmt_lo.cpu()
    up = kin.lmt_up.cpu()
    qdot_max = kin.qdot_max.cpu()
    qddot_max = torch.tensor(QDDOT_MAX)

    print(f"\n[limits] samples: n_q={q.shape[0]}  n_qdot={qd.shape[0]}  "
          f"n_qddot={qdd.shape[0]}")

    print(f"\n=== joint position vs limits ===")
    print(f"  {'joint':<6} {'q_lo':>8} {'min':>8} {'p1':>8} {'p99':>8} "
          f"{'max':>8} {'q_up':>8}  {'margin_lo':>9} {'margin_up':>9}  status")
    for j in range(7):
        col = q[:, j]
        mn, mx = col.min().item(), col.max().item()
        p1, p99 = _pct(col, 0.01), _pct(col, 0.99)
        m_lo = mn - lo[j].item()
        m_up = up[j].item() - mx
        bad = (m_lo < 0) or (m_up < 0)
        warn = (m_lo < 0.1) or (m_up < 0.1)
        tag = "VIOL" if bad else ("warn" if warn else "ok")
        print(f"  j{j+1:<5} {lo[j].item():>+8.3f} {mn:>+8.3f} {p1:>+8.3f} "
              f"{p99:>+8.3f} {mx:>+8.3f} {up[j].item():>+8.3f}  "
              f"{m_lo:>+9.3f} {m_up:>+9.3f}  {tag}")

    print(f"\n=== joint velocity |q̇| vs limit (rad/s) ===")
    print(f"  {'joint':<6} {'p50':>8} {'p99':>8} {'max':>8}  {'limit':>8}  "
          f"{'max/lim':>7}  status")
    for j in range(7):
        col = qd[:, j]
        p50 = _pct(col, 0.50)
        p99 = _pct(col, 0.99)
        mx = col.max().item()
        lim = qdot_max[j].item()
        ratio = mx / lim
        bad = ratio > 1.0
        warn = ratio > 0.7
        tag = "VIOL" if bad else ("warn" if warn else "ok")
        print(f"  j{j+1:<5} {p50:>8.3f} {p99:>8.3f} {mx:>8.3f}  "
              f"{lim:>8.3f}  {ratio:>6.1%}  {tag}")

    print(f"\n=== joint acceleration |q̈| vs limit (rad/s²) ===")
    print(f"  {'joint':<6} {'p50':>8} {'p99':>8} {'max':>8}  {'limit':>8}  "
          f"{'max/lim':>7}  status")
    for j in range(7):
        col = qdd[:, j]
        p50 = _pct(col, 0.50)
        p99 = _pct(col, 0.99)
        mx = col.max().item()
        lim = qddot_max[j].item()
        ratio = mx / lim
        bad = ratio > 1.0
        warn = ratio > 0.7
        tag = "VIOL" if bad else ("warn" if warn else "ok")
        print(f"  j{j+1:<5} {p50:>8.3f} {p99:>8.3f} {mx:>8.3f}  "
              f"{lim:>8.3f}  {ratio:>6.1%}  {tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-envs", type=int, default=200)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] ckpt={args.ckpt}")
    print(f"[eval] config={args.config}  device={device}  n_envs={args.n_envs}")

    env = _build_env(args.config, n_envs=args.n_envs, device=device, seed=0)
    agent = Agent(obs_dim=env.obs_dim, act_dim=env.act_dim,
                  hidden_dim=512, init_log_std=-1.0).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    agent.load_state_dict(state)
    agent.eval()

    print(f"[eval] dt={env.cfg.dt}, v={env.cfg.v}, max_steps={env.max_steps}")
    out = rollout_and_record(env, agent)
    ep_len = out["ep_len"].cpu()
    ep_term = out["ep_term"].cpu()
    print(f"[eval] ep_len: mean={ep_len.float().mean():.1f}  "
          f"median={ep_len.float().median():.0f}  "
          f"max={ep_len.max().item()}")
    term_counts = {TERM_NAMES.get(int(c), str(int(c))): int((ep_term == c).sum().item())
                   for c in ep_term.unique().tolist()}
    print(f"[eval] term breakdown: {term_counts}")

    stats = analyze(out["q_seq"], out["ep_len"], env.kin, env.cfg.dt)
    report(stats, env.kin)


if __name__ == "__main__":
    main()

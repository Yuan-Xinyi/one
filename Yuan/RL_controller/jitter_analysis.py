"""Quantify |q_dot| and |q_ddot| of a trained policy across the eval holdout.

Reuses `plot_joint_trajectories.rollout_record` so case ids match eval.py and
the cached `joint_tcp_cache*.npz` (if present) is hit without re-rolling.

`q_dot` and `q_ddot` are computed by finite differences on the q trajectories
recorded each env step, with steps past per-env `ep_len` masked out (env freezes
done envs, so the raw qs would otherwise drag a trail of zeros into RMS).

Output: a JSON with per-joint max + RMS of |q_dot| and |q_ddot|, per-joint
FR3-datasheet compliance for |q_ddot|, and aggregate stats. Use this on every
sweep model to compare against v8 directly.

Usage:
    python -m Yuan.RL_controller.jitter_analysis \\
        --config Yuan/RL_controller/runs/.../config.yaml \\
        --ckpt   Yuan/RL_controller/runs/.../agent.pt \\
        --out    Yuan/RL_controller/runs/.../qddot_stats.json
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
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from Yuan.RL_controller.env.env import NSRLBatchedEnv, EnvConfig
from Yuan.RL_controller.env.line_distribution import (
    LineDistribution, ScriptedLineDistribution,
)
from Yuan.RL_controller.ppo import Agent
from Yuan.RL_controller.plot_joint_trajectories import (
    rollout_record, _rl_action_fn,
)


# FR3 datasheet (rad/s^2) — accel limits, per-joint.
QDDOT_LIMIT = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0], dtype=np.float64)


def _load_rl_qs_from_cache(cache_path: Path):
    """Return (qs, ep_len) from joint_tcp_cache*.npz if it contains RL fields."""
    if not cache_path.exists():
        return None
    npz = np.load(cache_path)
    if "rl_qs" not in npz.files or "rl_ep_len" not in npz.files:
        return None
    return npz["rl_qs"], npz["rl_ep_len"]


def _rollout_rl(cfg_yaml, ckpt_path: Path, device, n_holdout: int):
    """Roll out RL policy only (no classical / GPM — we only need qs for jitter)."""
    env_cfg = EnvConfig(**{**cfg_yaml["env"], "n_envs": n_holdout})
    env = NSRLBatchedEnv(env_cfg, line_dist=None, device=device)

    line_cfg = cfg_yaml["line_distribution"]
    eval_cfg = cfg_yaml["eval"]
    threshold_m = (float(line_cfg["feasibility_threshold_m"])
                   if line_cfg.get("feasibility_filter", False) else None)
    sampler = LineDistribution.load_or_build(
        kin=env.kin, collision=env.collision,
        n_pool=line_cfg["n_pool"],
        n_target_noise_deg=line_cfg["n_target_noise_deg"],
        seed=eval_cfg["holdout_seed"],
        env_cfg=env_cfg,
        feasibility_threshold_m=threshold_m,
    )
    holdout = sampler.sample(n_holdout)
    env.line_dist = ScriptedLineDistribution(
        {k: v.clone() for k, v in holdout.items()})

    agent = Agent(env.obs_dim, env.act_dim,
                  hidden_dim=cfg_yaml["ppo"]["hidden_dim"],
                  init_log_std=cfg_yaml["ppo"]["init_log_std"]).to(device)
    agent.load_state_dict(torch.load(ckpt_path, map_location=device))
    agent.eval()
    qs, ep_len, _ = rollout_record(env, _rl_action_fn(agent), env_cfg.max_steps)
    return qs, ep_len


def _compute_stats(qs: np.ndarray, ep_len: np.ndarray, dt: float):
    """qs (T+1, N, 7), ep_len (N,) → flat valid q_dot / q_ddot samples.

    For env i with length L:
      qdot[0..L-1] valid (L samples)
      qddot[0..L-2] valid (L-1 samples; need ≥2 steps to define accel)
    Returns dict of np arrays (samples, 7) — flattened across envs.
    """
    T_plus, N, _ = qs.shape
    qdot = (qs[1:] - qs[:-1]) / dt          # (T, N, 7)
    qddot = (qdot[1:] - qdot[:-1]) / dt     # (T-1, N, 7)

    qdot_samples = []
    qddot_samples = []
    for i in range(N):
        L = int(ep_len[i])
        if L >= 1:
            qdot_samples.append(qdot[:L, i, :])
        if L >= 2:
            qddot_samples.append(qddot[:L - 1, i, :])
    return (np.concatenate(qdot_samples, axis=0) if qdot_samples
            else np.zeros((0, 7), dtype=np.float64),
            np.concatenate(qddot_samples, axis=0) if qddot_samples
            else np.zeros((0, 7), dtype=np.float64))


def _summarize(qdot_samples: np.ndarray, qddot_samples: np.ndarray,
               ep_len: np.ndarray) -> dict:
    """Build the JSON-serializable summary dict."""
    qd_abs = np.abs(qdot_samples.astype(np.float64))   # (M, 7)
    qdd_abs = np.abs(qddot_samples.astype(np.float64)) # (M', 7)

    per_joint_qdot_max = qd_abs.max(axis=0).tolist() if qd_abs.size else [0.0] * 7
    per_joint_qdot_rms = (np.sqrt((qdotsq := qd_abs ** 2).mean(axis=0)).tolist()
                          if qd_abs.size else [0.0] * 7)
    per_joint_qddot_max = qdd_abs.max(axis=0).tolist() if qdd_abs.size else [0.0] * 7
    per_joint_qddot_rms = (np.sqrt((qdd_abs ** 2).mean(axis=0)).tolist()
                           if qdd_abs.size else [0.0] * 7)

    # Datasheet compliance for q_ddot: fraction of (sample, joint) pairs where
    # |q_ddot| ≤ limit, both per-joint and overall.
    if qdd_abs.size:
        per_joint_compliance = (qdd_abs <= QDDOT_LIMIT).mean(axis=0).tolist()
        overall_compliance = float((qdd_abs <= QDDOT_LIMIT).all(axis=1).mean())
        # Per-joint peak compliance: is the worst-case sample within limit?
        per_joint_peak_within = (np.array(per_joint_qddot_max) <= QDDOT_LIMIT).tolist()
        overall_qddot_rms = float(np.sqrt((qdd_abs ** 2).mean()))
        overall_qddot_max = float(qdd_abs.max())
    else:
        per_joint_compliance = [1.0] * 7
        overall_compliance = 1.0
        per_joint_peak_within = [True] * 7
        overall_qddot_rms = 0.0
        overall_qddot_max = 0.0

    # Headline aggregate: arithmetic mean of per-joint RMS / per-joint peak across
    # the 7 joints. Preferred over flat (s,t,j) RMS because flat squaring amplifies
    # whichever joint happens to be loudest (q5/q7) and hides the per-joint picture.
    per_joint_qdot_rms_avg = (float(np.mean(per_joint_qdot_rms))
                              if qd_abs.size else 0.0)
    per_joint_qdot_peak_avg = (float(np.mean(per_joint_qdot_max))
                               if qd_abs.size else 0.0)
    per_joint_qddot_rms_avg = (float(np.mean(per_joint_qddot_rms))
                               if qdd_abs.size else 0.0)
    per_joint_qddot_peak_avg = (float(np.mean(per_joint_qddot_max))
                                if qdd_abs.size else 0.0)

    return {
        "n_episodes": int(len(ep_len)),
        "n_qdot_samples": int(qd_abs.shape[0]),
        "n_qddot_samples": int(qdd_abs.shape[0]),
        "ep_len_mean": float(ep_len.mean()),
        "ep_len_min": int(ep_len.min()),
        "ep_len_max": int(ep_len.max()),
        "qdot": {
            "per_joint_max": per_joint_qdot_max,
            "per_joint_rms": per_joint_qdot_rms,
            "per_joint_rms_avg": per_joint_qdot_rms_avg,
            "per_joint_peak_avg": per_joint_qdot_peak_avg,
            "overall_max": float(qd_abs.max()) if qd_abs.size else 0.0,
            "overall_rms": float(np.sqrt((qd_abs ** 2).mean())) if qd_abs.size else 0.0,
        },
        "qddot": {
            "per_joint_max": per_joint_qddot_max,
            "per_joint_rms": per_joint_qddot_rms,
            "per_joint_rms_avg": per_joint_qddot_rms_avg,
            "per_joint_peak_avg": per_joint_qddot_peak_avg,
            "overall_max": overall_qddot_max,
            "overall_rms": overall_qddot_rms,
            "datasheet_limit": QDDOT_LIMIT.tolist(),
            "per_joint_sample_compliance": per_joint_compliance,
            "per_joint_peak_within_limit": per_joint_peak_within,
            "overall_sample_compliance": overall_compliance,
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True,
                   help="Output JSON path (parent dirs created if needed)")
    p.add_argument("--device", default=None)
    p.add_argument("--cache", default=None,
                   help="Optional joint_tcp_cache*.npz to read RL qs from. "
                        "Default: <ckpt_dir>/joint_tcp_cache_final.npz then "
                        "<ckpt_dir>/joint_tcp_cache.npz.")
    p.add_argument("--force-rollout", action="store_true",
                   help="Ignore cache, always re-roll.")
    args = p.parse_args()

    device = (torch.device(args.device) if args.device is not None
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    with open(args.config, "r") as f:
        cfg_yaml = yaml.safe_load(f)
    dt = float(cfg_yaml["env"]["dt"])
    n_holdout = int(cfg_yaml["eval"]["n_holdout"])
    ckpt_path = Path(args.ckpt)

    qs = ep_len = None
    if not args.force_rollout:
        for cand in ([Path(args.cache)] if args.cache else
                     [ckpt_path.parent / "joint_tcp_cache_final.npz",
                      ckpt_path.parent / "joint_tcp_cache.npz"]):
            hit = _load_rl_qs_from_cache(cand)
            if hit is not None:
                qs, ep_len = hit
                print(f"[jitter] loaded RL qs from cache: {cand}")
                break
    if qs is None:
        print(f"[jitter] no cache hit, rolling RL on {n_holdout} cases "
              f"(dt={dt}, max_steps={cfg_yaml['env']['max_steps']})")
        qs, ep_len = _rollout_rl(cfg_yaml, ckpt_path, device, n_holdout)

    qdot_samples, qddot_samples = _compute_stats(qs, ep_len, dt)
    summary = _summarize(qdot_samples, qddot_samples, ep_len)
    summary["meta"] = {
        "config": str(Path(args.config).resolve()),
        "ckpt": str(ckpt_path.resolve()),
        "dt": dt,
        "a_max": float(cfg_yaml["env"]["a_max"]),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    qd = summary["qdot"]
    qdd = summary["qddot"]
    print(f"[jitter] ep_len mean={summary['ep_len_mean']:.2f} "
          f"(min={summary['ep_len_min']}, max={summary['ep_len_max']}, "
          f"n_eps={summary['n_episodes']})")
    print(f"[jitter] |q_dot|  per-joint RMS avg ={qd['per_joint_rms_avg']:7.3f}  "
          f"peak avg ={qd['per_joint_peak_avg']:7.3f} rad/s")
    print(f"[jitter] |q_ddot| per-joint RMS avg ={qdd['per_joint_rms_avg']:7.2f}  "
          f"peak avg ={qdd['per_joint_peak_avg']:7.2f} rad/s^2  ← headline")
    print(f"[jitter] per-joint |q_ddot| RMS: "
          + " ".join(f"{x:6.2f}" for x in qdd["per_joint_rms"]))
    print(f"[jitter] per-joint |q_ddot| max: "
          + " ".join(f"{x:6.2f}" for x in qdd["per_joint_max"]))
    print(f"[jitter] datasheet limit:        "
          + " ".join(f"{x:6.2f}" for x in qdd["datasheet_limit"]))
    print(f"[jitter] per-joint peak within:  "
          + " ".join(f"{'  OK  ' if w else 'OVER  '}"
                     for w in qdd["per_joint_peak_within_limit"]))
    print(f"[jitter] overall sample compliance: "
          f"{100*qdd['overall_sample_compliance']:.1f}%  "
          f"(flat RMS for ref: {qdd['overall_rms']:.2f})")
    print(f"[jitter] wrote → {out_path}")


if __name__ == "__main__":
    main()

"""Variant B: step-level state-conditional RL ↔ Classical switching.

Fresh rollouts (not pure post-processing): once the policy switches mid-episode
the trajectory diverges from the cached one, so we must roll forward with the
hybrid policy from t=0.

Per-env hysteresis:
    state: using_rl (bool)
    at t=0: using_rl = (max|q_norm(q_0)| < tau_enter)        # start-in-classical
    at each step (q_t = env.q, before this step's action):
        cur_max_qn = max(|q_norm(q_t)|)
        if using_rl and cur_max_qn >= tau_enter: switch to Classical, q_ref := q_t
        elif not using_rl and cur_max_qn < tau_exit: switch back to RL
        a_t = RL(obs_t) if using_rl else Classical(q_t, line_dir, n_target, q_ref)

We batch all (tau_enter, tau_exit) pairs into ONE big env by tiling each cell's
N=10000 tasks K times (K = number of pairs). All cells run in parallel with
identical task IDs, only the per-env (tau_enter, tau_exit) differs.

Usage:
    python -m Yuan.RL_controller.eval_hybrid_steplevel \\
        --ckpt-dir Yuan/RL_controller/runs/p0_progress_only_30M_0520 \\
        --cache  Yuan/RL_controller/runs/p0_progress_only_30M_0520/diag_10000_classical/rollouts.npz \\
        --out Yuan/RL_controller/runs/p0_progress_only_30M_0520/diag_10000_classical/hybrid_variantB.npz
"""
from __future__ import annotations

# Self-relaunch with $CONDA_PREFIX/lib on LD_LIBRARY_PATH (same as train/eval).
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
import collections
import csv
import dataclasses
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from Yuan.RL_controller.env.env import (
    NSRLBatchedEnv, EnvConfig, TERM_NAMES, TERM_TRUNCATED, TERM_ALIVE,
    build_task_aligned_basis,
)
from Yuan.RL_controller.env.line_distribution import ScriptedLineDistribution
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.RL_controller.ppo import Agent


TERM_ORDER = ["cone", "jl", "lateral", "collision", "truncated", "alive"]


def make_tau_pairs(enters, offsets):
    """Build the (tau_enter, tau_exit) list. tau_exit = tau_enter - offset."""
    pairs = []
    for te in enters:
        for off in offsets:
            tx = te - off
            pairs.append((float(te), float(tx)))
    return pairs


def run_hybrid_rollout(env, agent, classical_ctrl, tau_enter, tau_exit,
                       n_tasks, n_pairs, device):
    """One big rollout with n_tasks*n_pairs envs in parallel.

    Returns dict with episode_len, term_reason, switch_count, started_in_cls
    (each shape (n_pairs, n_tasks)).
    """
    N_total = n_tasks * n_pairs

    # Reset to known q0/line_dir/n_target (env.line_dist holds the tiled spec).
    env.reset()

    # Per-env state for hysteresis. Initially using_rl unless init max|qn| >= tau_enter.
    q_mid = env.q_mid
    q_half = env.q_half

    def _max_abs_qn(q):
        return ((q - q_mid).abs() / q_half).max(dim=-1).values

    init_max_qn = _max_abs_qn(env.q)
    using_rl = init_max_qn < tau_enter
    started_in_cls = ~using_rl
    # q_ref: last "safe" attractor. At t=0 this is q_0 (matches default Classical).
    q_ref = env.q.clone()
    switch_count = torch.zeros(N_total, dtype=torch.long, device=device)

    episode_len = torch.full((N_total,), -1, dtype=torch.long, device=device)
    episode_term = torch.full((N_total,), -1, dtype=torch.long, device=device)
    finished = torch.zeros((N_total,), dtype=torch.bool, device=device)

    max_steps = env.max_steps
    for step_i in range(max_steps):
        # Threshold check on env.q (q_t before this step).
        cur_max_qn = _max_abs_qn(env.q)
        new_using_rl = torch.where(
            using_rl,
            cur_max_qn < tau_enter,       # currently RL: stay if < tau_enter
            cur_max_qn < tau_exit,        # currently Cls: switch back if < tau_exit
        )
        switched = new_using_rl != using_rl
        # On RL → Classical, snapshot q_ref := env.q.
        rl_to_cls = using_rl & (~new_using_rl)
        if rl_to_cls.any():
            q_ref = torch.where(rl_to_cls.unsqueeze(-1), env.q, q_ref)
        # Count switches only on still-active envs.
        active = ~finished
        switch_count = switch_count + (switched & active).long()
        using_rl = new_using_rl

        # Build actions for both branches; pick per env.
        obs = env.current_obs()
        with torch.no_grad():
            rl_act = agent.actor_mean(obs).clamp(-1.0, 1.0)

        # Classical action: q_dot_null projected onto task-aligned basis.
        # (Mirror of cn_action_fn but with our per-env q_ref.)
        with torch.no_grad():
            B_basis, _ = build_task_aligned_basis(
                env.kin, env.q, env.line_dir, env.n_target,
                env.kin.q_mid, env.q_half, env.cfg.manip_damping,
            )
        q_dot_raw = classical_ctrl.q_dot_null(
            env.q, env.line_dir, env.n_target, q_ref)
        with torch.no_grad():
            cls_act = (B_basis.transpose(-1, -2)
                       @ q_dot_raw.unsqueeze(-1)).squeeze(-1)
            cls_act = (cls_act / env.a_max).clamp(-1.0, 1.0)

        action = torch.where(using_rl.unsqueeze(-1), rl_act, cls_act)
        _, _, _, _, info = env.step(action, auto_reset=False)

        new_done = info["episode_done"]
        if new_done.any():
            episode_len[new_done] = env.t[new_done]
            episode_term[new_done] = info["term_reason"][new_done]
            finished = finished | new_done
        if bool(env.done_persistent.all().item()):
            break

    if (~finished).any():
        not_done = ~finished
        episode_len[not_done] = env.t[not_done]
        episode_term[not_done] = TERM_TRUNCATED

    return {
        "episode_len": episode_len.view(n_pairs, n_tasks).cpu().numpy(),
        "term_reason": episode_term.view(n_pairs, n_tasks).cpu().numpy(),
        "switch_count": switch_count.view(n_pairs, n_tasks).cpu().numpy(),
        "started_in_cls": started_in_cls.view(n_pairs, n_tasks).cpu().numpy(),
        "init_max_qn": init_max_qn.view(n_pairs, n_tasks).cpu().numpy(),
    }


def summarize_cell(name, T, term, T_base, switches, started_in_cls, dt, v):
    N = T.shape[0]
    progress = T.astype(np.float64) * dt * v
    ratio = T.astype(np.float64) / np.maximum(T_base.astype(np.float64), 1.0)
    worse = int((T < T_base).sum())
    term_hist = collections.Counter(TERM_NAMES.get(int(t), "?") for t in term)
    term_frac = {k: 100.0 * term_hist.get(k, 0) / N for k in TERM_ORDER}
    return {
        "name": name,
        "N": N,
        "mean_progress_m": float(progress.mean()),
        "median_progress_m": float(np.median(progress)),
        "mean_ratio_vs_classical": float(ratio.mean()),
        "median_ratio_vs_classical": float(np.median(ratio)),
        "frac_hybrid_worse_than_classical": 100.0 * worse / N,
        "term_frac": term_frac,
        "mean_switches": float(switches.mean()),
        "frac_zero_switches": 100.0 * float((switches == 0).mean()),
        "frac_started_in_cls": 100.0 * float(started_in_cls.mean()),
    }


def fmt_header():
    head = [
        f"{'row':<26s}",
        f"{'meanP':>5}", f"{'medP':>5}",
        f"{'meanR':>5}", f"{'medR':>5}",
        f"{'wrs%':>5}",
    ]
    for k in TERM_ORDER:
        head.append(f"{k[:5]:>5}")
    head.extend([f"{'sw/ep':>5}", f"{'0sw%':>5}", f"{'inCls%':>6}"])
    return "  ".join(head)


def fmt_row(r):
    cells = [
        f"{r['name']:<26s}",
        f"{r['mean_progress_m']:>5.3f}",
        f"{r['median_progress_m']:>5.3f}",
        f"{r['mean_ratio_vs_classical']:>5.3f}",
        f"{r['median_ratio_vs_classical']:>5.3f}",
        f"{r['frac_hybrid_worse_than_classical']:>4.1f}%",
    ]
    for k in TERM_ORDER:
        cells.append(f"{r['term_frac'][k]:>5.1f}")
    cells.append(f"{r['mean_switches']:>5.2f}")
    cells.append(f"{r['frac_zero_switches']:>4.1f}%")
    cells.append(f"{r['frac_started_in_cls']:>5.1f}%")
    return "  ".join(cells)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--enters", nargs="+", type=float,
                        default=[0.80, 0.85, 0.88, 0.90])
    parser.add_argument("--offsets", nargs="+", type=float,
                        default=[0.0, 0.03, 0.05, 0.08])
    parser.add_argument("--out", default=None)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--pairs-per-chunk", type=int, default=4,
                        help="cusolverDn batched eigvalsh has a ~64k batch "
                             "limit; chunk so n_tasks * pairs_per_chunk fits")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    cache_path = Path(args.cache)
    cfg_path = ckpt_dir / "config.yaml"
    ckpt_path = ckpt_dir / "agent.pt"
    device = (torch.device(args.device) if args.device is not None
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    with open(cfg_path) as f:
        cfg_yaml = yaml.safe_load(f)

    # Load cache and reconstruct the holdout task set.
    d = np.load(cache_path, allow_pickle=True)
    q0_np = d["q0"]
    line_dir_np = d["line_dir"]
    n_target_np = d["n_target"]
    T_base = d["episode_len_base"].astype(np.int64)
    n_tasks = q0_np.shape[0]
    dt = float(d["dt"])
    v_const = float(cfg_yaml["env"].get("v", 0.2))

    pairs = make_tau_pairs(args.enters, args.offsets)
    n_pairs = len(pairs)
    print(f"[hybridB] N_tasks={n_tasks}  n_pairs={n_pairs}  "
          f"chunk={args.pairs_per_chunk}  device={device}")
    print(f"[hybridB] pairs (enter, exit):")
    for te, tx in pairs:
        print(f"    enter={te:.2f}  exit={tx:.2f}  hysteresis={te-tx:.2f}")

    # Split pairs into chunks to stay under cusolverDn batched eigvalsh limit.
    chunks = [pairs[i:i + args.pairs_per_chunk]
              for i in range(0, n_pairs, args.pairs_per_chunk)]

    all_out = {
        "episode_len": np.zeros((n_pairs, n_tasks), dtype=np.int64),
        "term_reason": np.zeros((n_pairs, n_tasks), dtype=np.int64),
        "switch_count": np.zeros((n_pairs, n_tasks), dtype=np.int64),
        "started_in_cls": np.zeros((n_pairs, n_tasks), dtype=bool),
        "init_max_qn": np.zeros((n_pairs, n_tasks), dtype=np.float32),
    }

    # Build agent once (independent of chunk size).
    valid_keys = {f.name for f in dataclasses.fields(EnvConfig)}
    env_kw = {k: v for k, v in cfg_yaml["env"].items() if k in valid_keys}

    pair_idx_offset = 0
    for ck, chunk_pairs in enumerate(chunks):
        n_in_chunk = len(chunk_pairs)
        N_total = n_tasks * n_in_chunk
        env_cfg = EnvConfig(**{**env_kw, "n_envs": N_total})

        tau_enter = torch.empty(N_total, device=device, dtype=torch.float32)
        tau_exit = torch.empty(N_total, device=device, dtype=torch.float32)
        for i, (te, tx) in enumerate(chunk_pairs):
            tau_enter[i * n_tasks:(i + 1) * n_tasks] = te
            tau_exit[i * n_tasks:(i + 1) * n_tasks] = tx

        q0_tiled = torch.from_numpy(np.tile(q0_np, (n_in_chunk, 1))).to(device)
        line_tiled = torch.from_numpy(np.tile(line_dir_np, (n_in_chunk, 1))).to(device)
        n_tiled = torch.from_numpy(np.tile(n_target_np, (n_in_chunk, 1))).to(device)

        env = NSRLBatchedEnv(env_cfg, line_dist=None, device=device)
        env.line_dist = ScriptedLineDistribution(
            {"q0": q0_tiled.to(env.kin.dtype),
             "line_dir": line_tiled.to(env.kin.dtype),
             "n_target": n_tiled.to(env.kin.dtype)})

        if ck == 0:
            agent = Agent(env.obs_dim, env.act_dim,
                          hidden_dim=cfg_yaml["ppo"]["hidden_dim"],
                          init_log_std=cfg_yaml["ppo"]["init_log_std"]).to(device)
            state_dict = torch.load(ckpt_path, map_location=device)
            agent.load_state_dict(state_dict)
            agent.eval()

            classical_ctrl = ClassicalNullspaceController(env.kin)
            print(f"[hybridB] classical gains: manip={classical_ctrl.manip_gain}, "
                  f"jl={classical_ctrl.jl_gain}, "
                  f"angle_b={classical_ctrl.angle_boundary_gain}, "
                  f"k_null={classical_ctrl.k_null}")

        print(f"[hybridB] chunk {ck+1}/{len(chunks)}: "
              f"{n_in_chunk} pairs, N_total={N_total}, running...")
        t0 = time.time()
        out = run_hybrid_rollout(env, agent, classical_ctrl, tau_enter, tau_exit,
                                 n_tasks, n_in_chunk, device)
        dt_run = time.time() - t0
        print(f"[hybridB]   chunk done in {dt_run:.1f}s")

        for i in range(n_in_chunk):
            idx = pair_idx_offset + i
            all_out["episode_len"][idx] = out["episode_len"][i]
            all_out["term_reason"][idx] = out["term_reason"][i]
            all_out["switch_count"][idx] = out["switch_count"][i]
            all_out["started_in_cls"][idx] = out["started_in_cls"][i]
            all_out["init_max_qn"][idx] = out["init_max_qn"][i]
        pair_idx_offset += n_in_chunk

        del env, q0_tiled, line_tiled, n_tiled, tau_enter, tau_exit
        torch.cuda.empty_cache()

    out = all_out
    imq = out["init_max_qn"]
    assert np.allclose(imq, imq[0:1]), "init_max_qn mismatch across pair-blocks"

    rows = []
    for i, (te, tx) in enumerate(pairs):
        name = f"varB_te={te:.2f}_tx={tx:.2f}"
        r = summarize_cell(
            name, out["episode_len"][i], out["term_reason"][i], T_base,
            out["switch_count"][i], out["started_in_cls"][i], dt, v_const)
        r["tau_enter"] = te
        r["tau_exit"] = tx
        rows.append(r)

    print()
    print("# Variant B sweep")
    print(fmt_header())
    for r in rows:
        line = fmt_row(r)
        chatter = " <-- CHATTERING" if r["mean_switches"] > 3.0 else ""
        print(line + chatter)

    best_b = max(rows, key=lambda r: r["mean_progress_m"])
    print()
    print(f"[hybridB] best variant B: {best_b['name']}  "
          f"mean_P={best_b['mean_progress_m']:.4f}  "
          f"meanR={best_b['mean_ratio_vs_classical']:.3f}  "
          f"worse={best_b['frac_hybrid_worse_than_classical']:.1f}%  "
          f"sw/ep={best_b['mean_switches']:.2f}")

    if args.out:
        np.savez_compressed(
            args.out,
            episode_len=out["episode_len"],
            term_reason=out["term_reason"],
            switch_count=out["switch_count"],
            started_in_cls=out["started_in_cls"],
            init_max_qn=out["init_max_qn"][0],
            tau_enter=np.array([p[0] for p in pairs]),
            tau_exit=np.array([p[1] for p in pairs]),
            T_base=T_base,
        )
        print(f"[hybridB] saved raw → {args.out}")

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["row", "tau_enter", "tau_exit", "mean_progress_m",
                        "median_progress_m", "mean_ratio_vs_cls",
                        "median_ratio_vs_cls", "worse_pct",
                        "mean_switches", "zero_sw_pct", "started_in_cls_pct"]
                       + [f"term_{k}_pct" for k in TERM_ORDER])
            for r in rows:
                w.writerow([r["name"], r["tau_enter"], r["tau_exit"],
                            f"{r['mean_progress_m']:.4f}",
                            f"{r['median_progress_m']:.4f}",
                            f"{r['mean_ratio_vs_classical']:.4f}",
                            f"{r['median_ratio_vs_classical']:.4f}",
                            f"{r['frac_hybrid_worse_than_classical']:.2f}",
                            f"{r['mean_switches']:.3f}",
                            f"{r['frac_zero_switches']:.2f}",
                            f"{r['frac_started_in_cls']:.2f}"]
                           + [f"{r['term_frac'][k]:.2f}" for k in TERM_ORDER])
        print(f"[hybridB] saved csv → {args.out_csv}")


if __name__ == "__main__":
    main()

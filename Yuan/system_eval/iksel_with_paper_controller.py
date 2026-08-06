"""What does the iksel selector deliver under the paper's own controller?

The iksel campaign scored its selector under a different controller
(`unified_rl/runs/r2_grouped_best`, tau 0.985/0.96), so its 94.7% is not on the
same scale as anything in the paper. This rolls the selector's picked seed, the
full 48-candidate cone-IK pool and the pilot fallback through the controller
the paper actually uses (`p0_progress_only_30M_0520`, hybrid tau 0.98/0.94,
k_lateral = 0, i.e. the submitted configuration), so the ratio

    picked / max over the pool

is measured end to end under one controller.

    python -m Yuan.system_eval.iksel_with_paper_controller --n-tasks 2000
"""
from __future__ import annotations

import os, sys
_conda_lib = os.path.join(sys.prefix, "lib")
if _conda_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    _e = dict(os.environ)
    _e["LD_LIBRARY_PATH"] = _conda_lib + ":" + _e.get("LD_LIBRARY_PATH", "")
    if __spec__ is not None and __spec__.name != "__main__":
        _argv = [sys.executable, "-m", __spec__.name] + sys.argv[1:]
    else:
        _argv = [sys.executable] + sys.argv
    os.execvpe(sys.executable, _argv, _e)

import argparse
import dataclasses
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]

from Yuan.RL_controller.env.env import NSRLBatchedEnv, EnvConfig, TERM_NAMES
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.RL_controller.algorithms.ppo import Agent
from Yuan.system_eval.curvature_scan import rollout_chunk

IKSEL = "Yuan/unified_rl/runs/iksel_final_n48/"
RL_CKPT = "Yuan/RL_controller/runs/p0_progress_only_30M_0520"
ENV_YAML = "Yuan/RL_controller/config.yaml"


def boot(x, n=2000, seed=0):
    g = np.random.default_rng(seed)
    m = np.array([np.mean(x[g.integers(0, len(x), len(x))]) for _ in range(n)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tasks", type=int, default=2000)
    ap.add_argument("--task-seed", type=int, default=0)
    ap.add_argument("--k-lateral", type=float, default=0.0)
    ap.add_argument("--tau-enter", type=float, default=0.98)
    ap.add_argument("--tau-exit", type=float, default=0.94)
    ap.add_argument("--controller", default="hybrid",
                    choices=["hybrid", "classical", "rl"])
    ap.add_argument("--chunk", type=int, default=1300)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = torch.device(args.device)
    t0 = time.time()

    cand = np.load(REPO / IKSEL / "iksel_eval10k_candidates.npz")
    pick = np.load(REPO / IKSEL / "_picked_seed_tmp.npz")
    assert np.abs(cand["p0"] - pick["p0"]).max() == 0, "task order mismatch"

    n_all = len(cand["p0"])
    rng = np.random.default_rng(args.task_seed)
    tasks = np.sort(rng.choice(n_all, size=min(args.n_tasks, n_all),
                               replace=False))
    T = len(tasks)
    p0 = cand["p0"][tasks]
    d0 = cand["line_dir"][tasks]
    nt = cand["n_target"][tasks]

    # column 0 = the selector's pick, 1 = the pilot fallback, 2.. = the pool
    seeds = np.concatenate([
        pick["seeds"][tasks],                       # (T, 1, 7)
        cand["q0_pilot"][tasks][:, None, :],        # (T, 1, 7)
        cand["seeds"][tasks],                       # (T, 48, 7)
    ], axis=1)
    valid = np.concatenate([
        pick["ik_ok"][tasks],
        np.ones((T, 1), bool),
        cand["ik_ok"][tasks],
    ], axis=1)
    S = seeds.shape[1]
    print(f"[iksel] {T} tasks x {S} seeds (1 picked + 1 pilot + 48 pool); "
          f"picked valid {100*valid[:, 0].mean():.1f}%, "
          f"pool valid {valid[:, 2:].sum(1).mean():.1f}/48")

    with open(REPO / ENV_YAML) as f:
        y = yaml.safe_load(f)
    keys = {f.name for f in dataclasses.fields(EnvConfig)}
    kw = {k: v for k, v in y["env"].items() if k in keys}
    env = NSRLBatchedEnv(EnvConfig(**{**kw, "n_envs": args.chunk,
                                      "k_lateral": args.k_lateral}), None, dev)
    classical = ClassicalNullspaceController(env.kin)
    with open(REPO / RL_CKPT / "config.yaml") as f:
        rc = yaml.safe_load(f)
    agent = Agent(env.obs_dim, env.act_dim, hidden_dim=rc["ppo"]["hidden_dim"],
                  init_log_std=rc["ppo"]["init_log_std"]).to(dev)
    agent.load_state_dict(torch.load(REPO / RL_CKPT / "agent.pt", map_location=dev))
    agent.eval()
    print(f"[iksel] controller = {args.controller} from {RL_CKPT}, "
          f"tau {args.tau_enter}/{args.tau_exit}, k_lateral {args.k_lateral}")

    dt = env.kin.dtype
    # substitute an invalid slot with the task's first valid seed so the batched
    # linear algebra stays conditioned; those columns are masked out afterwards
    sfix = seeds.copy()
    for i in range(T):
        v = np.nonzero(valid[i])[0]
        sfix[i, ~valid[i]] = seeds[i, v[0]]
    q = torch.as_tensor(sfix.reshape(-1, 7), device=dev, dtype=dt)
    P = torch.as_tensor(np.repeat(p0, S, 0), device=dev, dtype=dt)
    D = torch.as_tensor(np.repeat(d0, S, 0), device=dev, dtype=dt)
    N_ = torch.as_tensor(np.repeat(nt, S, 0), device=dev, dtype=dt)
    kap = torch.zeros(args.chunk, device=dev, dtype=dt)
    N = T * S
    arc = np.zeros(N, np.float32)
    term = np.zeros(N, np.int8)
    nch = math.ceil(N / args.chunk)
    for c in range(nch):
        lo, hi = c * args.chunk, min((c + 1) * args.chunk, N)
        pad = args.chunk - (hi - lo)
        sl = lambda x: (torch.cat([x[lo:hi], x[hi - 1:hi].expand(pad, *x.shape[1:])])
                        if pad else x[lo:hi])
        r = rollout_chunk(env, args.controller, q0=sl(q), p0=sl(P), d0=sl(D),
                          n_target=sl(N_), kappa=kap, classical=classical,
                          agent=agent, tau_enter=args.tau_enter,
                          tau_exit=args.tau_exit)
        arc[lo:hi] = r["arc_m"][:hi - lo].float().cpu().numpy()
        term[lo:hi] = r["term_reason"][:hi - lo].cpu().numpy()
        if c % 10 == 0 or c == nch - 1:
            el = time.time() - t0
            print(f"[iksel]   chunk {c+1}/{nch}  {el:.0f}s  "
                  f"eta {el/(c+1)*(nch-c-1):.0f}s", flush=True)

    A = arc.reshape(T, S)
    A = np.where(valid, A, np.nan)
    picked = A[:, 0]
    pilot = A[:, 1]
    pool = A[:, 2:]
    ref = np.nanmax(np.concatenate([A[:, 1:2], pool], 1), axis=1)
    typ = np.nanmedian(pool, axis=1)
    keep = np.isfinite(picked) & np.isfinite(ref) & (ref > 0.02)
    print(f"\n[iksel] ==== {keep.sum()} tasks, controller = {args.controller} ====")
    for name, v in (("pool oracle (reference)", ref), ("iksel picked", picked),
                    ("pool median (uninformed)", typ), ("q_jl pilot", pilot)):
        vv = v[keep]
        print(f"  {name:<26} mean {np.nanmean(vv):.4f} m   "
              f"ratio to reference {100*np.nanmean(vv/ref[keep]):.2f}%")
    cap = ((np.nanmean(picked[keep]) - np.nanmean(typ[keep]))
           / max(np.nanmean(ref[keep]) - np.nanmean(typ[keep]), 1e-9))
    lo, hi = boot((picked / ref)[keep] * 100)
    print(f"\n  picked / reference : mean {100*np.nanmean((picked/ref)[keep]):.2f}%"
          f"   95% CI [{lo:.2f}, {hi:.2f}]")
    print(f"  capture over the uninformed pick : {100*cap:.1f}%")
    tt = term.reshape(T, S)[:, 0][keep]
    print("  termination of the picked seed: " + "  ".join(
        f"{TERM_NAMES.get(int(c), '?')} {100*np.mean(tt == c):.0f}%"
        for c in np.unique(tt)))
    if args.out:
        np.savez_compressed(REPO / args.out, arc=A, valid=valid, tasks=tasks,
                            term=term.reshape(T, S), controller=args.controller)
        print(f"[iksel] saved -> {args.out}")


if __name__ == "__main__":
    main()

"""Zero-shot curvature scan — decisive test for "is the workspace envelope the
binding constraint?".

Nothing is retrained. Straight-line-trained seeds and the straight-line-trained
RL policy are replayed on constant-curvature arcs that share the task's origin
p0, initial tangent d and plane normal n; only the curvature kappa changes.
The 31-D observation is unchanged, because the environment feeds the policy the
*instantaneous* tangent, which on a straight ray is what it always fed.

One sweep, three readings — all of them ratios:

  capacity      ell_ref(kappa) / ell_ref(0) per task, where ell_ref(kappa) is
                the largest arc length any seed in a curvature-agnostic
                candidate pool reaches at that curvature. This is a property
                of the task, not of a policy. If it grows strongly with
                |kappa|, the spatial extent of the path is the real
                constraint; if it is flat or falls, the joint limits and the
                tool cone are.

  controller    arc length of a fixed seed under a given controller, divided
                by ell_ref(kappa) of the same task at the same kappa.

  seed          arc length of the seed that was best at kappa = 0 -- i.e. what
                a perfect straight-line seed selector would pick -- divided by
                ell_ref(kappa). This measures how much of the loss is the seed
                module's, without depending on the diffusion checkpoint.

Ratios are only ever formed inside one kappa, so comparing across kappa is a
comparison of ratios.

The candidate pool is the cone-IK enumeration of the IK-pool line of work
(48 seeds/task, built from p0, n and the cone alone, then de-duplicated), so
its construction carries no straight-line bias.

Usage:
    python -m Yuan.system_eval.curvature_scan --n-tasks 400 --n-ik 16 \
        --out Yuan/system_eval/runs/curvature_scan/scan.npz
"""
from __future__ import annotations

# Self-relaunch with $CONDA_PREFIX/lib on LD_LIBRARY_PATH (same as eval.hybrid).
import os, sys
_conda_lib = os.path.join(sys.prefix, "lib")
if _conda_lib not in os.environ.get("LD_LIBRARY_PATH", ""):
    _new_env = dict(os.environ)
    _new_env["LD_LIBRARY_PATH"] = _conda_lib + ":" + _new_env.get("LD_LIBRARY_PATH", "")
    if __spec__ is not None and __spec__.name != "__main__":
        _argv = [sys.executable, "-m", __spec__.name] + sys.argv[1:]
    else:
        _argv = [sys.executable] + sys.argv
    os.execvpe(sys.executable, _argv, _new_env)

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from Yuan.RL_controller.env.env import (
    NSRLBatchedEnv, EnvConfig, TERM_TRUNCATED, build_task_aligned_basis,
)
from Yuan.RL_controller.env.classical_nullspace import ClassicalNullspaceController
from Yuan.RL_controller.env.line_distribution import ScriptedLineDistribution
from Yuan.RL_controller.algorithms.ppo import Agent


REPO = Path(__file__).resolve().parents[2]

DEF_CANDIDATES = "Yuan/unified_rl/runs/iksel_final_n48/iksel_eval10k_candidates.npz"
DEF_EVALSET = "Yuan/system_eval/runs/eval_10k_systematic/eval_set_10k.npz"
DEF_RL_CKPT = "Yuan/RL_controller/runs/p0_progress_only_30M_0520"
DEF_ENV_YAML = "Yuan/RL_controller/config.yaml"

# Seed index 0 of every task is the pilot seed q_jl; 1.. are the IK pool.
SRC_PILOT, SRC_IK = 0, 2


# ----------------------------------------------------------------- rollout

@torch.no_grad()
def rollout_chunk(env: NSRLBatchedEnv, mode: str, *,
                  q0, p0, d0, n_target, kappa,
                  classical: ClassicalNullspaceController,
                  agent: Agent | None,
                  tau_enter: float, tau_exit: float) -> dict:
    """One batched rollout of env.n_envs episodes. mode: classical|rl|hybrid."""
    n = env.n_envs
    env.line_dist = ScriptedLineDistribution(
        {"q0": q0, "line_dir": d0, "n_target": n_target,
         "p0": p0, "kappa": kappa})
    env.reset()

    q_mid, q_half = env.q_mid, env.q_half

    def _max_abs_qn(q):
        return ((q - q_mid).abs() / q_half).max(dim=-1).values

    using_rl = _max_abs_qn(env.q) < tau_enter
    switch_count = torch.zeros(n, dtype=torch.long, device=env.device)
    ep_len = torch.full((n,), -1, dtype=torch.long, device=env.device)
    term = torch.full((n,), -1, dtype=torch.long, device=env.device)
    finished = torch.zeros(n, dtype=torch.bool, device=env.device)
    lateral_max = 0.0

    for _ in range(env.max_steps + 1):
        if mode == "hybrid":
            cur_qn = _max_abs_qn(env.q)
            new_using_rl = torch.where(using_rl, cur_qn < tau_enter,
                                       cur_qn < tau_exit)
            switch_count += ((new_using_rl != using_rl) & ~finished).long()
            using_rl = new_using_rl

        # current_obs() also refreshes env.line_dir to the tangent at the
        # current TCP, so the classical branch below sees the right direction.
        obs = env.current_obs()

        rl_act = cls_act = None
        if mode in ("rl", "hybrid"):
            rl_act = agent.actor_mean(obs).clamp(-1.0, 1.0)
        if mode in ("classical", "hybrid"):
            B_basis, _ = build_task_aligned_basis(
                env.kin, env.q, env.line_dir, env.n_target,
                env.kin.q_mid, env.q_half, env.cfg.manip_damping)
            q_dot = classical.q_dot_null(env.q, env.line_dir, env.n_target)
            cls_act = (B_basis.transpose(-1, -2) @ q_dot.unsqueeze(-1)).squeeze(-1)
            cls_act = (cls_act / env.a_max).clamp(-1.0, 1.0)

        if mode == "rl":
            a = rl_act
        elif mode == "classical":
            a = cls_act
        else:
            a = torch.where(using_rl.unsqueeze(-1), rl_act, cls_act)

        _, _, _, _, info = env.step(a, auto_reset=False)
        lateral_max = max(lateral_max, info["lateral_err_max"])
        new_done = info["episode_done"]
        if bool(new_done.any().item()):
            ep_len[new_done] = env.t[new_done]
            term[new_done] = info["term_reason"][new_done]
            finished |= new_done
        if bool(env.done_persistent.all().item()):
            break

    if (~finished).any():
        nd = ~finished
        ep_len[nd] = env.t[nd]
        term[nd] = TERM_TRUNCATED

    return {
        "arc_m": env.arc_progress.clone(),
        "episode_len": ep_len,
        "term_reason": term,
        "switch_count": switch_count,
        "lateral_max": lateral_max,
    }


@torch.no_grad()
def seed_validity(env, q, p0, n_target) -> tuple[np.ndarray, np.ndarray]:
    """Collision-free, inside the tool cone, and FK(q) close to the task p0."""
    p, R, _, _ = env.kin.tcp_fk_jac(q)
    coll = env.collision.is_collided(env.kin.link_transforms(q))
    cos_a = (R[:, :, 2] * n_target).sum(-1)
    ok = (~coll) & (cos_a >= env.cos_cone) & ((p - p0).norm(dim=-1) < 5e-3)
    return ok.cpu().numpy(), (p - p0).norm(dim=-1).cpu().numpy()


# ----------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=DEF_CANDIDATES)
    ap.add_argument("--eval-set", default=DEF_EVALSET)
    ap.add_argument("--rl-ckpt", default=DEF_RL_CKPT)
    ap.add_argument("--env-yaml", default=DEF_ENV_YAML)
    ap.add_argument("--n-tasks", type=int, default=400)
    ap.add_argument("--n-ik", type=int, default=16,
                    help="IK-pool candidates kept per task (reference pool)")
    ap.add_argument("--task-seed", type=int, default=0)
    ap.add_argument("--kappas", nargs="+", type=float,
                    default=[0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0])
    ap.add_argument("--controllers", nargs="+", default=["hybrid", "classical"],
                    choices=["hybrid", "classical", "rl"])
    ap.add_argument("--k-lateral", type=float, default=5.0)
    ap.add_argument("--tau-enter", type=float, default=0.98)
    ap.add_argument("--tau-exit", type=float, default=0.94)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bench", action="store_true",
                    help="run one chunk per controller, report timing, exit")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)

    cand = np.load(REPO / args.candidates)
    evalset = np.load(REPO / args.eval_set, allow_pickle=True)
    tid = cand["task_indices"].astype(np.int64)
    assert np.abs(cand["p0"] - evalset["cs_p0"][tid]).max() == 0, \
        "candidate file is not aligned with this eval set"

    n_ok = cand["ik_ok"].sum(1)
    eligible = tid[n_ok >= args.n_ik]
    rng = np.random.default_rng(args.task_seed)
    tasks = np.sort(rng.choice(eligible, size=min(args.n_tasks, len(eligible)),
                               replace=False))
    n_tasks = len(tasks)
    row_of = np.full(int(tid.max()) + 1, -1, np.int64)
    row_of[tid] = np.arange(len(tid))
    rows = row_of[tasks]
    print(f"[scan] tasks with >= {args.n_ik} IK candidates: {len(eligible)}"
          f" -> sampled {n_tasks}")

    p0 = evalset["cs_p0"][tasks].astype(np.float64)
    d0 = evalset["cs_line_dir"][tasks].astype(np.float64)
    nt = evalset["cs_n_target"][tasks].astype(np.float64)
    bucket = evalset["bucket"][tasks]

    # ---- per-task seed list: pilot seed first, then the IK pool ----------
    K = args.n_ik
    ik_seeds, ik_ok = cand["seeds"][rows], cand["ik_ok"][rows]
    sel = np.zeros((n_tasks, K, 7), np.float32)
    for i in range(n_tasks):
        idx = np.nonzero(ik_ok[i])[0][:K]
        sel[i] = ik_seeds[i, idx]
    seeds = np.concatenate([cand["q0_pilot"][rows][:, None, :], sel], axis=1)
    src = np.concatenate([np.full(1, SRC_PILOT), np.full(K, SRC_IK)]).astype(np.int8)
    S = seeds.shape[1]

    # ---- env ------------------------------------------------------------
    with open(REPO / args.env_yaml) as f:
        env_yaml = yaml.safe_load(f)
    keys = {f.name for f in dataclasses.fields(EnvConfig)}
    env_kw = {k: v for k, v in env_yaml["env"].items() if k in keys}
    n_envs = min(args.chunk, n_tasks * S)
    env = NSRLBatchedEnv(
        EnvConfig(**{**env_kw, "n_envs": n_envs, "k_lateral": args.k_lateral}),
        line_dist=None, device=device)
    classical = ClassicalNullspaceController(env.kin)

    with open(REPO / args.rl_ckpt / "config.yaml") as f:
        rl_cfg = yaml.safe_load(f)
    agent = Agent(env.obs_dim, env.act_dim,
                  hidden_dim=rl_cfg["ppo"]["hidden_dim"],
                  init_log_std=rl_cfg["ppo"]["init_log_std"]).to(device)
    agent.load_state_dict(torch.load(REPO / args.rl_ckpt / "agent.pt",
                                     map_location=device))
    agent.eval()
    print(f"[scan] env: v={env.v} dt={env.dt} max_steps={env.max_steps} "
          f"a_max={env.a_max} k_lateral={args.k_lateral} | classical gains "
          f"{classical.manip_gain}/{classical.jl_gain}/"
          f"{classical.angle_boundary_gain} | tau {args.tau_enter}/{args.tau_exit}")

    # ---- flatten (task, seed) ------------------------------------------
    dt_t = env.kin.dtype
    flat_q = torch.as_tensor(seeds.reshape(-1, 7), device=device, dtype=dt_t)
    flat_p0 = torch.as_tensor(np.repeat(p0, S, 0), device=device, dtype=dt_t)
    flat_d0 = torch.as_tensor(np.repeat(d0, S, 0), device=device, dtype=dt_t)
    flat_nt = torch.as_tensor(np.repeat(nt, S, 0), device=device, dtype=dt_t)
    N = flat_q.shape[0]

    valid = np.zeros(N, bool)
    fk_err = np.zeros(N, np.float32)
    for lo in range(0, N, 4096):
        hi = min(lo + 4096, N)
        valid[lo:hi], fk_err[lo:hi] = seed_validity(
            env, flat_q[lo:hi], flat_p0[lo:hi], flat_nt[lo:hi])
    valid = valid.reshape(n_tasks, S)
    print(f"[scan] seeds/task = {S} (1 pilot + {K} IK-pool); start-state valid "
          f"{valid.mean()*100:.1f}% (pilot {valid[:, 0].mean()*100:.1f}%, "
          f"pool {valid[:, 1:].mean()*100:.1f}%); "
          f"median |FK(q)-p0| = {np.median(fk_err)*1e3:.2f} mm")

    kappas, ctrls = list(args.kappas), list(args.controllers)
    shape = (len(kappas), len(ctrls), n_tasks * S)
    arc = np.zeros(shape, np.float32)
    steps = np.zeros(shape, np.int32)
    term = np.zeros(shape, np.int8)
    swit = np.zeros(shape, np.int16)

    n_chunks = math.ceil(N / n_envs)
    total = len(kappas) * len(ctrls) * n_chunks
    print(f"[scan] {N} rollouts/cell x {len(kappas)} kappa x {len(ctrls)} "
          f"controllers = {total} chunks of {n_envs}", flush=True)

    done, t0 = 0, time.time()
    lat_worst = 0.0
    for ki, kap in enumerate(kappas):
        kap_t = torch.full((n_envs,), float(kap), device=device, dtype=dt_t)
        for ci, mode in enumerate(ctrls):
            for c in range(n_chunks):
                lo, hi = c * n_envs, min((c + 1) * n_envs, N)
                pad = n_envs - (hi - lo)

                def _sl(x):
                    s = x[lo:hi]
                    return torch.cat(
                        [s, x[hi - 1:hi].expand(pad, *x.shape[1:])]) if pad else s

                r = rollout_chunk(
                    env, mode, q0=_sl(flat_q), p0=_sl(flat_p0), d0=_sl(flat_d0),
                    n_target=_sl(flat_nt), kappa=kap_t, classical=classical,
                    agent=agent, tau_enter=args.tau_enter, tau_exit=args.tau_exit)
                w = hi - lo
                arc[ki, ci, lo:hi] = r["arc_m"][:w].float().cpu().numpy()
                steps[ki, ci, lo:hi] = r["episode_len"][:w].cpu().numpy()
                term[ki, ci, lo:hi] = r["term_reason"][:w].cpu().numpy()
                swit[ki, ci, lo:hi] = r["switch_count"][:w].cpu().numpy()
                lat_worst = max(lat_worst, r["lateral_max"])
                done += 1
                el = time.time() - t0
                print(f"[scan] kappa={kap:+.2f} {mode} chunk {c+1}/{n_chunks} "
                      f"| {done}/{total}, {el:.0f}s elapsed, eta "
                      f"{el/done*(total-done):.0f}s, worst lateral "
                      f"{lat_worst*1e3:.2f} mm", flush=True)
                if args.bench:
                    break
            if args.bench:
                break
        if args.bench:
            break

    if args.bench:
        print(f"[bench] one chunk of {n_envs} took "
              f"{(time.time()-t0):.1f}s -> full sweep "
              f"{(time.time()-t0)*total:.0f}s ({(time.time()-t0)*total/60:.1f} min)")
        return

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        arc_m=arc.reshape(len(kappas), len(ctrls), n_tasks, S),
        episode_len=steps.reshape(len(kappas), len(ctrls), n_tasks, S),
        term_reason=term.reshape(len(kappas), len(ctrls), n_tasks, S),
        switch_count=swit.reshape(len(kappas), len(ctrls), n_tasks, S),
        kappas=np.array(kappas, np.float32), controllers=np.array(ctrls),
        seed_src=src, seed_valid=valid, task_rows=tasks, bucket=bucket,
        p0=p0.astype(np.float32), line_dir=d0.astype(np.float32),
        n_target=nt.astype(np.float32), lateral_max=np.float32(lat_worst),
        config=json.dumps(vars(args)))
    print(f"[scan] saved -> {out}  ({time.time()-t0:.0f}s, "
          f"worst lateral {lat_worst*1e3:.2f} mm)")


if __name__ == "__main__":
    main()

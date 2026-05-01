"""Benchmark end-to-end policy inference latency.

Measures wall time for the full pipeline:
    task dict  ->  state vector  ->  policy.act  ->  IK projection  ->  q (7-DOF)

Reports per-stage and total at three batch sizes:
    N=1     single-task deployment latency (what real-time control sees)
    N=32    typical eval batch
    N=128   larger batch (training-time inference)

Each measurement is the average of `--repeats` runs after a warmup.

Usage:
    python -m Yuan.RL.bench_policy_latency
    python -m Yuan.RL.bench_policy_latency --repeats 100
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
import torch

import Yuan.RL.config as cfg
from Yuan.RL.env import FarsightedSeedEnv
from Yuan.RL.policy import make_policy
from Yuan.RL.batched_rollout import (
    branch_project_multistart, build_branch_rotmat_batch, _device_from_cfg,
)
from Yuan.RL.batched_fr3_kin import BatchedFR3Kinematics


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _load_policy(ckpt_path, env, device):
    q_mid = torch.as_tensor(env.action_mid, dtype=torch.float32, device=device)
    q_half = torch.as_tensor(env.action_half, dtype=torch.float32, device=device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy = make_policy(cfg.STATE_DIM, env.action_dim, q_mid, q_half,
                         policy_type=state.get("policy_type", "gaussian")).to(device)
    policy.load_state_dict(state["policy"])
    policy.eval()
    return policy


def _bench_once(policy, env, kin, tasks, device, device_kin):
    """Run the full pipeline once on a fixed list of tasks. Returns
    (total_ms, dict of per-stage ms)."""
    n = len(tasks)
    stages = {}

    _sync(); t0 = time.perf_counter()
    states = np.stack([env._state_vec(t) for t in tasks], axis=0).astype(np.float32)
    c_np = np.stack([t["c"] for t in tasks], axis=0).astype(np.float32)
    _sync(); t1 = time.perf_counter()
    stages["state_build"] = (t1 - t0) * 1000

    states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
    c_t = torch.as_tensor(c_np, dtype=torch.float32, device=device_kin)
    _sync(); t2 = time.perf_counter()
    stages["host_to_dev"] = (t2 - t1) * 1000

    with torch.no_grad():
        a, _ = policy.act(states_t, deterministic=True)
    _sync(); t3 = time.perf_counter()
    stages["policy_forward"] = (t3 - t2) * 1000

    a_t = a.to(device_kin) if device_kin != device else a
    p0_t = c_t[:, :3]; d_t = c_t[:, 3:6]; n_t = c_t[:, 6:9]
    R_tgt = build_branch_rotmat_batch(d_t, n_t, a_t)
    _sync(); t4 = time.perf_counter()
    stages["build_R_tgt"] = (t4 - t3) * 1000

    q_best, ik_ok, _ = branch_project_multistart(kin, p0_t, R_tgt, a_t)
    _sync(); t5 = time.perf_counter()
    stages["ik_project"] = (t5 - t4) * 1000

    q_np = q_best.cpu().numpy()
    _sync(); t6 = time.perf_counter()
    stages["dev_to_host"] = (t6 - t5) * 1000

    total_ms = (t6 - t0) * 1000
    return total_ms, stages, q_np, int(ik_ok.sum().item())


def _bench_batch(policy, env, kin, n_tasks, device, device_kin,
                 repeats: int, warmup: int, seed: int):
    rng_seed = seed
    # warmup
    for _ in range(warmup):
        env.rng = np.random.default_rng(rng_seed)
        tasks = env._sample_tasks(n_tasks)
        _bench_once(policy, env, kin, tasks, device, device_kin)
        rng_seed += 1

    totals = []
    stage_acc: dict[str, list[float]] = {}
    last_ok = 0
    for _ in range(repeats):
        env.rng = np.random.default_rng(rng_seed)
        tasks = env._sample_tasks(n_tasks)
        tot, stages, _, ok = _bench_once(policy, env, kin, tasks, device, device_kin)
        totals.append(tot)
        for k, v in stages.items():
            stage_acc.setdefault(k, []).append(v)
        last_ok = ok
        rng_seed += 1

    return {
        "n": n_tasks,
        "total_ms_mean": float(np.mean(totals)),
        "total_ms_std":  float(np.std(totals)),
        "total_ms_p50":  float(np.percentile(totals, 50)),
        "total_ms_p90":  float(np.percentile(totals, 90)),
        "stage_means":   {k: float(np.mean(v)) for k, v in stage_acc.items()},
        "ik_ok_count":   last_ok,
    }


def _print_report(results):
    print()
    print(f"  {'batch N':>8}  {'mean (ms)':>11}  {'std':>7}  "
          f"{'p50':>7}  {'p90':>7}  {'per-task (ms)':>15}")
    for r in results:
        per_task = r["total_ms_mean"] / r["n"]
        print(f"  {r['n']:>8d}  {r['total_ms_mean']:>11.3f}  "
              f"{r['total_ms_std']:>7.3f}  {r['total_ms_p50']:>7.3f}  "
              f"{r['total_ms_p90']:>7.3f}  {per_task:>15.3f}")
    print()
    # stage breakdown for each batch size
    for r in results:
        print(f"  --- N={r['n']:>3d} per-stage means (ms) ---")
        total = sum(r["stage_means"].values())
        for stage in ("state_build", "host_to_dev", "policy_forward",
                      "build_R_tgt", "ik_project", "dev_to_host"):
            v = r["stage_means"].get(stage, 0.0)
            frac = v / total if total > 0 else 0.0
            print(f"      {stage:>16}  {v:>8.3f}   {frac:>5.1%}")
        print(f"      {'sum-of-stages':>16}  {total:>8.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str,
                    default=os.path.join(cfg.CKPT_DIR, "ckpt_005000.pt"))
    ap.add_argument("--batches", type=str, default="1,32,128",
                    help="comma-separated batch sizes")
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_kin = _device_from_cfg()
    print(f"device (policy): {device}   device (kin): {device_kin}")
    env = FarsightedSeedEnv(seed=args.seed, randomize=True, use_collision=False)
    print(f"loading {args.ckpt}")
    policy = _load_policy(args.ckpt, env, device)
    kin = BatchedFR3Kinematics(device=device_kin)
    print(f"BRANCH_IK_NUM_STARTS = {cfg.BRANCH_IK_NUM_STARTS}    "
          f"warmup={args.warmup}  repeats={args.repeats}")

    batches = [int(b) for b in args.batches.split(",") if b.strip()]
    results = []
    for n in batches:
        r = _bench_batch(policy, env, kin, n, device, device_kin,
                         args.repeats, args.warmup, args.seed)
        results.append(r)
    _print_report(results)


if __name__ == "__main__":
    main()
